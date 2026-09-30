"""vLLM framework adapter for affinity selection and native delegation."""

from __future__ import annotations

import inspect
import logging
import os
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterator

from kunpeng_affinity.adapters.vllm_commit import (
    commit_vllm_nodes,
    release_invalid_vllm_transaction,
)
from kunpeng_affinity.adapters.vllm_lifecycle import VllmAffinityAdapter
from kunpeng_affinity.adapters.vllm_resolution import (
    resolve_automatic_nodes,
    revalidate_visibility,
)
from kunpeng_affinity.adapters.vllm_revalidation import (
    validate_inherited_vllm_transaction,
)
from kunpeng_affinity.config import (
    CpuPolicy,
    PluginMode,
    load_plugin_config,
    load_plugin_mode,
)
from kunpeng_affinity.core.errors import AffinityIntegrationError

logger = logging.getLogger(__name__)

_HOOK_MARKER = "__kunpeng_affinity_original__"
_REQUIRED_PARAMETERS = {
    "vllm_config",
    "local_rank",
    "dp_local_rank",
    "process_kind",
}
_TARGET_VERSION = "0.23.0"
_COMPATIBILITY_VERSION = "0.25.1"
_AUXILIARY_DEMO_VERSION = "0.26.0"
_FORCE_GENERIC_ENV = "KUNPENG_AFFINITY_VLLM_FORCE_GENERIC"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_SUPPORTED_BASE_VERSIONS = frozenset({_TARGET_VERSION, _COMPATIBILITY_VERSION, _AUXILIARY_DEMO_VERSION})


def _vllm_version() -> str:
    try:
        return version("vllm")
    except PackageNotFoundError:
        return "unknown"


def _is_supported_vllm_version(detected_version: str) -> bool:
    """Accept validated upstream versions and vendor-local builds of them."""
    if detected_version in _SUPPORTED_BASE_VERSIONS:
        return True
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        # Source-only checks may intentionally run without installing package
        # dependencies. Keep the fallback limited to the local-version form.
        return any(
            detected_version.startswith(f"{base}+") and bool(detected_version.removeprefix(f"{base}+"))
            for base in _SUPPORTED_BASE_VERSIONS
        )
    try:
        parsed = Version(detected_version)
    except InvalidVersion:
        return False
    return (
        parsed.base_version in _SUPPORTED_BASE_VERSIONS
        and parsed.local is not None
        and parsed.pre is None
        and parsed.post is None
        and parsed.dev is None
    )


def _force_generic_enabled() -> bool:
    return os.environ.get(_FORCE_GENERIC_ENV, "").strip().lower() in _TRUE_VALUES


def _current_platform() -> Any:
    from vllm.platforms import current_platform

    return current_platform


def install(*, mode: PluginMode | None = None) -> None:
    """Install the vLLM affinity decision Hook in the current process."""
    # Resolve plugin policy before importing framework or hardware-specific code.
    mode = mode or load_plugin_mode()
    if mode is PluginMode.OFF:
        return

    config = load_plugin_config()
    from vllm.utils import numa_utils

    # Reject configuration options that this adapter cannot preserve safely.
    if config.cpu_policy is CpuPolicy.EXACT:
        raise AffinityIntegrationError(
            "vLLM exact CPU policy is not implemented by this adapter",
            code="CPU_POLICY_UNSUPPORTED",
        )
    requested_provider = None if config.provider == "auto" else config.provider
    if requested_provider is not None:
        from kunpeng_affinity.adapters.vllm_devices import (
            VLLM_PROVIDER_NAMES,
        )

        if requested_provider not in VLLM_PROVIDER_NAMES:
            raise AffinityIntegrationError(
                f"provider {requested_provider!r} is not registered for vLLM",
                code="PROVIDER_NOT_FOUND",
            )

    # Validate the original context-manager contract before replacing it.
    current = numa_utils.configure_subprocess
    if hasattr(current, _HOOK_MARKER):
        return

    parameters = set(inspect.signature(current).parameters)
    missing = _REQUIRED_PARAMETERS - parameters
    if missing:
        message = "Unsupported vLLM configure_subprocess signature; missing parameters: " + ", ".join(sorted(missing))
        if mode is PluginMode.STRICT:
            raise AffinityIntegrationError(
                message,
                code="HOOK_SIGNATURE_MISMATCH",
            )
        logger.warning("[kunpeng-affinity] %s; plugin Hook not installed", message)
        return

    detected_version = _vllm_version()
    if not _is_supported_vllm_version(detected_version):
        if mode is PluginMode.STRICT:
            raise AffinityIntegrationError(
                f"unsupported vLLM version {detected_version}; validated base "
                f"versions are {sorted(_SUPPORTED_BASE_VERSIONS)}",
                code="FRAMEWORK_VERSION_UNSUPPORTED",
            )
        logger.warning(
            "[kunpeng-affinity] unvalidated vLLM version=%s; validated bases=%s, "
            "auxiliary-demo=%s; plugin Hook not installed",
            detected_version,
            _TARGET_VERSION,
            _AUXILIARY_DEMO_VERSION,
        )
        return

    adapter = VllmAffinityAdapter(
        current=current,
        numa_utils=numa_utils,
        mode=mode,
        force_generic=_force_generic_enabled(),
        requested_provider=requested_provider,
        detected_version=detected_version,
        platform_factory=_current_platform,
        resolver=resolve_automatic_nodes,
        visibility_validator=revalidate_visibility,
        commit_nodes=commit_vllm_nodes,
        inherited_validator=validate_inherited_vllm_transaction,
        transaction_releaser=release_invalid_vllm_transaction,
        logger=logger,
    )

    @contextmanager
    def configure_subprocess_wrapper(*args: Any, **kwargs: Any) -> Iterator[None]:
        with adapter.configure_subprocess(*args, **kwargs):
            yield

    setattr(configure_subprocess_wrapper, _HOOK_MARKER, current)
    numa_utils.configure_subprocess = configure_subprocess_wrapper
    logger.warning(
        "[kunpeng-affinity] installed vLLM subprocess hook pid=%s vllm=%s mode=%s",
        os.getpid(),
        detected_version,
        config.mode.value,
    )
