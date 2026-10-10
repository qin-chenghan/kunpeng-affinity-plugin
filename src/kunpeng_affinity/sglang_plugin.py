"""SGLang 0.5.18 general-plugin entry point for generic NUMA fallback."""

from __future__ import annotations

import inspect
import logging
import os
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from kunpeng_affinity.config import (
    CpuPolicy,
    PluginMode,
    load_plugin_config,
    load_plugin_mode,
)
from kunpeng_affinity.core.errors import (
    AffinityIntegrationError,
    DeviceMappingError,
    NativeContractError,
    PluginConfigError,
    PluginContractError,
)
from kunpeng_affinity.core.models import NativeOutcome, NativeStatus

logger = logging.getLogger(__name__)
_HOOK_MARKER = "__kunpeng_affinity_sglang_hook__"
_HOOK_TARGET = "sglang.srt.utils.numa_utils.get_numa_node_if_available"
_REQUIRED_HOOK_PARAMETERS = {"server_args", "gpu_id"}
_SUPPORTED_SGLANG_VERSIONS = frozenset({"0.5.17.dev386+gc5bd3d7dc", "0.5.18"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _sglang_version() -> str:
    try:
        return version("sglang")
    except PackageNotFoundError:
        return "unknown"


def _validate_hook_signature(target: Any) -> None:
    if not callable(target):
        raise AffinityIntegrationError(
            "SGLang NUMA query target is not callable",
            code="HOOK_SIGNATURE_MISMATCH",
        )
    parameters = set(inspect.signature(target).parameters)
    missing = _REQUIRED_HOOK_PARAMETERS - parameters
    if missing:
        raise AffinityIntegrationError(
            "Unsupported SGLang NUMA query signature; missing parameters: " + ", ".join(sorted(missing)),
            code="HOOK_SIGNATURE_MISMATCH",
        )


def _unsupported_generic_call_site(server_args: Any) -> str | None:
    if bool(getattr(server_args, "use_ray", False)):
        return "SGLang Ray NUMA fallback is not validated"
    dp_size = getattr(server_args, "dp_size", 1)
    if isinstance(dp_size, int) and not isinstance(dp_size, bool) and dp_size > 1:
        return f"SGLang data-parallel NUMA fallback is not validated (dp_size={dp_size})"
    if _env_bool("SGLANG_SET_CPU_AFFINITY", default=False):
        return "SGLANG_SET_CPU_AFFINITY would overwrite the NUMA-derived CPU affinity"
    return None


def _generic_node(gpu_id: int, provider: str) -> int:
    # Resolve one visible SGLang GPU through the shared runtime and Linux path.
    if provider not in {
        "auto",
        "sglang-runtime-pci",
        "iluvatar-runtime-pci",
        "ascend-sysfs-pci",
    }:
        raise PluginConfigError(
            f"provider {provider!r} is not registered for SGLang",
            code="PROVIDER_NOT_FOUND",
        )
    _check_generic_binding_prerequisites()
    from kunpeng_affinity.adapters.sglang_generic import resolve_sglang_numa_node

    return resolve_sglang_numa_node(
        gpu_id,
        ixsmi=os.environ.get("KUNPENG_AFFINITY_IXSMI", "ixsmi"),
        provider=provider,
    )


def _check_generic_binding_prerequisites() -> None:
    """Check only Linux binding capabilities before using generic topology."""
    from kunpeng_affinity.core.errors import DeviceMappingError

    node_root = Path("/sys") / "devices/system/node"
    numa_nodes = tuple(path for path in node_root.glob("node[0-9]*") if path.is_dir())
    if len(numa_nodes) < 2:
        raise DeviceMappingError(
            "generic SGLang NUMA binding requires multiple NUMA nodes",
            code="NUMA_NOT_AVAILABLE",
        )
    if _env_bool("SGLANG_NUMA_BIND_V2") and shutil.which("numactl") is None:
        raise DeviceMappingError(
            "numactl is not available on PATH",
            code="BIND_EXECUTOR_MISSING",
        )
    try:
        from sglang.srt.utils import numa_utils
    except ImportError as exc:
        raise DeviceMappingError(
            "SGLang NUMA utility module is unavailable",
            code="BIND_EXECUTOR_MISSING",
        ) from exc
    can_set_mempolicy = getattr(numa_utils, "_can_set_mempolicy", None)
    if not callable(can_set_mempolicy) or not can_set_mempolicy():
        raise DeviceMappingError(
            "NUMA memory policy is unavailable in the current process",
            code="MEMPOLICY_UNAVAILABLE",
        )


def _classify_native_node(
    value: Any,
    provider: str,
    *,
    fallback_proven: bool = False,
) -> NativeOutcome:
    """Classify SGLang's single-node native query before generic fallback."""
    if value is None:
        if not fallback_proven:
            return NativeOutcome(
                status=NativeStatus.PRESERVE_NATIVE,
                failure_code="NATIVE_RESULT_EMPTY",
                evidence=("SGLang returned no node; Runtime Provider coverage has not yet been proven",),
            )
        return NativeOutcome(
            status=NativeStatus.FALLBACK_ALLOWED,
            failure_code="NATIVE_QUERY_UNAVAILABLE",
            evidence=(f"SGLang returned no node and Runtime Provider {provider!r} proved a complete candidate result",),
        )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return NativeOutcome(
            status=NativeStatus.INVALID,
            failure_code="NATIVE_RESULT_INVALID",
            evidence=(f"native NUMA node={value!r}",),
        )
    return NativeOutcome(status=NativeStatus.VALID, nodes=(value,))


def _around_numa_query(original: Any, server_args: Any, gpu_id: int, *, mode: PluginMode, provider: str) -> int | None:
    """Keep explicit/native SGLang results and add only a fallback."""
    # Preserve an explicit node list and SGLang's own disabled behavior.
    if getattr(server_args, "numa_node", None) is not None:
        return original(server_args, gpu_id)
    if not _env_bool("SGLANG_AUTO_NUMA_BIND"):
        return original(server_args, gpu_id)

    # Let the framework-native query decide before using the generic fallback.
    native_value = original(server_args, gpu_id)
    native = _classify_native_node(native_value, provider)
    if native.status is NativeStatus.VALID:
        return native.nodes[0]
    if native.status is NativeStatus.INVALID:
        raise NativeContractError(
            "SGLang native NUMA query violates its return contract",
            code=native.failure_code or "NATIVE_RESULT_INVALID",
        )
    if native.status is not NativeStatus.PRESERVE_NATIVE:
        raise PluginContractError(
            "unknown SGLang native query classification",
            code="NATIVE_STATUS_INVALID",
        )
    unsupported = _unsupported_generic_call_site(server_args)
    if unsupported is not None:
        error = DeviceMappingError(unsupported, code="CALL_SITE_UNSUPPORTED")
        if mode is PluginMode.STRICT:
            raise error
        logger.warning(
            "[kunpeng-affinity] SGLang generic NUMA fallback skipped for GPU %s: %s",
            gpu_id,
            error,
        )
        return None
    try:
        # A complete candidate result also proves that the configured Runtime
        # Provider covers this process. Only then is native None fallback-safe.
        node = _generic_node(gpu_id, provider)
    except DeviceMappingError as exc:
        if mode is PluginMode.STRICT:
            raise
        logger.warning(
            "[kunpeng-affinity] SGLang generic NUMA fallback skipped for GPU %s: %s",
            gpu_id,
            exc,
        )
        return None
    native = _classify_native_node(
        native_value,
        provider,
        fallback_proven=True,
    )
    if native.status is not NativeStatus.FALLBACK_ALLOWED:
        raise PluginContractError(
            "SGLang fallback proof produced an invalid native classification",
            code="NATIVE_STATUS_INVALID",
        )
    logger.warning(
        "[kunpeng-affinity] SGLang generic NUMA fallback selected node=%s gpu=%s",
        node,
        gpu_id,
    )
    return node


def register() -> None:
    """Register the SGLang NUMA-query fallback through its plugin SPI."""
    # Read policy before installing a hook; off mode has no framework side effects.
    mode = load_plugin_mode()
    if mode is PluginMode.OFF:
        return
    config = load_plugin_config()
    if config.cpu_policy is CpuPolicy.EXACT:
        raise PluginConfigError(
            "SGLang NUMA query cannot express exact CPU policy",
            code="CPU_POLICY_UNSUPPORTED",
        )
    if config.provider not in {
        "auto",
        "sglang-runtime-pci",
        "iluvatar-runtime-pci",
        "ascend-sysfs-pci",
    }:
        raise PluginConfigError(
            f"provider {config.provider!r} is not registered for SGLang",
            code="PROVIDER_NOT_FOUND",
        )
    detected_version = _sglang_version()
    if detected_version not in _SUPPORTED_SGLANG_VERSIONS:
        message = (
            f"unsupported SGLang version {detected_version}; validated versions are "
            f"{sorted(_SUPPORTED_SGLANG_VERSIONS)}"
        )
        if mode is PluginMode.STRICT:
            raise AffinityIntegrationError(
                message,
                code="FRAMEWORK_VERSION_UNSUPPORTED",
            )
        logger.warning("[kunpeng-affinity] %s; plugin Hook not registered", message)
        return

    from sglang.srt.plugins import HookRegistry
    from sglang.srt.plugins.hook_registry import HookType
    from sglang.srt.utils import numa_utils

    if getattr(register, _HOOK_MARKER, False):
        return
    try:
        _validate_hook_signature(numa_utils.get_numa_node_if_available)
    except AffinityIntegrationError as exc:
        if mode is PluginMode.STRICT:
            raise
        logger.warning("[kunpeng-affinity] %s; plugin Hook not registered", exc)
        return

    def hook(original: Any, server_args: Any, gpu_id: int) -> int | None:
        return _around_numa_query(
            original,
            server_args,
            gpu_id,
            mode=mode,
            provider=config.provider,
        )

    # The hook changes only NUMA-node selection; SGLang keeps process launching
    # and the actual numactl operation in its original implementation.
    HookRegistry.register(_HOOK_TARGET, hook, HookType.AROUND)
    setattr(register, _HOOK_MARKER, True)
    logger.warning(
        "[kunpeng-affinity] registered SGLang NUMA query hook version=%s mode=%s provider=%s; "
        "SGLang confirms application with its Applied hook log",
        detected_version,
        mode.value,
        config.provider,
    )
