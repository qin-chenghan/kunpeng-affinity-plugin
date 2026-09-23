"""vLLM general-plugin entry point and affinity decision hook."""

from __future__ import annotations

import inspect
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterator

from kunpeng_affinity.config import (
    CpuPolicy,
    PluginMode,
    load_plugin_config,
    load_plugin_mode,
)
from kunpeng_affinity.core.errors import (
    AffinityDiscoveryError,
    AffinityIntegrationError,
)

logger = logging.getLogger(__name__)

_HOOK_MARKER = "__kunpeng_affinity_original__"
_REQUIRED_PARAMETERS = {
    "vllm_config",
    "local_rank",
    "dp_local_rank",
    "process_kind",
}
_TARGET_VERSION = "0.23.0"
_AUXILIARY_DEMO_VERSION = "0.26.0"
_FORCE_GENERIC_ENV = "KUNPENG_AFFINITY_VLLM_FORCE_GENERIC"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class _AutomaticAffinityResolution:
    nodes: tuple[int, ...]
    source: str
    visibility_fingerprint: str | None
    registry: Any | None
    requested_provider: str | None = None


def _vllm_version() -> str:
    try:
        return version("vllm")
    except PackageNotFoundError:
        return "unknown"


def _argument(
    args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str, default: Any
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


def _numa_bind_enabled(vllm_config: Any) -> Any:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return getattr(parallel_config, "numa_bind", "unknown")


def _force_generic_enabled() -> bool:
    return os.environ.get(_FORCE_GENERIC_ENV, "").strip().lower() in _TRUE_VALUES


def _current_platform() -> Any:
    from vllm.platforms import current_platform

    return current_platform


def _provider_registry(platform: Any, requested_provider: str | None = None) -> Any:
    from kunpeng_affinity.adapters import create_vllm_provider_registry

    return create_vllm_provider_registry(
        platform, requested_provider=requested_provider
    )


def _nodes_from_generic_batch(batch: Any) -> list[int]:
    nodes: list[int] = []
    for resolution in batch.ordered_results:
        node = resolution.affinity.numa_node
        if node is None:
            raise AffinityDiscoveryError(
                "generic NUMA resolution produced a result without a node",
                code="NUMA_UNKNOWN",
            )
        nodes.append(node)
    return nodes


def _resolve_generic_nodes(
    numa_utils: Any,
    platform: Any,
    registry: Any,
    *,
    requested_provider: str | None,
    process_kind: str,
    local_rank: int | None,
    dp_local_rank: int | None,
) -> _AutomaticAffinityResolution:
    # Check the host and process prerequisites before reading device topology.
    from kunpeng_affinity.adapters import (
        check_vllm_generic_eligibility,
        resolve_vllm_generic_affinity,
    )

    check_vllm_generic_eligibility(numa_utils)

    # Resolve every visible device as one batch so a partial result cannot be used.
    batch = resolve_vllm_generic_affinity(
        platform,
        registry=registry,
        requested_provider=requested_provider,
        process_kind=process_kind,
        local_rank=local_rank,
        dp_local_rank=dp_local_rank,
    )
    if batch.visibility_fingerprint is None:
        raise AffinityDiscoveryError(
            "generic NUMA resolution did not produce a visibility fingerprint",
            code="VISIBILITY_UNAVAILABLE",
        )
    return _AutomaticAffinityResolution(
        nodes=tuple(_nodes_from_generic_batch(batch)),
        source="generic",
        visibility_fingerprint=batch.visibility_fingerprint,
        registry=registry,
        requested_provider=requested_provider,
    )


def _resolve_automatic_nodes(
    numa_utils: Any,
    platform: Any,
    *,
    force_generic: bool,
    requested_provider: str | None = None,
    process_kind: str = "worker",
    local_rank: int | None = None,
    dp_local_rank: int | None = None,
) -> _AutomaticAffinityResolution:
    """Resolve a complete node list without mutating vLLM configuration."""
    from kunpeng_affinity.adapters import resolve_vllm_visibility_fingerprint

    # Keep one provider registry for mapping and the later visibility recheck.
    registry = _provider_registry(platform, requested_provider)
    if not force_generic:
        from kunpeng_affinity.adapters import resolve_vllm_native_nodes

        # A valid framework-native result has priority over the Linux fallback.
        try:
            nodes = resolve_vllm_native_nodes(numa_utils, platform)
            fingerprint = resolve_vllm_visibility_fingerprint(
                platform,
                registry=registry,
                requested_provider=requested_provider,
                process_kind=process_kind,
                local_rank=local_rank,
                dp_local_rank=dp_local_rank,
            )
            return _AutomaticAffinityResolution(
                nodes=tuple(nodes),
                source="native",
                visibility_fingerprint=fingerprint,
                registry=registry,
                requested_provider=requested_provider,
            )
        except AffinityDiscoveryError as exc:
            logger.warning(
                "[kunpeng-affinity] native NUMA result unavailable code=%s: %s; "
                "trying generic Linux topology",
                exc.code,
                exc,
            )

    # Native discovery was unavailable or explicitly bypassed; use the generic path.
    return _resolve_generic_nodes(
        numa_utils,
        platform,
        registry,
        requested_provider=requested_provider,
        process_kind=process_kind,
        local_rank=local_rank,
        dp_local_rank=dp_local_rank,
    )


def _revalidate_visibility(
    resolution: _AutomaticAffinityResolution,
    platform: Any,
    *,
    process_kind: str,
    local_rank: int | None,
    dp_local_rank: int | None,
) -> None:
    if resolution.visibility_fingerprint is None:
        return
    from kunpeng_affinity.adapters import resolve_vllm_visibility_fingerprint

    current = resolve_vllm_visibility_fingerprint(
        platform,
        registry=resolution.registry,
        requested_provider=resolution.requested_provider,
        process_kind=process_kind,
        local_rank=local_rank,
        dp_local_rank=dp_local_rank,
    )
    if current != resolution.visibility_fingerprint:
        raise AffinityDiscoveryError(
            "vLLM device visibility changed between resolution and commit",
            code="VISIBILITY_CHANGED",
        )


def _strict_failure(error: AffinityDiscoveryError) -> AffinityIntegrationError:
    return AffinityIntegrationError(
        f"strict vLLM affinity discovery failed [{error.code}]: {error}",
        code=error.code,
    )


def _should_resolve(parallel_config: Any, process_kind: str) -> bool:
    if parallel_config is None or not getattr(parallel_config, "numa_bind", False):
        return False
    if getattr(parallel_config, "numa_bind_nodes", None) is not None:
        return False
    if process_kind not in {"worker", "EngineCore"}:
        return False
    return True


@contextmanager
def _automatic_affinity_context(
    *,
    current: Any,
    numa_utils: Any,
    mode: PluginMode,
    force_generic: bool,
    requested_provider: str | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    parallel_config: Any,
    process_kind: str,
    local_rank: int | None,
    dp_local_rank: int | None,
) -> Iterator[None]:
    platform = _current_platform()
    try:
        # Resolve the complete NUMA result without changing framework state yet.
        resolution = _resolve_automatic_nodes(
            numa_utils,
            platform,
            force_generic=force_generic,
            requested_provider=requested_provider,
            process_kind=process_kind,
            local_rank=local_rank,
            dp_local_rank=dp_local_rank,
        )

        # Ensure the device-to-BDF view did not change during topology analysis.
        _revalidate_visibility(
            resolution,
            platform,
            process_kind=process_kind,
            local_rank=local_rank,
            dp_local_rank=dp_local_rank,
        )
        from kunpeng_affinity.adapters.vllm_commit import commit_vllm_nodes

        # Commit only after the complete result and its visibility are validated.
        commit = commit_vllm_nodes(parallel_config, list(resolution.nodes))
    except AffinityDiscoveryError as exc:
        # Discovery failures are recoverable in auto mode and fatal in strict mode.
        if force_generic or mode is PluginMode.STRICT:
            raise _strict_failure(exc) from exc
        logger.warning(
            "[kunpeng-affinity] automatic affinity skipped code=%s: %s; "
            "launching without additional binding",
            exc.code,
            exc,
        )
        yield
        return

    logger.warning(
        "[kunpeng-affinity] selected NUMA nodes=%s source=%s%s",
        list(resolution.nodes),
        resolution.source,
        "; native GPU NUMA query bypassed" if force_generic else "",
    )
    try:
        # The original vLLM context still owns process creation and numactl setup.
        manager = current(*args, **kwargs)
        manager.__enter__()
    except BaseException:
        commit.rollback()
        raise

    try:
        yield
    except BaseException:
        if not manager.__exit__(*sys.exc_info()):
            raise
    else:
        manager.__exit__(None, None, None)


def register() -> None:
    """Install the vLLM affinity decision Hook in the current process."""
    # Resolve plugin policy before importing framework or hardware-specific code.
    mode = load_plugin_mode()
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
        from kunpeng_affinity.providers import (
            IluvatarRuntimeProvider,
            VllmPlatformProvider,
        )

        if requested_provider not in {
            VllmPlatformProvider.name,
            IluvatarRuntimeProvider.name,
        }:
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
        message = (
            "Unsupported vLLM configure_subprocess signature; missing parameters: "
            + ", ".join(sorted(missing))
        )
        if mode is PluginMode.STRICT:
            raise AffinityIntegrationError(
                message,
                code="HOOK_SIGNATURE_MISMATCH",
            )
        logger.warning("[kunpeng-affinity] %s; plugin Hook not installed", message)
        return

    detected_version = _vllm_version()
    if detected_version not in {_TARGET_VERSION, _AUXILIARY_DEMO_VERSION}:
        if mode is PluginMode.STRICT:
            raise AffinityIntegrationError(
                f"unsupported vLLM version {detected_version}; validated versions "
                f"are {_TARGET_VERSION} and {_AUXILIARY_DEMO_VERSION}",
                code="FRAMEWORK_VERSION_UNSUPPORTED",
            )
        logger.warning(
            "[kunpeng-affinity] unvalidated vLLM version=%s; target=%s, "
            "auxiliary-demo=%s; plugin Hook not installed",
            detected_version,
            _TARGET_VERSION,
            _AUXILIARY_DEMO_VERSION,
        )
        return

    force_generic = _force_generic_enabled()

    @contextmanager
    def configure_subprocess_wrapper(*args: Any, **kwargs: Any) -> Iterator[None]:
        # Capture framework rank and process metadata at the subprocess boundary.
        vllm_config = _argument(args, kwargs, 0, "vllm_config", None)
        local_rank = _argument(args, kwargs, 1, "local_rank", None)
        dp_local_rank = _argument(args, kwargs, 2, "dp_local_rank", None)
        process_kind = _argument(args, kwargs, 3, "process_kind", "worker")
        logger.warning(
            "[kunpeng-affinity] pid=%s vllm=%s process_kind=%s "
            "local_rank=%s dp_local_rank=%s numa_bind=%s",
            os.getpid(),
            detected_version,
            process_kind,
            local_rank,
            dp_local_rank,
            _numa_bind_enabled(vllm_config),
        )
        parallel_config = getattr(vllm_config, "parallel_config", None)
        if _should_resolve(parallel_config, process_kind):
            # Automatic discovery is limited to eligible worker-like processes.
            with _automatic_affinity_context(
                current=current,
                numa_utils=numa_utils,
                mode=mode,
                force_generic=force_generic,
                requested_provider=requested_provider,
                args=args,
                kwargs=kwargs,
                parallel_config=parallel_config,
                process_kind=process_kind,
                local_rank=local_rank,
                dp_local_rank=dp_local_rank,
            ):
                yield
            return

        # All other calls retain the original vLLM behavior unchanged.
        with current(*args, **kwargs):
            yield

    setattr(configure_subprocess_wrapper, _HOOK_MARKER, current)
    numa_utils.configure_subprocess = configure_subprocess_wrapper
    logger.warning(
        "[kunpeng-affinity] installed vLLM subprocess hook pid=%s vllm=%s mode=%s",
        os.getpid(),
        detected_version,
        config.mode.value,
    )
