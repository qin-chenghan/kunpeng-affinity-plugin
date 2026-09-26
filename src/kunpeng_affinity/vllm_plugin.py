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

from kunpeng_affinity.adapters.vllm_commit import (
    release_invalid_vllm_transaction,
    validate_inherited_vllm_transaction,
)
from kunpeng_affinity.config import (
    CpuPolicy,
    PluginMode,
    load_plugin_config,
    load_plugin_mode,
)
from kunpeng_affinity.core.errors import (
    AffinityDiscoveryError,
    AffinityIntegrationError,
    NativeContractError,
)
from kunpeng_affinity.core.models import NativeOutcome, NativeStatus

logger = logging.getLogger(__name__)

_HOOK_MARKER = "__kunpeng_affinity_original__"
_TRANSACTION_MARKER = "_kunpeng_affinity_transaction"
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
_SUPPORTED_BASE_VERSIONS = frozenset({_TARGET_VERSION, _AUXILIARY_DEMO_VERSION})


@dataclass(frozen=True)
class _AutomaticAffinityResolution:
    nodes: tuple[int, ...]
    source: str
    visibility_fingerprint: str | None
    registry: Any | None
    requested_provider: str | None = None
    snapshot_json: str | None = None


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
            detected_version.startswith(f"{base}+")
            and bool(detected_version.removeprefix(f"{base}+"))
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
    if not batch.committable:
        summary = "; ".join(batch.failure_summary) or "incomplete topology result"
        raise AffinityDiscoveryError(
            f"generic vLLM affinity resolution is not committable: {summary}",
            code=batch.status.value.upper(),
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
        snapshot_json=batch.snapshot_json,
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
    registry = None
    if not force_generic:
        from kunpeng_affinity.adapters import classify_vllm_native_result

        # Classify the native result before deciding whether fallback is legal.
        outcome = classify_vllm_native_result(numa_utils, platform)
        if outcome.status is NativeStatus.ERROR:
            assert outcome.original_error is not None
            raise outcome.original_error
        if outcome.status is NativeStatus.INVALID:
            raise NativeContractError(
                "vLLM native NUMA result violates its return contract: "
                + (outcome.failure_code or "NATIVE_RESULT_INVALID"),
                code=outcome.failure_code or "NATIVE_RESULT_INVALID",
            )
        if outcome.status is NativeStatus.PRESERVE_NATIVE:
            return _AutomaticAffinityResolution(
                nodes=(),
                source="native-preserved",
                visibility_fingerprint=None,
                registry=None,
                requested_provider=requested_provider,
            )
        if outcome.status is NativeStatus.VALID:
            nodes = list(outcome.nodes)
            return _AutomaticAffinityResolution(
                nodes=tuple(nodes),
                source="native",
                visibility_fingerprint=None,
                registry=None,
                requested_provider=requested_provider,
            )
        logger.warning(
            "[kunpeng-affinity] native NUMA result unavailable code=%s; "
            "trying generic Linux topology",
            outcome.failure_code,
        )

    # Native discovery was unavailable or explicitly bypassed; only now build
    # the provider registry and inspect device identity.
    registry = _provider_registry(platform, requested_provider)
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
        include_topology=True,
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
        # Native results remain owned by vLLM. The plugin must not create a
        # marker or rewrite fields when the native query was usable, or when
        # vLLM deliberately returned an empty result that it owns.
        if resolution.source in {"native", "native-preserved"}:
            with current(*args, **kwargs):
                yield
            return
        from kunpeng_affinity.adapters.vllm_commit import commit_vllm_nodes

        # Commit only after the complete result and its visibility are validated.
        commit = commit_vllm_nodes(
            parallel_config,
            list(resolution.nodes),
            visibility_fingerprint=resolution.visibility_fingerprint,
            snapshot_json=resolution.snapshot_json,
            requested_provider=resolution.requested_provider,
        )
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
        # A user-supplied CPU policy still needs vLLM's original context even
        # when the plugin cannot add an automatically discovered NUMA node.
        if getattr(parallel_config, "numa_bind_cpus", None) is not None:
            with current(*args, **kwargs):
                yield
            return
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
        commit.mark_committed()
    except BaseException:
        error = sys.exc_info()
        try:
            manager.__exit__(*error)
        finally:
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
    if not _is_supported_vllm_version(detected_version):
        if mode is PluginMode.STRICT:
            raise AffinityIntegrationError(
                f"unsupported vLLM version {detected_version}; validated base "
                f"versions are {_TARGET_VERSION} and {_AUXILIARY_DEMO_VERSION}",
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

        # Reuse a committed plugin result before considering a new resolution.
        # This covers repeated use in one process and serialized child configs.
        if getattr(parallel_config, _TRANSACTION_MARKER, None) is not None:
            try:
                validate_inherited_vllm_transaction(
                    parallel_config,
                    numa_utils=numa_utils,
                    platform=_current_platform(),
                    local_rank=local_rank,
                    dp_local_rank=dp_local_rank,
                    process_kind=process_kind,
                )
            except AffinityDiscoveryError as exc:
                if mode is PluginMode.STRICT:
                    raise _strict_failure(exc) from exc
                logger.warning(
                    "[kunpeng-affinity] inherited transaction released code=%s: %s",
                    exc.code,
                    exc,
                )
                release_invalid_vllm_transaction(parallel_config)
                with current(*args, **kwargs):
                    yield
                return
            with current(*args, **kwargs):
                yield
            return

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
