"""vLLM-specific assembly for the generic Linux topology path."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kunpeng_affinity.core.errors import AffinityDiscoveryError
from kunpeng_affinity.core.models import (
    BatchAffinityResult,
    DeviceContext,
    DeviceResolution,
    NativeOutcome,
    NativeStatus,
)
from kunpeng_affinity.policy import GenericAffinityProvider
from kunpeng_affinity.providers import (
    IluvatarRuntimeProvider,
    ProviderRegistry,
    VllmPlatformProvider,
)
from kunpeng_affinity.topology.cpulist import CpuListError, parse_cpulist
from kunpeng_affinity.topology.analyzer import analyze_bdf


def _device_count(platform: Any) -> int:
    method = getattr(platform, "device_count", None)
    if not callable(method):
        raise AffinityDiscoveryError(
            "vLLM platform does not expose device_count",
            code="PROVIDER_UNAVAILABLE",
        )
    try:
        count = method()
    except (NotImplementedError, RuntimeError, OSError, ValueError) as exc:
        raise AffinityDiscoveryError(
            f"vLLM platform device count query failed: {exc}",
            code="PROVIDER_UNAVAILABLE",
        ) from exc
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise AffinityDiscoveryError(
            f"vLLM platform returned invalid device count {count!r}",
            code="DEVICE_COUNT_INVALID",
        )
    return count


def create_vllm_provider_registry(
    platform: Any,
    *,
    providers: Sequence[Any] = (),
    requested_provider: str | None = None,
) -> ProviderRegistry:
    """Build providers, preferring direct platform BDF over runtime fallback."""
    # Register the framework-native mapper first, then add the runtime mapper
    # only when a direct platform BDF mapping is unavailable or explicitly asked for.
    registry = ProviderRegistry(providers)
    registry.register(VllmPlatformProvider(platform))
    if requested_provider == IluvatarRuntimeProvider.name or not _has_direct_bdf(
        platform
    ):
        registry.register(
            IluvatarRuntimeProvider(
                platform,
                ixsmi=os.environ.get("KUNPENG_AFFINITY_IXSMI", "ixsmi"),
            )
        )
    return registry


def _has_direct_bdf(platform: Any) -> bool:
    method = getattr(platform, "get_all_gpu_pci_bus_ids", None)
    if not callable(method):
        return False
    try:
        result = method()
    except (NotImplementedError, RuntimeError, OSError, ValueError):
        return False
    return isinstance(result, Mapping) and bool(result)


def build_vllm_device_contexts(
    platform: Any,
    *,
    process_kind: str = "worker",
    local_rank: int | None = None,
    dp_local_rank: int | None = None,
    allowed_cpus: frozenset[int] | set[int] | None = None,
) -> tuple[DeviceContext, ...]:
    """Build one ordered context for every framework-visible device."""
    # Device IDs here are framework-visible IDs; Providers establish their
    # physical identity without relying on host PCI enumeration order.
    count = _device_count(platform)
    if allowed_cpus is not None:
        effective_allowed = frozenset(allowed_cpus)
    else:
        try:
            effective_allowed = frozenset(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            # Non-Linux hosts can still build and validate framework-neutral
            # contexts. Linux resolution paths supply or acquire this value.
            effective_allowed = frozenset()
    return tuple(
        DeviceContext(
            framework="vllm",
            logical_device_id=device_id,
            process_kind=process_kind,
            local_rank=local_rank,
            dp_local_rank=dp_local_rank,
            allowed_cpus=effective_allowed,
        )
        for device_id in range(count)
    )


def resolve_vllm_visibility_fingerprint(
    platform: Any,
    *,
    registry: ProviderRegistry | None = None,
    requested_provider: str | None = None,
    process_kind: str = "worker",
    local_rank: int | None = None,
    dp_local_rank: int | None = None,
    allowed_cpus: frozenset[int] | set[int] | None = None,
    include_topology: bool = False,
    sysfs_root: Path | str = Path("/sys"),
) -> str:
    """Capture the ordered visibility, optionally including Linux evidence."""
    # Mapping-only mode is useful for framework identity checks. Commit paths
    # request the full topology fingerprint through the same resolver.
    contexts = build_vllm_device_contexts(
        platform,
        process_kind=process_kind,
        local_rank=local_rank,
        dp_local_rank=dp_local_rank,
        allowed_cpus=allowed_cpus,
    )
    active_registry = registry or create_vllm_provider_registry(platform)
    mapper = active_registry.select(contexts, requested=requested_provider)
    resolver = GenericAffinityProvider(
        mapper,
        sysfs_root=sysfs_root,
        allowed_cpus=allowed_cpus,
        snapshot_metadata={
            "adapter": "vllm.configure_subprocess.v1",
        },
    )
    if not include_topology:
        _, fingerprint = resolver.mapping_snapshot(contexts)
        return fingerprint
    batch = resolver.resolve_all(contexts)
    if not batch.committable or batch.visibility_fingerprint is None:
        summary = "; ".join(batch.failure_summary) or "snapshot is not committable"
        raise AffinityDiscoveryError(
            f"cannot revalidate vLLM affinity snapshot: {summary}",
            code="SNAPSHOT_CHANGED",
        )
    return batch.visibility_fingerprint


def resolve_vllm_consumed_device(
    platform: Any,
    device_index: int,
    *,
    requested_provider: str | None = None,
    process_kind: str = "worker",
    local_rank: int | None = None,
    dp_local_rank: int | None = None,
    allowed_cpus: frozenset[int] | set[int] | None = None,
    sysfs_root: Path | str = Path("/sys"),
) -> DeviceResolution:
    """Re-sample one child-consumed device without comparing parent CPU sets."""
    contexts = build_vllm_device_contexts(
        platform,
        process_kind=process_kind,
        local_rank=local_rank,
        dp_local_rank=dp_local_rank,
        allowed_cpus=allowed_cpus,
    )
    if device_index < 0 or device_index >= len(contexts):
        raise AffinityDiscoveryError(
            f"vLLM consumed device index {device_index} is outside visible range",
            code="INHERITED_RESULT_INVALID",
        )
    registry = create_vllm_provider_registry(
        platform,
        requested_provider=requested_provider,
    )
    mapper = registry.select(contexts, requested=requested_provider)
    resolver = GenericAffinityProvider(
        mapper,
        sysfs_root=sysfs_root,
        allowed_cpus=allowed_cpus,
        snapshot_metadata={"adapter": "vllm.configure_subprocess.v1"},
    )
    mappings, _ = resolver.mapping_snapshot(contexts)
    context = contexts[device_index]
    mapping = mappings[device_index]
    affinity = analyze_bdf(
        mapping.pci_bdf,
        sysfs_root=sysfs_root,
        allowed_cpus=allowed_cpus,
        mapping_source=mapping.source,
    )
    if not affinity.bindable:
        raise AffinityDiscoveryError(
            f"inherited vLLM device topology is {affinity.status.value}",
            code="INHERITED_RESULT_INVALID",
        )
    return DeviceResolution(context=context, mapping=mapping, affinity=affinity)


def check_vllm_generic_eligibility(
    numa_utils: Any,
    *,
    sysfs_root: Path | str = Path("/sys"),
    allowed_cpus: frozenset[int] | set[int] | None = None,
) -> None:
    """Apply vLLM's vendor-neutral automatic-binding safety gates."""
    # Refuse generic binding when the host, memory policy, or binding
    # executable cannot support the operation safely. Existing CPU affinity
    # is handled by topology's allowed-CPU intersection below.
    root = Path(sysfs_root)
    node_root = root / "devices/system/node"
    numa_nodes = tuple(path for path in node_root.glob("node[0-9]*") if path.is_dir())
    if len(numa_nodes) < 2:
        raise AffinityDiscoveryError(
            "automatic NUMA binding requires more than one NUMA node",
            code="NUMA_NOT_AVAILABLE",
        )

    can_set_mempolicy = getattr(numa_utils, "_can_set_mempolicy", None)
    if not callable(can_set_mempolicy) or not can_set_mempolicy():
        raise AffinityDiscoveryError(
            "NUMA memory policy is unavailable in the current process",
            code="MEMPOLICY_UNAVAILABLE",
        )
    if shutil.which("numactl") is None:
        raise AffinityDiscoveryError(
            "numactl is not available on PATH",
            code="BIND_EXECUTOR_MISSING",
        )


def resolve_vllm_native_nodes(
    numa_utils: Any,
    platform: Any,
    *,
    sysfs_root: Path | str = Path("/sys"),
    allowed_cpus: frozenset[int] | set[int] | None = None,
) -> list[int]:
    """Call and validate vLLM's native GPU-to-NUMA query as a full batch."""
    # Treat the native result as a complete device-ordered list, not as a
    # per-device best effort result.
    query = getattr(numa_utils, "get_auto_numa_nodes", None)
    if not callable(query):
        raise AffinityDiscoveryError(
            "vLLM does not expose get_auto_numa_nodes",
            code="NATIVE_QUERY_UNAVAILABLE",
        )
    try:
        raw_nodes = query()
    except (NotImplementedError, RuntimeError, OSError, ValueError) as exc:
        raise AffinityDiscoveryError(
            f"vLLM native NUMA query failed: {exc}",
            code="NATIVE_QUERY_FAILED",
        ) from exc
    if raw_nodes is None:
        raise AffinityDiscoveryError(
            "vLLM native NUMA query returned no result",
            code="NATIVE_QUERY_UNAVAILABLE",
        )
    if not isinstance(raw_nodes, list):
        raise AffinityDiscoveryError(
            f"vLLM native NUMA query returned {type(raw_nodes).__name__}, not list",
            code="NATIVE_RESULT_INVALID",
        )

    count = _device_count(platform)
    if len(raw_nodes) != count:
        raise AffinityDiscoveryError(
            f"vLLM native NUMA query returned {len(raw_nodes)} nodes for "
            f"{count} visible devices",
            code="DEVICE_COUNT_MISMATCH",
        )

    root = Path(sysfs_root)
    try:
        online = parse_cpulist(
            (root / "devices/system/cpu/online").read_text(encoding="ascii")
        )
    except (OSError, CpuListError) as exc:
        raise AffinityDiscoveryError(
            f"cannot validate online CPUs: {exc}",
            code="CPU_LIST_INVALID",
        ) from exc
    try:
        effective_allowed = (
            frozenset(allowed_cpus)
            if allowed_cpus is not None
            else frozenset(os.sched_getaffinity(0))
        )
    except (AttributeError, OSError) as exc:
        raise AffinityDiscoveryError(
            "current process CPU affinity is unavailable",
            code="CPU_AFFINITY_UNAVAILABLE",
        ) from exc

    nodes: list[int] = []
    for index, node in enumerate(raw_nodes):
        if not isinstance(node, int) or isinstance(node, bool) or node < 0:
            raise AffinityDiscoveryError(
                f"vLLM native NUMA result at device {index} is invalid: {node!r}",
                code="NATIVE_RESULT_INVALID",
            )
        try:
            node_cpus = parse_cpulist(
                (root / f"devices/system/node/node{node}/cpulist").read_text(
                    encoding="ascii"
                )
            )
        except (OSError, CpuListError) as exc:
            raise AffinityDiscoveryError(
                f"cannot validate NUMA node {node}: {exc}",
                code="NUMA_NODE_INVALID",
            ) from exc
        if not node_cpus & online & effective_allowed:
            raise AffinityDiscoveryError(
                f"NUMA node {node} has no online CPUs allowed to this process",
                code="CPU_SET_EMPTY",
            )
        nodes.append(node)
    return nodes


def classify_vllm_native_result(
    numa_utils: Any,
    platform: Any,
) -> NativeOutcome:
    """Classify the native query without converting native errors to fallback."""
    query = getattr(numa_utils, "get_auto_numa_nodes", None)
    if not callable(query):
        return NativeOutcome(
            status=NativeStatus.FALLBACK_ALLOWED,
            failure_code="NATIVE_QUERY_UNAVAILABLE",
            evidence=("vLLM native query symbol is missing",),
        )

    try:
        raw_nodes = query()
    except Exception as exc:
        return NativeOutcome(
            status=NativeStatus.ERROR,
            failure_code="NATIVE_QUERY_FAILED",
            original_error=exc,
            evidence=("vLLM native query raised an exception",),
        )

    if raw_nodes is None or raw_nodes == []:
        if _native_platform_is_uncovered(platform):
            return NativeOutcome(
                status=NativeStatus.FALLBACK_ALLOWED,
                failure_code="NATIVE_QUERY_UNAVAILABLE",
                evidence=("platform lacks a direct PCI identity capability",),
            )
        return NativeOutcome(
            status=NativeStatus.PRESERVE_NATIVE,
            failure_code="NATIVE_RESULT_EMPTY",
            evidence=("native query returned no usable result",),
        )

    if not isinstance(raw_nodes, list):
        return NativeOutcome(
            status=NativeStatus.INVALID,
            failure_code="NATIVE_RESULT_INVALID",
            evidence=(f"native result type={type(raw_nodes).__name__}",),
        )

    try:
        count = _device_count(platform)
    except AffinityDiscoveryError as exc:
        return NativeOutcome(
            status=NativeStatus.ERROR,
            failure_code=exc.code,
            original_error=exc,
            evidence=("visible device count could not be validated",),
        )
    if len(raw_nodes) != count:
        return NativeOutcome(
            status=NativeStatus.INVALID,
            failure_code="DEVICE_COUNT_MISMATCH",
            evidence=(f"native_count={len(raw_nodes)} visible_count={count}",),
        )
    if any(
        not isinstance(node, int) or isinstance(node, bool) or node < 0
        for node in raw_nodes
    ):
        return NativeOutcome(
            status=NativeStatus.INVALID,
            failure_code="NATIVE_RESULT_INVALID",
            evidence=("native result contains an invalid NUMA node",),
        )
    return NativeOutcome(
        status=NativeStatus.VALID,
        nodes=tuple(raw_nodes),
        evidence=("vLLM native NUMA result satisfies its return contract",),
    )


def _native_platform_is_uncovered(platform: Any) -> bool:
    """Use an observable platform capability gap as the fallback proof."""
    method = getattr(platform, "get_all_gpu_pci_bus_ids", None)
    if not callable(method):
        return True
    try:
        result = method()
    except (NotImplementedError, RuntimeError, OSError, ValueError):
        return True
    return not isinstance(result, Mapping) or not bool(result)


def resolve_vllm_generic_affinity(
    platform: Any,
    *,
    sysfs_root: Path | str = Path("/sys"),
    allowed_cpus: frozenset[int] | set[int] | None = None,
    registry: ProviderRegistry | None = None,
    requested_provider: str | None = None,
    process_kind: str = "worker",
    local_rank: int | None = None,
    dp_local_rank: int | None = None,
) -> BatchAffinityResult:
    """Resolve every vLLM-visible device through BDF and Linux sysfs."""
    # Build contexts, select one complete Provider mapping, and run the shared
    # batch resolver. No framework configuration is mutated in this function.
    contexts = build_vllm_device_contexts(
        platform,
        process_kind=process_kind,
        local_rank=local_rank,
        dp_local_rank=dp_local_rank,
        allowed_cpus=allowed_cpus,
    )
    active_registry = registry or create_vllm_provider_registry(platform)
    batch = GenericAffinityProvider(
        registry=active_registry,
        requested_provider=requested_provider,
        sysfs_root=sysfs_root,
        allowed_cpus=allowed_cpus,
        snapshot_metadata={
            "adapter": "vllm.configure_subprocess.v1",
        },
    ).resolve_all(contexts)
    return batch
