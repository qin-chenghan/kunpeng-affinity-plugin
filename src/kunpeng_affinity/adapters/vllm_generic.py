"""vLLM-specific assembly for the generic Linux topology path."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kunpeng_affinity.core.errors import AffinityDiscoveryError
from kunpeng_affinity.core.models import BatchAffinityResult, DeviceContext
from kunpeng_affinity.policy import GenericAffinityProvider
from kunpeng_affinity.providers import (
    IluvatarRuntimeProvider,
    ProviderRegistry,
    VllmPlatformProvider,
)
from kunpeng_affinity.topology.cpulist import CpuListError, parse_cpulist


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
        registry.register(IluvatarRuntimeProvider(platform))
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
) -> str:
    """Capture the ordered logical-device mapping without topology I/O."""
    # The fingerprint records the mapping that topology analysis is about to use.
    contexts = build_vllm_device_contexts(
        platform,
        process_kind=process_kind,
        local_rank=local_rank,
        dp_local_rank=dp_local_rank,
        allowed_cpus=allowed_cpus,
    )
    active_registry = registry or create_vllm_provider_registry(platform)
    mapper = active_registry.select(contexts, requested=requested_provider)
    _, fingerprint = GenericAffinityProvider(mapper).mapping_snapshot(contexts)
    return fingerprint


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
    if not (root / "devices/system/node/node1").is_dir():
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
    mapper = active_registry.select(contexts, requested=requested_provider)
    batch = GenericAffinityProvider(
        mapper,
        sysfs_root=sysfs_root,
        allowed_cpus=allowed_cpus,
    ).resolve_all(contexts)
    if not batch.committable:
        summary = "; ".join(batch.failure_summary) or "incomplete topology result"
        raise AffinityDiscoveryError(
            f"generic vLLM affinity resolution is not committable: {summary}",
            code="BATCH_NOT_COMMITTABLE",
        )
    return batch
