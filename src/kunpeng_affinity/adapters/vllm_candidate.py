"""Build and re-sample vLLM generic-affinity candidates."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kunpeng_affinity.adapters.vllm_contract import CAPABILITY_PROFILE
from kunpeng_affinity.core.errors import AffinityDiscoveryError
from kunpeng_affinity.core.models import (
    BatchAffinityResult,
    DeviceContext,
    DeviceResolution,
)
from kunpeng_affinity.policy import GenericAffinityProvider
from kunpeng_affinity.providers import (
    IluvatarRuntimeProvider,
    ProviderRegistry,
    VllmPlatformProvider,
)
from kunpeng_affinity.topology.analyzer import analyze_bdf


def vllm_device_count(platform: Any) -> int:
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
    requested_provider: str | None = None,
) -> ProviderRegistry:
    """Build providers, preferring direct platform BDF over runtime fallback."""
    registry = ProviderRegistry()
    if requested_provider == IluvatarRuntimeProvider.name:
        registry.register(
            IluvatarRuntimeProvider(
                platform,
                ixsmi=os.environ.get("KUNPENG_AFFINITY_IXSMI", "ixsmi"),
            )
        )
    elif requested_provider == VllmPlatformProvider.name or _has_direct_bdf(platform):
        registry.register(VllmPlatformProvider(platform))
    else:
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
    count = vllm_device_count(platform)
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
    active_registry = registry or create_vllm_provider_registry(
        platform,
        requested_provider=requested_provider,
    )
    mapper = active_registry.select(contexts, requested=requested_provider)
    resolver = GenericAffinityProvider(
        mapper,
        sysfs_root=sysfs_root,
        allowed_cpus=allowed_cpus,
        snapshot_metadata={
            "adapter": CAPABILITY_PROFILE,
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
        snapshot_metadata={"adapter": CAPABILITY_PROFILE},
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
    active_registry = registry or create_vllm_provider_registry(
        platform,
        requested_provider=requested_provider,
    )
    batch = GenericAffinityProvider(
        registry=active_registry,
        requested_provider=requested_provider,
        sysfs_root=sysfs_root,
        allowed_cpus=allowed_cpus,
        snapshot_metadata={
            "adapter": CAPABILITY_PROFILE,
        },
    ).resolve_all(contexts)
    return batch
