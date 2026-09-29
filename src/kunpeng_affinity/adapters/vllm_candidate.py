"""Build and re-sample vLLM generic-affinity candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kunpeng_affinity.adapters.vllm_contract import CAPABILITY_PROFILE
from kunpeng_affinity.adapters.vllm_devices import (
    build_vllm_device_contexts,
    create_vllm_provider_registry,
)
from kunpeng_affinity.core.errors import AffinityDiscoveryError
from kunpeng_affinity.core.models import (
    BatchAffinityResult,
    DeviceResolution,
)
from kunpeng_affinity.policy import GenericAffinityProvider
from kunpeng_affinity.providers import ProviderRegistry
from kunpeng_affinity.topology.analyzer import analyze_bdf


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
        sysfs_root=sysfs_root,
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
        sysfs_root=sysfs_root,
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
        sysfs_root=sysfs_root,
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
