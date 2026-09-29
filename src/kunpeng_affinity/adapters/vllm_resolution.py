"""Resolve vLLM native or generic NUMA affinity candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from kunpeng_affinity.adapters import vllm_candidate
from kunpeng_affinity.adapters.vllm_eligibility import (
    check_vllm_generic_eligibility,
)
from kunpeng_affinity.adapters.vllm_native import classify_vllm_native_result
from kunpeng_affinity.core.errors import (
    AffinityDiscoveryError,
    NativeContractError,
)
from kunpeng_affinity.core.models import NativeStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutomaticAffinityResolution:
    """Complete NUMA result produced before vLLM configuration is changed."""

    nodes: tuple[int, ...]
    source: str
    visibility_fingerprint: str | None
    registry: Any | None
    requested_provider: str | None = None
    snapshot_json: str | None = None


def provider_registry(platform: Any, requested_provider: str | None = None) -> Any:
    """Create the provider registry used by vLLM resolution."""
    return vllm_candidate.create_vllm_provider_registry(
        platform,
        requested_provider=requested_provider,
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


def resolve_generic_nodes(
    numa_utils: Any,
    platform: Any,
    registry: Any,
    *,
    requested_provider: str | None,
    process_kind: str,
    local_rank: int | None,
    dp_local_rank: int | None,
    eligibility_checker: Any | None = None,
    generic_resolver: Any | None = None,
) -> AutomaticAffinityResolution:
    """Resolve a complete generic NUMA result through the provider pipeline."""
    if eligibility_checker is None:
        eligibility_checker = check_vllm_generic_eligibility
    if generic_resolver is None:
        generic_resolver = vllm_candidate.resolve_vllm_generic_affinity
    eligibility_checker(numa_utils)

    # Resolve every visible device as one batch so a partial result cannot be used.
    batch = generic_resolver(
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
    return AutomaticAffinityResolution(
        nodes=tuple(_nodes_from_generic_batch(batch)),
        source="generic",
        visibility_fingerprint=batch.visibility_fingerprint,
        registry=registry,
        requested_provider=requested_provider,
        snapshot_json=batch.snapshot_json,
    )


def resolve_automatic_nodes(
    numa_utils: Any,
    platform: Any,
    *,
    force_generic: bool,
    requested_provider: str | None = None,
    process_kind: str = "worker",
    local_rank: int | None = None,
    dp_local_rank: int | None = None,
    native_classifier: Any | None = None,
    registry_factory: Any | None = None,
    generic_resolution: Any | None = None,
) -> AutomaticAffinityResolution:
    """Select native NUMA results or resolve a complete generic candidate."""
    if native_classifier is None:
        native_classifier = classify_vllm_native_result
    if registry_factory is None:
        registry_factory = provider_registry
    if generic_resolution is None:
        generic_resolution = resolve_generic_nodes
    registry = None
    if not force_generic:
        # Classify the native result before deciding whether fallback is legal.
        outcome = native_classifier(numa_utils, platform)
        if outcome.status is NativeStatus.ERROR:
            assert outcome.original_error is not None
            raise outcome.original_error
        if outcome.status is NativeStatus.INVALID:
            raise NativeContractError(
                "vLLM native NUMA result violates its return contract: "
                + (outcome.failure_code or "NATIVE_RESULT_INVALID"),
                code=outcome.failure_code or "NATIVE_RESULT_INVALID",
            )
        if outcome.status is NativeStatus.VALID:
            return AutomaticAffinityResolution(
                nodes=tuple(outcome.nodes),
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
    registry = registry_factory(platform, requested_provider)
    return generic_resolution(
        numa_utils,
        platform,
        registry,
        requested_provider=requested_provider,
        process_kind=process_kind,
        local_rank=local_rank,
        dp_local_rank=dp_local_rank,
    )


def revalidate_visibility(
    resolution: AutomaticAffinityResolution,
    platform: Any,
    *,
    process_kind: str,
    local_rank: int | None,
    dp_local_rank: int | None,
) -> None:
    """Ensure device identity and topology stayed stable before commit."""
    if resolution.visibility_fingerprint is None:
        return

    current = vllm_candidate.resolve_vllm_visibility_fingerprint(
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
