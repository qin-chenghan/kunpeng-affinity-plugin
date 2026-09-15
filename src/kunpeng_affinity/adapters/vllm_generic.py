"""vLLM-specific assembly for the generic Linux topology path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kunpeng_affinity.core.errors import AffinityDiscoveryError
from kunpeng_affinity.core.models import BatchAffinityResult, DeviceContext
from kunpeng_affinity.policy import GenericAffinityProvider
from kunpeng_affinity.providers import VllmPlatformProvider


def resolve_vllm_generic_affinity(
    platform: Any,
    *,
    sysfs_root: Path | str = Path("/sys"),
    allowed_cpus: frozenset[int] | set[int] | None = None,
) -> BatchAffinityResult:
    """Resolve every vLLM-visible device through BDF and Linux sysfs."""
    device_count_method = getattr(platform, "device_count", None)
    if not callable(device_count_method):
        raise AffinityDiscoveryError(
            "vLLM platform does not expose device_count",
            code="PROVIDER_UNAVAILABLE",
        )
    try:
        device_count = device_count_method()
    except (NotImplementedError, RuntimeError, OSError) as exc:
        raise AffinityDiscoveryError(
            f"vLLM platform device count query failed: {exc}",
            code="PROVIDER_UNAVAILABLE",
        ) from exc
    if not isinstance(device_count, int) or device_count <= 0:
        raise AffinityDiscoveryError(
            f"vLLM platform returned invalid device count {device_count!r}",
            code="DEVICE_COUNT_INVALID",
        )

    contexts = tuple(
        DeviceContext(framework="vllm", logical_device_id=device_id)
        for device_id in range(device_count)
    )
    batch = GenericAffinityProvider(
        VllmPlatformProvider(platform),
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
