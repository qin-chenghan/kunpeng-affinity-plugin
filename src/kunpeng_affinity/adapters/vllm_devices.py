"""Build vLLM-visible device contexts and identity providers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kunpeng_affinity.core.errors import AffinityDiscoveryError
from kunpeng_affinity.core.models import DeviceContext
from kunpeng_affinity.providers import (
    AscendSysfsProvider,
    IluvatarRuntimeProvider,
    ProviderRegistry,
    VllmPlatformProvider,
)

VLLM_PROVIDER_NAMES = frozenset(
    {
        AscendSysfsProvider.name,
        IluvatarRuntimeProvider.name,
        VllmPlatformProvider.name,
    }
)


def vllm_device_count(platform: Any) -> int:
    """Read and validate the framework-visible accelerator count."""
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
    sysfs_root: Path | str = Path("/sys"),
) -> ProviderRegistry:
    """Build identity providers, preferring direct platform BDF support."""
    registry = ProviderRegistry()
    if requested_provider == AscendSysfsProvider.name:
        registry.register(AscendSysfsProvider(sysfs_root=sysfs_root))
    elif requested_provider == IluvatarRuntimeProvider.name:
        registry.register(
            IluvatarRuntimeProvider(
                platform,
                ixsmi=os.environ.get("KUNPENG_AFFINITY_IXSMI", "ixsmi"),
            )
        )
    elif requested_provider == VllmPlatformProvider.name or _has_direct_bdf(platform):
        registry.register(VllmPlatformProvider(platform))
    elif os.environ.get("ASCEND_RT_VISIBLE_DEVICES") is not None:
        registry.register(AscendSysfsProvider(sysfs_root=sysfs_root))
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
    count = vllm_device_count(platform)
    if allowed_cpus is not None:
        effective_allowed = frozenset(allowed_cpus)
    else:
        try:
            effective_allowed = frozenset(os.sched_getaffinity(0))
        except (AttributeError, OSError):
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
