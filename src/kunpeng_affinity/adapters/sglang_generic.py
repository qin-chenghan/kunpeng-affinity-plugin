"""SGLang runtime identity adapter for the generic Linux topology path."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kunpeng_affinity.core.errors import DeviceMappingError
from kunpeng_affinity.core.models import DeviceContext, DeviceMapping, ProbeResult
from kunpeng_affinity.providers.iluvatar_runtime import IluvatarRuntimeProvider
from kunpeng_affinity.topology.analyzer import normalize_bdf


class SglangTorchPlatform:
    """Small platform facade built from the SGLang process's torch runtime.

    SGLang does not expose the vLLM platform SPI. The facade keeps framework
    imports at the adapter boundary and lets the existing provider contract be
    reused. UUID lookup is intentionally runtime-based; visible device indexes
    are never joined to host inventory by position.
    """

    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module

    def get_device_uuid(self, device_id: int) -> str:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is None:
            raise DeviceMappingError("torch runtime has no cuda module", code="PROVIDER_UNAVAILABLE")
        for name in ("get_device_uuid", "_get_device_uuid"):
            method = getattr(cuda, name, None)
            if callable(method):
                return _as_text(method(device_id))
        properties = cuda.get_device_properties(device_id)
        for name in ("uuid", "device_uuid"):
            value = getattr(properties, name, None)
            if value is not None:
                return _as_text(value)
        raise DeviceMappingError(
            "torch device properties expose no GPU UUID",
            code="RUNTIME_IDENTITY_UNAVAILABLE",
        )

    def device_count(self) -> int:
        cuda = getattr(self.torch, "cuda", None)
        method = getattr(cuda, "device_count", None)
        if not callable(method):
            raise DeviceMappingError(
                "torch runtime does not expose device_count",
                code="PROVIDER_UNAVAILABLE",
            )
        try:
            count = method()
        except (NotImplementedError, RuntimeError, OSError, ValueError) as exc:
            raise DeviceMappingError(
                f"torch runtime device count query failed: {exc}",
                code="PROVIDER_UNAVAILABLE",
            ) from exc
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise DeviceMappingError(
                f"torch runtime returned invalid device count {count!r}",
                code="DEVICE_COUNT_INVALID",
            )
        return count

    def get_device_bdf(self, device_id: int) -> str | None:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is None:
            return None
        for name in ("get_device_pci_bus_id", "_get_device_pci_bus_id"):
            method = getattr(cuda, name, None)
            if callable(method):
                try:
                    value = method(device_id)
                except (NotImplementedError, RuntimeError, OSError, ValueError) as exc:
                    raise DeviceMappingError(
                        f"torch PCI BDF query failed for device {device_id}: {exc}",
                        code="RUNTIME_IDENTITY_UNAVAILABLE",
                    ) from exc
                return _valid_bdf(value)
        try:
            properties = cuda.get_device_properties(device_id)
        except (AttributeError, RuntimeError, OSError, ValueError) as exc:
            raise DeviceMappingError(
                f"torch device properties query failed for device {device_id}: {exc}",
                code="RUNTIME_IDENTITY_UNAVAILABLE",
            ) from exc
        for name in ("pci_bus_id", "pci_bus_id_string", "bus_id"):
            value = getattr(properties, name, None)
            if value is not None:
                return _valid_bdf(value)
        return None


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii")
    return value if isinstance(value, str) else str(value)


def _valid_bdf(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return normalize_bdf(value)
    except ValueError:
        return None


class SglangRuntimeProvider:
    """Map SGLang visible GPU ids using runtime BDF or UUID evidence."""

    name = "sglang-runtime-pci"
    supports_shared_bdf = False

    def __init__(
        self,
        platform: SglangTorchPlatform,
        *,
        ixsmi: str = "ixsmi",
        timeout: float = 10.0,
        command_runner: Any | None = None,
    ) -> None:
        self.platform = platform
        self.ixsmi = ixsmi
        self.timeout = timeout
        self.command_runner = command_runner

    def probe(self, contexts: Sequence[DeviceContext]) -> ProbeResult:
        try:
            self.map_all(contexts)
        except DeviceMappingError as exc:
            return ProbeResult(provider=self.name, supported=False, reason=str(exc))
        return ProbeResult(provider=self.name, supported=True)

    def map_all(self, contexts: Sequence[DeviceContext]) -> tuple[DeviceMapping, ...]:
        # Prefer a complete direct runtime BDF mapping when the runtime exposes it.
        direct = tuple(self.platform.get_device_bdf(c.logical_device_id) for c in contexts)
        if all(bdf is not None for bdf in direct):
            return tuple(
                DeviceMapping(
                    logical_device_id=context.logical_device_id,
                    pci_bdf=bdf,
                    source=self.name,
                    evidence=("torch.cuda PCI BDF runtime API",),
                )
                for context, bdf in zip(contexts, direct, strict=True)
            )
        if any(bdf is not None for bdf in direct):
            raise DeviceMappingError(
                "SGLang runtime returned only a partial direct BDF mapping",
                code="DEVICE_MAPPING_PARTIAL",
            )
        # A complete direct mapping is required; otherwise use UUID-to-BDF evidence.
        provider = IluvatarRuntimeProvider(
            self.platform,
            ixsmi=self.ixsmi,
            timeout=self.timeout,
            command_runner=self.command_runner,
        )
        mappings = provider.map_all(contexts)
        return tuple(
            DeviceMapping(
                logical_device_id=item.logical_device_id,
                pci_bdf=item.pci_bdf,
                source=self.name,
                physical_device_id=item.physical_device_id,
                evidence=("torch runtime UUID", *item.evidence),
            )
            for item in mappings
        )


def resolve_sglang_numa_node(
    gpu_id: int,
    *,
    torch_module: Any | None = None,
    sysfs_root: Path | str = Path("/sys"),
    ixsmi: str = "ixsmi",
) -> int:
    """Resolve one visible SGLang GPU after validating the complete batch."""
    # Reuse the shared batch resolver so SGLang and vLLM apply the same
    # mapping validation and topology rules.
    if torch_module is None:
        import torch

        torch_module = torch
    from kunpeng_affinity.policy import GenericAffinityProvider

    platform = SglangTorchPlatform(torch_module)
    count = platform.device_count()
    if gpu_id < 0 or gpu_id >= count:
        raise DeviceMappingError(
            f"SGLang GPU id {gpu_id} is outside visible range 0..{count - 1}",
            code="DEVICE_INDEX_INVALID",
        )
    contexts = tuple(
        DeviceContext(framework="sglang", logical_device_id=device_id)
        for device_id in range(count)
    )
    resolver = GenericAffinityProvider(
        SglangRuntimeProvider(platform, ixsmi=ixsmi),
        sysfs_root=sysfs_root,
        snapshot_metadata={
            "adapter": "sglang.numa_query.v1",
        },
    )
    batch = resolver.resolve_all(contexts)
    if not batch.committable:
        reason = "; ".join(batch.failure_summary) or "topology result is not committable"
        raise DeviceMappingError(reason, code="BATCH_NOT_COMMITTABLE")
    # Re-sample the same complete batch immediately before returning the node.
    current = resolver.resolve_all(contexts)
    if (
        not current.committable
        or current.visibility_fingerprint != batch.visibility_fingerprint
        or tuple(item.affinity.numa_node for item in current.ordered_results)
        != tuple(item.affinity.numa_node for item in batch.ordered_results)
    ):
        raise DeviceMappingError(
            "SGLang device visibility or NUMA topology changed during resolution",
            code="SNAPSHOT_CHANGED",
        )
    node = batch.ordered_results[gpu_id].affinity.numa_node
    if node is None:
        raise DeviceMappingError(
            f"NUMA node is unproven for SGLang GPU {gpu_id}",
            code="NUMA_UNKNOWN",
        )
    return node
