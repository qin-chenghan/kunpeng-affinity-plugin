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

    def get_device_bdf(self, device_id: int) -> str | None:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is None:
            return None
        for name in ("get_device_pci_bus_id", "_get_device_pci_bus_id"):
            method = getattr(cuda, name, None)
            if callable(method):
                value = method(device_id)
                return _valid_bdf(value)
        properties = cuda.get_device_properties(device_id)
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
    """Resolve one SGLang GPU id to a proven Linux NUMA node."""
    # Reuse the shared batch resolver so SGLang and vLLM apply the same
    # mapping validation and topology rules.
    if torch_module is None:
        import torch

        torch_module = torch
    from kunpeng_affinity.policy import GenericAffinityProvider

    platform = SglangTorchPlatform(torch_module)
    context = DeviceContext(framework="sglang", logical_device_id=gpu_id)
    batch = GenericAffinityProvider(
        SglangRuntimeProvider(platform, ixsmi=ixsmi),
        sysfs_root=sysfs_root,
    ).resolve_all((context,))
    if not batch.committable:
        reason = "; ".join(batch.failure_summary) or "topology result is not committable"
        raise DeviceMappingError(reason, code="BATCH_NOT_COMMITTABLE")
    node = batch.ordered_results[0].affinity.numa_node
    if node is None:
        raise DeviceMappingError(
            f"NUMA node is unproven for SGLang GPU {gpu_id}",
            code="NUMA_UNKNOWN",
        )
    return node
