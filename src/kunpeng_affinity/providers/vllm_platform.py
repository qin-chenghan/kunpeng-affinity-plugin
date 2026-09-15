"""vLLM platform adapter for logical-device to PCI-BDF mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kunpeng_affinity.core.errors import DeviceMappingError
from kunpeng_affinity.core.models import DeviceContext, DeviceMapping, ProbeResult
from kunpeng_affinity.topology.analyzer import normalize_bdf


class VllmPlatformProvider:
    """Use vLLM's platform identity APIs without importing vLLM in core code.

    vLLM 0.23 platforms expose ``get_all_gpu_pci_bus_ids`` for physical GPU
    indices and ``device_id_to_physical_device_id`` for visible-device
    translation. A platform that does not implement either contract is not
    silently mapped by logical index.
    """

    name = "vllm-platform-pci"
    supports_shared_bdf = False

    def __init__(self, platform: Any) -> None:
        self.platform = platform

    def probe(self, contexts: Sequence[DeviceContext]) -> ProbeResult:
        try:
            self.map_all(contexts)
        except DeviceMappingError as exc:
            return ProbeResult(
                provider=self.name,
                supported=False,
                reason=str(exc),
            )
        return ProbeResult(provider=self.name, supported=True)

    def map_all(self, contexts: Sequence[DeviceContext]) -> tuple[DeviceMapping, ...]:
        bus_ids = self._bus_ids()
        mappings: list[DeviceMapping] = []
        seen_bdfs: set[str] = set()
        for context in contexts:
            physical_id = self._physical_id(context)
            try:
                raw_bdf = bus_ids[physical_id]
            except (KeyError, TypeError) as exc:
                raise DeviceMappingError(
                    f"vLLM platform has no PCI BDF for physical device "
                    f"{physical_id}",
                    code="DEVICE_MAPPING_MISSING",
                ) from exc
            try:
                bdf = normalize_bdf(raw_bdf)
            except (TypeError, ValueError) as exc:
                raise DeviceMappingError(
                    f"vLLM platform returned invalid BDF for physical device "
                    f"{physical_id}: {raw_bdf!r}",
                    code="BDF_INVALID",
                ) from exc
            if bdf in seen_bdfs:
                raise DeviceMappingError(
                    f"BDF {bdf} is mapped more than once",
                    code="DEVICE_MAPPING_CONFLICT",
                )
            seen_bdfs.add(bdf)
            mappings.append(
                DeviceMapping(
                    logical_device_id=context.logical_device_id,
                    pci_bdf=bdf,
                    source=self.name,
                    physical_device_id=physical_id,
                    evidence=(
                        "platform.get_all_gpu_pci_bus_ids",
                        f"physical_device_id={physical_id}",
                    ),
                )
            )
        return tuple(mappings)

    def _bus_ids(self) -> Mapping[int, str]:
        method = getattr(self.platform, "get_all_gpu_pci_bus_ids", None)
        if not callable(method):
            raise DeviceMappingError(
                "vLLM platform does not expose get_all_gpu_pci_bus_ids",
                code="PROVIDER_UNAVAILABLE",
            )
        try:
            bus_ids = method()
        except (NotImplementedError, RuntimeError, OSError) as exc:
            raise DeviceMappingError(
                f"vLLM platform PCI BDF query failed: {exc}",
                code="PROVIDER_UNAVAILABLE",
            ) from exc
        if not isinstance(bus_ids, Mapping) or not bus_ids:
            raise DeviceMappingError(
                "vLLM platform returned no PCI BDF mapping",
                code="DEVICE_MAPPING_MISSING",
            )
        return bus_ids

    def _physical_id(self, context: DeviceContext) -> int:
        method = getattr(self.platform, "device_id_to_physical_device_id", None)
        if not callable(method):
            raise DeviceMappingError(
                "vLLM platform does not expose visible-to-physical device mapping",
                code="PROVIDER_UNAVAILABLE",
            )
        try:
            physical_id = method(context.logical_device_id)
        except (IndexError, TypeError, ValueError, RuntimeError) as exc:
            raise DeviceMappingError(
                f"vLLM platform could not map logical device "
                f"{context.logical_device_id} to a physical device: {exc}",
                code="DEVICE_MAPPING_UNAVAILABLE",
            ) from exc
        if not isinstance(physical_id, int) or physical_id < 0:
            raise DeviceMappingError(
                f"vLLM platform returned invalid physical device ID {physical_id!r}",
                code="DEVICE_MAPPING_INVALID",
            )
        return physical_id
