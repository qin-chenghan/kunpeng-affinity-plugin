"""Explicit mapping provider used for deployment configuration and tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from kunpeng_affinity.core.errors import DeviceMappingError
from kunpeng_affinity.core.models import DeviceContext, DeviceMapping, ProbeResult
from kunpeng_affinity.topology.analyzer import normalize_bdf


class StaticMappingProvider:
    """Map logical IDs using a validated, deployment-supplied mapping.

    This provider deliberately does not infer device order. It is also useful
    as a contract test double before a target runtime provider exists.
    """

    name = "static"
    supports_shared_bdf = False

    def __init__(self, mapping: Mapping[int, str]) -> None:
        self._mapping = dict(mapping)

    def probe(self, contexts: Sequence[DeviceContext]) -> ProbeResult:
        missing = [
            context.logical_device_id
            for context in contexts
            if context.explicit_bdf is None and context.logical_device_id not in self._mapping
        ]
        if missing:
            return ProbeResult(
                provider=self.name,
                supported=False,
                reason=f"missing logical device mappings: {missing}",
            )
        return ProbeResult(provider=self.name, supported=True)

    def map_all(self, contexts: Sequence[DeviceContext]) -> tuple[DeviceMapping, ...]:
        mappings: list[DeviceMapping] = []
        seen_bdfs: set[str] = set()
        for context in contexts:
            raw_bdf = context.explicit_bdf or self._mapping.get(context.logical_device_id)
            if raw_bdf is None:
                raise DeviceMappingError(
                    f"no PCI BDF mapping for logical device {context.logical_device_id}",
                    code="DEVICE_MAPPING_MISSING",
                )
            try:
                bdf = normalize_bdf(raw_bdf)
            except (TypeError, ValueError) as exc:
                raise DeviceMappingError(
                    f"invalid BDF for logical device {context.logical_device_id}: {raw_bdf!r}",
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
                    source="explicit-config",
                    evidence=("deployment mapping",),
                )
            )
        return tuple(mappings)
