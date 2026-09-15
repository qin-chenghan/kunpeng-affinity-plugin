"""Framework-neutral BDF provider backed by explicit Linux context facts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from kunpeng_affinity.core.errors import DeviceMappingError
from kunpeng_affinity.core.models import DeviceContext, DeviceMapping, ProbeResult
from kunpeng_affinity.topology.analyzer import normalize_bdf


class LinuxContextProvider:
    """Map devices only when the runtime context carries a provable PCI BDF.

    This provider deliberately does not infer a BDF from a logical device ID,
    directory ordering, PCI bus numbers, or device names. A target GPU runtime
    provider can enrich ``DeviceContext`` and use the same mapper contract.
    """

    name = "linux-context"
    supports_shared_bdf = False

    def __init__(self, *, sysfs_root: Path | str = Path("/sys")) -> None:
        self.sysfs_root = Path(sysfs_root)

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
        mappings: list[DeviceMapping] = []
        seen_bdfs: set[str] = set()
        for context in contexts:
            raw_bdf, source, evidence = self._candidate(context)
            try:
                bdf = normalize_bdf(raw_bdf)
            except (TypeError, ValueError) as exc:
                raise DeviceMappingError(
                    f"invalid BDF for logical device {context.logical_device_id}: "
                    f"{raw_bdf!r}",
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
                    source=source,
                    physical_device_id=context.runtime_device_id,
                    evidence=evidence,
                )
            )
        return tuple(mappings)

    def _candidate(self, context: DeviceContext) -> tuple[str, str, tuple[str, ...]]:
        if context.explicit_bdf is not None:
            return (
                context.explicit_bdf,
                "explicit-config",
                ("context.explicit_bdf",),
            )

        runtime_bdf = self._normalize_candidate(context.runtime_device_id)
        if runtime_bdf is not None:
            return (
                runtime_bdf,
                "framework-context",
                ("context.runtime_device_id",),
            )

        node_bdf = self._bdf_from_device_node(context.device_node)
        if node_bdf is not None:
            return (
                node_bdf,
                "linux-device-context",
                ("context.device_node",),
            )

        raise DeviceMappingError(
            f"logical device {context.logical_device_id} has no provable PCI BDF "
            "in explicit_bdf, runtime_device_id, or device_node",
            code="DEVICE_MAPPING_UNAVAILABLE",
        )

    @staticmethod
    def _normalize_candidate(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        try:
            return normalize_bdf(value)
        except ValueError:
            return None

    def _bdf_from_device_node(self, value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            return None
        devices_root = (self.sysfs_root / "devices").resolve()
        try:
            relative = resolved.relative_to(devices_root)
        except ValueError:
            return None
        for component in reversed(relative.parts):
            candidate = self._normalize_candidate(component)
            if candidate is not None:
                return candidate
        return None
