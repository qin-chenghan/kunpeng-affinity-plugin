"""Ascend logical-device to PCI-BDF mapping through environment and sysfs."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from kunpeng_affinity.core.errors import DeviceMappingError
from kunpeng_affinity.core.models import DeviceContext, DeviceMapping, ProbeResult
from kunpeng_affinity.topology.analyzer import normalize_bdf

ASCEND_VISIBLE_DEVICES = "ASCEND_RT_VISIBLE_DEVICES"
_BDF_TO_DEVID = "devdrv_sysfs_bdf_to_devid"
_TABLE_ROW = re.compile(
    r"^\s*(?P<bdf>[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:"
    r"[0-9a-fA-F]{2}\.[0-7])\s*(?:-+>|=+>|->)\s*(?P<dev_id>\d+)\s*$"
)


def parse_bdf_to_devid_table(text: str) -> dict[int, str]:
    """Parse one Ascend driver's BDF-to-device-id sysfs table."""
    result: dict[int, str] = {}
    bdfs: set[str] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _TABLE_ROW.fullmatch(line)
        if match is None:
            raise DeviceMappingError(
                f"invalid Ascend BDF-to-device-id row: {line!r}",
                code="RUNTIME_IDENTITY_INVALID",
            )
        try:
            bdf = normalize_bdf(match.group("bdf"))
        except ValueError as exc:
            raise DeviceMappingError(
                f"invalid Ascend PCI BDF in sysfs table: {match.group('bdf')!r}",
                code="BDF_INVALID",
            ) from exc
        dev_id = int(match.group("dev_id"))
        if dev_id in result or bdf in bdfs:
            raise DeviceMappingError(
                "Ascend sysfs table contains duplicate device identity",
                code="RUNTIME_IDENTITY_CONFLICT",
            )
        result[dev_id] = bdf
        bdfs.add(bdf)
    if not result:
        raise DeviceMappingError(
            "Ascend sysfs BDF-to-device-id table is empty",
            code="RUNTIME_IDENTITY_UNAVAILABLE",
        )
    return result


def parse_visible_device_ids(value: object) -> tuple[int, ...]:
    """Parse the ordered physical IDs accepted by Ascend Runtime."""
    if not isinstance(value, str) or not value.strip():
        raise DeviceMappingError(
            f"{ASCEND_VISIBLE_DEVICES} must be explicitly set",
            code="RUNTIME_VISIBILITY_UNAVAILABLE",
        )
    ids: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token or not token.isdecimal():
            raise DeviceMappingError(
                f"invalid {ASCEND_VISIBLE_DEVICES} entry: {token!r}",
                code="RUNTIME_VISIBILITY_INVALID",
            )
        device_id = int(token)
        if device_id in ids:
            raise DeviceMappingError(
                f"{ASCEND_VISIBLE_DEVICES} contains duplicate device id {device_id}",
                code="RUNTIME_VISIBILITY_CONFLICT",
            )
        ids.append(device_id)
    return tuple(ids)


class AscendSysfsProvider:
    """Map Ascend visible IDs to BDFs without calling an Ascend runtime API.

    The visibility variable supplies the ordered physical device IDs. The
    Ascend driver supplies a read-only BDF-to-device-ID table through sysfs;
    the generic topology layer consumes the resulting BDFs afterward.
    """

    name = "ascend-sysfs-pci"
    supports_shared_bdf = False

    def __init__(
        self,
        *,
        sysfs_root: Path | str = Path("/sys"),
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.sysfs_root = Path(sysfs_root)
        self.environ = os.environ if environ is None else environ
        self._probe_cache: tuple[
            tuple[DeviceContext, ...], tuple[DeviceMapping, ...]
        ] | None = None

    def probe(self, contexts: Sequence[DeviceContext]) -> ProbeResult:
        ordered_contexts = tuple(contexts)
        try:
            mappings = self._map_all(ordered_contexts)
        except DeviceMappingError as exc:
            self._probe_cache = None
            return ProbeResult(
                provider=self.name,
                supported=False,
                reason=str(exc),
            )
        self._probe_cache = (ordered_contexts, mappings)
        return ProbeResult(provider=self.name, supported=True)

    def map_all(self, contexts: Sequence[DeviceContext]) -> tuple[DeviceMapping, ...]:
        ordered_contexts = tuple(contexts)
        cached = self._probe_cache
        self._probe_cache = None
        if cached is not None and cached[0] == ordered_contexts:
            return cached[1]
        return self._map_all(ordered_contexts)

    def _map_all(
        self, contexts: tuple[DeviceContext, ...]
    ) -> tuple[DeviceMapping, ...]:
        visible_ids = parse_visible_device_ids(
            self.environ.get(ASCEND_VISIBLE_DEVICES)
        )
        if len(visible_ids) != len(contexts):
            raise DeviceMappingError(
                f"{ASCEND_VISIBLE_DEVICES} exposes {len(visible_ids)} devices, "
                f"but the framework exposes {len(contexts)}",
                code="DEVICE_COUNT_MISMATCH",
            )
        dev_id_to_bdf = self._read_mapping()
        mappings: list[DeviceMapping] = []
        for context in contexts:
            logical_id = context.logical_device_id
            if not isinstance(logical_id, int) or isinstance(logical_id, bool):
                raise DeviceMappingError(
                    f"invalid logical device ID {logical_id!r}",
                    code="DEVICE_MAPPING_INVALID",
                )
            if logical_id < 0 or logical_id >= len(visible_ids):
                raise DeviceMappingError(
                    f"logical device ID {logical_id} is outside visible range",
                    code="DEVICE_MAPPING_INVALID",
                )
            physical_id = visible_ids[logical_id]
            try:
                bdf = dev_id_to_bdf[physical_id]
            except KeyError as exc:
                raise DeviceMappingError(
                    f"Ascend sysfs has no BDF for physical device {physical_id}",
                    code="DEVICE_MAPPING_MISSING",
                ) from exc
            mappings.append(
                DeviceMapping(
                    logical_device_id=context.logical_device_id,
                    pci_bdf=bdf,
                    source=self.name,
                    physical_device_id=physical_id,
                    evidence=(
                        f"environment:{ASCEND_VISIBLE_DEVICES}",
                        "sysfs:devdrv_sysfs_bdf_to_devid",
                        f"physical_device_id={physical_id}",
                    ),
                )
            )
        return tuple(mappings)

    def _read_mapping(self) -> dict[int, str]:
        attribute_paths = sorted(
            (
                path
                for path in (
                    self.sysfs_root
                    / "bus/pci/devices"
                ).glob(f"*/{_BDF_TO_DEVID}")
                if path.is_file()
            ),
            key=lambda path: str(path),
        )
        if not attribute_paths:
            raise DeviceMappingError(
                f"Ascend sysfs attribute {_BDF_TO_DEVID!r} was not found",
                code="RUNTIME_IDENTITY_UNAVAILABLE",
            )
        tables: list[dict[int, str]] = []
        for path in attribute_paths:
            try:
                tables.append(parse_bdf_to_devid_table(path.read_text(encoding="ascii")))
            except UnicodeDecodeError as exc:
                raise DeviceMappingError(
                    f"Ascend sysfs attribute is not ASCII: {path}",
                    code="RUNTIME_IDENTITY_INVALID",
                ) from exc
            except OSError as exc:
                raise DeviceMappingError(
                    f"cannot read Ascend sysfs attribute {path}: {exc}",
                    code="RUNTIME_QUERY_UNAVAILABLE",
                ) from exc
        first = tables[0]
        if any(table != first for table in tables[1:]):
            raise DeviceMappingError(
                "Ascend sysfs BDF-to-device-id tables disagree",
                code="RUNTIME_IDENTITY_CONFLICT",
            )
        return first
