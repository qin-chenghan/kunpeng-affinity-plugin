"""Iluvatar runtime provider for logical-device to PCI-BDF mapping."""

from __future__ import annotations

import csv
import io
import re
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from kunpeng_affinity.core.errors import DeviceMappingError
from kunpeng_affinity.core.models import DeviceContext, DeviceMapping, ProbeResult
from kunpeng_affinity.topology.analyzer import normalize_bdf

_UUID_RE = re.compile(
    r"^(?:GPU-)?(?P<value>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)


def normalize_uuid(value: object) -> str:
    """Normalize UUIDs returned by vLLM, torch-compatible runtimes or ixsmi."""
    if not isinstance(value, str):
        raise ValueError(f"GPU UUID must be a string, got {type(value).__name__}")
    match = _UUID_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid GPU UUID: {value!r}")
    return match.group("value").lower()


def parse_ixsmi_rows(output: str) -> tuple[tuple[int, str, str], ...]:
    """Parse ixsmi CSV rows into physical index, UUID and normalized BDF."""
    rows: list[tuple[int, str, str]] = []
    seen_uuids: set[str] = set()
    seen_bdfs: set[str] = set()
    for fields in csv.reader(io.StringIO(output)):
        values = [field.strip() for field in fields]
        if not values:
            continue
        uuid_index = next(
            (index for index, value in enumerate(values) if _UUID_RE.fullmatch(value)),
            None,
        )
        bdf_index = next(
            (index for index, value in enumerate(values) if _is_bdf(value)),
            None,
        )
        if uuid_index is None or bdf_index is None:
            continue
        try:
            gpu_uuid = normalize_uuid(values[uuid_index])
            bdf = normalize_bdf(values[bdf_index])
        except (TypeError, ValueError) as exc:
            raise DeviceMappingError(
                f"invalid ixsmi identity row: {values!r}",
                code="RUNTIME_IDENTITY_INVALID",
            ) from exc
        if bdf is None:
            raise DeviceMappingError(
                f"ixsmi returned an invalid PCI BDF: {values[bdf_index]!r}",
                code="BDF_INVALID",
            )
        index = _physical_index(values, uuid_index, bdf_index)
        if gpu_uuid in seen_uuids or bdf in seen_bdfs:
            raise DeviceMappingError(
                "ixsmi returned duplicate GPU UUID or PCI BDF",
                code="RUNTIME_IDENTITY_CONFLICT",
            )
        seen_uuids.add(gpu_uuid)
        seen_bdfs.add(bdf)
        rows.append((index, gpu_uuid, bdf))
    if not rows:
        raise DeviceMappingError(
            "ixsmi returned no rows containing UUID and PCI BDF",
            code="RUNTIME_IDENTITY_UNAVAILABLE",
        )
    return tuple(rows)


def _is_bdf(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]",
            value,
        )
    )


def _physical_index(values: list[str], uuid_index: int, bdf_index: int) -> int:
    for value in values[: min(uuid_index, bdf_index)]:
        if value.isdigit():
            return int(value)
    raise DeviceMappingError(
        f"ixsmi row has no physical GPU index: {values!r}",
        code="RUNTIME_IDENTITY_INVALID",
    )


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class IluvatarRuntimeProvider:
    """Resolve vLLM logical devices through UUID and the Iluvatar utility.

    vLLM supplies the current process's logical-device UUID, which follows
    visibility remapping. ``ixsmi`` supplies the host-wide UUID-to-BDF table.
    The provider never uses ixsmi's physical enumeration order as a logical
    device mapping.
    """

    name = "iluvatar-runtime-pci"
    supports_shared_bdf = False

    def __init__(
        self,
        platform: Any,
        *,
        ixsmi: str = "ixsmi",
        timeout: float = 10.0,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.platform = platform
        self.ixsmi = ixsmi
        self.timeout = timeout
        self._command_runner = command_runner or subprocess.run

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
        # Query the host-wide inventory once, then join each visible device by UUID.
        uuid_to_row = self._inventory()
        mappings: list[DeviceMapping] = []
        seen_bdfs: set[str] = set()
        for context in contexts:
            # Runtime UUID follows visibility remapping; physical index order does not.
            gpu_uuid = self._device_uuid(context.logical_device_id)
            try:
                physical_id, normalized_uuid, bdf = uuid_to_row[gpu_uuid]
            except KeyError as exc:
                raise DeviceMappingError(
                    f"ixsmi has no PCI BDF for vLLM device UUID {gpu_uuid}",
                    code="DEVICE_MAPPING_MISSING",
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
                        "vllm.platform.get_device_uuid",
                        "ixsmi --query-gpu=uuid,pci.bus_id",
                        f"uuid={normalized_uuid}",
                    ),
                )
            )
        return tuple(mappings)

    def _device_uuid(self, logical_device_id: int) -> str:
        method = getattr(self.platform, "get_device_uuid", None)
        if not callable(method):
            raise DeviceMappingError(
                "vLLM platform does not expose get_device_uuid",
                code="PROVIDER_UNAVAILABLE",
            )
        try:
            return normalize_uuid(method(logical_device_id))
        except (TypeError, ValueError, NotImplementedError, RuntimeError, OSError) as exc:
            raise DeviceMappingError(
                f"vLLM GPU UUID query failed for logical device "
                f"{logical_device_id}: {exc}",
                code="RUNTIME_IDENTITY_UNAVAILABLE",
            ) from exc

    def _inventory(self) -> dict[str, tuple[int, str, str]]:
        # ixsmi is used only as a read-only UUID-to-BDF inventory source.
        try:
            completed = self._command_runner(
                [
                    self.ixsmi,
                    "--query-gpu=index,uuid,pci.bus_id",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
            raise DeviceMappingError(
                f"cannot execute Iluvatar utility {self.ixsmi!r}: {exc}",
                code="RUNTIME_QUERY_UNAVAILABLE",
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise DeviceMappingError(
                f"Iluvatar utility exited with status {completed.returncode}: {detail}",
                code="RUNTIME_QUERY_FAILED",
            )
        rows = parse_ixsmi_rows(completed.stdout)
        return {gpu_uuid: (physical_id, gpu_uuid, bdf) for physical_id, gpu_uuid, bdf in rows}
