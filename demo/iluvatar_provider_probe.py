#!/usr/bin/env python3
"""Read-only Iluvatar logical-device to NUMA affinity probe.

This command intentionally stops before any affinity or memory-policy change.
It validates the complete discovery path used by the runtime provider:

    SGLang/vLLM visible device -> runtime UUID -> ixsmi BDF -> Linux topology
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kunpeng_affinity.core.models import DeviceContext
from kunpeng_affinity.providers.iluvatar_runtime import IluvatarRuntimeProvider
from kunpeng_affinity.topology.analyzer import analyze_bdf


def _load_platform() -> Any:
    try:
        from vllm.platforms import current_platform
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not importable; run this probe in the target runtime "
            "environment or provide its source on PYTHONPATH"
        ) from exc
    return current_platform


def _device_count(platform: Any) -> int:
    method = getattr(platform, "device_count", None)
    if not callable(method):
        raise RuntimeError("vLLM platform does not expose device_count()")
    count = method()
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise RuntimeError(f"invalid visible device count: {count!r}")
    return count


def probe(
    platform: Any,
    *,
    sysfs_root: Path | str = Path("/sys"),
    ixsmi: str = "ixsmi",
    device_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Resolve and analyze selected visible devices without changing the host."""
    count = _device_count(platform)
    selected = tuple(range(count)) if device_ids is None else tuple(device_ids)
    if not selected or any(device_id < 0 or device_id >= count for device_id in selected):
        raise RuntimeError(
            f"device ids must be in [0, {count}); got {list(selected)!r}"
        )

    contexts = tuple(
        DeviceContext(framework="vllm", logical_device_id=device_id)
        for device_id in selected
    )
    provider = IluvatarRuntimeProvider(platform, ixsmi=ixsmi)
    mappings = provider.map_all(contexts)
    records: list[dict[str, Any]] = []
    for context, mapping in zip(contexts, mappings, strict=True):
        affinity = analyze_bdf(
            mapping.pci_bdf,
            sysfs_root=sysfs_root,
            mapping_source=mapping.source,
        )
        record = {
            "logical_device_id": context.logical_device_id,
            "runtime_uuid": next(
                (
                    evidence.removeprefix("uuid=")
                    for evidence in mapping.evidence
                    if evidence.startswith("uuid=")
                ),
                None,
            ),
            "ixsmi_physical_index": mapping.physical_device_id,
            "pci_bdf": mapping.pci_bdf,
            "mapping_source": mapping.source,
            **affinity.to_dict(),
        }
        records.append(record)
    return records


def _format_text(records: Sequence[dict[str, Any]]) -> str:
    lines = [
        "Kunpeng affinity Iluvatar Runtime Provider probe (read-only)",
        "logical_device_id | runtime_uuid | physical_index | pci_bdf | numa | target_cpus | status",
    ]
    for record in records:
        lines.append(
            "{logical_device_id} | {runtime_uuid} | {ixsmi_physical_index} | "
            "{pci_bdf} | {numa_node} | {target_cpus} | {status}".format(**record)
        )
        path = " -> ".join(node["bdf"] for node in record["pci_path"])
        lines.append(f"  pci_path: {path or '<unavailable>'}")
        lines.append(f"  numa_source: {record['numa_source'] or '<unproven>'}")
        for diagnostic in record["diagnostics"]:
            lines.append(f"  diagnostic: {diagnostic}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ixsmi", default="ixsmi")
    parser.add_argument("--sysfs-root", default="/sys")
    parser.add_argument("--device", dest="devices", type=int, action="append")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        records = probe(
            _load_platform(),
            sysfs_root=args.sysfs_root,
            ixsmi=args.ixsmi,
            device_ids=args.devices,
        )
    except Exception as exc:
        print(f"probe_status=failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        print(_format_text(records))
    return 0 if all(record["bindable"] for record in records) else 3


if __name__ == "__main__":
    raise SystemExit(main())
