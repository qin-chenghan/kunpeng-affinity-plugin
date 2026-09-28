#!/usr/bin/env python3
"""Read-only Ascend logical-device to NUMA affinity probe.

This command validates the Ascend provider without starting vLLM or changing
CPU or memory affinity:

    ASCEND_RT_VISIBLE_DEVICES -> sysfs dev_id -> PCI BDF -> Linux topology
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from kunpeng_affinity.core.models import DeviceContext
from kunpeng_affinity.providers.ascend_sysfs import AscendSysfsProvider
from kunpeng_affinity.topology.analyzer import analyze_bdf


def _load_platform() -> Any:
    try:
        from vllm.platforms import current_platform
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not importable; run this probe in the target runtime "
            "environment"
        ) from exc
    return current_platform


def probe(
    platform: Any,
    *,
    sysfs_root: Path | str = Path("/sys"),
) -> list[dict[str, Any]]:
    """Resolve and analyze every visible Ascend device without side effects."""
    count_method = getattr(platform, "device_count", None)
    if not callable(count_method):
        raise RuntimeError("vLLM platform does not expose device_count()")
    count = count_method()
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise RuntimeError(f"invalid visible device count: {count!r}")

    contexts = tuple(
        DeviceContext(framework="vllm", logical_device_id=device_id)
        for device_id in range(count)
    )
    provider = AscendSysfsProvider(sysfs_root=sysfs_root)
    mappings = provider.map_all(contexts)
    records: list[dict[str, Any]] = []
    for context, mapping in zip(contexts, mappings, strict=True):
        affinity = analyze_bdf(
            mapping.pci_bdf,
            sysfs_root=sysfs_root,
            mapping_source=mapping.source,
        )
        records.append(
            {
                "logical_device_id": context.logical_device_id,
                "physical_device_id": mapping.physical_device_id,
                "pci_bdf": mapping.pci_bdf,
                "mapping_source": mapping.source,
                **affinity.to_dict(),
            }
        )
    return records


def _format_text(records: list[dict[str, Any]]) -> str:
    lines = [
        "Kunpeng affinity Ascend Sysfs Provider probe (read-only)",
        "logical_device_id | physical_device_id | pci_bdf | numa | target_cpus | status",
    ]
    for record in records:
        lines.append(
            "{logical_device_id} | {physical_device_id} | {pci_bdf} | "
            "{numa_node} | {target_cpus} | {status}".format(**record)
        )
        path = " -> ".join(node["bdf"] for node in record["pci_path"])
        lines.append(f"  pci_path: {path or '<unavailable>'}")
        lines.append(f"  numa_source: {record['numa_source'] or '<unproven>'}")
        for diagnostic in record["diagnostics"]:
            lines.append(f"  diagnostic: {diagnostic}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sysfs-root", default="/sys")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        records = probe(_load_platform(), sysfs_root=args.sysfs_root)
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
