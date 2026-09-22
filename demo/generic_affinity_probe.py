#!/usr/bin/env python3
"""Read-only, framework-independent GPU affinity probe.

The probe accepts trusted BDFs or discovers PCI accelerator candidates. It
reuses the project's Linux topology analyzer and never imports a framework,
calls a vendor utility, or changes process affinity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from kunpeng_affinity.topology.analyzer import analyze_bdf, normalize_bdf
from kunpeng_affinity.topology.cpulist import CpuListError, format_cpulist, parse_cpulist
from kunpeng_affinity.topology.models import AffinityResult, ResultStatus

_DEFAULT_CLASS_BASES = ("03", "12")


@dataclass(frozen=True)
class PciCandidate:
    """PCI metadata used to explain automatic candidate discovery."""

    bdf: str
    pci_class: str
    vendor: str
    device: str
    driver: str
    numa_node: str


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii").strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return "unknown"


def _driver_name(path: Path) -> str:
    try:
        return path.resolve(strict=True).name
    except (FileNotFoundError, OSError, RuntimeError):
        return "unknown"


def discover_pci_candidates(
    sysfs_root: Path,
    *,
    class_bases: Sequence[str] = _DEFAULT_CLASS_BASES,
    vendor_id: str | None = None,
    driver: str | None = None,
) -> tuple[PciCandidate, ...]:
    """Discover PCI candidates without claiming a logical GPU mapping."""
    wanted_classes = frozenset(item.lower().removeprefix("0x") for item in class_bases)
    wanted_vendor = vendor_id.lower() if vendor_id else None
    wanted_driver = driver.lower() if driver else None
    devices_root = sysfs_root / "bus/pci/devices"
    try:
        entries = sorted(devices_root.iterdir(), key=lambda path: path.name)
    except (FileNotFoundError, OSError):
        return ()

    candidates: list[PciCandidate] = []
    for entry in entries:
        pci_class = _read(entry / "class").lower()
        class_code = pci_class.removeprefix("0x")
        if len(class_code) < 2 or class_code[:2] not in wanted_classes:
            continue
        vendor = _read(entry / "vendor").lower()
        device_driver = _driver_name(entry / "driver").lower()
        if wanted_vendor and vendor != wanted_vendor:
            continue
        if wanted_driver and device_driver != wanted_driver:
            continue
        try:
            bdf = normalize_bdf(entry.name)
        except ValueError:
            continue
        candidates.append(
            PciCandidate(
                bdf=bdf,
                pci_class=pci_class,
                vendor=vendor,
                device=_read(entry / "device"),
                driver=device_driver,
                numa_node=_read(entry / "numa_node"),
            )
        )
    return tuple(candidates)


def _parse_bdfs(value: str) -> tuple[str, ...]:
    return tuple(item for item in re.split(r"[\s,]+", value.strip()) if item)


def _candidate_dict(candidate: PciCandidate) -> dict[str, str]:
    return asdict(candidate)


def _result_dict(
    index: int,
    bdf: str,
    result: AffinityResult,
    candidate: PciCandidate | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "logical_device_id": index,
        "bdf": bdf,
        "result": result.to_dict(),
    }
    if candidate is not None:
        payload["pci"] = _candidate_dict(candidate)
    return payload


def _print_result(index: int, result: AffinityResult, candidate: PciCandidate | None) -> None:
    print(f"\n=== Device [{index}] ===")
    if candidate is not None:
        print(
            f"candidate:     bdf={candidate.bdf} class={candidate.pci_class} "
            f"vendor={candidate.vendor} device={candidate.device} "
            f"driver={candidate.driver} numa={candidate.numa_node}"
        )
    print(f"status:        {result.status.value}")
    print(f"bindable:      {'yes' if result.bindable else 'no'}")
    print(f"bdf:           {result.normalized_bdf or '-'}")
    print("pcie_path:")
    for path_index, node in enumerate(result.pci_path):
        print(
            f"  [{path_index}] {node.bdf} role={node.role} "
            f"class={node.pci_class or '-'} numa={node.numa_node}"
        )
    print(f"root_bus:      {result.root_bus_path or '-'}")
    print(f"numa_node:     {result.numa_node if result.numa_node is not None else '-'}")
    print(f"numa_source:   {result.numa_source or '-'}")
    print(f"node_cpus:     {format_cpulist(result.node_cpus) or '-'}")
    print(f"online_cpus:   {format_cpulist(result.online_cpus) or '-'}")
    print(f"allowed_cpus:  {format_cpulist(result.allowed_cpus) or '-'}")
    print(f"target_cpus:   {format_cpulist(result.target_cpus) or '-'}")
    for diagnostic in result.diagnostics:
        print(f"diagnostic:    {diagnostic}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only generic GPU PCIe/NUMA affinity probe. "
            "It never changes process affinity."
        )
    )
    parser.add_argument(
        "--bdf",
        action="append",
        help="trusted PCI BDF; repeat for visible-device order",
    )
    parser.add_argument(
        "--sysfs-root", type=Path, default=Path("/sys"), help="default: /sys"
    )
    parser.add_argument(
        "--class-base",
        action="append",
        dest="class_bases",
        help="PCI base class to discover; repeatable, default: 03 and 12",
    )
    parser.add_argument("--vendor-id", help="filter automatic discovery by PCI vendor")
    parser.add_argument("--driver", help="filter automatic discovery by driver name")
    parser.add_argument(
        "--allowed-cpus",
        help="override current CPU affinity for diagnostics/tests, e.g. 0-31,64-95",
    )
    parser.add_argument("--json", action="store_true", help="emit structured JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        allowed_cpus = (
            parse_cpulist(args.allowed_cpus)
            if args.allowed_cpus is not None
            else None
        )
    except CpuListError as exc:
        print(f"invalid --allowed-cpus: {exc}", file=sys.stderr)
        return 3

    env_bdfs = _parse_bdfs(os.environ.get("AFFINITY_BDFS", ""))
    explicit_bdfs = tuple(args.bdf or ()) or env_bdfs
    candidate_by_bdf: dict[str, PciCandidate] = {}
    if explicit_bdfs:
        bdfs = explicit_bdfs
        source = "explicit-bdf"
    else:
        candidates = discover_pci_candidates(
            args.sysfs_root,
            class_bases=tuple(args.class_bases or _DEFAULT_CLASS_BASES),
            vendor_id=args.vendor_id,
            driver=args.driver,
        )
        bdfs = tuple(candidate.bdf for candidate in candidates)
        candidate_by_bdf = {candidate.bdf: candidate for candidate in candidates}
        source = "pci-class-candidate"

    if not bdfs:
        message = (
            "no PCI candidates found; provide trusted BDFs with --bdf or "
            "AFFINITY_BDFS"
        )
        if args.json:
            print(json.dumps({"status": "failed", "diagnostic": message}, indent=2))
        else:
            print(f"probe_status:  failed\ndiagnostic:    {message}")
        return 3

    devices: list[dict[str, object]] = []
    results: list[AffinityResult] = []
    for index, bdf in enumerate(bdfs):
        result = analyze_bdf(
            bdf,
            sysfs_root=args.sysfs_root,
            allowed_cpus=allowed_cpus,
            mapping_source=source,
        )
        results.append(result)
        devices.append(_result_dict(index, bdf, result, candidate_by_bdf.get(bdf)))

    complete = all(result.status is ResultStatus.SUCCESS for result in results)
    payload = {
        "schema_version": 1,
        "tool": "kunpeng-affinity-generic-probe",
        "read_only": True,
        "sysfs_root": str(args.sysfs_root),
        "device_source": source,
        "device_count": len(devices),
        "complete": complete,
        "devices": devices,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Kunpeng generic GPU affinity probe (read-only)")
        print(f"sysfs_root:    {args.sysfs_root}")
        print(f"device_source: {source}")
        if source == "pci-class-candidate":
            print("note:          automatic PCI order is not a logical GPU mapping")
        for index, result in enumerate(results):
            _print_result(index, result, candidate_by_bdf.get(bdfs[index]))
        if complete:
            print("\nrecommendation: complete per-device NUMA/CPU results are available")
            print(
                "numa_nodes:    "
                + " ".join(str(result.numa_node) for result in results)
            )
            print(
                "target_cpus:   "
                + " | ".join(format_cpulist(result.target_cpus) for result in results)
            )
        else:
            print(
                "\nrecommendation: no complete result; do not construct an automatic "
                "binding command"
            )
        print(f"\nprobe_status:  {'success' if complete else 'failed'}")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
