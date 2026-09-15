"""CLI for validating the topology analyzer against the current Linux host."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kunpeng_affinity.topology.analyzer import analyze_bdf, normalize_bdf
from kunpeng_affinity.topology.cli import _print_result
from kunpeng_affinity.topology.cpulist import format_cpulist
from kunpeng_affinity.topology.models import ResultStatus

_CANDIDATE_BASE_CLASSES = {"03", "12"}


@dataclass(frozen=True)
class PciCandidate:
    bdf: str
    pci_class: str
    vendor: str
    device: str
    numa_node: str


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii").strip()
    except (FileNotFoundError, OSError):
        return "unknown"


def discover_pci_candidates(sysfs_root: Path) -> tuple[PciCandidate, ...]:
    """Find display controllers and processing accelerators in Linux PCI sysfs."""
    devices_root = sysfs_root / "bus/pci/devices"
    candidates: list[PciCandidate] = []
    try:
        entries = sorted(devices_root.iterdir(), key=lambda path: path.name)
    except (FileNotFoundError, OSError):
        return ()

    for entry in entries:
        pci_class = _read(entry / "class").lower()
        class_code = pci_class.removeprefix("0x")
        if len(class_code) < 2 or class_code[:2] not in _CANDIDATE_BASE_CLASSES:
            continue
        try:
            bdf = normalize_bdf(entry.name)
        except ValueError:
            continue
        candidates.append(
            PciCandidate(
                bdf=bdf,
                pci_class=pci_class,
                vendor=_read(entry / "vendor"),
                device=_read(entry / "device"),
                numa_node=_read(entry / "numa_node"),
            )
        )
    return tuple(candidates)


def _explicit_bdfs(value: str) -> tuple[str, ...]:
    return tuple(item for item in re.split(r"[\s,]+", value.strip()) if item)


def _print_host_inventory(sysfs_root: Path) -> None:
    print(f"sysfs_root:    {sysfs_root}")
    print(f"online_cpus:   {_read(sysfs_root / 'devices/system/cpu/online')}")
    try:
        allowed = format_cpulist(frozenset(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        allowed = "unavailable"
    print(f"allowed_cpus:  {allowed}")

    node_root = sysfs_root / "devices/system/node"
    try:
        nodes = sorted(node_root.glob("node[0-9]*"), key=lambda path: path.name)
    except OSError:
        nodes = []
    if not nodes:
        print("numa_nodes:    none")
    for node in nodes:
        print(f"numa_node:     {node.name} cpus={_read(node / 'cpulist')}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the read-only topology analyzer against this Linux host."
    )
    parser.add_argument(
        "--sysfs-root", type=Path, default=Path("/sys"), help="default: /sys"
    )
    parser.add_argument(
        "--bdf",
        action="append",
        help="trusted accelerator BDF; repeat for multiple devices",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    env_bdfs = _explicit_bdfs(os.environ.get("AFFINITY_BDFS", ""))
    requested_bdfs = tuple(args.bdf or ()) or env_bdfs

    print("=== Live Linux topology inventory ===")
    _print_host_inventory(args.sysfs_root)

    if requested_bdfs:
        source = "explicit-bdf"
        bdfs = requested_bdfs
        print("device_source: explicit --bdf/AFFINITY_BDFS")
        for bdf in bdfs:
            print(f"candidate:     bdf={bdf}")
    else:
        source = "pci-class-candidate"
        candidates = discover_pci_candidates(args.sysfs_root)
        bdfs = tuple(candidate.bdf for candidate in candidates)
        print("device_source: PCI class candidate discovery (not logical GPU mapping)")
        for candidate in candidates:
            print(
                f"candidate:     bdf={candidate.bdf} class={candidate.pci_class} "
                f"vendor={candidate.vendor} device={candidate.device} "
                f"numa={candidate.numa_node}"
            )

    if not bdfs:
        print("probe_status:  failed")
        print("diagnostic:    no display-controller or processing-accelerator PCI candidate found")
        print("diagnostic:    set AFFINITY_BDFS to trusted comma-separated PCI BDFs")
        return 3

    results = []
    for index, bdf in enumerate(bdfs):
        print(f"\n=== Topology analysis [{index}] ===")
        result = analyze_bdf(bdf, sysfs_root=args.sysfs_root, mapping_source=source)
        _print_result(result)
        results.append(result)

    if all(result.status is ResultStatus.SUCCESS for result in results):
        print("\nprobe_status:  success")
        return 0
    print("\nprobe_status:  failed")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
