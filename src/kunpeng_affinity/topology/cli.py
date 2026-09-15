"""Command-line interface for read-only topology analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from kunpeng_affinity.topology.analyzer import analyze_bdf
from kunpeng_affinity.topology.cpulist import CpuListError, format_cpulist, parse_cpulist
from kunpeng_affinity.topology.models import AffinityResult, ResultStatus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read Linux sysfs and suggest CPUs for trusted PCI BDFs. "
            "This command never changes process affinity."
        )
    )
    parser.add_argument(
        "--bdf",
        action="append",
        required=True,
        help="trusted PCI BDF; repeat for multiple devices",
    )
    parser.add_argument(
        "--sysfs-root",
        type=Path,
        default=Path("/sys"),
        help="sysfs mount or fixture root (default: /sys)",
    )
    parser.add_argument(
        "--allowed-cpus",
        help="override current sched_getaffinity CPU list for diagnostics/tests",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def _print_result(result: AffinityResult) -> None:
    print(f"status:        {result.status.value}")
    print(f"bindable:      {'yes' if result.bindable else 'no'}")
    print(f"input_bdf:     {result.input_bdf}")
    print(f"bdf:           {result.normalized_bdf or '-'}")
    print(f"mapping:       {result.device_mapping_source}")
    print("pcie_path:")
    for index, node in enumerate(result.pci_path):
        print(
            f"  [{index}] {node.bdf} role={node.role} "
            f"class={node.pci_class or '-'} numa={node.numa_node}"
        )
    print(f"root_bus:      {result.root_bus_path or '-'}")
    print(f"numa_node:     {result.numa_node if result.numa_node is not None else '-'}")
    print(f"numa_source:   {result.numa_source or '-'}")
    print(f"node_cpus:     {format_cpulist(result.node_cpus) or '-'}")
    print(f"online_cpus:   {format_cpulist(result.online_cpus) or '-'}")
    print(f"allowed_cpus:  {format_cpulist(result.allowed_cpus) or '-'}")
    print(f"suggested:     {format_cpulist(result.target_cpus) or '-'}")
    for diagnostic in result.diagnostics:
        print(f"diagnostic:    {diagnostic}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        allowed = (
            parse_cpulist(args.allowed_cpus)
            if args.allowed_cpus is not None
            else None
        )
    except CpuListError as exc:
        parser.error(str(exc))

    results = [
        analyze_bdf(
            bdf,
            sysfs_root=args.sysfs_root,
            allowed_cpus=allowed,
        )
        for bdf in args.bdf
    ]
    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        for index, result in enumerate(results):
            if index:
                print()
            _print_result(result)
    return 0 if all(result.status is ResultStatus.SUCCESS for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
