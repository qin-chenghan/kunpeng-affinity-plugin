from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from kunpeng_affinity.topology.analyzer import analyze_bdf, normalize_bdf
from kunpeng_affinity.topology.cli import main
from kunpeng_affinity.topology.cpulist import CpuListError, format_cpulist, parse_cpulist
from kunpeng_affinity.topology.models import ResultStatus


class SysfsFixture:
    def __init__(self, root: Path, node_cpus: dict[int, str], online: str = "0-15"):
        self.root = root
        (root / "bus/pci/devices").mkdir(parents=True)
        (root / "devices/system/cpu").mkdir(parents=True)
        (root / "devices/system/cpu/online").write_text(online, encoding="ascii")
        for node, cpus in node_cpus.items():
            node_dir = root / "devices/system/node" / f"node{node}"
            node_dir.mkdir(parents=True)
            (node_dir / "cpulist").write_text(cpus, encoding="ascii")

    def add_path(
        self,
        bdfs: list[str],
        *,
        numa_nodes: list[int | None],
        local_cpulist: str | None = None,
    ) -> None:
        if len(bdfs) != len(numa_nodes):
            raise ValueError("one NUMA value is required for each BDF")
        root_bus = self.root / "devices" / f"pci{bdfs[-1][:7]}"
        root_bus.mkdir(parents=True, exist_ok=True)
        parent = root_bus
        real_paths: dict[str, Path] = {}
        for reverse_index, bdf in enumerate(reversed(bdfs)):
            path = parent / bdf
            path.mkdir()
            original_index = len(bdfs) - reverse_index - 1
            pci_class = "0x030200" if original_index == 0 else "0x060400"
            (path / "class").write_text(pci_class, encoding="ascii")
            numa = numa_nodes[original_index]
            if numa is not None:
                (path / "numa_node").write_text(str(numa), encoding="ascii")
            real_paths[bdf] = path
            parent = path
        endpoint = real_paths[bdfs[0]]
        if local_cpulist is not None:
            (endpoint / "local_cpulist").write_text(local_cpulist, encoding="ascii")
        (self.root / "bus/pci/devices" / bdfs[0]).symlink_to(endpoint)


class CpuListTest(unittest.TestCase):
    def test_parse_and_format(self) -> None:
        cpus = parse_cpulist("0-3,7,9-10")
        self.assertEqual(cpus, frozenset({0, 1, 2, 3, 7, 9, 10}))
        self.assertEqual(format_cpulist(cpus), "0-3,7,9-10")

    def test_invalid_range(self) -> None:
        with self.assertRaises(CpuListError):
            parse_cpulist("4-2")


class BdfTest(unittest.TestCase):
    def test_normalizes_vendor_style_domain(self) -> None:
        self.assertEqual(normalize_bdf("00000000:AB:00.0"), "0000:ab:00.0")
        self.assertEqual(normalize_bdf("AB:00.0"), "0000:ab:00.0")

    def test_rejects_nonzero_high_domain(self) -> None:
        with self.assertRaises(ValueError):
            normalize_bdf("00010000:ab:00.0")


class TopologyAnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_direct_path_and_cpu_intersection(self) -> None:
        fixture = SysfsFixture(self.root, {0: "0-3", 1: "4-7"}, online="0-7")
        fixture.add_path(
            ["0000:01:00.0", "0000:00:01.0"],
            numa_nodes=[0, 0],
            local_cpulist="0-3",
        )

        result = analyze_bdf(
            "00000000:01:00.0", sysfs_root=self.root, allowed_cpus={2, 3, 4}
        )

        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.numa_node, 0)
        self.assertEqual(result.numa_source, "endpoint")
        self.assertEqual(result.target_cpus, frozenset({2, 3}))
        self.assertEqual(
            [node.role for node in result.pci_path], ["endpoint", "pci-bridge"]
        )

    def test_cli_emits_structured_result(self) -> None:
        fixture = SysfsFixture(self.root, {0: "0-3"}, online="0-3")
        fixture.add_path(
            ["0000:01:00.0", "0000:00:01.0"],
            numa_nodes=[0, 0],
        )
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "--bdf",
                    "0000:01:00.0",
                    "--sysfs-root",
                    str(self.root),
                    "--allowed-cpus",
                    "1-2",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload[0]["status"], "success")
        self.assertEqual(payload[0]["target_cpus"], "1-2")

    def test_single_switch_uses_nearest_ancestor(self) -> None:
        fixture = SysfsFixture(self.root, {0: "0-3", 1: "4-7"}, online="0-7")
        fixture.add_path(
            [
                "0000:04:00.0",
                "0000:03:00.0",
                "0000:02:00.0",
                "0000:00:01.0",
            ],
            numa_nodes=[-1, 1, 1, 1],
            local_cpulist="4-7",
        )

        result = analyze_bdf("0000:04:00.0", sysfs_root=self.root, allowed_cpus=set(range(8)))

        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.numa_node, 1)
        self.assertEqual(result.numa_source, "ancestor:0000:03:00.0")
        self.assertEqual(len(result.pci_path), 4)

    def test_multi_level_switch_has_no_depth_limit(self) -> None:
        fixture = SysfsFixture(self.root, {2: "8-15"})
        path = [
            "0000:08:00.0",
            "0000:07:00.0",
            "0000:06:00.0",
            "0000:05:00.0",
            "0000:04:00.0",
            "0000:00:02.0",
        ]
        fixture.add_path(path, numa_nodes=[-1, 2, 2, 2, 2, 2])

        result = analyze_bdf("0000:08:00.0", sysfs_root=self.root, allowed_cpus=set(range(16)))

        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.numa_node, 2)
        self.assertEqual(len(result.pci_path), len(path))

    def test_conflicting_numa_evidence_fails(self) -> None:
        fixture = SysfsFixture(self.root, {0: "0-3", 1: "4-7"}, online="0-7")
        fixture.add_path(
            ["0000:01:00.0", "0000:00:01.0"],
            numa_nodes=[0, 1],
        )

        result = analyze_bdf("0000:01:00.0", sysfs_root=self.root, allowed_cpus=set(range(8)))

        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertIn("conflicting NUMA evidence", result.diagnostics[0])

    def test_unknown_numa_is_partial_and_not_bindable(self) -> None:
        fixture = SysfsFixture(self.root, {0: "0-3"}, online="0-3")
        fixture.add_path(
            ["0000:01:00.0", "0000:00:01.0"],
            numa_nodes=[-1, -1],
        )

        result = analyze_bdf("0000:01:00.0", sysfs_root=self.root, allowed_cpus=set(range(4)))

        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertEqual(result.failure_code, "NUMA_UNKNOWN")
        self.assertFalse(result.bindable)

    def test_broken_parent_chain_fails(self) -> None:
        fixture = SysfsFixture(self.root, {0: "0-3"}, online="0-3")
        endpoint = self.root / "devices" / "0000:01:00.0"
        endpoint.mkdir()
        (endpoint / "numa_node").write_text("0", encoding="ascii")
        (self.root / "bus/pci/devices/0000:01:00.0").symlink_to(endpoint)

        result = analyze_bdf("0000:01:00.0", sysfs_root=self.root, allowed_cpus=set(range(4)))

        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertIn("incomplete", result.diagnostics[0])

    def test_empty_effective_cpu_set_fails(self) -> None:
        fixture = SysfsFixture(self.root, {0: "0-3", 1: "4-7"}, online="0-7")
        fixture.add_path(
            ["0000:01:00.0", "0000:00:01.0"],
            numa_nodes=[0, 0],
        )

        result = analyze_bdf("0000:01:00.0", sysfs_root=self.root, allowed_cpus={4, 5})

        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertIn("no CPUs", result.diagnostics[0])
        self.assertEqual(result.numa_node, 0)
        self.assertEqual(result.node_cpus, frozenset({0, 1, 2, 3}))
        self.assertEqual(result.allowed_cpus, frozenset({4, 5}))

    def test_unrelated_memory_only_node_does_not_fail(self) -> None:
        fixture = SysfsFixture(self.root, {0: "0-3", 1: ""}, online="0-3")
        fixture.add_path(
            ["0000:01:00.0", "0000:00:01.0"],
            numa_nodes=[0, 0],
        )

        result = analyze_bdf("0000:01:00.0", sysfs_root=self.root, allowed_cpus=set(range(4)))

        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.numa_node, 0)
