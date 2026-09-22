from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from demo.generic_affinity_probe import discover_pci_candidates, main


class GenericAffinityProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "bus/pci/devices").mkdir(parents=True)
        (self.root / "devices/system/cpu").mkdir(parents=True)
        (self.root / "devices/system/cpu/online").write_text("0-7", encoding="ascii")
        (self.root / "devices/system/node/node0").mkdir(parents=True)
        (self.root / "devices/system/node/node0/cpulist").write_text("0-3", encoding="ascii")
        (self.root / "devices/system/node/node1").mkdir(parents=True)
        (self.root / "devices/system/node/node1/cpulist").write_text("4-7", encoding="ascii")

        root_bus = self.root / "devices/pci0000:00"
        endpoint = root_bus / "0000:00:01.0" / "0000:01:00.0"
        endpoint.mkdir(parents=True)
        (endpoint / "class").write_text("0x030200", encoding="ascii")
        (endpoint / "numa_node").write_text("0", encoding="ascii")
        (endpoint / "local_cpulist").write_text("0-3", encoding="ascii")
        link = self.root / "bus/pci/devices/0000:01:00.0"
        link.symlink_to(endpoint)

        device = self.root / "bus/pci/devices/0000:01:00.0"
        (device / "class").write_text("0x030200", encoding="ascii")
        (device / "vendor").write_text("0x1234", encoding="ascii")
        (device / "device").write_text("0xabcd", encoding="ascii")
        (device / "numa_node").write_text("0", encoding="ascii")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_candidate_discovery_is_metadata_only(self) -> None:
        candidates = discover_pci_candidates(self.root)

        self.assertEqual([candidate.bdf for candidate in candidates], ["0000:01:00.0"])
        self.assertEqual(candidates[0].vendor, "0x1234")

    def test_explicit_bdf_produces_structured_complete_result(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--sysfs-root",
                    str(self.root),
                    "--bdf",
                    "00000000:01:00.0",
                    "--allowed-cpus",
                    "0-3",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["device_source"], "explicit-bdf")
        self.assertEqual(payload["devices"][0]["result"]["numa_node"], 0)
        self.assertEqual(payload["devices"][0]["result"]["target_cpus"], "0-3")


if __name__ == "__main__":
    unittest.main()
