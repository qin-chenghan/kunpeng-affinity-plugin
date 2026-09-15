from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from kunpeng_affinity.topology.environment import discover_pci_candidates, main


class EnvironmentProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "bus/pci/devices").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _device(
        self,
        bdf: str,
        pci_class: str,
        *,
        vendor: str = "0x1234",
        device: str = "0x5678",
        numa_node: str = "0",
    ) -> None:
        path = self.root / "bus/pci/devices" / bdf
        path.mkdir()
        for name, value in {
            "class": pci_class,
            "vendor": vendor,
            "device": device,
            "numa_node": numa_node,
        }.items():
            (path / name).write_text(value, encoding="ascii")

    def test_discovers_display_and_processing_accelerator_classes(self) -> None:
        self._device("0000:01:00.0", "0x030200", vendor="0x10de")
        self._device("0000:02:00.0", "0x120000", vendor="0x1abc")
        self._device("0000:03:00.0", "0x020000")

        candidates = discover_pci_candidates(self.root)

        self.assertEqual(
            [candidate.bdf for candidate in candidates],
            ["0000:01:00.0", "0000:02:00.0"],
        )
        self.assertEqual(candidates[0].vendor, "0x10de")
        self.assertEqual(candidates[1].pci_class, "0x120000")

    def test_no_candidate_is_an_explicit_probe_failure(self) -> None:
        self._device("0000:03:00.0", "0x020000")
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["--sysfs-root", str(self.root)])

        self.assertEqual(exit_code, 3)
        self.assertIn("probe_status:  failed", output.getvalue())
        self.assertIn("AFFINITY_BDFS", output.getvalue())


if __name__ == "__main__":
    unittest.main()
