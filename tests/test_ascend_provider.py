from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kunpeng_affinity.core.models import DeviceContext
from kunpeng_affinity.providers.ascend_sysfs import (
    ASCEND_VISIBLE_DEVICES,
    AscendSysfsProvider,
    parse_bdf_to_devid_table,
    parse_visible_device_ids,
)


TABLE = """0000:c1:00.0 ---> 0
0000:c2:00.0 ---> 1
0000:81:00.0 ---> 2
0000:82:00.0 ---> 3
"""


class AscendSysfsProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        devices = self.root / "bus/pci/devices"
        for bdf in ("0000:c1:00.0", "0000:c2:00.0"):
            path = devices / bdf
            path.mkdir(parents=True)
            (path / "devdrv_sysfs_bdf_to_devid").write_text(
                TABLE, encoding="ascii"
            )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def contexts(self, count: int = 2) -> tuple[DeviceContext, ...]:
        return tuple(
            DeviceContext(framework="vllm", logical_device_id=index)
            for index in range(count)
        )

    def test_maps_non_contiguous_visible_ids_without_runtime_import(self) -> None:
        provider = AscendSysfsProvider(
            sysfs_root=self.root,
            environ={ASCEND_VISIBLE_DEVICES: "2,0"},
        )

        mappings = provider.map_all(self.contexts())

        self.assertEqual(
            [item.pci_bdf for item in mappings],
            ["0000:81:00.0", "0000:c1:00.0"],
        )
        self.assertEqual([item.physical_device_id for item in mappings], [2, 0])
        self.assertEqual(mappings[0].source, provider.name)

    def test_maps_by_logical_id_when_contexts_are_reordered(self) -> None:
        provider = AscendSysfsProvider(
            sysfs_root=self.root,
            environ={ASCEND_VISIBLE_DEVICES: "2,0"},
        )

        mappings = provider.map_all((self.contexts()[1], self.contexts()[0]))

        self.assertEqual(
            [(item.logical_device_id, item.physical_device_id) for item in mappings],
            [(1, 0), (0, 2)],
        )

    def test_probe_requires_explicit_visibility(self) -> None:
        provider = AscendSysfsProvider(sysfs_root=self.root, environ={})

        result = provider.probe(self.contexts())

        self.assertFalse(result.supported)
        self.assertIn(ASCEND_VISIBLE_DEVICES, result.reason or "")

    def test_probe_rejects_visibility_count_mismatch(self) -> None:
        provider = AscendSysfsProvider(
            sysfs_root=self.root,
            environ={ASCEND_VISIBLE_DEVICES: "0"},
        )

        result = provider.probe(self.contexts())

        self.assertFalse(result.supported)
        self.assertIn("exposes 1 devices", result.reason or "")

    def test_probe_rejects_disagreeing_sysfs_tables(self) -> None:
        second = self.root / "bus/pci/devices/0000:c2:00.0/devdrv_sysfs_bdf_to_devid"
        second.write_text("0000:c1:00.0 ---> 1\n", encoding="ascii")
        provider = AscendSysfsProvider(
            sysfs_root=self.root,
            environ={ASCEND_VISIBLE_DEVICES: "0,1"},
        )

        result = provider.probe(self.contexts())

        self.assertFalse(result.supported)
        self.assertIn("disagree", result.reason or "")

    def test_probe_rejects_malformed_bdf(self) -> None:
        path = self.root / "bus/pci/devices/0000:c1:00.0/devdrv_sysfs_bdf_to_devid"
        path.write_text("00000000:10000:00.0 ---> 0\n", encoding="ascii")
        provider = AscendSysfsProvider(
            sysfs_root=self.root,
            environ={ASCEND_VISIBLE_DEVICES: "0,1"},
        )

        result = provider.probe(self.contexts())

        self.assertFalse(result.supported)
        self.assertIn("invalid Ascend", result.reason or "")

    def test_mapping_reuses_probe_sample_once(self) -> None:
        provider = AscendSysfsProvider(
            sysfs_root=self.root,
            environ={ASCEND_VISIBLE_DEVICES: "0,1"},
        )
        contexts = self.contexts()

        self.assertTrue(provider.probe(contexts).supported)
        first = provider.map_all(contexts)
        second = provider.map_all(contexts)

        self.assertEqual(first, second)


class AscendParsingTest(unittest.TestCase):
    def test_parse_bdf_to_devid_table(self) -> None:
        self.assertEqual(
            parse_bdf_to_devid_table("00000000:C1:00.0 ---> 0\n"),
            {0: "0000:c1:00.0"},
        )

    def test_parse_visible_device_ids(self) -> None:
        self.assertEqual(parse_visible_device_ids("1, 4,7"), (1, 4, 7))

    def test_parse_visible_device_ids_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            parse_visible_device_ids("1,1")


if __name__ == "__main__":
    unittest.main()
