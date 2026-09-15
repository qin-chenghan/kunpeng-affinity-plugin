from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kunpeng_affinity.core.models import DeviceContext, DeviceMapping
from kunpeng_affinity.policy import GenericAffinityProvider
from kunpeng_affinity.providers import ProviderRegistry, StaticMappingProvider
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

    def add_path(self, bdfs: list[str], numa_nodes: list[int]) -> None:
        root_bus = self.root / "devices" / f"pci{bdfs[-1][:7]}"
        root_bus.mkdir(parents=True, exist_ok=True)
        parent = root_bus
        real_paths: dict[str, Path] = {}
        for reverse_index, bdf in enumerate(reversed(bdfs)):
            path = parent / bdf
            path.mkdir()
            original_index = len(bdfs) - reverse_index - 1
            (path / "class").write_text(
                "0x030200" if original_index == 0 else "0x060400",
                encoding="ascii",
            )
            (path / "numa_node").write_text(str(numa_nodes[original_index]), encoding="ascii")
            real_paths[bdf] = path
            parent = path
        (self.root / "bus/pci/devices" / bdfs[0]).symlink_to(real_paths[bdfs[0]])


def context(device_id: int, *, fingerprint: str = "visible-a") -> DeviceContext:
    return DeviceContext(
        framework="test",
        logical_device_id=device_id,
        visibility_fingerprint=fingerprint,
    )


class ProviderRegistryTest(unittest.TestCase):
    def test_static_provider_preserves_context_order(self) -> None:
        provider = StaticMappingProvider({0: "AB:00.0", 1: "AC:00.0"})
        mappings = provider.map_all((context(1), context(0)))
        self.assertEqual([mapping.logical_device_id for mapping in mappings], [1, 0])
        self.assertEqual([mapping.pci_bdf for mapping in mappings], ["0000:ac:00.0", "0000:ab:00.0"])

    def test_registry_rejects_ambiguous_auto_selection(self) -> None:
        first = StaticMappingProvider({0: "01:00.0"})
        second = StaticMappingProvider({0: "02:00.0"})
        second.name = "second"
        registry = ProviderRegistry((first, second))
        with self.assertRaisesRegex(RuntimeError, "multiple providers"):
            registry.select((context(0),))


class GenericAffinityProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.fixture = SysfsFixture(self.root, {0: "0-3", 1: "4-7"}, online="0-7")
        self.fixture.add_path(["0000:01:00.0", "0000:00:01.0"], [0, 0])
        self.fixture.add_path(["0000:02:00.0", "0000:00:02.0"], [1, 1])

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_resolves_all_devices_in_input_order(self) -> None:
        resolver = GenericAffinityProvider(
            StaticMappingProvider({0: "01:00.0", 1: "02:00.0"}),
            sysfs_root=self.root,
            allowed_cpus=set(range(8)),
        )
        result = resolver.resolve_all((context(1), context(0)))
        self.assertTrue(result.committable)
        self.assertEqual(
            [item.context.logical_device_id for item in result.ordered_results], [1, 0]
        )
        self.assertEqual(
            [item.affinity.numa_node for item in result.ordered_results], [1, 0]
        )
        self.assertEqual(
            [item.affinity.status for item in result.ordered_results],
            [ResultStatus.SUCCESS, ResultStatus.SUCCESS],
        )

    def test_any_topology_failure_blocks_the_whole_batch(self) -> None:
        resolver = GenericAffinityProvider(
            StaticMappingProvider({0: "01:00.0", 1: "03:00.0"}),
            sysfs_root=self.root,
            allowed_cpus=set(range(8)),
        )
        result = resolver.resolve_all((context(0), context(1)))
        self.assertFalse(result.committable)
        self.assertEqual(len(result.ordered_results), 2)
        self.assertTrue(result.failure_summary)

    def test_missing_mapping_does_not_produce_partial_results(self) -> None:
        resolver = GenericAffinityProvider(
            StaticMappingProvider({0: "01:00.0"}),
            sysfs_root=self.root,
            allowed_cpus=set(range(8)),
        )
        result = resolver.resolve_all((context(0), context(1)))
        self.assertFalse(result.committable)
        self.assertEqual(result.ordered_results, ())
        self.assertIn("DEVICE_MAPPING_MISSING", result.failure_summary[0])

    def test_malformed_mapping_does_not_reach_sysfs(self) -> None:
        resolver = GenericAffinityProvider(
            StaticMappingProvider({0: "not-a-bdf"}),
            sysfs_root=self.root,
            allowed_cpus=set(range(8)),
        )
        result = resolver.resolve_all((context(0),))
        self.assertFalse(result.committable)
        self.assertEqual(result.ordered_results, ())
        self.assertIn("BDF_INVALID", result.failure_summary[0])

    def test_duplicate_mapping_is_not_committable(self) -> None:
        resolver = GenericAffinityProvider(
            StaticMappingProvider({0: "01:00.0", 1: "01:00.0"}),
            sysfs_root=self.root,
            allowed_cpus=set(range(8)),
        )
        result = resolver.resolve_all((context(0), context(1)))
        self.assertFalse(result.committable)
        self.assertEqual(result.ordered_results, ())
        self.assertIn("DEVICE_MAPPING_CONFLICT", result.failure_summary[0])

    def test_fingerprint_mismatch_blocks_the_whole_batch(self) -> None:
        resolver = GenericAffinityProvider(
            StaticMappingProvider({0: "01:00.0", 1: "02:00.0"}),
            sysfs_root=self.root,
            allowed_cpus=set(range(8)),
        )
        result = resolver.resolve_all((context(0), context(1, fingerprint="visible-b")))
        self.assertFalse(result.committable)
        self.assertIn("VISIBILITY_CHANGED", result.failure_summary[0])

    def test_provider_bdf_is_canonicalized_before_analysis(self) -> None:
        class UppercaseProvider(StaticMappingProvider):
            def map_all(self, contexts):
                return tuple(
                    DeviceMapping(
                        logical_device_id=context.logical_device_id,
                        pci_bdf="00000000:01:00.0",
                        source="test",
                    )
                    for context in contexts
                )

        resolver = GenericAffinityProvider(
            UppercaseProvider({}),
            sysfs_root=self.root,
            allowed_cpus=set(range(8)),
        )
        result = resolver.resolve_all((context(0),))
        self.assertTrue(result.committable)
        self.assertEqual(result.ordered_results[0].mapping.pci_bdf, "0000:01:00.0")


if __name__ == "__main__":
    unittest.main()
