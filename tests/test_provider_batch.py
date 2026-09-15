from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kunpeng_affinity.core.models import DeviceContext, DeviceMapping
from kunpeng_affinity.policy import GenericAffinityProvider
from kunpeng_affinity.providers import (
    LinuxContextProvider,
    ProviderRegistry,
    StaticMappingProvider,
    VllmPlatformProvider,
)
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


class LinuxContextProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "devices/pci0000:00/0000:00:01.0/0000:01:00.0").mkdir(
            parents=True
        )
        (self.root / "bus/pci/devices").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_prefers_explicit_bdf_and_normalizes_it(self) -> None:
        provider = LinuxContextProvider(sysfs_root=self.root)
        mappings = provider.map_all(
            (
                DeviceContext(
                    framework="test",
                    logical_device_id=0,
                    explicit_bdf="00000000:AB:00.0",
                    runtime_device_id="0000:CD:00.0",
                ),
            )
        )

        self.assertEqual(mappings[0].pci_bdf, "0000:ab:00.0")
        self.assertEqual(mappings[0].source, "explicit-config")

    def test_accepts_bdf_from_runtime_context(self) -> None:
        provider = LinuxContextProvider(sysfs_root=self.root)
        mappings = provider.map_all(
            (
                DeviceContext(
                    framework="test",
                    logical_device_id=0,
                    runtime_device_id="AB:00.0",
                ),
            )
        )

        self.assertEqual(mappings[0].pci_bdf, "0000:ab:00.0")
        self.assertEqual(mappings[0].source, "framework-context")

    def test_resolves_bdf_from_a_device_path(self) -> None:
        device_path = self.root / "devices/pci0000:00/0000:00:01.0/0000:01:00.0"
        provider = LinuxContextProvider(sysfs_root=self.root)

        mappings = provider.map_all(
            (
                DeviceContext(
                    framework="test",
                    logical_device_id=0,
                    device_node=str(device_path),
                ),
            )
        )

        self.assertEqual(mappings[0].pci_bdf, "0000:01:00.0")
        self.assertEqual(mappings[0].source, "linux-device-context")

    def test_does_not_infer_bdf_from_logical_id(self) -> None:
        provider = LinuxContextProvider(sysfs_root=self.root)

        with self.assertRaisesRegex(RuntimeError, "no provable PCI BDF"):
            provider.map_all((context(0),))

    def test_probe_rejects_incomplete_context(self) -> None:
        provider = LinuxContextProvider(sysfs_root=self.root)

        result = provider.probe((context(0),))

        self.assertFalse(result.supported)
        self.assertIn("no provable PCI BDF", result.reason or "")

    def test_probe_rejects_invalid_explicit_bdf(self) -> None:
        provider = LinuxContextProvider(sysfs_root=self.root)
        invalid = DeviceContext(
            framework="test",
            logical_device_id=0,
            explicit_bdf="not-a-bdf",
        )

        result = provider.probe((invalid,))

        self.assertFalse(result.supported)
        self.assertIn("invalid BDF", result.reason or "")

    def test_probe_rejects_duplicate_bdfs(self) -> None:
        provider = LinuxContextProvider(sysfs_root=self.root)
        contexts = (
            DeviceContext(
                framework="test", logical_device_id=0, explicit_bdf="01:00.0"
            ),
            DeviceContext(
                framework="test", logical_device_id=1, explicit_bdf="01:00.0"
            ),
        )

        result = provider.probe(contexts)

        self.assertFalse(result.supported)
        self.assertIn("mapped more than once", result.reason or "")


class VllmPlatformProviderTest(unittest.TestCase):
    class FakePlatform:
        @classmethod
        def get_all_gpu_pci_bus_ids(cls):
            return {0: "0000:ab:00.0", 1: "00000000:cd:00.0"}

        @classmethod
        def device_id_to_physical_device_id(cls, device_id: int) -> int:
            return {0: 1, 1: 0}[device_id]

    def test_maps_visible_ids_through_physical_ids(self) -> None:
        provider = VllmPlatformProvider(self.FakePlatform)
        mappings = provider.map_all(
            (
                context(0),
                context(1),
            )
        )

        self.assertEqual(
            [mapping.pci_bdf for mapping in mappings],
            ["0000:cd:00.0", "0000:ab:00.0"],
        )
        self.assertEqual(
            [mapping.physical_device_id for mapping in mappings], [1, 0]
        )
        self.assertEqual(mappings[0].source, "vllm-platform-pci")

    def test_probe_rejects_platform_without_identity_api(self) -> None:
        provider = VllmPlatformProvider(object())

        result = provider.probe((context(0),))

        self.assertFalse(result.supported)
        self.assertIn("get_all_gpu_pci_bus_ids", result.reason or "")

    def test_rejects_invalid_platform_bdf(self) -> None:
        class InvalidPlatform(self.FakePlatform):
            @classmethod
            def get_all_gpu_pci_bus_ids(cls):
                return {0: "not-a-bdf"}

        provider = VllmPlatformProvider(InvalidPlatform)

        with self.assertRaisesRegex(RuntimeError, "invalid BDF"):
            provider.map_all((context(1),))


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

    def test_resolves_context_bdf_through_linux_provider(self) -> None:
        resolver = GenericAffinityProvider(
            LinuxContextProvider(),
            sysfs_root=self.root,
            allowed_cpus=set(range(8)),
        )
        result = resolver.resolve_all(
            (
                DeviceContext(
                    framework="test",
                    logical_device_id=0,
                    runtime_device_id="00000000:01:00.0",
                    visibility_fingerprint="visible-a",
                ),
            )
        )

        self.assertTrue(result.committable)
        self.assertEqual(result.ordered_results[0].mapping.source, "framework-context")
        self.assertEqual(result.ordered_results[0].affinity.numa_node, 0)

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
