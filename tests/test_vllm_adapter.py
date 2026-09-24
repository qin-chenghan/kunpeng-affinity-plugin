from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from kunpeng_affinity.adapters import (
    build_vllm_device_contexts,
    check_vllm_generic_eligibility,
    create_vllm_provider_registry,
    resolve_vllm_generic_affinity,
    resolve_vllm_native_nodes,
    resolve_vllm_visibility_fingerprint,
)
from kunpeng_affinity.core.errors import AffinityDiscoveryError
from kunpeng_affinity.providers import ProviderRegistry, StaticMappingProvider


class FakePlatform:
    @classmethod
    def device_count(cls):
        return 1

    @classmethod
    def get_all_gpu_pci_bus_ids(cls):
        return {0: "00000000:AB:00.0"}

    @classmethod
    def device_id_to_physical_device_id(cls, device_id):
        return device_id


class VllmGenericAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        endpoint = self.root / "devices/pci0000:aa/0000:aa:00.0/0000:ab:00.0"
        endpoint.mkdir(parents=True)
        (endpoint / "class").write_text("0x030200", encoding="ascii")
        (endpoint / "numa_node").write_text("3", encoding="ascii")
        bridge = endpoint.parent
        (bridge / "class").write_text("0x060400", encoding="ascii")
        (bridge / "numa_node").write_text("3", encoding="ascii")
        device_links = self.root / "bus/pci/devices"
        device_links.mkdir(parents=True)
        (device_links / "0000:ab:00.0").symlink_to(endpoint)
        node = self.root / "devices/system/node/node3"
        node.mkdir(parents=True)
        (node / "cpulist").write_text("8-11", encoding="ascii")
        node1 = self.root / "devices/system/node/node1"
        node1.mkdir(parents=True)
        (node1 / "cpulist").write_text("0-3", encoding="ascii")
        cpu = self.root / "devices/system/cpu"
        cpu.mkdir(parents=True)
        (cpu / "online").write_text("0-11", encoding="ascii")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_resolves_vllm_visible_device_to_linux_affinity(self) -> None:
        batch = resolve_vllm_generic_affinity(
            FakePlatform,
            sysfs_root=self.root,
            allowed_cpus={9, 10, 11},
        )

        self.assertTrue(batch.committable)
        result = batch.ordered_results[0]
        self.assertEqual(result.mapping.pci_bdf, "0000:ab:00.0")
        self.assertEqual(result.affinity.numa_node, 3)
        self.assertEqual(result.affinity.target_cpus, {9, 10, 11})
        self.assertIsNotNone(batch.visibility_fingerprint)

    def test_generic_path_uses_injected_provider_registry(self) -> None:
        registry = ProviderRegistry(
            (StaticMappingProvider({0: "0000:ab:00.0"}),)
        )

        batch = resolve_vllm_generic_affinity(
            FakePlatform,
            sysfs_root=self.root,
            allowed_cpus={9, 10, 11},
            registry=registry,
            requested_provider="static",
        )

        self.assertTrue(batch.committable)
        self.assertEqual(batch.ordered_results[0].mapping.source, "explicit-config")

    def test_iluvatar_provider_uses_configured_runtime_utility(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"KUNPENG_AFFINITY_IXSMI": "/tmp/test-ixsmi"},
                clear=False,
            ),
            patch(
                "kunpeng_affinity.adapters.vllm_generic.IluvatarRuntimeProvider"
            ) as provider_type,
        ):
            provider_type.name = "iluvatar-runtime-pci"
            provider_type.return_value.name = "iluvatar-runtime-pci"
            create_vllm_provider_registry(
                FakePlatform,
                requested_provider="iluvatar-runtime-pci",
            )

        provider_type.assert_called_once_with(FakePlatform, ixsmi="/tmp/test-ixsmi")

    def test_visibility_fingerprint_changes_with_visible_order(self) -> None:
        class ReorderedPlatform:
            visible = (0, 1)

            @classmethod
            def device_count(cls):
                return 2

            @classmethod
            def get_all_gpu_pci_bus_ids(cls):
                return {0: "01:00.0", 1: "02:00.0"}

            @classmethod
            def device_id_to_physical_device_id(cls, device_id):
                return cls.visible[device_id]

        first = resolve_vllm_visibility_fingerprint(ReorderedPlatform)
        ReorderedPlatform.visible = (1, 0)
        second = resolve_vllm_visibility_fingerprint(ReorderedPlatform)

        self.assertNotEqual(first, second)

    def test_context_builder_preserves_process_context(self) -> None:
        contexts = build_vllm_device_contexts(
            FakePlatform,
            process_kind="EngineCore",
            local_rank=4,
            dp_local_rank=2,
            allowed_cpus={9, 10},
        )

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].process_kind, "EngineCore")
        self.assertEqual(contexts[0].local_rank, 4)
        self.assertEqual(contexts[0].dp_local_rank, 2)
        self.assertEqual(contexts[0].allowed_cpus, frozenset({9, 10}))

    def test_context_builder_marks_unavailable_cpu_affinity_as_unknown(self) -> None:
        with patch.object(
            os,
            "sched_getaffinity",
            side_effect=AttributeError,
            create=True,
        ):
            contexts = build_vllm_device_contexts(FakePlatform)

        self.assertEqual(contexts[0].allowed_cpus, frozenset())

    def test_rejects_invalid_device_count(self) -> None:
        class EmptyPlatform(FakePlatform):
            @classmethod
            def device_count(cls):
                return 0

        with self.assertRaisesRegex(AffinityDiscoveryError, "device count"):
            resolve_vllm_generic_affinity(EmptyPlatform, sysfs_root=self.root)

    def test_validates_native_result_as_a_complete_batch(self) -> None:
        numa_utils = types.SimpleNamespace(get_auto_numa_nodes=lambda: [3])

        nodes = resolve_vllm_native_nodes(
            numa_utils,
            FakePlatform,
            sysfs_root=self.root,
            allowed_cpus={9, 10},
        )

        self.assertEqual(nodes, [3])

    def test_rejects_incomplete_native_result(self) -> None:
        numa_utils = types.SimpleNamespace(get_auto_numa_nodes=lambda: [])

        with self.assertRaisesRegex(AffinityDiscoveryError, "visible devices"):
            resolve_vllm_native_nodes(
                numa_utils,
                FakePlatform,
                sysfs_root=self.root,
                allowed_cpus={9, 10},
            )

    def test_rejects_native_node_without_allowed_cpu(self) -> None:
        numa_utils = types.SimpleNamespace(get_auto_numa_nodes=lambda: [3])

        with self.assertRaisesRegex(AffinityDiscoveryError, "no online CPUs"):
            resolve_vllm_native_nodes(
                numa_utils,
                FakePlatform,
                sysfs_root=self.root,
                allowed_cpus={0, 1},
            )

    def test_generic_eligibility_uses_vendor_neutral_gates(self) -> None:
        numa_utils = types.SimpleNamespace(_can_set_mempolicy=lambda: True)
        with patch(
            "kunpeng_affinity.adapters.vllm_generic.shutil.which",
            return_value="/usr/bin/numactl",
        ):
            check_vllm_generic_eligibility(
                numa_utils,
                sysfs_root=self.root,
                allowed_cpus={0, 1},
            )

    def test_generic_eligibility_allows_inherited_cpu_restriction(self) -> None:
        numa_utils = types.SimpleNamespace(_can_set_mempolicy=lambda: True)
        with (
            patch(
                "kunpeng_affinity.adapters.vllm_generic.os.sched_getaffinity",
                side_effect=AssertionError("generic eligibility must not reject cpuset"),
                create=True,
            ),
            patch(
                "kunpeng_affinity.adapters.vllm_generic.shutil.which",
                return_value="/usr/bin/numactl",
            ),
        ):
            check_vllm_generic_eligibility(numa_utils, sysfs_root=self.root)

    def test_generic_eligibility_rejects_missing_mempolicy(self) -> None:
        numa_utils = types.SimpleNamespace(_can_set_mempolicy=lambda: False)
        with self.assertRaisesRegex(AffinityDiscoveryError, "memory policy"):
            check_vllm_generic_eligibility(
                numa_utils,
                sysfs_root=self.root,
                allowed_cpus={0, 1},
            )
