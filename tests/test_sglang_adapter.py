from __future__ import annotations

import subprocess
import types
import unittest
from unittest.mock import patch

from kunpeng_affinity.adapters.sglang_generic import (
    SglangRuntimeProvider,
    SglangTorchPlatform,
)
from kunpeng_affinity.core.models import DeviceContext
from kunpeng_affinity.sglang_plugin import _around_numa_query
from kunpeng_affinity.config import PluginMode


UUID_0 = "8631681a-860d-5d5c-8937-fc4efe2beea4"


class FakeProperties:
    def __init__(self, uuid: str = UUID_0, pci_bus_id: str | None = None) -> None:
        self.uuid = uuid
        self.pci_bus_id = pci_bus_id


class FakeCuda:
    def __init__(self, properties: FakeProperties) -> None:
        self.properties = properties

    def get_device_properties(self, _device_id: int) -> FakeProperties:
        return self.properties


class FakeTorch:
    def __init__(self, properties: FakeProperties) -> None:
        self.cuda = FakeCuda(properties)


def completed(output: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ixsmi"], returncode=0, stdout=output, stderr=""
    )


class SglangRuntimeProviderTest(unittest.TestCase):
    def test_direct_runtime_bdf_is_used_when_available(self) -> None:
        platform = SglangTorchPlatform(
            FakeTorch(FakeProperties(pci_bus_id="00000000:45:00.0"))
        )
        provider = SglangRuntimeProvider(platform)

        mappings = provider.map_all(
            (DeviceContext(framework="sglang", logical_device_id=0),)
        )

        self.assertEqual(mappings[0].pci_bdf, "0000:45:00.0")
        self.assertIn("PCI BDF", mappings[0].evidence[0])

    def test_uuid_fallback_uses_runtime_identity_not_inventory_order(self) -> None:
        output = (
            f"0, 00000000:45:00.0, GPU-{UUID_0}, Iluvatar\n"
            f"1, 00000000:48:00.0, GPU-62e9c670-d402-58b9-972f-f855b246d309, Iluvatar\n"
        )
        platform = SglangTorchPlatform(FakeTorch(FakeProperties()))
        provider = SglangRuntimeProvider(platform, command_runner=lambda *a, **k: completed(output))

        mappings = provider.map_all(
            (DeviceContext(framework="sglang", logical_device_id=0),)
        )

        self.assertEqual(mappings[0].pci_bdf, "0000:45:00.0")
        self.assertEqual(mappings[0].source, provider.name)


class SglangHookDecisionTest(unittest.TestCase):
    def test_explicit_node_is_preserved(self) -> None:
        server_args = types.SimpleNamespace(numa_node=[7])
        with patch("kunpeng_affinity.sglang_plugin._generic_node") as generic:
            result = _around_numa_query(
                lambda _args, _gpu: 7,
                server_args,
                0,
                mode=PluginMode.AUTO,
                provider="auto",
            )

        self.assertEqual(result, 7)
        generic.assert_not_called()

    def test_native_node_wins_over_generic_fallback(self) -> None:
        server_args = types.SimpleNamespace(numa_node=None)
        with patch(
            "kunpeng_affinity.sglang_plugin._generic_node", return_value=4
        ) as generic:
            result = _around_numa_query(
                lambda _args, _gpu: 2,
                server_args,
                0,
                mode=PluginMode.AUTO,
                provider="auto",
            )

        self.assertEqual(result, 2)
        generic.assert_not_called()

    def test_generic_node_is_used_when_native_query_is_empty(self) -> None:
        server_args = types.SimpleNamespace(numa_node=None)
        with patch(
            "kunpeng_affinity.sglang_plugin._generic_node", return_value=4
        ) as generic:
            result = _around_numa_query(
                lambda _args, _gpu: None,
                server_args,
                0,
                mode=PluginMode.AUTO,
                provider="auto",
            )

        self.assertEqual(result, 4)
        generic.assert_called_once_with(0, "auto")

    def test_auto_failure_skips_and_strict_failure_raises(self) -> None:
        server_args = types.SimpleNamespace(numa_node=None)
        error = RuntimeError("identity unavailable")
        with patch(
            "kunpeng_affinity.sglang_plugin._generic_node", side_effect=error
        ):
            self.assertIsNone(
                _around_numa_query(
                    lambda _args, _gpu: None,
                    server_args,
                    0,
                    mode=PluginMode.AUTO,
                    provider="auto",
                )
            )
            with self.assertRaises(RuntimeError):
                _around_numa_query(
                    lambda _args, _gpu: None,
                    server_args,
                    0,
                    mode=PluginMode.STRICT,
                    provider="auto",
                )


if __name__ == "__main__":
    unittest.main()
