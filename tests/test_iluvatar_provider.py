from __future__ import annotations

import subprocess
import unittest

from kunpeng_affinity.core.models import DeviceContext
from kunpeng_affinity.providers.iluvatar_runtime import (
    IluvatarRuntimeProvider,
    normalize_uuid,
    parse_ixsmi_rows,
)


UUID_0 = "8631681a-860d-5d5c-8937-fc4efe2beea4"
UUID_1 = "62e9c670-d402-58b9-972f-f855b246d309"


class FakePlatform:
    uuids = (f"GPU-{UUID_1}", UUID_0)

    @classmethod
    def get_device_uuid(cls, device_id: int) -> str:
        return cls.uuids[device_id]


def completed(output: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ixsmi"], returncode=returncode, stdout=output, stderr="query failed"
    )


class IluvatarRuntimeProviderTest(unittest.TestCase):
    output = (
        f"0, 00000000:45:00.0, GPU-{UUID_0}, Iluvatar BI-V150\n"
        f"1, 00000000:48:00.0, GPU-{UUID_1}, Iluvatar BI-V150\n"
    )

    def contexts(self) -> tuple[DeviceContext, ...]:
        return (
            DeviceContext(framework="vllm", logical_device_id=0),
            DeviceContext(framework="vllm", logical_device_id=1),
        )

    def runner(self, output: str = output, returncode: int = 0):
        def run(*args, **kwargs):
            return completed(output, returncode)

        return run

    def test_normalizes_uuid_prefix_and_case(self) -> None:
        self.assertEqual(normalize_uuid(f"GPU-{UUID_0.upper()}"), UUID_0)
        self.assertEqual(normalize_uuid(UUID_1), UUID_1)

    def test_parses_uuid_to_bdf_without_using_row_order(self) -> None:
        rows = parse_ixsmi_rows(self.output)

        self.assertEqual(rows[0], (0, UUID_0, "0000:45:00.0"))
        self.assertEqual(rows[1], (1, UUID_1, "0000:48:00.0"))

    def test_maps_vllm_logical_devices_by_uuid_after_reordering(self) -> None:
        provider = IluvatarRuntimeProvider(
            FakePlatform, command_runner=self.runner()
        )

        mappings = provider.map_all(self.contexts())

        self.assertEqual(
            [item.pci_bdf for item in mappings],
            ["0000:48:00.0", "0000:45:00.0"],
        )
        self.assertEqual([item.physical_device_id for item in mappings], [1, 0])
        self.assertTrue(all(item.source == provider.name for item in mappings))

    def test_probe_rejects_missing_runtime_uuid(self) -> None:
        class NoUuidPlatform:
            pass

        provider = IluvatarRuntimeProvider(
            NoUuidPlatform, command_runner=self.runner()
        )

        result = provider.probe(self.contexts())

        self.assertFalse(result.supported)
        self.assertIn("get_device_uuid", result.reason or "")

    def test_probe_rejects_uuid_not_in_ixsmi_inventory(self) -> None:
        class UnknownUuidPlatform:
            @classmethod
            def get_device_uuid(cls, device_id: int) -> str:
                return "GPU-00000000-0000-0000-0000-000000000000"

        provider = IluvatarRuntimeProvider(
            UnknownUuidPlatform, command_runner=self.runner()
        )

        result = provider.probe(
            (DeviceContext(framework="vllm", logical_device_id=0),)
        )

        self.assertFalse(result.supported)
        self.assertIn("no PCI BDF", result.reason or "")

    def test_probe_rejects_duplicate_inventory_identity(self) -> None:
        duplicate = (
            f"0, 00000000:45:00.0, GPU-{UUID_0}\n"
            f"1, 00000000:48:00.0, GPU-{UUID_0}\n"
        )
        provider = IluvatarRuntimeProvider(
            FakePlatform, command_runner=self.runner(duplicate)
        )

        result = provider.probe(self.contexts())

        self.assertFalse(result.supported)
        self.assertIn("duplicate", result.reason or "")

    def test_probe_reports_command_failure(self) -> None:
        provider = IluvatarRuntimeProvider(
            FakePlatform, command_runner=self.runner(returncode=1)
        )

        result = provider.probe(self.contexts())

        self.assertFalse(result.supported)
        self.assertIn("status 1", result.reason or "")


if __name__ == "__main__":
    unittest.main()
