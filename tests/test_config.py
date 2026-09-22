from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from kunpeng_affinity.config import (
    CpuPolicy,
    DiagnosticLevel,
    PluginMode,
    load_plugin_config,
    load_plugin_mode,
)
from kunpeng_affinity.core.errors import PluginConfigError

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 has no standard-library TOML parser.
    tomllib = None


class PluginConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write(self, value: object) -> Path:
        path = self.root / "config.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_defaults_are_immutable(self) -> None:
        config = load_plugin_config({})

        self.assertEqual(config.mode, PluginMode.AUTO)
        self.assertEqual(config.provider, "auto")
        self.assertEqual(config.cpu_policy, CpuPolicy.NODE)
        self.assertEqual(config.diagnostic_level, DiagnosticLevel.SUMMARY)
        with self.assertRaises(FrozenInstanceError):
            config.provider = "changed"  # type: ignore[misc]

    def test_environment_overrides_json_file(self) -> None:
        path = self._write(
            {
                "mode": "strict",
                "provider": "runtime-a",
                "cpu_policy": "exact",
                "diagnostic_level": "detail",
            }
        )

        config = load_plugin_config(
            {
                "KUNPENG_AFFINITY_CONFIG": str(path),
                "KUNPENG_AFFINITY_MODE": "auto",
                "KUNPENG_AFFINITY_PROVIDER": "runtime-b",
            }
        )

        self.assertEqual(config.mode, PluginMode.AUTO)
        self.assertEqual(config.provider, "runtime-b")
        self.assertEqual(config.cpu_policy, CpuPolicy.EXACT)
        self.assertEqual(config.diagnostic_level, DiagnosticLevel.DETAIL)
        self.assertEqual(config.config_file, path)

    def test_unknown_file_field_is_rejected(self) -> None:
        path = self._write({"mode": "auto", "unexpected": True})

        with self.assertRaisesRegex(PluginConfigError, "unknown.*unexpected"):
            load_plugin_config({"KUNPENG_AFFINITY_CONFIG": str(path)})

    def test_duplicate_file_field_is_rejected(self) -> None:
        path = self.root / "config.json"
        path.write_text('{"mode":"auto","mode":"strict"}', encoding="utf-8")

        with self.assertRaisesRegex(PluginConfigError, "duplicate.*mode"):
            load_plugin_config({"KUNPENG_AFFINITY_CONFIG": str(path)})

    def test_invalid_values_are_rejected(self) -> None:
        cases = (
            {"KUNPENG_AFFINITY_MODE": "sometimes"},
            {"KUNPENG_AFFINITY_PROVIDER": "bad provider"},
            {"KUNPENG_AFFINITY_CPU_POLICY": "socket"},
            {"KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL": "verbose"},
        )
        for environ in cases:
            with self.subTest(environ=environ):
                with self.assertRaises(PluginConfigError):
                    load_plugin_config(environ)

    def test_invalid_file_is_rejected(self) -> None:
        missing = self.root / "missing.json"
        with self.assertRaisesRegex(PluginConfigError, "cannot read"):
            load_plugin_config({"KUNPENG_AFFINITY_CONFIG": str(missing)})

        path = self.root / "config.json"
        path.write_bytes(b"\xff")
        with self.assertRaisesRegex(PluginConfigError, "UTF-8"):
            load_plugin_config({"KUNPENG_AFFINITY_CONFIG": str(path)})

        path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(PluginConfigError, "JSON object"):
            load_plugin_config({"KUNPENG_AFFINITY_CONFIG": str(path)})

    def test_environment_off_does_not_read_config_file(self) -> None:
        mode = load_plugin_mode(
            {
                "KUNPENG_AFFINITY_MODE": "off",
                "KUNPENG_AFFINITY_CONFIG": str(self.root / "missing.json"),
            }
        )

        self.assertEqual(mode, PluginMode.OFF)

    def test_file_off_is_resolved_without_environment_override(self) -> None:
        path = self._write({"mode": "off"})

        mode = load_plugin_mode({"KUNPENG_AFFINITY_CONFIG": str(path)})

        self.assertEqual(mode, PluginMode.OFF)

    def test_package_import_does_not_import_optional_frameworks(self) -> None:
        script = (
            "import os, sys; "
            "os.environ['KUNPENG_AFFINITY_MODE'] = 'off'; "
            "import kunpeng_affinity; "
            "assert 'vllm' not in sys.modules; "
            "assert 'sglang' not in sys.modules; "
            "from kunpeng_affinity.vllm_plugin import register; "
            "register(); "
            "assert 'vllm' not in sys.modules"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

        result = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_file_off_register_does_not_import_vllm(self) -> None:
        path = self._write({"mode": "off"})
        script = (
            "import sys; "
            "from kunpeng_affinity.vllm_plugin import register; "
            "register(); "
            "assert 'vllm' not in sys.modules"
        )
        environment = dict(os.environ)
        environment.pop("KUNPENG_AFFINITY_MODE", None)
        environment["KUNPENG_AFFINITY_CONFIG"] = str(path)
        environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

        result = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_vllm_entry_point_metadata_targets_register(self) -> None:
        if tomllib is None:
            self.skipTest("tomllib is not available on Python 3.10")
        project = Path(__file__).parents[1]
        with (project / "pyproject.toml").open("rb") as source:
            metadata = tomllib.load(source)

        entry_points = metadata["project"]["entry-points"]["vllm.general_plugins"]
        self.assertEqual(
            entry_points,
            {"kunpeng_affinity": "kunpeng_affinity.vllm_plugin:register"},
        )

    def test_sglang_entry_point_metadata_targets_register(self) -> None:
        if tomllib is None:
            self.skipTest("tomllib is not available on Python 3.10")
        metadata = tomllib.loads(
            (Path(__file__).parents[1] / "pyproject.toml").read_text()
        )
        entry_points = metadata["project"]["entry-points"]["sglang.srt.plugins"]
        self.assertEqual(
            entry_points,
            {"kunpeng_affinity": "kunpeng_affinity.sglang_plugin:register"},
        )


if __name__ == "__main__":
    unittest.main()
