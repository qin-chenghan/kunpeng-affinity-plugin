from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "validation" / "iluvatar" / "run.sh"
CONFIG = REPO_ROOT / "validation" / "iluvatar" / "config.env.example"


class IluvatarValidationSuiteTest(unittest.TestCase):
    def _dry_run(self, config: Path = CONFIG) -> str:
        completed = subprocess.run(
            ["bash", str(RUNNER), "--config", str(config), "--dry-run"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    def test_default_config_keeps_environment_changes_disabled(self) -> None:
        output = self._dry_run()

        self.assertIn(f"Configuration: {CONFIG}", output)
        self.assertIn("Result directory: /tmp/kunpeng-affinity-validation-", output)
        self.assertIn("Iluvatar Provider all devices", output)
        self.assertIn(
            "Editable install and entry points SKIP", output
        )
        self.assertIn(
            "vLLM forced-generic dummy spawn  SKIP", output
        )

    def test_enabled_config_selects_the_complete_validation_suite(self) -> None:
        config_text = CONFIG.read_text(encoding="utf-8").replace(
            "ENABLE_ENVIRONMENT_CHANGES=0",
            "ENABLE_ENVIRONMENT_CHANGES=1",
        )
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.env"
            config.write_text(config_text, encoding="utf-8")
            output = self._dry_run(config)

        self.assertIn("Editable install and entry points DRY-RUN", output)
        self.assertIn("vLLM forced-generic dummy spawn  DRY-RUN", output)
        self.assertIn("vLLM automatic fallback dummy spawn DRY-RUN", output)

    def test_relative_result_directory_is_resolved_from_repository_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as directory:
            result_name = f"{Path(directory).name}/results"
            config_text = CONFIG.read_text(encoding="utf-8").replace(
                "RESULT_DIR=",
                f"RESULT_DIR={result_name}",
            )
            config = Path(directory) / "config.env"
            config.write_text(config_text, encoding="utf-8")
            output = self._dry_run(config)

            expected = REPO_ROOT / result_name
            self.assertIn(f"Result directory: {expected}", output)
            self.assertTrue(expected.is_dir())

    def test_reorder_count_ignores_vllm_stdout_banner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sysfs = root / "sys"
            (sysfs / "bus/pci/devices").mkdir(parents=True)
            fake_python = root / "python3"
            fake_python.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-c" && "${2:-}" == *current_platform.device_count* ]]; then
  echo 'INFO platform plugin iluvatar is activated'
  echo 'KUNPENG_VISIBLE_DEVICE_COUNT=4'
  exit 0
fi
if [[ "${1:-}" == "-c" ]]; then
  exit 0
fi
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  echo 'python_executable=fake'
  echo 'python_version=3.12.0'
  echo 'vllm_version=0.23.0+corex.5.0.0'
  echo 'vllm_location=/fake'
  exit 0
fi
exit 0
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            fake_ixsmi = root / "ixsmi"
            fake_ixsmi.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_ixsmi.chmod(0o755)
            result_dir = root / "results"
            config = root / "config.env"
            config.write_text(
                "\n".join(
                    (
                        f"PYTHON_BIN={fake_python}",
                        f"IXSMI_BIN={fake_ixsmi}",
                        f"SYSFS_ROOT={sysfs}",
                        f"RESULT_DIR={result_dir}",
                        "RUN_UNIT_TESTS=0",
                        "RUN_TOPOLOGY_PROBE=0",
                        "RUN_PROVIDER_PROBE=0",
                        "RUN_REORDER_PROBE=1",
                        "RUN_SOURCE_INSTALL=0",
                        "RUN_FORCED_GENERIC_SPAWN=0",
                        "RUN_AUTO_FALLBACK_SPAWN=0",
                        "ENABLE_ENVIRONMENT_CHANGES=0",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                ["bash", str(RUNNER), "--config", str(config)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )

        self.assertIn("Iluvatar Provider visibility reorder PASS", completed.stdout)
        self.assertNotIn("fewer than two devices visible", completed.stdout)


if __name__ == "__main__":
    unittest.main()
