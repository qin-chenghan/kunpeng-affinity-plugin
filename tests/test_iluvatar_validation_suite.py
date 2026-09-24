from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
