from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from kunpeng_affinity import vllm_plugin


class VllmPluginTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calls = []

        @contextmanager
        def original(
            vllm_config,
            local_rank,
            dp_local_rank=None,
            process_kind="worker",
        ):
            self.calls.append(
                (vllm_config, local_rank, dp_local_rank, process_kind)
            )
            yield

        self.numa_utils = types.ModuleType("vllm.utils.numa_utils")
        self.numa_utils.configure_subprocess = original
        self.vllm = types.ModuleType("vllm")
        self.utils = types.ModuleType("vllm.utils")
        self.utils.numa_utils = self.numa_utils

        self.modules = patch.dict(
            sys.modules,
            {
                "vllm": self.vllm,
                "vllm.utils": self.utils,
                "vllm.utils.numa_utils": self.numa_utils,
            },
        )
        self.modules.start()

    def tearDown(self) -> None:
        self.modules.stop()

    def test_register_wraps_and_delegates(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(numa_bind=True)
        )
        with patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"):
            vllm_plugin.register()

        with self.assertLogs(vllm_plugin.logger, level="WARNING") as captured:
            with self.numa_utils.configure_subprocess(
                config, 2, dp_local_rank=1, process_kind="EngineCore"
            ):
                pass

        self.assertEqual(self.calls, [(config, 2, 1, "EngineCore")])
        self.assertIn("process_kind=EngineCore", "\n".join(captured.output))
        self.assertIn("numa_bind=True", "\n".join(captured.output))

    def test_register_is_idempotent(self) -> None:
        with patch.object(vllm_plugin, "_vllm_version", return_value="0.26.0"):
            vllm_plugin.register()
            first_wrapper = self.numa_utils.configure_subprocess
            vllm_plugin.register()

        self.assertIs(self.numa_utils.configure_subprocess, first_wrapper)

    def test_incompatible_signature_fails_closed(self) -> None:
        @contextmanager
        def incompatible(vllm_config, local_rank):
            yield

        self.numa_utils.configure_subprocess = incompatible
        with self.assertRaisesRegex(RuntimeError, "missing parameters"):
            vllm_plugin.register()


if __name__ == "__main__":
    unittest.main()
