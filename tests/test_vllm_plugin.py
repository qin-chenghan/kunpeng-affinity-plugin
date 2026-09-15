from __future__ import annotations

import os
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

    def test_forced_generic_path_commits_nodes_before_delegating(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=None,
            )
        )
        with (
            patch.dict(
                os.environ,
                {"KUNPENG_AFFINITY_VLLM_FORCE_GENERIC": "1"},
            ),
            patch.object(vllm_plugin, "_generic_numa_nodes", return_value=[3]),
            patch.object(vllm_plugin, "_vllm_version", return_value="0.26.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        self.assertEqual(config.parallel_config.numa_bind_nodes, [3])
        self.assertEqual(self.calls, [(config, 0, None, "worker")])

    def test_forced_generic_path_preserves_explicit_nodes(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=[2],
            )
        )
        with (
            patch.dict(
                os.environ,
                {"KUNPENG_AFFINITY_VLLM_FORCE_GENERIC": "true"},
            ),
            patch.object(vllm_plugin, "_generic_numa_nodes") as resolver,
            patch.object(vllm_plugin, "_vllm_version", return_value="0.26.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        resolver.assert_not_called()
        self.assertEqual(config.parallel_config.numa_bind_nodes, [2])

    def test_forced_generic_path_respects_disabled_numa_bind(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=False,
                numa_bind_nodes=None,
            )
        )
        with (
            patch.dict(
                os.environ,
                {"KUNPENG_AFFINITY_VLLM_FORCE_GENERIC": "yes"},
            ),
            patch.object(vllm_plugin, "_generic_numa_nodes") as resolver,
            patch.object(vllm_plugin, "_vllm_version", return_value="0.26.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        resolver.assert_not_called()
        self.assertIsNone(config.parallel_config.numa_bind_nodes)

    def test_forced_generic_failure_does_not_delegate_or_mutate(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=None,
            )
        )
        with (
            patch.dict(
                os.environ,
                {"KUNPENG_AFFINITY_VLLM_FORCE_GENERIC": "on"},
            ),
            patch.object(
                vllm_plugin,
                "_generic_numa_nodes",
                side_effect=RuntimeError("generic failed"),
            ),
            patch.object(vllm_plugin, "_vllm_version", return_value="0.26.0"),
        ):
            vllm_plugin.register()
            with self.assertRaisesRegex(RuntimeError, "generic failed"):
                with self.numa_utils.configure_subprocess(config, 0):
                    pass

        self.assertIsNone(config.parallel_config.numa_bind_nodes)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
