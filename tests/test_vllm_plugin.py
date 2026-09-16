from __future__ import annotations

import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from kunpeng_affinity import vllm_plugin
from kunpeng_affinity.core.errors import (
    AffinityConfigurationError,
    AffinityDiscoveryError,
    AffinityIntegrationError,
)


class VllmPluginTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calls = []
        self.environment = patch.dict(
            os.environ,
            {
                "KUNPENG_AFFINITY_MODE": "auto",
                "KUNPENG_AFFINITY_VLLM_FORCE_GENERIC": "",
            },
        )
        self.environment.start()

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
        self.environment.stop()

    @staticmethod
    def resolution(nodes, source):
        return vllm_plugin._AutomaticAffinityResolution(
            nodes=tuple(nodes),
            source=source,
            visibility_fingerprint=None,
            registry=None,
        )

    def test_register_wraps_and_delegates(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=[0],
            )
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

    def test_incompatible_signature_auto_mode_does_not_install(self) -> None:
        @contextmanager
        def incompatible(vllm_config, local_rank):
            yield

        self.numa_utils.configure_subprocess = incompatible
        with self.assertLogs(vllm_plugin.logger, level="WARNING") as captured:
            vllm_plugin.register()

        self.assertIs(self.numa_utils.configure_subprocess, incompatible)
        self.assertIn("Hook not installed", "\n".join(captured.output))

    def test_incompatible_signature_strict_mode_fails(self) -> None:
        @contextmanager
        def incompatible(vllm_config, local_rank):
            yield

        self.numa_utils.configure_subprocess = incompatible
        with patch.dict(os.environ, {"KUNPENG_AFFINITY_MODE": "strict"}):
            with self.assertRaisesRegex(
                AffinityIntegrationError,
                "missing parameters",
            ):
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
            patch.object(vllm_plugin, "_current_platform", return_value=object()),
            patch.object(
                vllm_plugin,
                "_resolve_automatic_nodes",
                return_value=self.resolution([3], "generic"),
            ) as resolver,
            patch.object(vllm_plugin, "_vllm_version", return_value="0.26.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        self.assertEqual(config.parallel_config.numa_bind_nodes, [3])
        self.assertEqual(self.calls, [(config, 0, None, "worker")])
        resolver.assert_called_once()
        self.assertTrue(resolver.call_args.kwargs["force_generic"])

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
            patch.object(vllm_plugin, "_resolve_automatic_nodes") as resolver,
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
            patch.object(vllm_plugin, "_resolve_automatic_nodes") as resolver,
            patch.object(vllm_plugin, "_vllm_version", return_value="0.26.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        resolver.assert_not_called()
        self.assertIsNone(config.parallel_config.numa_bind_nodes)

    def test_forced_generic_failure_is_strict(self) -> None:
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
            patch.object(vllm_plugin, "_current_platform", return_value=object()),
            patch.object(
                vllm_plugin,
                "_resolve_automatic_nodes",
                side_effect=AffinityDiscoveryError(
                    "generic failed",
                    code="DEVICE_MAPPING_MISSING",
                ),
            ),
            patch.object(vllm_plugin, "_vllm_version", return_value="0.26.0"),
        ):
            vllm_plugin.register()
            with self.assertRaisesRegex(
                AffinityIntegrationError,
                "DEVICE_MAPPING_MISSING",
            ):
                with self.numa_utils.configure_subprocess(config, 0):
                    pass

        self.assertIsNone(config.parallel_config.numa_bind_nodes)
        self.assertEqual(self.calls, [])

    def test_native_success_commits_nodes_and_delegates(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=None,
                numa_bind_cpus=None,
            )
        )
        with (
            patch.object(vllm_plugin, "_current_platform", return_value=object()),
            patch.object(
                vllm_plugin,
                "_resolve_automatic_nodes",
                return_value=self.resolution([1], "native"),
            ) as resolver,
            patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        self.assertEqual(config.parallel_config.numa_bind_nodes, [1])
        self.assertEqual(self.calls, [(config, 0, None, "worker")])
        self.assertFalse(resolver.call_args.kwargs["force_generic"])

    def test_native_failure_falls_back_to_generic(self) -> None:
        native_error = AffinityDiscoveryError(
            "native unavailable",
            code="NATIVE_QUERY_UNAVAILABLE",
        )
        with (
            patch(
                "kunpeng_affinity.adapters.resolve_vllm_native_nodes",
                side_effect=native_error,
            ) as native,
            patch.object(
                vllm_plugin,
                "_resolve_generic_nodes",
                return_value=self.resolution([3], "generic"),
            ) as generic,
        ):
            resolution = vllm_plugin._resolve_automatic_nodes(
                self.numa_utils,
                object(),
                force_generic=False,
            )

        self.assertEqual(resolution.nodes, (3,))
        self.assertEqual(resolution.source, "generic")
        native.assert_called_once()
        generic.assert_called_once()

    def test_auto_failure_skips_binding_and_continues(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=None,
                numa_bind_cpus=None,
            )
        )
        with (
            patch.object(vllm_plugin, "_current_platform", return_value=object()),
            patch.object(
                vllm_plugin,
                "_resolve_automatic_nodes",
                side_effect=AffinityDiscoveryError(
                    "no trusted BDF",
                    code="DEVICE_MAPPING_MISSING",
                ),
            ),
            patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"),
        ):
            vllm_plugin.register()
            with self.assertLogs(vllm_plugin.logger, level="WARNING") as captured:
                with self.numa_utils.configure_subprocess(config, 0):
                    pass

        self.assertIsNone(config.parallel_config.numa_bind_nodes)
        self.assertEqual(self.calls, [])
        self.assertIn("launching without additional binding", "\n".join(captured.output))

    def test_visibility_change_skips_commit_in_auto_mode(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=None,
            )
        )
        resolution = vllm_plugin._AutomaticAffinityResolution(
            nodes=(1,),
            source="generic",
            visibility_fingerprint="before",
            registry=object(),
        )
        with (
            patch.object(vllm_plugin, "_current_platform", return_value=object()),
            patch.object(
                vllm_plugin,
                "_resolve_automatic_nodes",
                return_value=resolution,
            ),
            patch.object(
                vllm_plugin,
                "_revalidate_visibility",
                side_effect=AffinityDiscoveryError(
                    "visibility changed",
                    code="VISIBILITY_CHANGED",
                ),
            ),
            patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        self.assertIsNone(config.parallel_config.numa_bind_nodes)
        self.assertEqual(self.calls, [])

    def test_strict_failure_stops_before_delegating(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=None,
            )
        )
        with (
            patch.dict(os.environ, {"KUNPENG_AFFINITY_MODE": "strict"}),
            patch.object(vllm_plugin, "_current_platform", return_value=object()),
            patch.object(
                vllm_plugin,
                "_resolve_automatic_nodes",
                side_effect=AffinityDiscoveryError(
                    "topology unknown",
                    code="NUMA_UNKNOWN",
                ),
            ),
            patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"),
        ):
            vllm_plugin.register()
            with self.assertRaisesRegex(AffinityIntegrationError, "NUMA_UNKNOWN"):
                with self.numa_utils.configure_subprocess(config, 0):
                    pass

        self.assertIsNone(config.parallel_config.numa_bind_nodes)
        self.assertEqual(self.calls, [])

    def test_off_mode_preserves_vllm_native_behavior(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=None,
            )
        )
        with (
            patch.dict(os.environ, {"KUNPENG_AFFINITY_MODE": "off"}),
            patch.object(vllm_plugin, "_resolve_automatic_nodes") as resolver,
            patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        resolver.assert_not_called()
        self.assertEqual(self.calls, [(config, 0, None, "worker")])

    def test_off_mode_is_not_overridden_by_force_generic(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=None,
            )
        )
        with (
            patch.dict(
                os.environ,
                {
                    "KUNPENG_AFFINITY_MODE": "off",
                    "KUNPENG_AFFINITY_VLLM_FORCE_GENERIC": "1",
                },
            ),
            patch.object(vllm_plugin, "_resolve_automatic_nodes") as resolver,
            patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        resolver.assert_not_called()
        self.assertIsNone(config.parallel_config.numa_bind_nodes)
        self.assertEqual(self.calls, [(config, 0, None, "worker")])

    def test_explicit_cpus_are_preserved_when_nodes_are_resolved(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=None,
                numa_bind_cpus=["4-7"],
            )
        )
        with (
            patch.object(vllm_plugin, "_current_platform", return_value=object()),
            patch.object(
                vllm_plugin,
                "_resolve_automatic_nodes",
                return_value=self.resolution([1], "generic"),
            ),
            patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        self.assertEqual(config.parallel_config.numa_bind_nodes, [1])
        self.assertEqual(config.parallel_config.numa_bind_cpus, ["4-7"])

    def test_framework_binding_error_is_not_swallowed(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=None,
                numa_bind_cpus=None,
            )
        )

        @contextmanager
        def failing_original(
            vllm_config,
            local_rank,
            dp_local_rank=None,
            process_kind="worker",
        ):
            raise RuntimeError("numactl execution failed")
            yield

        self.numa_utils.configure_subprocess = failing_original
        with (
            patch.object(vllm_plugin, "_current_platform", return_value=object()),
            patch.object(
                vllm_plugin,
                "_resolve_automatic_nodes",
                return_value=self.resolution([1], "generic"),
            ),
            patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"),
        ):
            vllm_plugin.register()
            with self.assertRaisesRegex(RuntimeError, "numactl execution failed"):
                with self.numa_utils.configure_subprocess(config, 0):
                    pass

        self.assertIsNone(config.parallel_config.numa_bind_nodes)

    def test_invalid_mode_fails_during_registration(self) -> None:
        with (
            patch.dict(os.environ, {"KUNPENG_AFFINITY_MODE": "invalid"}),
            patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"),
        ):
            with self.assertRaisesRegex(AffinityConfigurationError, "expected off"):
                vllm_plugin.register()

    def test_unvalidated_version_auto_mode_does_not_install(self) -> None:
        original = self.numa_utils.configure_subprocess
        with (
            patch.object(vllm_plugin, "_vllm_version", return_value="9.9.9"),
            self.assertLogs(vllm_plugin.logger, level="WARNING") as captured,
        ):
            vllm_plugin.register()

        self.assertIs(self.numa_utils.configure_subprocess, original)
        self.assertIn("Hook not installed", "\n".join(captured.output))

    def test_unvalidated_version_strict_mode_fails(self) -> None:
        with (
            patch.dict(os.environ, {"KUNPENG_AFFINITY_MODE": "strict"}),
            patch.object(vllm_plugin, "_vllm_version", return_value="9.9.9"),
        ):
            with self.assertRaisesRegex(
                AffinityIntegrationError,
                "unsupported vLLM version",
            ):
                vllm_plugin.register()


if __name__ == "__main__":
    unittest.main()
