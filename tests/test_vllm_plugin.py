from __future__ import annotations

import json
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import ANY, Mock, patch

from kunpeng_affinity import vllm_plugin
from kunpeng_affinity import adapters
from kunpeng_affinity.core.identity import serialized_snapshot_fingerprint
from kunpeng_affinity.core.errors import (
    AffinityDiscoveryError,
    AffinityIntegrationError,
    PluginConfigError,
)


class VllmPluginTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calls = []
        self.environment = patch.dict(
            os.environ,
            {
                "KUNPENG_AFFINITY_CONFIG": "",
                "KUNPENG_AFFINITY_MODE": "auto",
                "KUNPENG_AFFINITY_PROVIDER": "auto",
                "KUNPENG_AFFINITY_CPU_POLICY": "node",
                "KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL": "summary",
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
    def resolution(nodes, source, fingerprint=None):
        snapshot_json = None
        if source == "generic" and fingerprint is not None:
            snapshot_json = json.dumps(
                {
                    "metadata": {"adapter": "vllm.configure_subprocess.v1"},
                    "mappings": [
                        {"logical_device_id": index, "pci_bdf": f"0000:{index + 1:02x}:00.0"}
                        for index, _ in enumerate(nodes)
                    ],
                    "resolutions": [
                        {"numa_node": node} for node in nodes
                    ],
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            fingerprint = serialized_snapshot_fingerprint(snapshot_json)
        return vllm_plugin._AutomaticAffinityResolution(
            nodes=tuple(nodes),
            source=source,
            visibility_fingerprint=fingerprint,
            registry=None,
            snapshot_json=snapshot_json,
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

    def test_generic_resolution_carries_snapshot_to_transaction(self) -> None:
        snapshot_json = '{"snapshot":"complete"}'
        batch = types.SimpleNamespace(
            committable=True,
            ordered_results=(
                types.SimpleNamespace(
                    affinity=types.SimpleNamespace(numa_node=3),
                ),
            ),
            visibility_fingerprint="fingerprint",
            snapshot_json=snapshot_json,
        )
        registry = object()

        with (
            patch.object(adapters, "check_vllm_generic_eligibility"),
            patch.object(
                adapters,
                "resolve_vllm_generic_affinity",
                return_value=batch,
            ) as resolver,
        ):
            result = vllm_plugin._resolve_generic_nodes(
                self.numa_utils,
                object(),
                registry,
                requested_provider="test-provider",
                process_kind="worker",
                local_rank=0,
                dp_local_rank=None,
            )

        self.assertEqual(result.nodes, (3,))
        self.assertEqual(result.visibility_fingerprint, "fingerprint")
        self.assertEqual(result.snapshot_json, snapshot_json)
        resolver.assert_called_once_with(
            ANY,
            registry=registry,
            requested_provider="test-provider",
            process_kind="worker",
            local_rank=0,
            dp_local_rank=None,
        )

    def test_transaction_commit_failure_exits_native_context_and_rolls_back(
        self,
    ) -> None:
        events = []

        @contextmanager
        def original(*args, **kwargs):
            events.append("entered")
            try:
                yield
            finally:
                events.append("exited")

        commit = types.SimpleNamespace(
            mark_committed=Mock(side_effect=RuntimeError("marker conflict")),
            rollback=Mock(side_effect=lambda: events.append("rolled-back")),
        )
        parallel_config = types.SimpleNamespace(numa_bind_cpus=None)

        with (
            patch.object(vllm_plugin, "_current_platform", return_value=object()),
            patch.object(
                vllm_plugin,
                "_resolve_automatic_nodes",
                return_value=self.resolution([3], "generic", "candidate"),
            ),
            patch.object(vllm_plugin, "_revalidate_visibility"),
            patch(
                "kunpeng_affinity.adapters.vllm_commit.commit_vllm_nodes",
                return_value=commit,
            ),
            self.assertRaisesRegex(RuntimeError, "marker conflict"),
        ):
            with vllm_plugin._automatic_affinity_context(
                current=original,
                numa_utils=self.numa_utils,
                mode=vllm_plugin.PluginMode.AUTO,
                force_generic=False,
                requested_provider=None,
                args=(),
                kwargs={},
                parallel_config=parallel_config,
                process_kind="worker",
                local_rank=0,
                dp_local_rank=None,
            ):
                pass

        self.assertEqual(events, ["entered", "exited", "rolled-back"])
        commit.rollback.assert_called_once_with()

    def test_vendor_local_version_installs_hook(self) -> None:
        with patch.object(
            vllm_plugin,
            "_vllm_version",
            return_value="0.23.0+corex.5.0.0",
        ):
            vllm_plugin.register()

        self.assertTrue(
            hasattr(
                self.numa_utils.configure_subprocess,
                "__kunpeng_affinity_original__",
            )
        )

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
            with self.numa_utils.configure_subprocess(config, 0, dp_local_rank=2):
                pass

        self.assertEqual(config.parallel_config.numa_bind_nodes, [3])
        self.assertEqual(self.calls, [(config, 0, 2, "worker")])
        resolver.assert_called_once()
        self.assertTrue(resolver.call_args.kwargs["force_generic"])
        self.assertEqual(resolver.call_args.kwargs["dp_local_rank"], 2)

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

    def test_native_success_is_delegated_without_plugin_transaction(self) -> None:
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

        self.assertIsNone(config.parallel_config.numa_bind_nodes)
        self.assertFalse(hasattr(config.parallel_config, "_kunpeng_affinity_transaction"))
        self.assertEqual(self.calls, [(config, 0, None, "worker")])
        self.assertFalse(resolver.call_args.kwargs["force_generic"])

    def test_committed_plugin_marker_is_reused_without_resolving_again(self) -> None:
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
                return_value=self.resolution(
                    [1], "generic", fingerprint="test-fingerprint"
                ),
            ) as resolver,
            patch.object(vllm_plugin, "_revalidate_visibility"),
            patch(
                "kunpeng_affinity.adapters.vllm_generic.resolve_vllm_visibility_fingerprint",
                side_effect=lambda *args, **kwargs: config.parallel_config._kunpeng_affinity_transaction[
                    "visibility_fingerprint"
                ],
            ),
            patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        resolver.assert_called_once()
        self.assertEqual(config.parallel_config.numa_bind_nodes, [1])

    def test_explicit_vllm_provider_reaches_resolution(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=None,
                numa_bind_cpus=None,
            )
        )
        with (
            patch.dict(
                os.environ,
                {"KUNPENG_AFFINITY_PROVIDER": "vllm-platform-pci"},
            ),
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

        self.assertEqual(
            resolver.call_args.kwargs["requested_provider"],
            "vllm-platform-pci",
        )

    def test_native_capability_gap_falls_back_to_generic(self) -> None:
        with (
            patch(
                "kunpeng_affinity.adapters.classify_vllm_native_result",
                return_value=vllm_plugin.NativeOutcome(
                    status=vllm_plugin.NativeStatus.FALLBACK_ALLOWED,
                    failure_code="NATIVE_QUERY_UNAVAILABLE",
                ),
            ),
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
                dp_local_rank=3,
            )

        self.assertEqual(resolution.nodes, (3,))
        self.assertEqual(resolution.source, "generic")
        generic.assert_called_once()
        self.assertEqual(generic.call_args.kwargs["dp_local_rank"], 3)

    def test_native_exception_is_not_converted_to_generic_fallback(self) -> None:
        native_error = RuntimeError("native query failed")
        with (
            patch(
                "kunpeng_affinity.adapters.classify_vllm_native_result",
                return_value=vllm_plugin.NativeOutcome(
                    status=vllm_plugin.NativeStatus.ERROR,
                    original_error=native_error,
                    failure_code="NATIVE_QUERY_FAILED",
                ),
            ),
            patch.object(vllm_plugin, "_resolve_generic_nodes") as generic,
        ):
            with self.assertRaises(RuntimeError) as captured:
                vllm_plugin._resolve_automatic_nodes(
                    self.numa_utils,
                    object(),
                    force_generic=False,
                )

        self.assertIs(captured.exception, native_error)
        generic.assert_not_called()

    def test_native_empty_result_is_preserved_when_platform_is_supported(self) -> None:
        class SupportedPlatform:
            get_all_gpu_pci_bus_ids = staticmethod(lambda: {0: "0000:01:00.0"})

        with (
            patch(
                "kunpeng_affinity.adapters.classify_vllm_native_result",
                return_value=vllm_plugin.NativeOutcome(
                    status=vllm_plugin.NativeStatus.PRESERVE_NATIVE,
                    failure_code="NATIVE_RESULT_EMPTY",
                ),
            ),
            patch.object(vllm_plugin, "_resolve_generic_nodes") as generic,
        ):
            resolution = vllm_plugin._resolve_automatic_nodes(
                self.numa_utils,
                SupportedPlatform,
                force_generic=False,
            )

        self.assertEqual(resolution.source, "native-preserved")
        generic.assert_not_called()

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

    def test_auto_failure_preserves_explicit_cpu_policy_through_native_context(self) -> None:
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
                side_effect=AffinityDiscoveryError(
                    "no trusted BDF", code="DEVICE_MAPPING_MISSING"
                ),
            ),
            patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        self.assertEqual(self.calls, [(config, 0, None, "worker")])
        self.assertEqual(config.parallel_config.numa_bind_cpus, ["4-7"])

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
        original = self.numa_utils.configure_subprocess
        with (
            patch.dict(os.environ, {"KUNPENG_AFFINITY_MODE": "off"}),
            patch.object(vllm_plugin, "_resolve_automatic_nodes") as resolver,
            patch.object(vllm_plugin, "_vllm_version", return_value="0.23.0"),
        ):
            vllm_plugin.register()
            with self.numa_utils.configure_subprocess(config, 0):
                pass

        resolver.assert_not_called()
        self.assertIs(self.numa_utils.configure_subprocess, original)
        self.assertEqual(self.calls, [(config, 0, None, "worker")])

    def test_off_mode_is_not_overridden_by_force_generic(self) -> None:
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                numa_bind=True,
                numa_bind_nodes=None,
            )
        )
        original = self.numa_utils.configure_subprocess
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
        self.assertIs(self.numa_utils.configure_subprocess, original)
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
            with self.assertRaisesRegex(PluginConfigError, "expected off"):
                vllm_plugin.register()

    def test_exact_cpu_policy_is_rejected_before_hook_installation(self) -> None:
        original = self.numa_utils.configure_subprocess
        with (
            patch.dict(os.environ, {"KUNPENG_AFFINITY_CPU_POLICY": "exact"}),
            self.assertRaisesRegex(AffinityIntegrationError, "exact CPU policy"),
        ):
            vllm_plugin.register()

        self.assertIs(self.numa_utils.configure_subprocess, original)

    def test_unknown_explicit_provider_is_rejected_before_hook_installation(self) -> None:
        original = self.numa_utils.configure_subprocess
        with (
            patch.dict(os.environ, {"KUNPENG_AFFINITY_PROVIDER": "missing"}),
            self.assertRaisesRegex(AffinityIntegrationError, "not registered"),
        ):
            vllm_plugin.register()

        self.assertIs(self.numa_utils.configure_subprocess, original)

    def test_unvalidated_version_auto_mode_does_not_install(self) -> None:
        original = self.numa_utils.configure_subprocess
        with (
            patch.object(vllm_plugin, "_vllm_version", return_value="9.9.9"),
            self.assertLogs(vllm_plugin.logger, level="WARNING") as captured,
        ):
            vllm_plugin.register()

        self.assertIs(self.numa_utils.configure_subprocess, original)
        self.assertIn("Hook not installed", "\n".join(captured.output))

    def test_upstream_postrelease_is_not_accepted_as_validated(self) -> None:
        original = self.numa_utils.configure_subprocess
        with (
            patch.object(
                vllm_plugin,
                "_vllm_version",
                return_value="0.23.0.post1",
            ),
            self.assertLogs(vllm_plugin.logger, level="WARNING"),
        ):
            vllm_plugin.register()

        self.assertIs(self.numa_utils.configure_subprocess, original)

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
