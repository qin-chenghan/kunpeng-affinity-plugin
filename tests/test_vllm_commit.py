from __future__ import annotations

import types
import unittest
import copy
import json
import os
from unittest.mock import patch

from kunpeng_affinity.adapters.vllm_commit import (
    commit_vllm_nodes,
    release_invalid_vllm_transaction,
    validate_inherited_vllm_transaction,
)
from kunpeng_affinity.core.errors import AffinityDiscoveryError
from kunpeng_affinity.core.identity import serialized_snapshot_fingerprint


class VllmConfigCommitTest(unittest.TestCase):
    @staticmethod
    def _snapshot(nodes: list[int]) -> str:
        return json.dumps(
            {
                "metadata": {"adapter": "vllm.configure_subprocess.v1"},
                "mappings": [
                    {
                        "logical_device_id": index,
                        "pci_bdf": f"0000:{index + 1:02x}:00.0",
                    }
                    for index, _ in enumerate(nodes)
                ],
                "resolutions": [{"numa_node": node} for node in nodes],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def test_commits_and_rolls_back_plugin_owned_nodes(self) -> None:
        config = types.SimpleNamespace(numa_bind_nodes=None)

        commit = commit_vllm_nodes(config, [1, 2], visibility_fingerprint="fp")
        self.assertEqual(config.numa_bind_nodes, [1, 2])
        self.assertEqual(config._kunpeng_affinity_transaction["status"], "APPLIED")
        copy.deepcopy(config._kunpeng_affinity_transaction)
        commit.mark_committed()
        self.assertEqual(config._kunpeng_affinity_transaction["status"], "COMMITTED")
        commit.rollback()

        self.assertIsNone(config.numa_bind_nodes)
        self.assertFalse(hasattr(config, "_kunpeng_affinity_transaction"))

    def test_reuses_equivalent_concurrent_commit(self) -> None:
        config = types.SimpleNamespace(numa_bind_nodes=[1])

        commit = commit_vllm_nodes(config, [1])
        commit.rollback()

        self.assertEqual(config.numa_bind_nodes, [1])

    def test_rejects_conflicting_concurrent_commit(self) -> None:
        config = types.SimpleNamespace(numa_bind_nodes=[2])

        with self.assertRaisesRegex(
            AffinityDiscoveryError, "changed concurrently"
        ) as captured:
            commit_vllm_nodes(config, [1])

        self.assertEqual(captured.exception.code, "CONCURRENT_CONFIG_CONFLICT")
        self.assertEqual(config.numa_bind_nodes, [2])

    def test_rollback_does_not_overwrite_later_configuration(self) -> None:
        config = types.SimpleNamespace(numa_bind_nodes=None)
        commit = commit_vllm_nodes(config, [1])
        config.numa_bind_nodes = [3]

        commit.rollback()

        self.assertEqual(config.numa_bind_nodes, [3])

    def test_marker_written_snapshot_is_not_mutated_by_config_changes(self) -> None:
        config = types.SimpleNamespace(numa_bind_nodes=None)
        snapshot = self._snapshot([1, 2])
        commit = commit_vllm_nodes(
            config,
            [1, 2],
            visibility_fingerprint=serialized_snapshot_fingerprint(snapshot),
            snapshot_json=snapshot,
        )
        commit.mark_committed()

        config.numa_bind_nodes.append(3)

        with self.assertRaisesRegex(AffinityDiscoveryError, "modified externally"):
            validate_inherited_vllm_transaction(
                config,
                numa_utils=types.SimpleNamespace(),
                platform=object(),
                local_rank=0,
                dp_local_rank=None,
                process_kind="EngineCore",
            )
        release_invalid_vllm_transaction(config)
        self.assertEqual(config.numa_bind_nodes, [1, 2, 3])
        self.assertFalse(hasattr(config, "_kunpeng_affinity_transaction"))

    def test_malformed_marker_release_does_not_guess_field_ownership(self) -> None:
        markers = (
            "not-a-mapping",
            {"status": "COMMITTED", "written": []},
            {
                "status": "COMMITTED",
                "written": {
                    "numa_bind_nodes": [1],
                    "numa_bind_cpus": None,
                },
                "previous": {
                    "numa_bind_nodes": None,
                    "numa_bind_cpus": None,
                },
                "previous_present": {"numa_bind_cpus": False},
            },
        )
        for marker in markers:
            with self.subTest(marker=marker):
                config = types.SimpleNamespace(
                    numa_bind_nodes=[1],
                    _kunpeng_affinity_transaction=marker,
                )

                release_invalid_vllm_transaction(config)

                self.assertEqual(config.numa_bind_nodes, [1])
                self.assertFalse(hasattr(config, "_kunpeng_affinity_transaction"))

    def test_marker_requires_owner_and_snapshot_schema(self) -> None:
        config = types.SimpleNamespace(
            numa_bind_nodes=[1],
            _kunpeng_affinity_transaction={
                "status": "COMMITTED",
                "contract_version": "1",
                "transaction_id": "tx",
                "visibility_fingerprint": "fp",
                "written": {
                    "numa_bind_nodes": [1],
                    "numa_bind_cpus": None,
                },
                "previous": {
                    "numa_bind_nodes": None,
                    "numa_bind_cpus": None,
                },
                "previous_present": {
                    "numa_bind_nodes": True,
                    "numa_bind_cpus": False,
                },
            },
        )

        with self.assertRaisesRegex(AffinityDiscoveryError, "owner PID"):
            validate_inherited_vllm_transaction(
                config,
                numa_utils=types.SimpleNamespace(),
                platform=object(),
                local_rank=0,
                dp_local_rank=None,
                process_kind="EngineCore",
            )

    def test_same_process_reuse_requires_matching_complete_fingerprint(self) -> None:
        config = types.SimpleNamespace(numa_bind_nodes=None)
        snapshot = self._snapshot([1])
        commit = commit_vllm_nodes(
            config,
            [1],
            visibility_fingerprint=serialized_snapshot_fingerprint(snapshot),
            snapshot_json=snapshot,
        )
        commit.mark_committed()

        with (
            patch(
                "kunpeng_affinity.adapters.vllm_generic.resolve_vllm_visibility_fingerprint",
                return_value="changed",
            ),
            self.assertRaisesRegex(AffinityDiscoveryError, "same-process reuse"),
        ):
            validate_inherited_vllm_transaction(
                config,
                numa_utils=types.SimpleNamespace(),
                platform=object(),
                local_rank=0,
                dp_local_rank=None,
                process_kind="worker",
            )

    def test_spawn_reuse_revalidates_consumed_device_identity(self) -> None:
        config = types.SimpleNamespace(numa_bind_nodes=None)
        snapshot = self._snapshot([1])
        commit = commit_vllm_nodes(
            config,
            [1],
            visibility_fingerprint=serialized_snapshot_fingerprint(snapshot),
            snapshot_json=snapshot,
        )
        commit.mark_committed()
        config._kunpeng_affinity_transaction["pid"] = os.getpid() + 1
        current = types.SimpleNamespace(
            mapping=types.SimpleNamespace(
                pci_bdf="0000:02:00.0",
                physical_device_id=None,
                instance_id=None,
            ),
            affinity=types.SimpleNamespace(numa_node=1),
        )

        with (
            patch(
                "kunpeng_affinity.adapters.vllm_commit.os.sched_getaffinity",
                return_value={0},
                create=True,
            ),
            patch(
                "kunpeng_affinity.adapters.vllm_generic.resolve_vllm_consumed_device",
                return_value=current,
            ),
            self.assertRaisesRegex(AffinityDiscoveryError, "identity or NUMA"),
        ):
            validate_inherited_vllm_transaction(
                config,
                numa_utils=types.SimpleNamespace(_get_gpu_index=lambda *args: 0),
                platform=object(),
                local_rank=0,
                dp_local_rank=None,
                process_kind="worker",
            )


if __name__ == "__main__":
    unittest.main()
