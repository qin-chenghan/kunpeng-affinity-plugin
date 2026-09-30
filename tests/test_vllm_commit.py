from __future__ import annotations

import copy
import types
import unittest

from kunpeng_affinity.adapters.vllm_commit import (
    commit_vllm_nodes,
    release_invalid_vllm_transaction,
)
from kunpeng_affinity.core.errors import AffinityDiscoveryError


class VllmConfigCommitTest(unittest.TestCase):
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

        with self.assertRaisesRegex(AffinityDiscoveryError, "changed concurrently") as captured:
            commit_vllm_nodes(config, [1])

        self.assertEqual(captured.exception.code, "CONCURRENT_CONFIG_CONFLICT")
        self.assertEqual(config.numa_bind_nodes, [2])

    def test_rollback_does_not_overwrite_later_configuration(self) -> None:
        config = types.SimpleNamespace(numa_bind_nodes=None)
        commit = commit_vllm_nodes(config, [1])
        config.numa_bind_nodes = [3]

        commit.rollback()

        self.assertEqual(config.numa_bind_nodes, [3])

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


if __name__ == "__main__":
    unittest.main()
