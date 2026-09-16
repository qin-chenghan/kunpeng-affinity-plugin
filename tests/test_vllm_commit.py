from __future__ import annotations

import types
import unittest

from kunpeng_affinity.adapters.vllm_commit import commit_vllm_nodes
from kunpeng_affinity.core.errors import AffinityDiscoveryError


class VllmConfigCommitTest(unittest.TestCase):
    def test_commits_and_rolls_back_plugin_owned_nodes(self) -> None:
        config = types.SimpleNamespace(numa_bind_nodes=None)

        commit = commit_vllm_nodes(config, [1, 2])
        self.assertEqual(config.numa_bind_nodes, [1, 2])
        commit.rollback()

        self.assertIsNone(config.numa_bind_nodes)

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


if __name__ == "__main__":
    unittest.main()
