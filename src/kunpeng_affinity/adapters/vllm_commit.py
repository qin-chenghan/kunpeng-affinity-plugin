"""Atomic mutation of the vLLM NUMA fields owned by this plugin."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from kunpeng_affinity.core.errors import AffinityDiscoveryError


_CONFIG_COMMIT_LOCK = threading.RLock()


@dataclass
class VllmConfigCommit:
    """A committed node value that can roll back only plugin-owned state."""

    parallel_config: Any
    nodes: list[int]
    previous_nodes: Any
    had_nodes_attribute: bool
    wrote_nodes: bool

    def rollback(self) -> None:
        if not self.wrote_nodes:
            return
        with _CONFIG_COMMIT_LOCK:
            if getattr(self.parallel_config, "numa_bind_nodes", None) != self.nodes:
                return
            if self.had_nodes_attribute:
                self.parallel_config.numa_bind_nodes = self.previous_nodes
            else:
                delattr(self.parallel_config, "numa_bind_nodes")
            self.wrote_nodes = False


def commit_vllm_nodes(parallel_config: Any, nodes: list[int]) -> VllmConfigCommit:
    """Commit nodes once, reusing an equivalent concurrent result."""
    candidate = list(nodes)
    with _CONFIG_COMMIT_LOCK:
        had_attribute = hasattr(parallel_config, "numa_bind_nodes")
        existing = getattr(parallel_config, "numa_bind_nodes", None)
        if existing is not None and existing != candidate:
            raise AffinityDiscoveryError(
                f"NUMA nodes changed concurrently from resolved {candidate} "
                f"to {existing}",
                code="CONCURRENT_CONFIG_CONFLICT",
            )

        wrote_nodes = existing is None
        if wrote_nodes:
            try:
                parallel_config.numa_bind_nodes = candidate
            except (AttributeError, TypeError, ValueError) as exc:
                raise AffinityDiscoveryError(
                    f"cannot commit NUMA nodes to vLLM configuration: {exc}",
                    code="CONFIG_COMMIT_FAILED",
                ) from exc

        return VllmConfigCommit(
            parallel_config=parallel_config,
            nodes=candidate,
            previous_nodes=existing,
            had_nodes_attribute=had_attribute,
            wrote_nodes=wrote_nodes,
        )
