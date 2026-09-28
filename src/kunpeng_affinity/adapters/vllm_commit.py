"""Atomic ownership and mutation of vLLM NUMA node configuration."""

from __future__ import annotations

import copy
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from kunpeng_affinity.core.errors import AffinityDiscoveryError


TRANSACTION_MARKER = "_kunpeng_affinity_transaction"
CONTRACT_VERSION = "1"
CAPABILITY_PROFILE = "vllm.configure_subprocess.v1"
_COMMIT_LOCK = threading.RLock()


@dataclass
class VllmConfigCommit:
    """A node transaction that restores only values still owned by the plugin."""

    parallel_config: Any
    transaction_id: str
    nodes: list[int]
    previous_nodes: Any
    had_nodes_attribute: bool
    wrote_nodes: bool

    @property
    def marker(self) -> dict[str, Any]:
        value = getattr(self.parallel_config, TRANSACTION_MARKER, None)
        return value if isinstance(value, dict) else {}

    def mark_committed(self) -> None:
        if not self.wrote_nodes:
            return
        with _COMMIT_LOCK:
            marker = self.marker
            if marker.get("transaction_id") != self.transaction_id:
                raise AffinityDiscoveryError(
                    "vLLM affinity transaction marker changed before commit",
                    code="TRANSACTION_MARKER_CONFLICT",
                )
            if marker.get("status") != "APPLIED":
                raise AffinityDiscoveryError(
                    "vLLM affinity transaction is not in APPLIED state",
                    code="TRANSACTION_STATE_INVALID",
                )
            setattr(
                self.parallel_config,
                TRANSACTION_MARKER,
                {**marker, "status": "COMMITTED"},
            )

    def rollback(self) -> None:
        if not self.wrote_nodes:
            return
        with _COMMIT_LOCK:
            if getattr(self.parallel_config, "numa_bind_nodes", None) == self.nodes:
                if self.had_nodes_attribute:
                    self.parallel_config.numa_bind_nodes = self.previous_nodes
                else:
                    delattr(self.parallel_config, "numa_bind_nodes")
            if self.marker.get("transaction_id") == self.transaction_id:
                delattr(self.parallel_config, TRANSACTION_MARKER)
            self.wrote_nodes = False


def commit_vllm_nodes(
    parallel_config: Any,
    nodes: list[int],
    *,
    visibility_fingerprint: str | None = None,
    snapshot_json: str | None = None,
    requested_provider: str | None = None,
) -> VllmConfigCommit:
    """Compare and write the plugin-owned NUMA node list as one transaction."""
    candidate = list(nodes)
    with _COMMIT_LOCK:
        had_nodes = hasattr(parallel_config, "numa_bind_nodes")
        previous_nodes = getattr(parallel_config, "numa_bind_nodes", None)
        if previous_nodes is not None and previous_nodes != candidate:
            raise AffinityDiscoveryError(
                f"NUMA nodes changed concurrently from resolved {candidate} "
                f"to {previous_nodes}",
                code="CONCURRENT_CONFIG_CONFLICT",
            )

        transaction_id = uuid.uuid4().hex
        wrote_nodes = previous_nodes is None
        commit = VllmConfigCommit(
            parallel_config=parallel_config,
            transaction_id=transaction_id,
            nodes=candidate,
            previous_nodes=previous_nodes,
            had_nodes_attribute=had_nodes,
            wrote_nodes=wrote_nodes,
        )
        if not wrote_nodes:
            return commit

        marker = {
            "transaction_id": transaction_id,
            "status": "PREPARED",
            "pid": os.getpid(),
            "capability_profile": CAPABILITY_PROFILE,
            "contract_version": CONTRACT_VERSION,
            "visibility_fingerprint": visibility_fingerprint,
            "snapshot_json": snapshot_json,
            "requested_provider": requested_provider,
            "previous_nodes": copy.deepcopy(previous_nodes),
            "previous_nodes_present": had_nodes,
            "written_nodes": list(candidate),
        }
        try:
            setattr(parallel_config, TRANSACTION_MARKER, marker)
            parallel_config.numa_bind_nodes = list(candidate)
            setattr(
                parallel_config,
                TRANSACTION_MARKER,
                {**marker, "status": "APPLIED"},
            )
        except (AttributeError, TypeError, ValueError) as exc:
            commit.rollback()
            raise AffinityDiscoveryError(
                f"cannot commit NUMA nodes to vLLM configuration: {exc}",
                code="CONFIG_COMMIT_FAILED",
            ) from exc
        return commit


def release_invalid_vllm_transaction(parallel_config: Any) -> None:
    """Release a marker and restore its node field only when ownership is proven."""
    marker = getattr(parallel_config, TRANSACTION_MARKER, None)
    if marker is None:
        return
    if isinstance(marker, dict):
        written = marker.get("written_nodes")
        previous_present = marker.get("previous_nodes_present")
        if (
            isinstance(written, list)
            and isinstance(previous_present, bool)
            and getattr(parallel_config, "numa_bind_nodes", None) == written
        ):
            if previous_present:
                parallel_config.numa_bind_nodes = copy.deepcopy(
                    marker.get("previous_nodes")
                )
            elif hasattr(parallel_config, "numa_bind_nodes"):
                delattr(parallel_config, "numa_bind_nodes")
    if hasattr(parallel_config, TRANSACTION_MARKER):
        delattr(parallel_config, TRANSACTION_MARKER)
