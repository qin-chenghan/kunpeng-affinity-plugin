"""Validate the committed vLLM affinity marker contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from kunpeng_affinity.adapters.vllm_contract import (
    CAPABILITY_PROFILE,
    CONTRACT_VERSION,
    TRANSACTION_MARKER,
)
from kunpeng_affinity.core.errors import AffinityDiscoveryError
from kunpeng_affinity.core.identity import serialized_snapshot_fingerprint


@dataclass(frozen=True)
class CommittedVllmMarker:
    """Validated fields needed to reuse a committed affinity transaction."""

    owner_pid: int
    nodes: tuple[int, ...]
    mappings: tuple[Any, ...]
    resolutions: tuple[Any, ...]
    visibility_fingerprint: str
    requested_provider: str | None


def _invalid(
    message: str,
    *,
    code: str = "INHERITED_RESULT_INVALID",
) -> AffinityDiscoveryError:
    return AffinityDiscoveryError(message, code=code)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _invalid(message)


def read_committed_vllm_marker(
    parallel_config: Any,
) -> CommittedVllmMarker | None:
    """Validate and decode the plugin marker attached to a vLLM config."""
    marker = getattr(parallel_config, TRANSACTION_MARKER, None)
    if marker is None:
        return None
    _require(
        isinstance(marker, dict),
        "vLLM affinity transaction marker is not a mapping",
    )
    if marker.get("status") in {"PREPARED", "APPLIED"}:
        raise _invalid(
            "vLLM affinity transaction is still being committed",
            code="TRANSACTION_IN_PROGRESS",
        )
    _require(
        marker.get("status") == "COMMITTED",
        "vLLM affinity transaction is not committed",
    )
    _require(
        marker.get("contract_version") == CONTRACT_VERSION,
        "vLLM affinity transaction contract version is unsupported",
    )
    owner_pid = marker.get("pid")
    _require(
        isinstance(owner_pid, int) and not isinstance(owner_pid, bool),
        "vLLM affinity transaction has no valid owner PID",
    )
    _require(
        marker.get("capability_profile") == CAPABILITY_PROFILE,
        "vLLM affinity transaction capability profile is unsupported",
    )
    _require(
        isinstance(marker.get("transaction_id"), str) and bool(marker["transaction_id"]),
        "vLLM affinity transaction has no transaction ID",
    )

    fingerprint = marker.get("visibility_fingerprint")
    snapshot_json = marker.get("snapshot_json")
    _require(
        isinstance(fingerprint, str) and bool(fingerprint),
        "vLLM affinity transaction has no snapshot fingerprint",
    )
    _require(
        isinstance(snapshot_json, str) and bool(snapshot_json),
        "vLLM affinity transaction has no serialized snapshot",
    )
    _require(
        serialized_snapshot_fingerprint(snapshot_json) == fingerprint,
        "vLLM affinity transaction snapshot digest does not match",
    )
    try:
        snapshot = json.loads(snapshot_json)
    except (TypeError, ValueError) as exc:
        raise _invalid("vLLM affinity transaction snapshot is not valid JSON") from exc
    _require(
        isinstance(snapshot, dict),
        "vLLM affinity transaction snapshot is not a mapping",
    )

    mappings = snapshot.get("mappings")
    resolutions = snapshot.get("resolutions")
    metadata = snapshot.get("metadata")
    _require(
        isinstance(mappings, list)
        and isinstance(resolutions, list)
        and isinstance(metadata, dict)
        and metadata.get("adapter") == CAPABILITY_PROFILE
        and len(mappings) == len(resolutions),
        "vLLM affinity transaction snapshot schema is invalid",
    )

    nodes = marker.get("written_nodes")
    _require(
        isinstance(nodes, list) and bool(nodes),
        "vLLM affinity transaction has no written NUMA nodes",
    )
    _require(
        all(isinstance(node, int) and not isinstance(node, bool) and node >= 0 for node in nodes),
        "vLLM affinity transaction has invalid written NUMA nodes",
    )
    _require(
        len(nodes) == len(resolutions),
        "vLLM affinity transaction node count does not match its snapshot",
    )
    _require(
        getattr(parallel_config, "numa_bind_nodes", None) == nodes,
        "vLLM affinity transaction field numa_bind_nodes was modified externally",
    )
    requested_provider = marker.get("requested_provider")
    _require(
        requested_provider is None or isinstance(requested_provider, str),
        "vLLM affinity transaction requested provider is invalid",
    )
    return CommittedVllmMarker(
        owner_pid=owner_pid,
        nodes=tuple(nodes),
        mappings=tuple(mappings),
        resolutions=tuple(resolutions),
        visibility_fingerprint=fingerprint,
        requested_provider=requested_provider,
    )
