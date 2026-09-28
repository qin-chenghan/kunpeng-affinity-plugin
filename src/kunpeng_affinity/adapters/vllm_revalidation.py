"""Validate committed vLLM affinity results before reuse."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from kunpeng_affinity.adapters.vllm_candidate import (
    resolve_vllm_consumed_device,
    resolve_vllm_visibility_fingerprint,
)
from kunpeng_affinity.adapters.vllm_contract import (
    CAPABILITY_PROFILE,
    CONTRACT_VERSION,
    TRANSACTION_MARKER,
)
from kunpeng_affinity.core.errors import AffinityDiscoveryError
from kunpeng_affinity.core.identity import serialized_snapshot_fingerprint


def _invalid(
    message: str,
    *,
    code: str = "INHERITED_RESULT_INVALID",
) -> AffinityDiscoveryError:
    return AffinityDiscoveryError(message, code=code)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _invalid(message)


def _read_marker(
    parallel_config: Any,
) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    marker = getattr(parallel_config, TRANSACTION_MARKER, None)
    if marker is None:
        return {}, [], {}
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
    _require(
        isinstance(marker.get("pid"), int) and not isinstance(marker.get("pid"), bool),
        "vLLM affinity transaction has no valid owner PID",
    )
    _require(
        marker.get("capability_profile") == CAPABILITY_PROFILE,
        "vLLM affinity transaction capability profile is unsupported",
    )
    _require(
        isinstance(marker.get("transaction_id"), str)
        and bool(marker["transaction_id"]),
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
        all(
            isinstance(node, int) and not isinstance(node, bool) and node >= 0
            for node in nodes
        ),
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
    return marker, nodes, snapshot


def _consumed_indices(
    numa_utils: Any,
    parallel_config: Any,
    nodes: list[int],
    *,
    process_kind: str,
    local_rank: int | None,
    dp_local_rank: int | None,
) -> list[int]:
    if process_kind == "EngineCore":
        return list(range(len(nodes)))
    _require(
        process_kind == "worker",
        f"unsupported process kind for inherited affinity: {process_kind!r}",
    )
    _require(
        isinstance(local_rank, int) and local_rank >= 0,
        "worker local_rank is unavailable for inherited affinity validation",
    )
    get_gpu_index = getattr(numa_utils, "_get_gpu_index", None)
    try:
        index = (
            get_gpu_index(parallel_config, local_rank, dp_local_rank)
            if callable(get_gpu_index)
            else local_rank
        )
    except (IndexError, TypeError, ValueError, AttributeError) as exc:
        raise _invalid("vLLM could not compute the inherited worker GPU index") from exc
    _require(
        isinstance(index, int)
        and not isinstance(index, bool)
        and 0 <= index < len(nodes),
        f"inherited worker GPU index {index!r} is outside marker nodes",
    )
    return [index]


def validate_inherited_vllm_transaction(
    parallel_config: Any,
    *,
    numa_utils: Any,
    platform: Any,
    local_rank: int | None,
    dp_local_rank: int | None,
    process_kind: str,
    sysfs_root: Path | str = Path("/sys"),
) -> None:
    """Re-sample a committed result in the process that will consume it."""
    marker, nodes, snapshot = _read_marker(parallel_config)
    if not marker:
        return

    if marker["pid"] == os.getpid():
        current = resolve_vllm_visibility_fingerprint(
            platform,
            requested_provider=marker.get("requested_provider"),
            process_kind=process_kind,
            local_rank=local_rank,
            dp_local_rank=dp_local_rank,
            include_topology=True,
            sysfs_root=sysfs_root,
        )
        if current != marker["visibility_fingerprint"]:
            raise _invalid("vLLM affinity snapshot changed before same-process reuse")
        return

    try:
        allowed_cpus = frozenset(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise _invalid(
            "cannot read child CPU constraints for inherited affinity"
        ) from exc

    mappings = snapshot["mappings"]
    resolutions = snapshot["resolutions"]
    for index in _consumed_indices(
        numa_utils,
        parallel_config,
        nodes,
        process_kind=process_kind,
        local_rank=local_rank,
        dp_local_rank=dp_local_rank,
    ):
        expected_mapping = mappings[index]
        expected_resolution = resolutions[index]
        _require(
            isinstance(expected_mapping, dict)
            and isinstance(expected_resolution, dict),
            "vLLM affinity transaction contains an invalid device snapshot",
        )
        current = resolve_vllm_consumed_device(
            platform,
            index,
            requested_provider=marker.get("requested_provider"),
            process_kind=process_kind,
            local_rank=local_rank,
            dp_local_rank=dp_local_rank,
            allowed_cpus=allowed_cpus,
            sysfs_root=sysfs_root,
        )
        physical_id = (
            None
            if current.mapping.physical_device_id is None
            else {
                "type": type(current.mapping.physical_device_id).__name__,
                "value": str(current.mapping.physical_device_id),
            }
        )
        _require(
            expected_mapping.get("logical_device_id") == index
            and expected_mapping.get("pci_bdf") == current.mapping.pci_bdf
            and expected_mapping.get("physical_device_id") == physical_id
            and expected_mapping.get("instance_id") == current.mapping.instance_id
            and expected_resolution.get("numa_node") == current.affinity.numa_node
            and nodes[index] == current.affinity.numa_node,
            "vLLM inherited device identity or NUMA result changed",
        )
