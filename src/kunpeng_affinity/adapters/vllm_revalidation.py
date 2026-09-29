"""Validate committed vLLM affinity results before reuse."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kunpeng_affinity.adapters.vllm_candidate import (
    resolve_vllm_consumed_device,
    resolve_vllm_visibility_fingerprint,
)
from kunpeng_affinity.adapters.vllm_marker import read_committed_vllm_marker
from kunpeng_affinity.core.errors import AffinityDiscoveryError


def _invalid(
    message: str,
    *,
    code: str = "INHERITED_RESULT_INVALID",
) -> AffinityDiscoveryError:
    return AffinityDiscoveryError(message, code=code)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _invalid(message)


def _consumed_indices(
    numa_utils: Any,
    parallel_config: Any,
    nodes: Sequence[int],
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
    marker = read_committed_vllm_marker(parallel_config)
    if marker is None:
        return

    if marker.owner_pid == os.getpid():
        current = resolve_vllm_visibility_fingerprint(
            platform,
            requested_provider=marker.requested_provider,
            process_kind=process_kind,
            local_rank=local_rank,
            dp_local_rank=dp_local_rank,
            include_topology=True,
            sysfs_root=sysfs_root,
        )
        if current != marker.visibility_fingerprint:
            raise _invalid("vLLM affinity snapshot changed before same-process reuse")
        return

    try:
        allowed_cpus = frozenset(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise _invalid(
            "cannot read child CPU constraints for inherited affinity"
        ) from exc

    for index in _consumed_indices(
        numa_utils,
        parallel_config,
        marker.nodes,
        process_kind=process_kind,
        local_rank=local_rank,
        dp_local_rank=dp_local_rank,
    ):
        expected_mapping = marker.mappings[index]
        expected_resolution = marker.resolutions[index]
        _require(
            isinstance(expected_mapping, dict)
            and isinstance(expected_resolution, dict),
            "vLLM affinity transaction contains an invalid device snapshot",
        )
        current = resolve_vllm_consumed_device(
            platform,
            index,
            requested_provider=marker.requested_provider,
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
            and marker.nodes[index] == current.affinity.numa_node,
            "vLLM inherited device identity or NUMA result changed",
        )
