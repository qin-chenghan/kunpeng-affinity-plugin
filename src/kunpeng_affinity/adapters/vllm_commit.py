"""Atomic mutation of vLLM NUMA fields owned by this plugin."""

from __future__ import annotations

import copy
import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kunpeng_affinity.core.errors import AffinityDiscoveryError
from kunpeng_affinity.core.identity import serialized_snapshot_fingerprint
from kunpeng_affinity.topology.cpulist import CpuListError, parse_cpulist


_CONFIG_COMMIT_LOCK = threading.RLock()
_MARKER_FIELD = "_kunpeng_affinity_transaction"
_CONTRACT_VERSION = "1"
_CAPABILITY_PROFILE = "vllm.configure_subprocess.v1"


@dataclass
class VllmConfigCommit:
    """A transaction that can restore only values written by this plugin."""

    parallel_config: Any
    transaction_id: str
    nodes: list[int]
    cpus: Any
    previous_nodes: Any
    previous_cpus: Any
    had_nodes_attribute: bool
    had_cpus_attribute: bool
    wrote_nodes: bool
    wrote_cpus: bool

    @property
    def marker(self) -> dict[str, Any]:
        value = getattr(self.parallel_config, _MARKER_FIELD, None)
        return value if isinstance(value, dict) else {}

    def mark_committed(self) -> None:
        """Advance APPLIED to COMMITTED after the original context enters."""
        if not (self.wrote_nodes or self.wrote_cpus):
            return
        with _CONFIG_COMMIT_LOCK:
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
            updated = dict(marker)
            updated["status"] = "COMMITTED"
            setattr(self.parallel_config, _MARKER_FIELD, updated)

    def rollback(self) -> None:
        """Restore only fields that still contain this transaction's values."""
        if not (self.wrote_nodes or self.wrote_cpus):
            return
        with _CONFIG_COMMIT_LOCK:
            if (
                self.wrote_nodes
                and getattr(self.parallel_config, "numa_bind_nodes", None) == self.nodes
            ):
                if self.had_nodes_attribute:
                    self.parallel_config.numa_bind_nodes = self.previous_nodes
                else:
                    delattr(self.parallel_config, "numa_bind_nodes")
            if (
                self.wrote_cpus
                and getattr(self.parallel_config, "numa_bind_cpus", None) == self.cpus
            ):
                if self.had_cpus_attribute:
                    self.parallel_config.numa_bind_cpus = self.previous_cpus
                else:
                    delattr(self.parallel_config, "numa_bind_cpus")
            marker = self.marker
            if marker.get("transaction_id") == self.transaction_id:
                delattr(self.parallel_config, _MARKER_FIELD)
            self.wrote_nodes = False
            self.wrote_cpus = False


# Keep the design name available without making the vLLM-specific API opaque.
ConfigTransaction = VllmConfigCommit


def commit_vllm_nodes(
    parallel_config: Any,
    nodes: list[int],
    *,
    visibility_fingerprint: str | None = None,
    snapshot_json: str | None = None,
    requested_provider: str | None = None,
) -> VllmConfigCommit:
    """Commit a complete node list and record a serializable ownership marker."""
    return commit_vllm_affinity(
        parallel_config,
        nodes,
        visibility_fingerprint=visibility_fingerprint,
        snapshot_json=snapshot_json,
        requested_provider=requested_provider,
    )


def commit_vllm_affinity(
    parallel_config: Any,
    nodes: list[int],
    *,
    cpus: Any = None,
    visibility_fingerprint: str | None = None,
    snapshot_json: str | None = None,
    requested_provider: str | None = None,
    capability_profile: str = _CAPABILITY_PROFILE,
) -> VllmConfigCommit:
    """Apply plugin-owned fields in one locked compare-and-write operation."""
    candidate_nodes = list(nodes)
    candidate_cpus = list(cpus) if cpus is not None else None
    with _CONFIG_COMMIT_LOCK:
        had_nodes = hasattr(parallel_config, "numa_bind_nodes")
        had_cpus = hasattr(parallel_config, "numa_bind_cpus")
        existing_nodes = getattr(parallel_config, "numa_bind_nodes", None)
        existing_cpus = getattr(parallel_config, "numa_bind_cpus", None)
        if existing_nodes is not None and existing_nodes != candidate_nodes:
            raise AffinityDiscoveryError(
                f"NUMA nodes changed concurrently from resolved {candidate_nodes} "
                f"to {existing_nodes}",
                code="CONCURRENT_CONFIG_CONFLICT",
            )
        if (
            candidate_cpus is not None
            and existing_cpus is not None
            and existing_cpus != candidate_cpus
        ):
            raise AffinityDiscoveryError(
                "NUMA CPUs changed concurrently before plugin commit",
                code="CONCURRENT_CONFIG_CONFLICT",
            )

        transaction_id = uuid.uuid4().hex
        wrote_nodes = existing_nodes is None
        wrote_cpus = candidate_cpus is not None and existing_cpus is None
        commit = VllmConfigCommit(
            parallel_config=parallel_config,
            transaction_id=transaction_id,
            nodes=candidate_nodes,
            cpus=candidate_cpus,
            previous_nodes=existing_nodes,
            previous_cpus=existing_cpus,
            had_nodes_attribute=had_nodes,
            had_cpus_attribute=had_cpus,
            wrote_nodes=wrote_nodes,
            wrote_cpus=wrote_cpus,
        )
        if not (wrote_nodes or wrote_cpus):
            return commit
        marker = {
            "transaction_id": transaction_id,
            "status": "APPLIED",
            "pid": os.getpid(),
            "capability_profile": capability_profile,
            "contract_version": _CONTRACT_VERSION,
            "visibility_fingerprint": visibility_fingerprint,
            "snapshot_json": snapshot_json,
            "requested_provider": requested_provider,
            "previous": {
                "numa_bind_nodes": copy.deepcopy(existing_nodes),
                "numa_bind_cpus": copy.deepcopy(existing_cpus),
            },
            "previous_present": {
                "numa_bind_nodes": had_nodes,
                "numa_bind_cpus": had_cpus,
            },
            "written": {
                "numa_bind_nodes": list(candidate_nodes),
                "numa_bind_cpus": copy.deepcopy(candidate_cpus),
            },
        }
        try:
            prepared_marker = dict(marker)
            prepared_marker["status"] = "PREPARED"
            setattr(parallel_config, _MARKER_FIELD, prepared_marker)
            if wrote_nodes:
                parallel_config.numa_bind_nodes = list(candidate_nodes)
            if wrote_cpus:
                parallel_config.numa_bind_cpus = copy.deepcopy(candidate_cpus)
            applied_marker = dict(prepared_marker)
            applied_marker["status"] = "APPLIED"
            setattr(parallel_config, _MARKER_FIELD, applied_marker)
        except (AttributeError, TypeError, ValueError) as exc:
            commit.rollback()
            raise AffinityDiscoveryError(
                f"cannot commit NUMA affinity to vLLM configuration: {exc}",
                code="CONFIG_COMMIT_FAILED",
            ) from exc
        return commit


def release_invalid_vllm_transaction(parallel_config: Any) -> None:
    """Release plugin-owned fields without overwriting external changes."""
    marker = getattr(parallel_config, _MARKER_FIELD, None)
    if marker is None:
        return
    if not isinstance(marker, dict):
        if hasattr(parallel_config, _MARKER_FIELD):
            delattr(parallel_config, _MARKER_FIELD)
        return
    previous = marker.get("previous", {})
    previous_present = marker.get("previous_present", {})
    written = marker.get("written", {})
    fields = ("numa_bind_nodes", "numa_bind_cpus")
    if (
        not isinstance(previous, dict)
        or not isinstance(previous_present, dict)
        or not isinstance(written, dict)
        or any(field not in previous for field in fields)
        or any(field not in written for field in fields)
        or any(not isinstance(previous_present.get(field), bool) for field in fields)
    ):
        if hasattr(parallel_config, _MARKER_FIELD):
            delattr(parallel_config, _MARKER_FIELD)
        return
    for field in fields:
        if not hasattr(parallel_config, field):
            continue
        if getattr(parallel_config, field) != written.get(field):
            continue
        if previous_present.get(field) is True:
            setattr(parallel_config, field, previous.get(field))
        else:
            delattr(parallel_config, field)
    if hasattr(parallel_config, _MARKER_FIELD):
        delattr(parallel_config, _MARKER_FIELD)


def _parse_cpu_value(value: Any) -> frozenset[int]:
    if isinstance(value, str):
        try:
            return frozenset(parse_cpulist(value))
        except CpuListError as exc:
            raise AffinityDiscoveryError(
                f"invalid inherited CPU list {value!r}",
                code="INHERITED_RESULT_INVALID",
            ) from exc
    if isinstance(value, (list, tuple, set, frozenset)):
        if any(
            not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0
            for cpu in value
        ):
            raise AffinityDiscoveryError(
                f"invalid inherited CPU list {value!r}",
                code="INHERITED_RESULT_INVALID",
            )
        return frozenset(value)
    raise AffinityDiscoveryError(
        f"invalid inherited CPU list {value!r}",
        code="INHERITED_RESULT_INVALID",
    )


def _validate_inherited_nodes(nodes: list[Any]) -> None:
    try:
        online = parse_cpulist(
            (Path("/sys") / "devices/system/cpu/online").read_text(encoding="ascii")
        )
        allowed = frozenset(os.sched_getaffinity(0))
    except (AttributeError, OSError, CpuListError) as exc:
        raise AffinityDiscoveryError(
            "cannot validate inherited NUMA CPU constraints",
            code="INHERITED_RESULT_INVALID",
        ) from exc
    for node in nodes:
        if not isinstance(node, int) or isinstance(node, bool) or node < 0:
            raise AffinityDiscoveryError(
                f"invalid inherited NUMA node {node!r}",
                code="INHERITED_RESULT_INVALID",
            )
        try:
            node_cpus = parse_cpulist(
                (Path("/sys") / f"devices/system/node/node{node}/cpulist").read_text(
                    encoding="ascii"
                )
            )
        except (OSError, CpuListError) as exc:
            raise AffinityDiscoveryError(
                f"cannot validate inherited NUMA node {node}",
                code="INHERITED_RESULT_INVALID",
            ) from exc
        if not node_cpus & online & allowed:
            raise AffinityDiscoveryError(
                f"inherited NUMA node {node} has no usable CPUs",
                code="INHERITED_RESULT_INVALID",
            )


def validate_inherited_vllm_transaction(
    parallel_config: Any,
    *,
    numa_utils: Any,
    platform: Any,
    local_rank: int | None,
    dp_local_rank: int | None,
    process_kind: str,
) -> None:
    """Validate a committed marker before the current process consumes it."""
    marker = getattr(parallel_config, _MARKER_FIELD, None)
    if marker is None:
        return
    if not isinstance(marker, dict):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction marker is not a mapping",
            code="INHERITED_RESULT_INVALID",
        )
    if marker.get("status") != "COMMITTED":
        raise AffinityDiscoveryError(
            "vLLM affinity transaction is not committed",
            code="INHERITED_RESULT_INVALID",
        )
    if marker.get("contract_version") != _CONTRACT_VERSION:
        raise AffinityDiscoveryError(
            "vLLM affinity transaction contract version is unsupported",
            code="INHERITED_RESULT_INVALID",
        )
    if not isinstance(marker.get("pid"), int) or isinstance(marker.get("pid"), bool):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction has no valid owner PID",
            code="INHERITED_RESULT_INVALID",
        )
    if marker.get("capability_profile") != _CAPABILITY_PROFILE:
        raise AffinityDiscoveryError(
            "vLLM affinity transaction capability profile is unsupported",
            code="INHERITED_RESULT_INVALID",
        )
    if not isinstance(marker.get("transaction_id"), str) or not marker.get(
        "transaction_id"
    ):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction has no transaction ID",
            code="INHERITED_RESULT_INVALID",
        )
    fingerprint = marker.get("visibility_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise AffinityDiscoveryError(
            "vLLM affinity transaction has no snapshot fingerprint",
            code="INHERITED_RESULT_INVALID",
        )
    snapshot_json = marker.get("snapshot_json")
    if not isinstance(snapshot_json, str) or not snapshot_json:
        raise AffinityDiscoveryError(
            "vLLM affinity transaction has no serialized snapshot",
            code="INHERITED_RESULT_INVALID",
        )
    if serialized_snapshot_fingerprint(snapshot_json) != fingerprint:
        raise AffinityDiscoveryError(
            "vLLM affinity transaction snapshot digest does not match",
            code="INHERITED_RESULT_INVALID",
        )
    try:
        snapshot = json.loads(snapshot_json)
    except (TypeError, ValueError) as exc:
        raise AffinityDiscoveryError(
            "vLLM affinity transaction snapshot is not valid JSON",
            code="INHERITED_RESULT_INVALID",
        ) from exc
    if not isinstance(snapshot, dict):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction snapshot is not a mapping",
            code="INHERITED_RESULT_INVALID",
        )
    mappings = snapshot.get("mappings")
    resolutions = snapshot.get("resolutions")
    metadata = snapshot.get("metadata")
    if (
        not isinstance(mappings, list)
        or not isinstance(resolutions, list)
        or not isinstance(metadata, dict)
        or metadata.get("adapter") != _CAPABILITY_PROFILE
        or len(mappings) != len(resolutions)
    ):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction snapshot schema is invalid",
            code="INHERITED_RESULT_INVALID",
        )
    written = marker.get("written")
    previous = marker.get("previous")
    previous_present = marker.get("previous_present")
    if not isinstance(written, dict):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction has no written-field snapshot",
            code="INHERITED_RESULT_INVALID",
        )
    if not isinstance(previous, dict):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction has no previous-field snapshot",
            code="INHERITED_RESULT_INVALID",
        )
    if not isinstance(previous_present, dict):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction has no field-presence snapshot",
            code="INHERITED_RESULT_INVALID",
        )
    fields = ("numa_bind_nodes", "numa_bind_cpus")
    if any(field not in written for field in fields):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction has an incomplete written-field snapshot",
            code="INHERITED_RESULT_INVALID",
        )
    if any(field not in previous for field in fields):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction has an incomplete previous-field snapshot",
            code="INHERITED_RESULT_INVALID",
        )
    if any(not isinstance(previous_present.get(field), bool) for field in fields):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction has an invalid field-presence snapshot",
            code="INHERITED_RESULT_INVALID",
        )
    nodes = written.get("numa_bind_nodes")
    if not isinstance(nodes, list) or not nodes:
        raise AffinityDiscoveryError(
            "vLLM affinity transaction has no written NUMA nodes",
            code="INHERITED_RESULT_INVALID",
        )
    if any(
        not isinstance(node, int) or isinstance(node, bool) or node < 0
        for node in nodes
    ):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction has invalid written NUMA nodes",
            code="INHERITED_RESULT_INVALID",
        )
    if len(nodes) != len(resolutions):
        raise AffinityDiscoveryError(
            "vLLM affinity transaction node count does not match its snapshot",
            code="INHERITED_RESULT_INVALID",
        )
    for field in fields:
        expected = written.get(field)
        if expected is not None and getattr(parallel_config, field, None) != expected:
            raise AffinityDiscoveryError(
                f"vLLM affinity transaction field {field} was modified externally",
                code="INHERITED_RESULT_INVALID",
            )

    from kunpeng_affinity.adapters.vllm_generic import (
        resolve_vllm_consumed_device,
        resolve_vllm_visibility_fingerprint,
    )

    if marker["pid"] == os.getpid():
        current_fingerprint = resolve_vllm_visibility_fingerprint(
            platform,
            requested_provider=marker.get("requested_provider"),
            process_kind=process_kind,
            local_rank=local_rank,
            dp_local_rank=dp_local_rank,
            include_topology=True,
        )
        if current_fingerprint != fingerprint:
            raise AffinityDiscoveryError(
                "vLLM affinity snapshot changed before same-process reuse",
                code="INHERITED_RESULT_INVALID",
            )
        return

    if process_kind == "worker":
        if not isinstance(local_rank, int) or local_rank < 0:
            raise AffinityDiscoveryError(
                "worker local_rank is unavailable for inherited affinity validation",
                code="INHERITED_RESULT_INVALID",
            )
        get_gpu_index = getattr(numa_utils, "_get_gpu_index", None)
        try:
            gpu_index = (
                get_gpu_index(parallel_config, local_rank, dp_local_rank)
                if callable(get_gpu_index)
                else local_rank
            )
        except (IndexError, TypeError, ValueError, AttributeError) as exc:
            raise AffinityDiscoveryError(
                "vLLM could not compute the inherited worker GPU index",
                code="INHERITED_RESULT_INVALID",
            ) from exc
        if not isinstance(gpu_index, int) or gpu_index < 0 or gpu_index >= len(nodes):
            raise AffinityDiscoveryError(
                f"inherited worker GPU index {gpu_index!r} is outside marker nodes",
                code="INHERITED_RESULT_INVALID",
            )
        selected_indices = [gpu_index]
    elif process_kind == "EngineCore":
        selected_indices = list(range(len(nodes)))
    else:
        raise AffinityDiscoveryError(
            f"unsupported process kind for inherited affinity: {process_kind!r}",
            code="INHERITED_RESULT_INVALID",
        )

    try:
        current_allowed = frozenset(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise AffinityDiscoveryError(
            "cannot read child CPU constraints for inherited affinity",
            code="INHERITED_RESULT_INVALID",
        ) from exc
    for device_index in selected_indices:
        expected_mapping = mappings[device_index]
        expected_resolution = resolutions[device_index]
        if not isinstance(expected_mapping, dict) or not isinstance(
            expected_resolution, dict
        ):
            raise AffinityDiscoveryError(
                "vLLM affinity transaction contains an invalid device snapshot",
                code="INHERITED_RESULT_INVALID",
            )
        current = resolve_vllm_consumed_device(
            platform,
            device_index,
            requested_provider=marker.get("requested_provider"),
            process_kind=process_kind,
            local_rank=local_rank,
            dp_local_rank=dp_local_rank,
            allowed_cpus=current_allowed,
        )
        current_physical_id = (
            None
            if current.mapping.physical_device_id is None
            else {
                "type": type(current.mapping.physical_device_id).__name__,
                "value": str(current.mapping.physical_device_id),
            }
        )
        if (
            expected_mapping.get("logical_device_id") != device_index
            or expected_mapping.get("pci_bdf") != current.mapping.pci_bdf
            or expected_mapping.get("physical_device_id") != current_physical_id
            or expected_mapping.get("instance_id") != current.mapping.instance_id
            or expected_resolution.get("numa_node") != current.affinity.numa_node
            or nodes[device_index] != current.affinity.numa_node
        ):
            raise AffinityDiscoveryError(
                "vLLM inherited device identity or NUMA result changed",
                code="INHERITED_RESULT_INVALID",
            )

    _validate_inherited_nodes([nodes[index] for index in selected_indices])
    cpus = written.get("numa_bind_cpus")
    if cpus is not None:
        requested = _parse_cpu_value(cpus)
        if not requested:
            raise AffinityDiscoveryError(
                "inherited exact CPU list is empty",
                code="INHERITED_RESULT_INVALID",
            )
        try:
            allowed = frozenset(os.sched_getaffinity(0))
            online = parse_cpulist(
                (Path("/sys") / "devices/system/cpu/online").read_text(
                    encoding="ascii"
                )
            )
        except (AttributeError, OSError, CpuListError) as exc:
            raise AffinityDiscoveryError(
                "cannot validate inherited exact CPU list",
                code="INHERITED_RESULT_INVALID",
            ) from exc
        if not requested <= allowed & online:
            raise AffinityDiscoveryError(
                "inherited exact CPU list exceeds current CPU constraints",
                code="INHERITED_RESULT_INVALID",
            )
