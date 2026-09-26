"""Read-only Linux sysfs PCIe/NUMA topology analyzer."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from kunpeng_affinity.topology.cpulist import CpuListError, parse_cpulist
from kunpeng_affinity.topology.models import (
    AffinityResult,
    PciPathNode,
    ResultStatus,
)

_BDF_RE = re.compile(
    r"^(?:(?P<domain>[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):)?"
    r"(?P<bus>[0-9a-fA-F]{2}):(?P<slot>[0-9a-fA-F]{2})\."
    r"(?P<function>[0-7])$"
)
_ROOT_BUS_RE = re.compile(r"^pci[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}$")


class TopologyError(RuntimeError):
    """A topology fact is invalid or contradictory."""


@dataclass(frozen=True)
class _PathData:
    nodes: tuple[PciPathNode, ...]
    root_bus_path: str
    real_endpoint: Path


def normalize_bdf(value: str) -> str:
    """Normalize a PCI address to Linux ``dddd:bb:ss.f`` form."""
    match = _BDF_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid PCI BDF {value!r}")

    raw_domain = match.group("domain") or "0000"
    domain_number = int(raw_domain, 16)
    if domain_number > 0xFFFF:
        raise ValueError(
            f"PCI domain {raw_domain!r} cannot be represented by Linux "
            "dddd:bb:ss.f sysfs names"
        )
    return (
        f"{domain_number:04x}:{match.group('bus').lower()}:"
        f"{match.group('slot').lower()}.{match.group('function')}"
    )


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise TopologyError(f"cannot read {path}: {exc}") from exc


def _read_numa_node(path: Path) -> int | None:
    value = _read_optional(path)
    if value is None or value == "-1":
        return None
    try:
        node = int(value)
    except ValueError as exc:
        raise TopologyError(f"invalid NUMA node {value!r} in {path}") from exc
    if node < 0:
        raise TopologyError(f"invalid NUMA node {value!r} in {path}")
    return node


def _read_cpulist(path: Path, *, required: bool) -> frozenset[int] | None:
    value = _read_optional(path)
    if value is None:
        if required:
            raise TopologyError(f"required CPU list is missing: {path}")
        return None
    try:
        return parse_cpulist(value)
    except CpuListError as exc:
        raise TopologyError(f"invalid CPU list in {path}: {exc}") from exc


def _role(index: int, pci_class: str | None) -> str:
    if index == 0:
        return "endpoint"
    if pci_class is not None and pci_class.lower().startswith("0x0604"):
        return "pci-bridge"
    return "pci-function"


def _collect_path(sysfs_root: Path, bdf: str) -> _PathData:
    # Follow the real sysfs links from the endpoint through every PCI ancestor.
    # The traversal naturally covers direct connections and any switch depth.
    device_link = sysfs_root / "bus/pci/devices" / bdf
    try:
        endpoint = device_link.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise TopologyError(f"PCI device {bdf} is not resolvable at {device_link}") from exc

    devices_root = (sysfs_root / "devices").resolve(strict=True)
    try:
        endpoint.relative_to(devices_root)
    except ValueError as exc:
        raise TopologyError(
            f"PCI device {bdf} resolves outside {devices_root}: {endpoint}"
        ) from exc
    try:
        endpoint_bdf = normalize_bdf(endpoint.name)
    except ValueError as exc:
        raise TopologyError(f"PCI link target is not a function path: {endpoint}") from exc
    if endpoint_bdf != bdf:
        raise TopologyError(
            f"PCI link {device_link} resolves to mismatched function {endpoint_bdf}"
        )

    records: list[tuple[str, Path, str | None, int | None]] = []
    current = endpoint
    while True:
        try:
            current_bdf = normalize_bdf(current.name)
        except ValueError as exc:
            raise TopologyError(
                f"PCI parent chain is incomplete before root bus: {current}"
            ) from exc
        records.append(
            (
                current_bdf,
                current,
                _read_optional(current / "class"),
                _read_numa_node(current / "numa_node"),
            )
        )
        parent = current.parent
        if _ROOT_BUS_RE.fullmatch(parent.name):
            root_bus = parent
            break
        try:
            parent.relative_to(devices_root)
        except ValueError as exc:
            raise TopologyError(
                f"PCI parent chain leaves sysfs devices tree at {parent}"
            ) from exc
        current = parent

    nodes = tuple(
        PciPathNode(
            bdf=record_bdf,
            sysfs_path=str(path),
            pci_class=pci_class,
            role=_role(index, pci_class),
            numa_node=numa_node,
        )
        for index, (record_bdf, path, pci_class, numa_node) in enumerate(records)
    )
    return _PathData(nodes=nodes, root_bus_path=str(root_bus), real_endpoint=endpoint)


def _node_inventory(sysfs_root: Path) -> dict[int, frozenset[int]]:
    # Build the NUMA-node CPU inventory used by both evidence resolution and
    # the final process-allowed CPU intersection.
    node_root = sysfs_root / "devices/system/node"
    inventory: dict[int, frozenset[int]] = {}
    try:
        paths = tuple(node_root.glob("node[0-9]*"))
    except OSError as exc:
        raise TopologyError(f"cannot enumerate NUMA nodes at {node_root}: {exc}") from exc
    for path in paths:
        suffix = path.name.removeprefix("node")
        if not suffix.isdigit():
            continue
        cpus = _read_cpulist(path / "cpulist", required=True)
        assert cpus is not None
        inventory[int(suffix)] = cpus
    return inventory


def _resolve_numa(
    path_data: _PathData,
    inventory: dict[int, frozenset[int]],
) -> tuple[int | None, str | None, tuple[str, ...]]:
    # Combine endpoint, PCI ancestor, and local_cpulist evidence. Conflicting
    # evidence is rejected instead of being resolved by an arbitrary preference.
    candidates: list[tuple[int, str]] = []
    diagnostics: list[str] = []
    for index, path_node in enumerate(path_data.nodes):
        if path_node.numa_node is None:
            continue
        if path_node.numa_node not in inventory:
            raise TopologyError(
                f"{path_node.bdf} references NUMA node {path_node.numa_node}, "
                "but that node is missing or has no CPUs"
            )
        source = "endpoint" if index == 0 else f"ancestor:{path_node.bdf}"
        candidates.append((path_node.numa_node, source))

    local_cpus = _read_cpulist(
        path_data.real_endpoint / "local_cpulist", required=False
    )
    if local_cpus:
        matching_nodes = [
            node for node, node_cpus in inventory.items() if local_cpus <= node_cpus
        ]
        if len(matching_nodes) == 1:
            candidates.append((matching_nodes[0], "endpoint:local_cpulist"))
        elif not matching_nodes:
            raise TopologyError(
                "endpoint local_cpulist is not contained in any single NUMA node"
            )
        else:
            diagnostics.append(
                "endpoint local_cpulist matches multiple NUMA nodes and is not "
                "used as unique evidence"
            )

    if not candidates:
        return None, None, tuple(diagnostics)
    candidate_nodes = {node for node, _ in candidates}
    if len(candidate_nodes) != 1:
        evidence = ", ".join(f"{source}={node}" for node, source in candidates)
        raise TopologyError(f"conflicting NUMA evidence: {evidence}")

    selected = candidates[0][0]
    preferred_sources = [source for node, source in candidates if node == selected]
    source = next(
        (item for item in preferred_sources if item == "endpoint"),
        next(
            (item for item in preferred_sources if item.startswith("ancestor:")),
            preferred_sources[0],
        ),
    )
    return selected, source, tuple(diagnostics)


def _current_allowed_cpus() -> frozenset[int]:
    try:
        return frozenset(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise TopologyError("current process CPU affinity is unavailable") from exc


def _result(
    *,
    input_bdf: str,
    normalized_bdf: str | None,
    mapping_source: str,
    status: ResultStatus,
    failure_code: str | None = None,
    path_data: _PathData | None = None,
    diagnostics: tuple[str, ...] = (),
    numa_node: int | None = None,
    numa_source: str | None = None,
    node_cpus: frozenset[int] = frozenset(),
    online_cpus: frozenset[int] = frozenset(),
    allowed_cpus: frozenset[int] = frozenset(),
    target_cpus: frozenset[int] = frozenset(),
) -> AffinityResult:
    return AffinityResult(
        input_bdf=input_bdf,
        normalized_bdf=normalized_bdf,
        device_mapping_source=mapping_source,
        status=status,
        failure_code=failure_code,
        pci_path=path_data.nodes if path_data else (),
        root_bus_path=path_data.root_bus_path if path_data else None,
        numa_node=numa_node,
        numa_source=numa_source,
        node_cpus=node_cpus,
        online_cpus=online_cpus,
        allowed_cpus=allowed_cpus,
        target_cpus=target_cpus,
        cpu_source=(
            "node cpulist intersect online CPUs intersect current affinity"
            if target_cpus
            else None
        ),
        diagnostics=diagnostics,
    )


def analyze_bdf(
    bdf: str,
    *,
    sysfs_root: Path | str = Path("/sys"),
    allowed_cpus: frozenset[int] | set[int] | None = None,
    mapping_source: str = "explicit-bdf",
) -> AffinityResult:
    """Analyze one trusted PCI BDF without executing a binding operation."""
    root = Path(sysfs_root)
    normalized: str | None = None
    path_data: _PathData | None = None
    diagnostics: tuple[str, ...] = ()
    numa_node: int | None = None
    numa_source: str | None = None
    node_cpus: frozenset[int] = frozenset()
    online_cpus: frozenset[int] = frozenset()
    effective_allowed: frozenset[int] = frozenset()
    failure_code = "TOPOLOGY_INVALID"
    try:
        # Resolve the PCI path and prove a unique NUMA node before reading CPU sets.
        failure_code = "BDF_INVALID"
        normalized = normalize_bdf(bdf)
        failure_code = "INCOMPLETE_PCI_PATH"
        path_data = _collect_path(root, normalized)
        failure_code = "NUMA_INVENTORY_INVALID"
        inventory = _node_inventory(root)
        failure_code = "NUMA_EVIDENCE_INVALID"
        numa_node, numa_source, diagnostics = _resolve_numa(path_data, inventory)
        if numa_node is None:
            return _result(
                input_bdf=bdf,
                normalized_bdf=normalized,
                mapping_source=mapping_source,
                status=ResultStatus.FAILED,
                failure_code="NUMA_UNKNOWN",
                path_data=path_data,
                diagnostics=diagnostics
                + ("NUMA node could not be proven from PCI sysfs",),
            )

        # Restrict the suggested CPUs to CPUs that are online and allowed here.
        failure_code = "CPUSET_INVALID"
        node_cpus = inventory[numa_node]
        if not node_cpus:
            raise TopologyError(f"NUMA node {numa_node} has an empty CPU list")
        online_cpus = _read_cpulist(
            root / "devices/system/cpu/online", required=True
        )
        assert online_cpus is not None
        if not online_cpus:
            raise TopologyError("online CPU list is empty")
        effective_allowed = (
            frozenset(allowed_cpus)
            if allowed_cpus is not None
            else _current_allowed_cpus()
        )
        if not effective_allowed:
            raise TopologyError("current process CPU affinity is empty")
        target = node_cpus & online_cpus & effective_allowed
        if not target:
            raise TopologyError(
                f"NUMA node {numa_node} has no CPUs in the online and allowed sets"
            )
        return _result(
            input_bdf=bdf,
            normalized_bdf=normalized,
            mapping_source=mapping_source,
            status=ResultStatus.SUCCESS,
            path_data=path_data,
            diagnostics=diagnostics,
            numa_node=numa_node,
            numa_source=numa_source,
            node_cpus=node_cpus,
            online_cpus=online_cpus,
            allowed_cpus=effective_allowed,
            target_cpus=target,
        )
    except (TopologyError, ValueError) as exc:
        return _result(
            input_bdf=bdf,
            normalized_bdf=normalized,
            mapping_source=mapping_source,
            status=ResultStatus.FAILED,
            failure_code=failure_code,
            path_data=path_data,
            diagnostics=diagnostics + (str(exc),),
            numa_node=numa_node,
            numa_source=numa_source,
            node_cpus=node_cpus,
            online_cpus=online_cpus,
            allowed_cpus=effective_allowed,
        )
