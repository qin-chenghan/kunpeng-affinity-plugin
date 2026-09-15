"""Stable result types returned by the topology analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kunpeng_affinity.topology.cpulist import format_cpulist


class ResultStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class PciPathNode:
    bdf: str
    sysfs_path: str
    pci_class: str | None
    role: str
    numa_node: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bdf": self.bdf,
            "sysfs_path": self.sysfs_path,
            "pci_class": self.pci_class,
            "role": self.role,
            "numa_node": self.numa_node,
        }


@dataclass(frozen=True)
class AffinityResult:
    input_bdf: str
    normalized_bdf: str | None
    device_mapping_source: str
    status: ResultStatus
    pci_path: tuple[PciPathNode, ...] = ()
    root_bus_path: str | None = None
    numa_node: int | None = None
    numa_source: str | None = None
    node_cpus: frozenset[int] = field(default_factory=frozenset)
    online_cpus: frozenset[int] = field(default_factory=frozenset)
    allowed_cpus: frozenset[int] = field(default_factory=frozenset)
    target_cpus: frozenset[int] = field(default_factory=frozenset)
    cpu_source: str | None = None
    diagnostics: tuple[str, ...] = ()

    @property
    def bindable(self) -> bool:
        return self.status is ResultStatus.SUCCESS and bool(self.target_cpus)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_bdf": self.input_bdf,
            "normalized_bdf": self.normalized_bdf,
            "device_mapping_source": self.device_mapping_source,
            "status": self.status.value,
            "bindable": self.bindable,
            "pci_path": [node.to_dict() for node in self.pci_path],
            "root_bus_path": self.root_bus_path,
            "numa_node": self.numa_node,
            "numa_source": self.numa_source,
            "node_cpus": format_cpulist(self.node_cpus),
            "online_cpus": format_cpulist(self.online_cpus),
            "allowed_cpus": format_cpulist(self.allowed_cpus),
            "target_cpus": format_cpulist(self.target_cpus),
            "cpu_source": self.cpu_source,
            "diagnostics": list(self.diagnostics),
        }
