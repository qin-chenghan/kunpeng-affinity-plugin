"""Stable, framework-neutral models used by providers and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kunpeng_affinity.topology.models import AffinityResult


@dataclass(frozen=True)
class DeviceContext:
    """Identity and process context for one framework-visible device."""

    framework: str
    logical_device_id: int
    runtime_device_id: str | int | None = None
    device_node: str | None = None
    explicit_bdf: str | None = None
    process_kind: str = "worker"
    local_rank: int | None = None
    dp_local_rank: int | None = None
    allowed_cpus: frozenset[int] = field(default_factory=frozenset)
    visibility_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_cpus", frozenset(self.allowed_cpus))


@dataclass(frozen=True)
class DeviceMapping:
    """A provider-proven mapping from one logical device to one PCI function."""

    logical_device_id: int
    pci_bdf: str
    source: str
    physical_device_id: str | int | None = None
    evidence: tuple[str, ...] = ()
    instance_id: str | None = None


@dataclass(frozen=True)
class ProbeResult:
    """Read-only provider capability probe result."""

    provider: str
    supported: bool
    reason: str | None = None


@dataclass(frozen=True)
class DeviceResolution:
    """The complete result for one input context."""

    context: DeviceContext
    mapping: DeviceMapping
    affinity: AffinityResult

    @property
    def bindable(self) -> bool:
        return self.affinity.bindable


@dataclass(frozen=True)
class BatchAffinityResult:
    """Ordered multi-device result with an explicit commit decision."""

    ordered_results: tuple[DeviceResolution, ...]
    expected_device_count: int
    visibility_fingerprint: str | None
    committable: bool
    failure_summary: tuple[str, ...] = field(default_factory=tuple)

    @property
    def bindable(self) -> bool:
        return self.committable

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_device_count": self.expected_device_count,
            "visibility_fingerprint": self.visibility_fingerprint,
            "committable": self.committable,
            "failure_summary": list(self.failure_summary),
            "ordered_results": [
                {
                    "logical_device_id": item.context.logical_device_id,
                    "mapping": {
                        "pci_bdf": item.mapping.pci_bdf,
                        "source": item.mapping.source,
                        "physical_device_id": item.mapping.physical_device_id,
                        "instance_id": item.mapping.instance_id,
                    },
                    "affinity": item.affinity.to_dict(),
                }
                for item in self.ordered_results
            ],
        }
