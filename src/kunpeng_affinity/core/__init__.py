"""Framework-independent affinity domain models and errors."""

from kunpeng_affinity.core.identity import (
    affinity_snapshot_fingerprint,
    affinity_snapshot_json,
    mapping_fingerprint,
    serialized_snapshot_fingerprint,
)
from kunpeng_affinity.core.models import (
    BatchAffinityResult,
    BatchStatus,
    DeviceContext,
    DeviceMapping,
    DeviceResolution,
    NativeOutcome,
    NativeStatus,
    ProbeStatus,
    ProbeResult,
)

__all__ = [
    "BatchAffinityResult",
    "BatchStatus",
    "DeviceContext",
    "DeviceMapping",
    "DeviceResolution",
    "NativeOutcome",
    "NativeStatus",
    "ProbeStatus",
    "mapping_fingerprint",
    "affinity_snapshot_fingerprint",
    "affinity_snapshot_json",
    "serialized_snapshot_fingerprint",
    "ProbeResult",
]
