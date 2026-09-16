"""Framework-independent affinity domain models and errors."""

from kunpeng_affinity.core.identity import mapping_fingerprint
from kunpeng_affinity.core.models import (
    BatchAffinityResult,
    DeviceContext,
    DeviceMapping,
    DeviceResolution,
    ProbeResult,
)

__all__ = [
    "BatchAffinityResult",
    "DeviceContext",
    "DeviceMapping",
    "DeviceResolution",
    "mapping_fingerprint",
    "ProbeResult",
]
