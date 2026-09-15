"""Provider SPI for framework-visible device identity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from kunpeng_affinity.core.models import DeviceContext, DeviceMapping, ProbeResult


class DeviceMapper(Protocol):
    """Map framework logical devices to verified PCI functions."""

    name: str
    supports_shared_bdf: bool

    def probe(self, contexts: Sequence[DeviceContext]) -> ProbeResult:
        """Report capability without modifying system state."""

    def map_all(self, contexts: Sequence[DeviceContext]) -> Sequence[DeviceMapping]:
        """Return one mapping per context, in exactly the input order."""
