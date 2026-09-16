"""Stable identities for ordered framework-visible device mappings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from kunpeng_affinity.core.models import DeviceMapping


def mapping_fingerprint(mappings: Sequence[DeviceMapping]) -> str:
    """Return a stable digest for one ordered logical-device visibility view."""
    payload = [
        {
            "logical_device_id": mapping.logical_device_id,
            "pci_bdf": mapping.pci_bdf,
            "physical_device_id": (
                None
                if mapping.physical_device_id is None
                else {
                    "type": type(mapping.physical_device_id).__name__,
                    "value": str(mapping.physical_device_id),
                }
            ),
            "instance_id": mapping.instance_id,
        }
        for mapping in mappings
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
