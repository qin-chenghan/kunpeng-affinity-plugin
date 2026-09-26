"""Stable identities for ordered framework-visible device mappings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from kunpeng_affinity.core.models import DeviceMapping, DeviceResolution


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


def affinity_snapshot_fingerprint(
    mappings: Sequence[DeviceMapping],
    resolutions: Sequence[DeviceResolution],
    *,
    metadata: dict[str, str] | None = None,
) -> str:
    """Digest device identity plus the Linux evidence used for binding."""
    encoded = affinity_snapshot_json(
        mappings,
        resolutions,
        metadata=metadata,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def affinity_snapshot_json(
    mappings: Sequence[DeviceMapping],
    resolutions: Sequence[DeviceResolution],
    *,
    metadata: dict[str, str] | None = None,
) -> str:
    """Serialize the complete candidate snapshot for cross-process validation."""
    payload = {
        "metadata": dict(metadata or {}),
        "mappings": [
            {
                "logical_device_id": mapping.logical_device_id,
                "pci_bdf": mapping.pci_bdf,
                "source": mapping.source,
                "evidence": list(mapping.evidence),
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
        ],
        "resolutions": [
            {
                "input_bdf": item.affinity.input_bdf,
                "normalized_bdf": item.affinity.normalized_bdf,
                "root_bus_path": item.affinity.root_bus_path,
                "pci_path": [
                    {
                        "bdf": node.bdf,
                        "sysfs_path": node.sysfs_path,
                        "pci_class": node.pci_class,
                        "numa_node": node.numa_node,
                    }
                    for node in item.affinity.pci_path
                ],
                "numa_node": item.affinity.numa_node,
                "node_cpus": sorted(item.affinity.node_cpus),
                "online_cpus": sorted(item.affinity.online_cpus),
                "allowed_cpus": sorted(item.affinity.allowed_cpus),
                "target_cpus": sorted(item.affinity.target_cpus),
            }
            for item in resolutions
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def serialized_snapshot_fingerprint(snapshot_json: str) -> str:
    """Digest a canonical snapshot serialization."""
    return hashlib.sha256(snapshot_json.encode("ascii")).hexdigest()
