"""Preconditions for vLLM generic NUMA affinity discovery."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from kunpeng_affinity.core.errors import AffinityDiscoveryError


def check_vllm_generic_eligibility(
    numa_utils: Any,
    *,
    sysfs_root: Path | str = Path("/sys"),
) -> None:
    """Check framework execution prerequisites before candidate generation."""
    node_root = Path(sysfs_root) / "devices/system/node"
    nodes = tuple(path for path in node_root.glob("node[0-9]*") if path.is_dir())
    if len(nodes) < 2:
        raise AffinityDiscoveryError(
            "automatic NUMA binding requires more than one NUMA node",
            code="NUMA_NOT_AVAILABLE",
        )
    can_set_mempolicy = getattr(numa_utils, "_can_set_mempolicy", None)
    if not callable(can_set_mempolicy) or not can_set_mempolicy():
        raise AffinityDiscoveryError(
            "NUMA memory policy is unavailable in the current process",
            code="MEMPOLICY_UNAVAILABLE",
        )
    if shutil.which("numactl") is None:
        raise AffinityDiscoveryError(
            "numactl is not available on PATH",
            code="BIND_EXECUTOR_MISSING",
        )
