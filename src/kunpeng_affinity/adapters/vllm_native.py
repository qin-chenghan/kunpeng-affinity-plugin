"""Validation of vLLM's native NUMA discovery result."""

from __future__ import annotations

from typing import Any

from kunpeng_affinity.core.errors import AffinityDiscoveryError
from kunpeng_affinity.core.models import NativeOutcome, NativeStatus


def classify_vllm_native_result(numa_utils: Any, platform: Any) -> NativeOutcome:
    """Classify the native query without converting errors into fallback."""
    query = getattr(numa_utils, "get_auto_numa_nodes", None)
    if not callable(query):
        return NativeOutcome(
            status=NativeStatus.FALLBACK_ALLOWED,
            failure_code="NATIVE_QUERY_UNAVAILABLE",
            evidence=("vLLM native query symbol is missing",),
        )
    try:
        raw_nodes = query()
    except Exception as exc:
        return NativeOutcome(
            status=NativeStatus.ERROR,
            failure_code="NATIVE_QUERY_FAILED",
            original_error=exc,
            evidence=("vLLM native query raised an exception",),
        )
    if raw_nodes is None or raw_nodes == []:
        return NativeOutcome(
            status=NativeStatus.FALLBACK_ALLOWED,
            failure_code="NATIVE_QUERY_UNAVAILABLE",
            evidence=("native query returned no usable result",),
        )
    if not isinstance(raw_nodes, list):
        return NativeOutcome(
            status=NativeStatus.INVALID,
            failure_code="NATIVE_RESULT_INVALID",
            evidence=(f"native result type={type(raw_nodes).__name__}",),
        )
    from kunpeng_affinity.adapters.vllm_devices import vllm_device_count

    try:
        count = vllm_device_count(platform)
    except AffinityDiscoveryError as exc:
        return NativeOutcome(
            status=NativeStatus.ERROR,
            failure_code=exc.code,
            original_error=exc,
            evidence=("visible device count could not be validated",),
        )
    if len(raw_nodes) != count:
        return NativeOutcome(
            status=NativeStatus.INVALID,
            failure_code="DEVICE_COUNT_MISMATCH",
            evidence=(f"native_count={len(raw_nodes)} visible_count={count}",),
        )
    if any(not isinstance(node, int) or isinstance(node, bool) or node < 0 for node in raw_nodes):
        return NativeOutcome(
            status=NativeStatus.INVALID,
            failure_code="NATIVE_RESULT_INVALID",
            evidence=("native result contains an invalid NUMA node",),
        )
    return NativeOutcome(
        status=NativeStatus.VALID,
        nodes=tuple(raw_nodes),
        evidence=("vLLM native NUMA result satisfies its return contract",),
    )
