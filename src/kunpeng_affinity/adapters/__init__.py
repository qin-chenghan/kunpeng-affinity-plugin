"""Framework adapters that assemble the provider-neutral affinity core."""

from kunpeng_affinity.adapters.vllm_generic import (
    check_vllm_generic_eligibility,
    resolve_vllm_generic_affinity,
    resolve_vllm_native_nodes,
)

__all__ = [
    "check_vllm_generic_eligibility",
    "resolve_vllm_generic_affinity",
    "resolve_vllm_native_nodes",
]
