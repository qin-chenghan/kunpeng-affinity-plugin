"""Framework adapters that assemble the provider-neutral affinity core."""

from kunpeng_affinity.adapters.vllm_generic import (
    build_vllm_device_contexts,
    check_vllm_generic_eligibility,
    create_vllm_provider_registry,
    resolve_vllm_generic_affinity,
    resolve_vllm_native_nodes,
    resolve_vllm_visibility_fingerprint,
)
from kunpeng_affinity.adapters.sglang_generic import (
    SglangRuntimeProvider,
    SglangTorchPlatform,
    resolve_sglang_numa_node,
)

__all__ = [
    "build_vllm_device_contexts",
    "check_vllm_generic_eligibility",
    "create_vllm_provider_registry",
    "resolve_vllm_generic_affinity",
    "resolve_vllm_native_nodes",
    "resolve_vllm_visibility_fingerprint",
    "SglangRuntimeProvider",
    "SglangTorchPlatform",
    "resolve_sglang_numa_node",
]
