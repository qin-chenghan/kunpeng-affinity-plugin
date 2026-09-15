"""Framework adapters that assemble the provider-neutral affinity core."""

from kunpeng_affinity.adapters.vllm_generic import resolve_vllm_generic_affinity

__all__ = ["resolve_vllm_generic_affinity"]
