"""SGLang 0.5.18 general-plugin entry point for generic NUMA fallback."""

from __future__ import annotations

import logging
import os
from typing import Any

from kunpeng_affinity.config import PluginMode, load_plugin_config, load_plugin_mode
from kunpeng_affinity.core.errors import AffinityDiscoveryError

logger = logging.getLogger(__name__)
_HOOK_MARKER = "__kunpeng_affinity_sglang_hook__"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _generic_node(gpu_id: int, provider: str) -> int:
    # Resolve one visible SGLang GPU through the shared runtime and Linux path.
    if provider not in {"auto", "sglang-runtime-pci", "iluvatar-runtime-pci"}:
        raise AffinityDiscoveryError(
            f"provider {provider!r} is not registered for SGLang",
            code="PROVIDER_NOT_FOUND",
        )
    from kunpeng_affinity.adapters.sglang_generic import resolve_sglang_numa_node

    return resolve_sglang_numa_node(gpu_id, ixsmi=os.environ.get("KUNPENG_AFFINITY_IXSMI", "ixsmi"))


def _around_numa_query(original: Any, server_args: Any, gpu_id: int, *, mode: PluginMode, provider: str) -> int | None:
    """Keep explicit/native SGLang results and add only a fallback."""
    # Preserve an explicit node list and SGLang's own disabled behavior.
    if getattr(server_args, "numa_node", None) is not None:
        return original(server_args, gpu_id)
    if not _env_bool("SGLANG_AUTO_NUMA_BIND"):
        return original(server_args, gpu_id)

    # Let the framework-native query decide before using the generic fallback.
    native = original(server_args, gpu_id)
    if native is not None:
        return native
    try:
        # The fallback returns a node only after identity and Linux topology pass.
        node = _generic_node(gpu_id, provider)
    except Exception as exc:
        if mode is PluginMode.STRICT:
            raise
        logger.warning(
            "[kunpeng-affinity] SGLang generic NUMA fallback skipped for GPU %s: %s",
            gpu_id,
            exc,
        )
        return None
    logger.warning(
        "[kunpeng-affinity] SGLang generic NUMA fallback selected node=%s gpu=%s",
        node,
        gpu_id,
    )
    return node


def register() -> None:
    """Register the SGLang NUMA-query fallback through its plugin SPI."""
    # Read policy before installing a hook; off mode has no framework side effects.
    mode = load_plugin_mode()
    if mode is PluginMode.OFF:
        return
    config = load_plugin_config()
    from sglang.srt.plugins import HookRegistry
    from sglang.srt.plugins.hook_registry import HookType

    target = "sglang.srt.utils.numa_utils.get_numa_node_if_available"
    if getattr(register, _HOOK_MARKER, False):
        return

    def hook(original: Any, server_args: Any, gpu_id: int) -> int | None:
        return _around_numa_query(
            original,
            server_args,
            gpu_id,
            mode=mode,
            provider=config.provider,
        )

    # The hook changes only NUMA-node selection; SGLang keeps process launching
    # and the actual numactl operation in its original implementation.
    HookRegistry.register(target, hook, HookType.AROUND)
    setattr(register, _HOOK_MARKER, True)
    logger.warning(
        "[kunpeng-affinity] installed SGLang NUMA query hook mode=%s provider=%s",
        mode.value,
        config.provider,
    )
