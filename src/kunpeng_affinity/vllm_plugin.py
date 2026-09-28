"""vLLM general-plugin entry point."""

from __future__ import annotations

from kunpeng_affinity.config import PluginMode, load_plugin_mode


def register() -> None:
    """Load and install the vLLM framework adapter."""
    mode = load_plugin_mode()
    if mode is PluginMode.OFF:
        return

    from kunpeng_affinity.adapters.vllm_adapter import install

    install(mode=mode)
