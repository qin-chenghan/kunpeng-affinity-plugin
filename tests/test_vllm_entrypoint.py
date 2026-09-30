from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from kunpeng_affinity import vllm_plugin
from kunpeng_affinity.config import PluginMode


class VllmEntrypointTest(unittest.TestCase):
    def test_off_mode_does_not_import_adapter(self) -> None:
        module_name = "kunpeng_affinity.adapters.vllm_adapter"
        with (
            patch.object(vllm_plugin, "load_plugin_mode", return_value=PluginMode.OFF),
            patch.dict(sys.modules, {module_name: None}),
        ):
            vllm_plugin.register()
            self.assertIsNone(sys.modules[module_name])

    def test_enabled_mode_delegates_pre_resolved_mode(self) -> None:
        with (
            patch.object(
                vllm_plugin,
                "load_plugin_mode",
                return_value=PluginMode.STRICT,
            ),
            patch("kunpeng_affinity.adapters.vllm_adapter.install") as install,
        ):
            vllm_plugin.register()

        install.assert_called_once_with(mode=PluginMode.STRICT)


if __name__ == "__main__":
    unittest.main()
