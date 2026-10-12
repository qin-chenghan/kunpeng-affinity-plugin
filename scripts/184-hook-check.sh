#!/usr/bin/env bash
# Temporary 184-only Hook diagnosis. No model, service, or binding is started.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p /home/qch/tools/kunpeng-affinity-results
REPORT="$(mktemp /home/qch/tools/kunpeng-affinity-results/184-hook-check-XXXXXX.log)"
echo "Diagnostic output: $REPORT"

docker exec -i \
    -e PYTHONPATH="$REPO_ROOT/src" \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_USE_V1=1 \
    -e VLLM_ENFORCE_CUDA_GRAPH=1 \
    -e KUNPENG_AFFINITY_MODE=strict \
    -e KUNPENG_AFFINITY_PROVIDER=iluvatar-runtime-pci \
    -e KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1 \
    corex5-v0.23.0 python3 -u - <<'PY' 2>&1 | tee "$REPORT"
import importlib
import inspect
import logging
import types

from kunpeng_affinity.adapters import vllm_adapter
from kunpeng_affinity.vllm_plugin import register

register()
from vllm.utils import numa_utils

installed = numa_utils.configure_subprocess
original = getattr(installed, "__kunpeng_affinity_original__", None)
print("plugin_file=", vllm_adapter.__file__)
print("numa_module_file=", numa_utils.__file__)

def snapshot(label):
    fn = numa_utils.configure_subprocess
    print(f"\n===== {label} =====")
    print("hook_present=", hasattr(fn, "__kunpeng_affinity_original__"),
          "same_as_installed=", fn is installed)
    print("function_file=", inspect.getsourcefile(inspect.unwrap(fn)))
    print("signature=", inspect.signature(fn))
    print("logging_global_disable=", logging.root.manager.disable)
    logger = vllm_adapter.logger
    print("plugin_warning_enabled=", logger.isEnabledFor(logging.WARNING))
    while logger is not None:
        print("logger=", logger.name, "disabled=", logger.disabled,
              "level=", logger.level, "propagate=", logger.propagate,
              "filters=", [type(f).__name__ for f in logger.filters],
              "handlers=", [(type(h).__name__, h.level) for h in logger.handlers])
        logger = logger.parent

snapshot("after registration")
for name in ("vllm.entrypoints.openai.api_server",
             "vllm.v1.engine.utils", "vllm.v1.executor.multiproc_executor"):
    module = importlib.import_module(name)
    snapshot("after import " + name)
    reference = getattr(module, "numa_utils", None)
    print("caller_numa_module_matches=", reference is numa_utils)
    for alias, value in vars(module).items():
        if value is installed or (original is not None and value is original):
            print("caller_function_alias=", alias,
                  "uses_installed_hook=", value is installed)
    lines = inspect.getsource(module).splitlines()
    for index, line in enumerate(lines):
        if "configure_subprocess" in line:
            print(f"--- {module.__file__}:{index + 1} ---")
            print("\n".join(lines[max(0, index - 2):index + 5]))

# Exercise the current entry with binding disabled; no subprocess is launched.
records = []
class Capture(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())

handler = Capture()
vllm_adapter.logger.addHandler(handler)
try:
    config = types.SimpleNamespace(parallel_config=types.SimpleNamespace(numa_bind=False))
    with numa_utils.configure_subprocess(config, 0, process_kind="EngineCore"):
        print("unbound_context_entered=True")
finally:
    vllm_adapter.logger.removeHandler(handler)
print("plugin_call_log_captured=", any("process_kind=EngineCore" in r for r in records))
print("Diagnostic complete; no model or service was started.")
PY
