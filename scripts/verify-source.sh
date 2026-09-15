#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python3}"

"$python_bin" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -m unittest discover -s tests -v

"$python_bin" - <<'PY'
from importlib.metadata import entry_points
from pathlib import Path

import kunpeng_affinity

matches = [
    item
    for item in entry_points(group="vllm.general_plugins")
    if item.name == "kunpeng_affinity"
]
if len(matches) != 1:
    raise SystemExit(
        "expected exactly one installed vllm.general_plugins entry point "
        f"named kunpeng_affinity, found {len(matches)}"
    )

source = Path(kunpeng_affinity.__file__).resolve()
expected_root = (Path.cwd() / "src").resolve()
if expected_root not in source.parents:
    raise SystemExit(f"package is not imported from this checkout: {source}")

print(f"source import: {source}")
print(f"vLLM entry point: {matches[0].value}")
print("source deployment verification passed")
PY
