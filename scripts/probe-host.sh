#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

"$python_bin" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
cd "$repo_root"
PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -m kunpeng_affinity.topology.environment "$@"
