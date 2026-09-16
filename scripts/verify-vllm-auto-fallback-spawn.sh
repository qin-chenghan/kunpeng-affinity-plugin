#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

cd "$repo_root"
KUNPENG_AFFINITY_VERIFY_PATH=auto-fallback \
  "$python_bin" scripts/verify-vllm-generic-spawn.py
