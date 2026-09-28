#!/usr/bin/env bash
set -euo pipefail

demo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$demo_dir/.." && pwd)
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

exec python3 "$demo_dir/ascend_provider_probe.py" "$@"
