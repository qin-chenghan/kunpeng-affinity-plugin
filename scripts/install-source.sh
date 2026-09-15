#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python3}"

"$python_bin" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
"$python_bin" -c 'from importlib.metadata import version; assert tuple(map(int, version("setuptools").split(".")[:2])) >= (64, 0), "setuptools 64+ is required"'
"$python_bin" -m pip install --no-deps --no-build-isolation -e .

echo "Editable source installation complete: $repo_root"
echo "Interpreter: $("$python_bin" -c 'import sys; print(sys.executable)')"
