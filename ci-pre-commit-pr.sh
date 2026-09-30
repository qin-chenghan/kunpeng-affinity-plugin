#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

target_ref="${TARGET_REF:-${TARGET_BRANCH:-origin/main}}"
if ! git rev-parse --verify "${target_ref}^{commit}" >/dev/null 2>&1; then
    echo "target Git ref is not available: ${target_ref}" >&2
    echo "fetch the target branch or set TARGET_REF to an available ref" >&2
    exit 2
fi

changed_files=()
while IFS= read -r path; do
    changed_files+=("$path")
done < <(git diff --name-only --diff-filter=ACMR "${target_ref}...HEAD" | sort -u)

if ((${#changed_files[@]} == 0)); then
    echo "no added, copied, modified, or renamed files; pre-commit check passed"
    exit 0
fi

echo "running pre-commit for ${#changed_files[@]} changed file(s) against ${target_ref}"
pre-commit run --show-diff-on-failure --files "${changed_files[@]}"
