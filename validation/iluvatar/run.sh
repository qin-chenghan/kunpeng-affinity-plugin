#!/usr/bin/env bash
set -uo pipefail

suite_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$suite_dir/../.." && pwd)"
local_config="$suite_dir/config.env"
template_config="$suite_dir/config.env.example"
if [[ -n "${KUNPENG_AFFINITY_VALIDATION_CONFIG:-}" ]]; then
  config_file="$KUNPENG_AFFINITY_VALIDATION_CONFIG"
elif [[ -f "$local_config" ]]; then
  config_file="$local_config"
else
  config_file="$template_config"
fi
read_only=0
dry_run=0

usage() {
  cat <<'EOF'
Usage: ./validation/iluvatar/run.sh [options]

Options:
  --config PATH  Load a different validation configuration.
  --read-only    Skip source installation and dummy spawn checks.
  --dry-run      Print the selected stages without executing them.
  -h, --help     Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --config)
      if (($# < 2)); then
        echo "--config requires a path" >&2
        exit 2
      fi
      config_file="$2"
      shift 2
      ;;
    --read-only)
      read_only=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$config_file" ]]; then
  echo "validation config not found: $config_file" >&2
  exit 2
fi

# The configuration is a repository-owned shell environment file. Keep it to
# assignments and comments; command-line overrides are intentionally avoided.
# shellcheck disable=SC1090
source "$config_file"

: "${PYTHON_BIN:=python3}"
: "${IXSMI_BIN:=ixsmi}"
: "${SYSFS_ROOT:=/sys}"
: "${PROBE_DEVICE:=0}"
: "${VISIBILITY_ENV:=CUDA_VISIBLE_DEVICES}"
: "${REORDER_VISIBLE_DEVICES:=1,0}"
: "${SPAWN_VISIBLE_DEVICES:=0}"
: "${AFFINITY_BDFS:=}"
: "${KUNPENG_AFFINITY_MODE:=auto}"
: "${KUNPENG_AFFINITY_PROVIDER:=iluvatar-runtime-pci}"
: "${KUNPENG_AFFINITY_CPU_POLICY:=node}"
: "${KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL:=detail}"
: "${KUNPENG_AFFINITY_VLLM_FORCE_GENERIC:=1}"
: "${VLLM_WORKER_MULTIPROC_METHOD:=spawn}"
: "${RUN_UNIT_TESTS:=1}"
: "${RUN_TOPOLOGY_PROBE:=1}"
: "${RUN_PROVIDER_PROBE:=1}"
: "${RUN_REORDER_PROBE:=1}"
: "${RUN_SOURCE_INSTALL:=1}"
: "${RUN_FORCED_GENERIC_SPAWN:=1}"
: "${RUN_AUTO_FALLBACK_SPAWN:=1}"
: "${ENABLE_ENVIRONMENT_CHANGES:=0}"
: "${RESULT_DIR:=}"

is_enabled() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    0|false|FALSE|no|NO|off|OFF) return 1 ;;
    *)
      echo "invalid boolean value: $1" >&2
      exit 2
      ;;
  esac
}

resolve_command() {
  local value="$1"
  if [[ "$value" == */* ]]; then
    [[ -x "$value" ]] || return 1
    printf '%s\n' "$value"
    return
  fi
  command -v -- "$value"
}

case "$VISIBILITY_ENV" in
  CUDA_VISIBLE_DEVICES|ILUVATAR_VISIBLE_DEVICES) ;;
  *)
    echo "VISIBILITY_ENV must be CUDA_VISIBLE_DEVICES or ILUVATAR_VISIBLE_DEVICES" >&2
    exit 2
    ;;
esac

if ((dry_run)); then
  python_bin="$PYTHON_BIN"
  ixsmi_bin="$IXSMI_BIN"
else
  python_bin="$(resolve_command "$PYTHON_BIN")" || {
    echo "Python interpreter is not executable: $PYTHON_BIN" >&2
    exit 2
  }
  ixsmi_bin="$(resolve_command "$IXSMI_BIN")" || {
    echo "Iluvatar utility is not executable: $IXSMI_BIN" >&2
    exit 2
  }
fi

if [[ -z "$RESULT_DIR" ]]; then
  RESULT_DIR="/tmp/kunpeng-affinity-validation-$(date +%Y%m%d-%H%M%S)-$$"
elif [[ "$RESULT_DIR" != /* ]]; then
  RESULT_DIR="$repo_root/$RESULT_DIR"
fi
mkdir -p "$RESULT_DIR"
RESULT_DIR="$(cd -- "$RESULT_DIR" && pwd)"

declare -a summary_names=()
declare -a summary_states=()
declare -a summary_logs=()
current_stage="not started"

print_summary() {
  echo
  echo "Validation summary"
  echo "------------------"
  local index
  for ((index = 0; index < ${#summary_names[@]}; index++)); do
    printf '%-32s %-8s %s\n' \
      "${summary_names[index]}" \
      "${summary_states[index]}" \
      "${summary_logs[index]}"
  done
  echo "Result directory: $RESULT_DIR"
}

record_skip() {
  summary_names+=("$1")
  summary_states+=("SKIP")
  summary_logs+=("$2")
}

run_stage() {
  local stage_id="$1"
  local stage_name="$2"
  shift 2
  local log_file="$RESULT_DIR/${stage_id}.log"
  current_stage="$stage_name"
  echo
  echo "==> $stage_name"
  if ((dry_run)); then
    summary_names+=("$stage_name")
    summary_states+=("DRY-RUN")
    summary_logs+=("$log_file")
    return 0
  fi

  (set -euo pipefail; "$@") 2>&1 | tee "$log_file"
  local command_status=${PIPESTATUS[0]}
  summary_names+=("$stage_name")
  summary_logs+=("$log_file")
  if ((command_status != 0)); then
    summary_states+=("FAIL($command_status)")
    print_summary
    echo "Stopped at stage: $stage_name" >&2
    exit "$command_status"
  fi
  summary_states+=("PASS")
}

run_with_visibility() {
  local visible_devices="$1"
  shift
  env \
    -u CUDA_VISIBLE_DEVICES \
    -u ILUVATAR_VISIBLE_DEVICES \
    "${VISIBILITY_ENV}=${visible_devices}" \
    "$@"
}

stage_preflight() {
  cd "$repo_root"
  echo "repo_root=$repo_root"
  echo "config_file=$config_file"
  echo "python_bin=$python_bin"
  echo "ixsmi_bin=$ixsmi_bin"
  echo "sysfs_root=$SYSFS_ROOT"
  echo "visibility_env=$VISIBILITY_ENV"
  echo "affinity_bdfs=${AFFINITY_BDFS:-<automatic PCI candidates>}"
  uname -a
  git status --short --branch
  git log -1 --oneline --decorate
  "$python_bin" - <<'PY'
import importlib.metadata as metadata
import sys

print(f"python_executable={sys.executable}")
print(f"python_version={sys.version}")
try:
    distribution = metadata.distribution("vllm")
except metadata.PackageNotFoundError as exc:
    raise SystemExit("vLLM is not installed in the selected Python environment") from exc
print(f"vllm_version={distribution.version}")
print(f"vllm_location={distribution.locate_file('')}")
PY
  [[ -d "$SYSFS_ROOT/bus/pci/devices" ]]
  "$ixsmi_bin" -L
  "$ixsmi_bin"
  if ((!read_only)) && is_enabled "$ENABLE_ENVIRONMENT_CHANGES" && \
     { is_enabled "$RUN_FORCED_GENERIC_SPAWN" || \
       is_enabled "$RUN_AUTO_FALLBACK_SPAWN"; }; then
    command -v numactl
  fi
}

stage_unit_tests() {
  cd "$repo_root"
  PYTHON_BIN="$python_bin" ./scripts/test.sh
}

stage_topology_probe() {
  cd "$repo_root"
  AFFINITY_BDFS="$AFFINITY_BDFS" \
    PYTHON_BIN="$python_bin" ./scripts/probe-host.sh
}

stage_provider_single() {
  cd "$repo_root"
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" demo/iluvatar_provider_probe.py \
      --ixsmi "$ixsmi_bin" \
      --sysfs-root "$SYSFS_ROOT" \
      --device "$PROBE_DEVICE"
}

stage_provider_all() {
  cd "$repo_root"
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" demo/iluvatar_provider_probe.py \
      --ixsmi "$ixsmi_bin" \
      --sysfs-root "$SYSFS_ROOT" \
      --json
}

stage_provider_reorder() {
  cd "$repo_root"
  run_with_visibility "$REORDER_VISIBLE_DEVICES" \
    env PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" demo/iluvatar_provider_probe.py \
      --ixsmi "$ixsmi_bin" \
      --sysfs-root "$SYSFS_ROOT" \
      --json
}

stage_source_install() {
  cd "$repo_root"
  PYTHON_BIN="$python_bin" ./scripts/install-source.sh
  PYTHON_BIN="$python_bin" ./scripts/verify-source.sh
}

stage_forced_generic_spawn() {
  cd "$repo_root"
  run_with_visibility "$SPAWN_VISIBLE_DEVICES" \
    env \
      KUNPENG_AFFINITY_MODE="$KUNPENG_AFFINITY_MODE" \
      KUNPENG_AFFINITY_PROVIDER="$KUNPENG_AFFINITY_PROVIDER" \
      KUNPENG_AFFINITY_CPU_POLICY="$KUNPENG_AFFINITY_CPU_POLICY" \
      KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL="$KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL" \
      KUNPENG_AFFINITY_VLLM_FORCE_GENERIC="$KUNPENG_AFFINITY_VLLM_FORCE_GENERIC" \
      VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" \
      PYTHON_BIN="$python_bin" \
      ./scripts/verify-vllm-generic-spawn.sh
}

stage_auto_fallback_spawn() {
  cd "$repo_root"
  run_with_visibility "$SPAWN_VISIBLE_DEVICES" \
    env \
      KUNPENG_AFFINITY_MODE="$KUNPENG_AFFINITY_MODE" \
      KUNPENG_AFFINITY_PROVIDER="$KUNPENG_AFFINITY_PROVIDER" \
      KUNPENG_AFFINITY_CPU_POLICY="$KUNPENG_AFFINITY_CPU_POLICY" \
      KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL="$KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL" \
      VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" \
      KUNPENG_AFFINITY_VERIFY_PATH=auto-fallback \
      PYTHON_BIN="$python_bin" \
      ./scripts/verify-vllm-generic-spawn.sh
}

echo "Kunpeng affinity Iluvatar validation"
echo "Configuration: $config_file"
echo "Result directory: $RESULT_DIR"
if ((read_only)); then
  echo "Mode: read-only"
elif is_enabled "$ENABLE_ENVIRONMENT_CHANGES"; then
  echo "Mode: full validation"
else
  echo "Mode: read-only stages; environment-changing stages are disabled"
fi

run_stage 00-preflight "Environment preflight" stage_preflight

if is_enabled "$RUN_UNIT_TESTS"; then
  run_stage 01-unit-tests "Source unit tests" stage_unit_tests
else
  record_skip "Source unit tests" "disabled by config"
fi

if is_enabled "$RUN_TOPOLOGY_PROBE"; then
  run_stage 02-topology "Linux topology probe" stage_topology_probe
else
  record_skip "Linux topology probe" "disabled by config"
fi

if is_enabled "$RUN_PROVIDER_PROBE"; then
  run_stage 03-provider-single "Iluvatar Provider single device" stage_provider_single
  run_stage 04-provider-all "Iluvatar Provider all devices" stage_provider_all
else
  record_skip "Iluvatar Provider single device" "disabled by config"
  record_skip "Iluvatar Provider all devices" "disabled by config"
fi

visible_count=0
if ((!dry_run)) && is_enabled "$RUN_REORDER_PROBE"; then
  visible_count="$("$python_bin" -c 'from vllm.platforms import current_platform; print(current_platform.device_count())')" || {
    echo "cannot determine visible device count for reorder validation" >&2
    exit 2
  }
fi
if is_enabled "$RUN_REORDER_PROBE" && ((dry_run || visible_count >= 2)); then
  run_stage 05-provider-reorder "Iluvatar Provider visibility reorder" stage_provider_reorder
elif is_enabled "$RUN_REORDER_PROBE"; then
  record_skip "Iluvatar Provider visibility reorder" "fewer than two devices visible"
else
  record_skip "Iluvatar Provider visibility reorder" "disabled by config"
fi

allow_environment_changes=0
if ((!read_only)) && is_enabled "$ENABLE_ENVIRONMENT_CHANGES"; then
  allow_environment_changes=1
fi

if is_enabled "$RUN_SOURCE_INSTALL" && ((allow_environment_changes)); then
  run_stage 06-source-install "Editable install and entry points" stage_source_install
elif is_enabled "$RUN_SOURCE_INSTALL"; then
  record_skip "Editable install and entry points" "ENABLE_ENVIRONMENT_CHANGES is disabled"
else
  record_skip "Editable install and entry points" "disabled by config"
fi

if is_enabled "$RUN_FORCED_GENERIC_SPAWN" && ((allow_environment_changes)); then
  run_stage 07-forced-generic "vLLM forced-generic dummy spawn" stage_forced_generic_spawn
elif is_enabled "$RUN_FORCED_GENERIC_SPAWN"; then
  record_skip "vLLM forced-generic dummy spawn" "ENABLE_ENVIRONMENT_CHANGES is disabled"
else
  record_skip "vLLM forced-generic dummy spawn" "disabled by config"
fi

if is_enabled "$RUN_AUTO_FALLBACK_SPAWN" && ((allow_environment_changes)); then
  run_stage 08-auto-fallback "vLLM automatic fallback dummy spawn" stage_auto_fallback_spawn
elif is_enabled "$RUN_AUTO_FALLBACK_SPAWN"; then
  record_skip "vLLM automatic fallback dummy spawn" "ENABLE_ENVIRONMENT_CHANGES is disabled"
else
  record_skip "vLLM automatic fallback dummy spawn" "disabled by config"
fi

print_summary
