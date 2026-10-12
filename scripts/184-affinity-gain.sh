#!/usr/bin/env bash
# Temporary 184-only vLLM affinity-gain check. This is not a formal test.

set -uo pipefail
: "${TEST_ID:?ERROR: set TEST_ID}"

CONTAINER=corex5-v0.23.0
MODEL_PATH=/home/model/Qwen3-32B-W8A8
SERVED_MODEL=Qwen3-32B-W8A8
GPU_LIST=0,1,2,3
PORT=18080
REQUESTS=32
CONCURRENCY=8
MODES="${MODES:-off on}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="/home/qch/tools/kunpeng-affinity-results/${TEST_ID}"
ACTIVE_PID_FILE=

if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: result directory already exists: $RUN_DIR" >&2
    exit 2
fi
mkdir -p "$RUN_DIR"
SUMMARY="$RUN_DIR/run-summary.txt"
: > "$SUMMARY"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"
}

container_exec() {
    docker exec \
        -e no_proxy=127.0.0.1,localhost \
        -e NO_PROXY=127.0.0.1,localhost \
        "$CONTAINER" bash -lc \
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; $1"
}

stop_service() {
    local pid_file="$1"
    local pid
    pid="$(container_exec "cat '$pid_file' 2>/dev/null" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        container_exec "kill -TERM '$pid' 2>/dev/null || true" >/dev/null 2>&1 || true
        sleep 8
        container_exec "kill -KILL '$pid' 2>/dev/null || true" >/dev/null 2>&1 || true
    fi
}

cleanup() {
    if [[ -n "$ACTIVE_PID_FILE" ]]; then
        stop_service "$ACTIVE_PID_FILE"
        ACTIVE_PID_FILE=
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_service() {
    local waited=0
    while ((waited < 600)); do
        if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    return 1
}

capture_processes() {
    container_exec '
for pid in $(pgrep -f "VLLM::EngineCore|VLLM::Worker|vllm.entrypoints.openai.api_server|vllm serve" 2>/dev/null); do
  echo "===== PID=$pid ====="
  ps -o pid=,ppid=,comm=,args= -p "$pid"
  grep -E "^(Cpus_allowed_list|Mems_allowed_list):" "/proc/$pid/status"
  grep -m 8 -E " (bind|interleave|prefer):" "/proc/$pid/numa_maps" 2>/dev/null || true
done'
}

start_service() {
    local mode="$1"
    local mode_dir="$RUN_DIR/service-$mode"
    local server_script="$mode_dir/server.sh"
    local pid_file="$mode_dir/server.pid"
    local plugin_env

    mkdir -p "$mode_dir"
    if [[ "$mode" == off ]]; then
        plugin_env='export KUNPENG_AFFINITY_MODE=off'
    else
        mkdir -p "$RUN_DIR/bootstrap"
        cat > "$RUN_DIR/bootstrap/sitecustomize.py" <<'PY'
from kunpeng_affinity.vllm_plugin import register
register()
PY
        plugin_env="export PYTHONPATH='$RUN_DIR/bootstrap:$REPO_ROOT/src'; export KUNPENG_AFFINITY_MODE=strict; export KUNPENG_AFFINITY_PROVIDER=iluvatar-runtime-pci; export KUNPENG_AFFINITY_IXSMI='$IXSMI_BIN'; export KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1"
    fi

    cat > "$server_script" <<EOF
#!/usr/bin/env bash
echo \$\$ > '$pid_file'
export CUDA_VISIBLE_DEVICES='$GPU_LIST'
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENFORCE_CUDA_GRAPH=1
$plugin_env
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy=127.0.0.1,localhost
exec vllm serve '$MODEL_PATH' \
  --served-model-name '$SERVED_MODEL' \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 --port '$PORT' \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  > '$mode_dir/server.log' 2>&1
EOF
    chmod +x "$server_script"

    log "start vLLM TP=4 mode=$mode"
    ACTIVE_PID_FILE="$pid_file"
    docker exec -d "$CONTAINER" bash "$server_script"
    if ! wait_for_service; then
        log "FAIL: mode=$mode service did not become ready"
        tail -n 80 "$mode_dir/server.log" || true
        return 1
    fi
    log "vLLM ready mode=$mode"
}

run_benchmark() {
    local mode="$1"
    local case_name="$2"
    local input_len="$3"
    local output_len="$4"
    local mode_dir="$RUN_DIR/service-$mode"
    local result_file="$mode_dir/bench_${case_name}.json"
    local log_file="$mode_dir/bench_${case_name}.log"

    log "benchmark mode=$mode case=$case_name input=$input_len output=$output_len"
    container_exec "vllm bench serve \
      --backend openai --base-url http://127.0.0.1:$PORT \
      --endpoint /v1/completions --model '$SERVED_MODEL' --tokenizer '$MODEL_PATH' \
      --dataset-name random --random-input-len $input_len --random-output-len $output_len \
      --num-warmups 4 --num-prompts $REQUESTS --max-concurrency $CONCURRENCY \
      --save-result --save-detailed --result-dir '$mode_dir' \
      --result-filename 'bench_${case_name}.json'" > "$log_file" 2>&1 || return 1

    [[ -s "$result_file" ]] || {
        tail -n 80 "$log_file" || true
        return 1
    }
}

summarize_case() {
    local mode="$1"
    local case_name="$2"
    local result_file="$RUN_DIR/service-$mode/bench_${case_name}.json"

    python3 - "$result_file" "$mode" "$case_name" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
mode, case_name = sys.argv[2:]

def get(*keys):
    for key in keys:
        if key in data:
            return float(data[key])
    raise KeyError(f"missing benchmark field: {keys}")

print(
    f"{case_name}_{mode}: "
    f"ttft_p50={get('median_ttft_ms'):.2f}ms "
    f"ttft_p95={get('p99_ttft_ms'):.2f}ms "
    f"tpot_p50={get('median_tpot_ms'):.2f}ms "
    f"throughput={get('output_throughput'):.2f}tok/s"
)
PY
}

compare_results() {
    python3 - "$RUN_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
fields = {
    "ttft_mean_ms": ("mean_ttft_ms", 1),
    "ttft_p50_ms": ("median_ttft_ms", 1),
    "ttft_p95_ms": ("p99_ttft_ms", 1),
    "tpot_mean_ms": ("mean_tpot_ms", 1),
    "tpot_p50_ms": ("median_tpot_ms", 1),
    "e2e_mean_s": ("mean_e2e_latency_ms", 0.001),
    "output_throughput_tok_s": ("output_throughput", 1),
}

for case_name in ("prefill", "decode"):
    off = json.loads((root / "service-off" / f"bench_{case_name}.json").read_text())
    on = json.loads((root / "service-on" / f"bench_{case_name}.json").read_text())
    print(f"===== {case_name.upper()} OFF / ON COMPARISON =====")
    for label, (key, scale) in fields.items():
        before = float(off[key]) * scale
        after = float(on[key]) * scale
        print(f"{label}: off={before:.3f} on={after:.3f} delta={(after / before - 1) * 100:+.2f}%")
PY
}

log "184 affinity-gain check"
log "prefill=input2048/output128; decode=input128/output2048"
log "requests=$REQUESTS concurrency=$CONCURRENCY modes='$MODES'"

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    log "FAIL: container not found: $CONTAINER"
    exit 2
fi
if ! container_exec "test -d '$MODEL_PATH' && test -d '$REPO_ROOT'"; then
    log "FAIL: model or repository is not visible in the container"
    exit 2
fi
if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    log "FAIL: port $PORT is already serving"
    exit 2
fi

if container_exec "test -x /home/qch/kunpeng-affinity-fix/ixsmi-run"; then
    IXSMI_BIN=/home/qch/kunpeng-affinity-fix/ixsmi-run
else
    IXSMI_BIN=ixsmi
fi

{
    echo "commit=$(git rev-parse HEAD)"
    git status --short
    docker inspect --format 'container={{.Name}} image={{.Config.Image}}' "$CONTAINER"
    container_exec "python3 -c 'import vllm; print(\"vllm=\" + vllm.__version__)'"
} > "$RUN_DIR/environment.txt" 2>&1

for mode in $MODES; do
    if [[ "$mode" != off && "$mode" != on ]]; then
        log "FAIL: invalid mode=$mode"
        exit 2
    fi
    start_service "$mode" || exit 1
    for case_spec in "prefill 2048 128" "decode 128 2048"; do
        read -r case_name input_len output_len <<< "$case_spec"
        run_benchmark "$mode" "$case_name" "$input_len" "$output_len" || exit 1
        summarize_case "$mode" "$case_name" | tee -a "$SUMMARY"
    done
    capture_processes > "$RUN_DIR/service-$mode/processes.txt" 2>&1
    if [[ "$mode" == on ]]; then
        grep -E "kunpeng-affinity|Binding (EngineCore|worker)" \
            "$RUN_DIR/service-on/server.log" > "$RUN_DIR/service-on/affinity.log" 2>/dev/null || true
    fi
    stop_service "$RUN_DIR/service-$mode/server.pid"
    ACTIVE_PID_FILE=
done

if [[ ! -f "$RUN_DIR/service-on/server.log" ]] || \
   ! grep -q "installed vLLM subprocess hook" "$RUN_DIR/service-on/server.log"; then
    log "FAIL: plugin Hook was not installed in mode=on"
    exit 1
fi
if ! grep -q "source=generic" "$RUN_DIR/service-on/server.log"; then
    log "BLOCKED: Hook loaded but real service did not reach generic path"
    log "classification=PLUGIN_LOADED_NATIVE_PRESERVED"
    exit 1
fi

compare_results | tee "$RUN_DIR/comparison.txt" | tee -a "$SUMMARY"
log "PASS: real TP=4 split prefill/decode benchmark completed"
log "result directory: $RUN_DIR"
