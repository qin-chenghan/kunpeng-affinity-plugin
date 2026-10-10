#!/usr/bin/env bash
# Temporary 165-only SGLang affinity-gain check. This is not a formal test.

set -uo pipefail

: "${TEST_ID:?ERROR: set TEST_ID}"

CONTAINER=sglang-serve
MODEL_PATH=/model/Qwen/Qwen3-32B-W8A8
SERVED_MODEL=Qwen3-32B-W8A8
GPU_LIST=0,1,2,3
TP_SIZE=4
PORT=8005
REQUESTS=32
CONCURRENCY=8
MODES="${MODES:-off on}"
LATENCY_SCRIPT=/home/qch/scripts/sglang_latency_online.py
RUN_DIR="/home/qch/timeline/${TEST_ID}"
LOG_DIR="$RUN_DIR/logs"
RESULT_DIR="$RUN_DIR/results"
CLEANED=0

CASES=(
    "prefill 2048 128"
    "decode 128 2048"
)

if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: result directory already exists: $RUN_DIR" >&2
    exit 2
fi
mkdir -p "$LOG_DIR" "$RESULT_DIR"

export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

container_exec() {
    docker exec \
        -e no_proxy=127.0.0.1,localhost \
        -e NO_PROXY=127.0.0.1,localhost \
        "$CONTAINER" bash -lc \
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; $1"
}

stop_server() {
    container_exec \
        "pkill -f -- 'sglang serve.*--port[ ]$PORT([ ]|\$)' 2>/dev/null; sleep 5; pkill -9 -f -- 'sglang serve.*--port[ ]$PORT([ ]|\$)' 2>/dev/null; true" \
        >/dev/null 2>&1 || true
    sleep 3
}

cleanup() {
    if [[ "$CLEANED" -eq 0 ]]; then
        CLEANED=1
        stop_server
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_server() {
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

start_server() {
    local mode="$1"
    local server_log="$LOG_DIR/server_${mode}.log"
    local bind_env

    if [[ "$mode" == on ]]; then
        bind_env='export KUNPENG_AFFINITY_MODE=auto KUNPENG_AFFINITY_PROVIDER=auto SGLANG_AUTO_NUMA_BIND=1 SGLANG_NUMA_BIND_V2=1'
    else
        bind_env='export KUNPENG_AFFINITY_MODE=off SGLANG_AUTO_NUMA_BIND=0 SGLANG_NUMA_BIND_V2=1'
    fi

    stop_server
    : > "$server_log"
    container_exec "cat > /tmp/165_affinity_gain_serve.sh <<'EOS'
#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=$GPU_LIST
export HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=1800 SOC_VERSION=ascend910b1 OMP_NUM_THREADS=4
export SGLANG_OPT_FUSE_WQA_WKV=0 SGLANG_OPT_FP8_WO_A_GEMM=0
$bind_env
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy=127.0.0.1,localhost
exec sglang serve '$MODEL_PATH' \
  --served-model-name '$SERVED_MODEL' \
  --tp-size $TP_SIZE \
  --host 127.0.0.1 --port $PORT \
  --trust-remote-code \
  --context-length 4096 \
  --mem-fraction-static 0.9 \
  --max-running-requests $CONCURRENCY \
  --attention-backend ascend \
  --random-seed 1 \
  --disable-radix-cache
EOS
chmod +x /tmp/165_affinity_gain_serve.sh"

    docker exec -d "$CONTAINER" bash -lc \
        "bash /tmp/165_affinity_gain_serve.sh > '$server_log' 2>&1"
    log "waiting for SGLang mode=$mode"
    if ! wait_for_server; then
        log "FAIL: mode=$mode did not become ready"
        tail -n 80 "$server_log" || true
        return 1
    fi
    log "SGLang ready mode=$mode"
}

capture_binding() {
    local mode="$1"
    local output="$LOG_DIR/process_affinity_${mode}.log"

    container_exec '
echo "===== NUMA NODE CPU LISTS ====="
for node in /sys/devices/system/node/node*; do
  [ -r "$node/cpulist" ] || continue
  echo "$(basename "$node") cpus=$(cat "$node/cpulist")"
done
echo "===== FULL PROCESS TABLE ====="
ps -eo pid=,ppid=,psr=,comm=,args=
echo "===== SGLANG PROCESS AFFINITY ====="
for pid in $(find /proc -maxdepth 1 -type d -name "[0-9]*" -printf "%f\n" | sort -n); do
  [ "$pid" = "$$" ] && continue
  [ -r "/proc/$pid/cmdline" ] || continue
  cmd=$(tr "\000" " " < "/proc/$pid/cmdline")
  case "$cmd" in
    *sglang*|*multiprocessing.spawn*) ;;
    *) continue ;;
  esac
  echo "----- PID=$pid -----"
  ps -o pid=,ppid=,psr=,comm=,args= -p "$pid"
  grep -E "^(Name|Cpus_allowed_list|Mems_allowed_list):" "/proc/$pid/status"
  awk "{count[\$2]++} END {for (policy in count) print \"numa_policy \" count[policy], policy}" "/proc/$pid/numa_maps" 2>/dev/null | sort
done
' > "$output" 2>&1
    log "captured process affinity: $output"
}

run_benchmark() {
    local mode="$1"
    local case_name="$2"
    local input_len="$3"
    local output_len="$4"
    local output_dir="$RESULT_DIR/${case_name}_${mode}"
    local result_json="$output_dir/latency.json"
    local bench_log="$LOG_DIR/bench_${case_name}_${mode}.log"
    local start_ns end_ns rc

    mkdir -p "$output_dir"
    container_exec "python3 '$LATENCY_SCRIPT' \
      --base-url http://127.0.0.1:$PORT --model '$SERVED_MODEL' \
      --input-len $input_len --output-len 32 --n 1 --concurrency 1 \
      --out /tmp/165_affinity_gain_warmup.json --tag warmup" \
        > "$LOG_DIR/warmup_${case_name}_${mode}.log" 2>&1 || return 1

    start_ns="$(date +%s%N)"
    container_exec "python3 '$LATENCY_SCRIPT' \
      --base-url http://127.0.0.1:$PORT --model '$SERVED_MODEL' \
      --input-len $input_len --output-len $output_len \
      --n $REQUESTS --concurrency $CONCURRENCY \
      --out '$result_json' --tag '${case_name}_${mode}'" > "$bench_log" 2>&1
    rc=$?
    end_ns="$(date +%s%N)"
    if [[ "$rc" -ne 0 || ! -f "$result_json" ]]; then
        tail -n 80 "$bench_log" || true
        return 1
    fi

    python3 - "$result_json" "$start_ns" "$end_ns" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
elapsed = (int(sys.argv[3]) - int(sys.argv[2])) / 1_000_000_000
data = json.loads(path.read_text())
aggregate = data["aggregate"]
samples = data.get("samples", [])
total_tokens = sum(sample.get("n_out", 0) for sample in samples if sample.get("ok"))
aggregate["measurement_wall_time_s"] = round(elapsed, 3)
aggregate["aggregate_output_throughput_tok_s"] = round(total_tokens / elapsed, 2)
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
print(
    "n_ok={}/{} ttft_p50={}ms ttft_p95={}ms tpot_p50={}ms "
    "aggregate_output_throughput={}tok/s".format(
        aggregate.get("n_ok"),
        aggregate.get("n_requested"),
        aggregate.get("ttft_p50_ms"),
        aggregate.get("ttft_p95_ms"),
        aggregate.get("tpot_p50_ms"),
        aggregate.get("aggregate_output_throughput_tok_s"),
    )
)
if aggregate.get("n_ok") != aggregate.get("n_requested"):
    raise SystemExit(1)
PY
}

validate_mode() {
    local mode="$1"
    local server_log="$LOG_DIR/server_${mode}.log"

    if grep -Eq '#cached-token: [1-9][0-9]*' "$server_log"; then
        log "FAIL: mode=$mode observed a nonzero cached-token count"
        return 1
    fi
    if [[ "$mode" == off ]]; then
        if grep -q '\[kunpeng-affinity\]' "$server_log"; then
            log "FAIL: off mode contains plugin logs"
            return 1
        fi
        return 0
    fi
    local expected
    for expected in \
        'selected node=6 gpu=0' \
        'selected node=6 gpu=1' \
        'selected node=4 gpu=2' \
        'selected node=4 gpu=3'; do
        if ! grep -q "$expected" "$server_log"; then
            log "FAIL: on mode missing plugin evidence: $expected"
            return 1
        fi
    done
}

compare_results() {
    python3 - "$RESULT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for case_name in ("prefill", "decode"):
    off = json.loads((root / f"{case_name}_off" / "latency.json").read_text())["aggregate"]
    on = json.loads((root / f"{case_name}_on" / "latency.json").read_text())["aggregate"]

    def delta(name):
        before = off[name]
        after = on[name]
        return (after / before - 1) * 100

    print(f"===== {case_name.upper()} OFF / ON COMPARISON =====")
    for name in (
        "ttft_mean_ms",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "tpot_mean_ms",
        "e2e_mean_s",
        "aggregate_output_throughput_tok_s",
    ):
        print(f"{name}: off={off[name]} on={on[name]} delta={delta(name):+.2f}%")
PY
}

log "165 affinity-gain check"
log "prefill=input2048/output128; decode=input128/output2048"
log "requests=$REQUESTS concurrency=$CONCURRENCY modes='$MODES'"
log "radix cache disabled; random seed fixed to 1"

for mode in $MODES; do
    if [[ "$mode" != off && "$mode" != on ]]; then
        log "FAIL: invalid mode=$mode"
        exit 2
    fi
    start_server "$mode" || exit 1
    capture_binding "$mode"
    for case_spec in "${CASES[@]}"; do
        read -r case_name input_len output_len <<< "$case_spec"
        log "benchmark mode=$mode case=$case_name input=$input_len output=$output_len"
        run_benchmark "$mode" "$case_name" "$input_len" "$output_len" || {
            log "FAIL: benchmark mode=$mode case=$case_name"
            exit 1
        }
    done
    validate_mode "$mode" || exit 1
    stop_server
done

if [[ -f "$RESULT_DIR/prefill_off/latency.json" && -f "$RESULT_DIR/prefill_on/latency.json" \
    && -f "$RESULT_DIR/decode_off/latency.json" && -f "$RESULT_DIR/decode_on/latency.json" ]]; then
    compare_results | tee "$RUN_DIR/comparison.txt"
fi

CLEANED=1
log "PASS: completed"
log "results: $RUN_DIR"
