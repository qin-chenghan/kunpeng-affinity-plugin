#!/usr/bin/env bash
# Temporary helper for the fixed 184 vLLM TP=4 environment.
# This is not a reusable or formal validation test.

set -uo pipefail

: "${TEST_ID:?ERROR: set TEST_ID, for example TEST_ID=tp4_$(date +%Y%m%d_%H%M%S)}"

CONTAINER=corex5-v0.23.0
MODEL_PATH=/home/model/Qwen3-32B-W8A8
SERVED_MODEL=Qwen3-32B-W8A8
GPU_LIST=0,1,2,3
PORT=18080
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="/home/qch/tools/kunpeng-affinity-results/${TEST_ID}"
AFFINITY_BDFS=0000:45:00.0,0000:48:00.0,0000:af:00.0,0000:b2:00.0
ACTIVE_PID_FILE=

mkdir -p "$RESULT_DIR"
SUMMARY="$RESULT_DIR/run-summary.txt"
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

run_service() {
    local mode="$1"
    local mode_dir="$RESULT_DIR/service-$mode"
    local server_script="$mode_dir/server.sh"
    local pid_file="$mode_dir/server.pid"
    local plugin_env

    mkdir -p "$mode_dir"
    : > "$mode_dir/server.log"
    if [[ "$mode" == off ]]; then
        plugin_env='export KUNPENG_AFFINITY_MODE=off'
    else
        mkdir -p "$RESULT_DIR/bootstrap"
        cat > "$RESULT_DIR/bootstrap/sitecustomize.py" <<'PY'
from kunpeng_affinity.vllm_plugin import register
register()
PY
        plugin_env="export PYTHONPATH='$RESULT_DIR/bootstrap:$REPO_ROOT/src'; export KUNPENG_AFFINITY_MODE=strict; export KUNPENG_AFFINITY_PROVIDER=iluvatar-runtime-pci; export KUNPENG_AFFINITY_IXSMI='$IXSMI_BIN'; export KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1"
    fi

    cat > "$server_script" <<EOF
#!/usr/bin/env bash
echo \$\$ > '$pid_file'
export CUDA_VISIBLE_DEVICES='$GPU_LIST'
export VLLM_WORKER_MULTIPROC_METHOD=spawn
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
  --numa-bind > '$mode_dir/server.log' 2>&1
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

    local round
    for round in 0 1 2 3; do
        curl -sf --noproxy '*' \
            -H 'Content-Type: application/json' \
            -X POST "http://127.0.0.1:${PORT}/v1/completions" \
            -d "{\"model\":\"$SERVED_MODEL\",\"prompt\":\"Explain NUMA affinity in one short paragraph.\",\"temperature\":0,\"seed\":1,\"max_tokens\":64}" \
            > "$mode_dir/request_${round}.json" || return 1
    done

    container_exec "vllm bench serve \
      --backend openai --base-url http://127.0.0.1:$PORT \
      --endpoint /v1/completions --model '$SERVED_MODEL' --tokenizer '$MODEL_PATH' \
      --dataset-name random --random-input-len 128 --random-output-len 512 \
      --num-warmups 4 --num-prompts 32 --max-concurrency 8 \
      --save-result --save-detailed --result-dir '$mode_dir' \
      --result-filename bench_${mode}.json" > "$mode_dir/bench.log" 2>&1 || return 1

    capture_processes > "$mode_dir/processes.txt" 2>&1
    stop_service "$pid_file"
    ACTIVE_PID_FILE=
    if [[ "$mode" == on ]]; then
        grep -E "kunpeng-affinity|Binding (EngineCore|worker)" "$mode_dir/server.log" \
            > "$mode_dir/affinity.log" 2>/dev/null || true
    fi
    log "PASS: mode=$mode service, requests and benchmark"
}

log "184 temporary validation start"
log "container=$CONTAINER model=$MODEL_PATH result=$RESULT_DIR"

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
} > "$RESULT_DIR/environment.txt" 2>&1

CONFIG_FILE="$RESULT_DIR/iluvatar.env"
cat > "$CONFIG_FILE" <<EOF
PYTHON_BIN=python3
IXSMI_BIN=$IXSMI_BIN
SYSFS_ROOT=/sys
PROBE_DEVICE=0
VISIBILITY_ENV=CUDA_VISIBLE_DEVICES
REORDER_VISIBLE_DEVICES=1,0
SPAWN_VISIBLE_DEVICES=0
AFFINITY_BDFS=$AFFINITY_BDFS
KUNPENG_AFFINITY_MODE=auto
KUNPENG_AFFINITY_PROVIDER=iluvatar-runtime-pci
KUNPENG_AFFINITY_CPU_POLICY=node
KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL=detail
KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
RUN_UNIT_TESTS=1
RUN_TOPOLOGY_PROBE=1
RUN_PROVIDER_PROBE=1
RUN_REORDER_PROBE=1
RUN_SOURCE_INSTALL=1
RUN_FORCED_GENERIC_SPAWN=1
RUN_AUTO_FALLBACK_SPAWN=1
ENABLE_ENVIRONMENT_CHANGES=1
RESULT_DIR=$RESULT_DIR/iluvatar-suite
EOF

log "run existing Iluvatar 9-stage suite"
if ! container_exec "cd '$REPO_ROOT' && ./validation/iluvatar/run.sh --config '$CONFIG_FILE'" \
    > "$RESULT_DIR/iluvatar-suite.log" 2>&1; then
    log "FAIL: Iluvatar suite"
    exit 1
fi
log "PASS: Iluvatar suite"

run_service off || { log "FAIL: mode=off"; exit 1; }
run_service on || { log "FAIL: mode=on"; exit 1; }

python3 - "$RESULT_DIR" <<'PY' > "$RESULT_DIR/request-comparison.txt"
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for round_id in range(1, 4):
    texts = []
    for mode in ("off", "on"):
        path = root / f"service-{mode}" / f"request_{round_id}.json"
        texts.append(json.loads(path.read_text())["choices"][0]["text"])
    print(
        f"round={round_id} equal={texts[0] == texts[1]} "
        f"sha256={hashlib.sha256(texts[0].encode()).hexdigest()}"
    )
    if texts[0] != texts[1]:
        raise SystemExit(1)
PY
comparison_rc=$?

ON_LOG="$RESULT_DIR/service-on/server.log"
if ! grep -q "installed vLLM subprocess hook" "$ON_LOG"; then
    log "FAIL: plugin Hook was not installed in mode=on"
    exit 1
fi
if ! grep -q "source=generic" "$ON_LOG"; then
    log "BLOCKED: Hook loaded but real service did not reach generic path"
    log "classification=PLUGIN_LOADED_NATIVE_PRESERVED"
    exit 1
fi
if ((comparison_rc != 0)); then
    log "FAIL: off/on deterministic responses differ"
    exit 1
fi

log "PASS: real TP=4 service reached plugin generic path"
log "PASS: all requested stages"
log "result directory: $RESULT_DIR"
