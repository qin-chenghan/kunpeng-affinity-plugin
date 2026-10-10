
#   export TEST_ID=20261009_timeline
#   ./timeline.sh
#
# 必传: TEST_ID
# 可选: CONTAINER / MODEL_PATH / SERVED_MODEL / GPU_LIST / TP_SIZE / PORT /
#       NUM_PROMPTS / MAX_CONCURRENCY / PROFILE_EVERY / CASE_INTERVAL / MODES /
#       SGLANG_LATENCY / LATENCY_N / LATENCY_CONCURRENCY
#
# 与原版 vLLM 脚本(184 timeline.sh)的差异 [适配]:
#   1. 通过 docker exec 调 sglang-serve 容器, 不在宿主直接 python
#   2. 用 ASCEND_RT_VISIBLE_DEVICES (NPU) 而非 CUDA_VISIBLE_DEVICES
#   3. 用 sglang serve 而非 vllm.entrypoints.openai.api_server
#      - --tp-size / --context-length / --mem-fraction-static / --attention-backend ascend
#      - SGLang v0.5.x 无 --profiler-config (server 端不做 profiling)
#   4. 时延采集走 sglang_latency_online.py 打 /v1/completions 流式,
#      不走 vllm bench serve
#   5. server 只启动一次, 不按 case 重启
#      (vLLM 版每 case 重启是为了让 PROFILE_DIRS 生效, SGLang 通过 --out 按 case 指定, 无需重启)
#
# 修复记录 (vs. 上一版):
#   - 清代理: 宿主/容器 http_proxy 会污染 curl 与 requests, 必须 unset
#   - curl --noproxy '*': 双保险
#   - start_server 用 docker exec -d: 避免前台 bash 卡住
#   - run_case 检查 n_ok: 全部请求失败视为 case 失败, 打印 bench 日志尾部
#   - pkill 端口号边界: 避免 --port 8005 误杀 --port 80050
#   - server 日志优先落宿主挂载路径: 等待期间宿主可直接 tail, 免 docker exec 慢
# =============================================================================

set -uo pipefail

# ---- 环境隔离: 本脚本所有 HTTP 都打本地, 不走代理 ----
# (165 宿主/容器环境存在 http_proxy, 会污染 curl 与 python requests)
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# ============ 必传环境变量校验 ============
: "${TEST_ID:?ERROR: 必须设置 TEST_ID}"

# ============ 可选环境变量(带默认值)============
# [适配] 165 上两个容器:
#   vLLM :  CONTAINER=vllm-deepseek
#   SGLang: CONTAINER=sglang-serve
CONTAINER="${CONTAINER:-sglang-serve}"
# [适配] 容器内模型路径 (165 宿主 /home/AgentDemo/Qwen/Qwen3-32B-W8A8 挂到容器 /model/Qwen/Qwen3-32B-W8A8)
MODEL_PATH="${MODEL_PATH:-/model/Qwen/Qwen3-32B-W8A8}"
SERVED_MODEL="${SERVED_MODEL:-Qwen3-32B-W8A8}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"                              # [适配] SGLang 用 --tp-size
PORT="${PORT:-8005}"                                 # [适配] 165 sglang-serve 默认端口
NUM_PROMPTS="${NUM_PROMPTS:-32}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
CONTEXT_LEN="${CONTEXT_LEN:-4096}"                   # [适配] SGLang 用 --context-length
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"                  # [适配] SGLang 用 --mem-fraction-static
# PROFILE_EVERY=1 → 每个 case 都标记 collect=ON
# PROFILE_EVERY=N → 每 N 个 case 标记一次(idx=0, N, 2N, ...)
# [适配] SGLang 版此处只用于日志标记; 时延采集是 bench 侧行为, 实际每次都执行。
PROFILE_EVERY="${PROFILE_EVERY:-1}"
# 每个 case 跑完后等待的间隔(秒),给 worker 资源释放时间
CASE_INTERVAL="${CASE_INTERVAL:-60}"
# off/on 双模式: off=普通启动(基线) / on=插件绑核(KUNPENG_AFFINITY_MODE=auto + SGLANG_AUTO_NUMA_BIND=1)
MODES="${MODES:-off on}"

# [适配] SGLang 时延采集参数 (对应原脚本 vllm bench serve 的 num-prompts/max-concurrency)
SGLANG_LATENCY="${SGLANG_LATENCY:-/home/qch/scripts/sglang_latency_online.py}"
LATENCY_N="${LATENCY_N:-3}"                          # 每 case 重复请求次数(够算 p50/p95)
LATENCY_CONCURRENCY="${LATENCY_CONCURRENCY:-1}"

# ============ 目录结构 ============
BASE_DIR="${BASE_DIR:-/home/qch}"                    # [适配] 165 上脚本/数据固定在 /home/qch
RUN_DIR="${BASE_DIR}/timeline/${TEST_ID}"
LOG_DIR="${RUN_DIR}/logs"
PROFILE_ROOT_DIR="${PROFILE_ROOT_DIR:-${RUN_DIR}/profile}"
PROFILE_DIRS="${PROFILE_DIRS:-${PROFILE_ROOT_DIR}}"
SERVER_LOG="${LOG_DIR}/server.log"
mkdir -p "$LOG_DIR" "$PROFILE_DIRS"
: > "$SERVER_LOG"                                    # 保证宿主有日志文件, 便于判断容器是否挂了

# ============ 测试用例矩阵(跟 vllmBench.sh 一致,9 case)============
# 模型:   Qwen3-32B-W8A8
# 调度:   async
# bs:     8
# num:    32
# input:  {128, 1024, 2048}
# output: {128, 1024, 2048}
#
# 注意: "2048 2048" 时 input+output == CONTEXT_LEN(4096), SGLang 通常会拒;
#       需要时可把 CONTEXT_LEN 提到 8192, 或换更小 output。
# CASES=(
#     "128 128"
#     "128 1024"
#     "128 2048"
#     "1024 128"
#     "1024 1024"
#     "1024 2048"
#     "2048 128"
#     "2048 1024"
#     "2048 2048"
# )

CASES=(
    "128 128"
)

# ============ 工具函数 ============

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# 是否标记某个 case collect=ON(沿用原代码接口名;PROFILE_EVERY 控制采样间隔)
should_profile_instance() {
    local idx="$1"
    [ $((idx % PROFILE_EVERY)) -eq 0 ]
}

# [适配] 在 sglang-serve 容器内执行一段命令(统一入口)
# 关键: 显式 unset 容器内的 http_proxy, 否则容器 profile 里的代理会污染
#       curl / python requests, 导致访问 127.0.0.1 走代理而失败。
container_exec() {
    if [ -z "$CONTAINER" ]; then
        log "ERROR: 未设置 CONTAINER (docker 容器名), 请先填!"
        return 2
    fi
    docker exec \
        -e no_proxy="127.0.0.1,localhost" \
        -e NO_PROXY="127.0.0.1,localhost" \
        "$CONTAINER" bash -lc "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; $1"
}

# [适配] 探针: 容器能否写宿主的 LOG_DIR(即 /home/qch 是否已挂载)
SERVER_LOG_HOST_VISIBLE=0
probe_host_mount() {
    if container_exec "test -d '$LOG_DIR' && test -w '$LOG_DIR'" >/dev/null 2>&1; then
        SERVER_LOG_HOST_VISIBLE=1
        log "容器可写宿主 $LOG_DIR, server 日志将直写此路径"
    else
        SERVER_LOG_HOST_VISIBLE=0
        log "警告: 容器不能写宿主 $LOG_DIR(可能未挂载 /home/qch),"
        log "      server 日志只保留在容器内 /tmp/timeline_serve.log,"
        log "      等待期间会通过 docker exec tail 拉取(每 60s 一次, 略慢)"
    fi
}

# 读取 server 日志尾部 3 行(优先宿主直读, 否则 docker exec)
tail_server_log() {
    if [ "$SERVER_LOG_HOST_VISIBLE" -eq 1 ] && [ -s "$SERVER_LOG" ]; then
        tail -n 3 "$SERVER_LOG" 2>/dev/null
    else
        container_exec "tail -n 3 /tmp/timeline_serve.log 2>/dev/null" 2>/dev/null || true
    fi
}

wait_for_server() {
    local port="$1"
    local max_wait=600
    local elapsed=0
    while [ "$elapsed" -lt "$max_wait" ]; do
        # --noproxy '*': 即使 no_proxy 未生效, 也强制绕过代理
        if curl -sf --noproxy '*' "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            log "端口 $port 已就绪 (${elapsed}s)"
            return 0
        fi
        if [ $((elapsed % 60)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
            log "  ... 仍在等待 server (${elapsed}s); server 日志尾部:"
            tail_server_log | sed 's/^/    /'
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "ERROR: 端口 $port 超时未就绪! 请查看 $SERVER_LOG"
    tail_server_log >&2 || true
    return 1
}

start_server() {
    local mode="${1:-off}"
    # 每个 mode 独立 server 日志, 避免 off/on 互相覆盖
    SERVER_LOG="${LOG_DIR}/server_${mode}.log"
    log "启动 SGLang 服务(mode=$mode) | container=$CONTAINER | model=$MODEL_PATH | gpu=$GPU_LIST | port=$PORT | tp=$TP_SIZE"
    # on 轮: 注入插件绑核 env(注意用 auto, 不是 on,它是非法 PluginMode)
    local bind_env=""
    if [ "$mode" = "on" ]; then
        bind_env="export KUNPENG_AFFINITY_MODE=auto; export KUNPENG_AFFINITY_PROVIDER=auto; export SGLANG_AUTO_NUMA_BIND=1"
    fi

    # 启动前清残留(端口号边界匹配, 避免 --port 8005 误杀 --port 80050)
    container_exec "pkill -f -- 'sglang serve.*--port[ ]$PORT([ ]|\$)' 2>/dev/null; sleep 3; true" || true

    # 把启动命令写进容器内脚本, 避免 bash -lc 多层引号嵌套
    container_exec "cat > /tmp/timeline_serve.sh <<'EOS'
#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=$GPU_LIST
export HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=1800 SOC_VERSION=ascend910b1 OMP_NUM_THREADS=4
export SGLANG_OPT_FUSE_WQA_WKV=0 SGLANG_OPT_FP8_WO_A_GEMM=0
${bind_env}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy=127.0.0.1,localhost
exec sglang serve '$MODEL_PATH' \\
  --served-model-name '$SERVED_MODEL' \\
  --tp-size $TP_SIZE \\
  --host 127.0.0.1 --port $PORT \\
  --trust-remote-code \\
  --context-length $CONTEXT_LEN \\
  --mem-fraction-static $GPU_MEM_UTIL \\
  --max-running-requests $MAX_CONCURRENCY \\
  --attention-backend ascend
EOS
chmod +x /tmp/timeline_serve.sh"

    # [关键] -d: detach, docker exec 立刻返回, 不会持有前台管道导致脚本卡住
    # 日志优先写到宿主挂载路径, 便于宿主直接 tail; 否则回退到容器内 /tmp
    if [ "$SERVER_LOG_HOST_VISIBLE" -eq 1 ]; then
        docker exec -d "$CONTAINER" bash -lc "bash /tmp/timeline_serve.sh > '$SERVER_LOG' 2>&1"
    else
        docker exec -d "$CONTAINER" bash -lc "bash /tmp/timeline_serve.sh > /tmp/timeline_serve.log 2>&1"
    fi

    log "服务已在容器内后台启动; 等待端口 $PORT 就绪..."
    if ! wait_for_server "$PORT"; then
        log "ERROR: server 未就绪 (端口 $PORT)"
        return 1
    fi
    log "服务启动完成; 日志: $SERVER_LOG"
}

stop_server() {
    # [适配] 165 用 docker exec pkill 清端口进程(端口号边界匹配)
    container_exec "pkill -f -- 'sglang serve.*--port[ ]$PORT([ ]|\$)' 2>/dev/null; sleep 5; pkill -9 -f -- 'sglang serve.*--port[ ]$PORT([ ]|\$)' 2>/dev/null; true" || true
    sleep 3
}

run_case() {
    local case_idx="$1"
    local input_len="$2"
    local output_len="$3"
    local mode="${4:-off}"
    local bench_log="${LOG_DIR}/bench_instance${case_idx}_${mode}.log"
    # [适配] SGLang 时延 JSON 输出到本 case 的 PROFILE_DIRS 下(依赖容器挂了 /home/qch)
    local latency_out="${PROFILE_DIRS}/latency.json"

    local profile_state
    if should_profile_instance "$case_idx"; then
        profile_state="ON"
    else
        profile_state="OFF"
    fi
    log "==== case $case_idx (mode=$mode): in=$input_len, out=$output_len (collect=$profile_state) ===="

    # [适配] SGLang: 在线时延采集(打已起 server, 不二次加载模型)
    container_exec "
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=$GPU_LIST
python3 $SGLANG_LATENCY \
  --base-url http://127.0.0.1:$PORT \
  --model '$SERVED_MODEL' \
  --input-len $input_len --output-len $output_len \
  --n $LATENCY_N --concurrency $LATENCY_CONCURRENCY \
  --out '$latency_out' --tag 'case${case_idx}_${mode}'" > "$bench_log" 2>&1

    local rc=$?

    # 解析 latency.json: 打印摘要 + 检查 n_ok
    if [ -f "$latency_out" ]; then
        python3 - "$latency_out" <<'PY'
import sys, json
try:
    with open(sys.argv[1]) as f:
        d = json.load(f).get("aggregate", {})
    n_ok = d.get("n_ok") or 0
    n_req = d.get("n_requested") or 0
    print("  n_ok={}/{} ttft_p50={}ms tpot_p50={}ms e2e_mean={}s thr={}tok/s".format(
        n_ok, n_req,
        d.get("ttft_p50_ms"), d.get("tpot_p50_ms"),
        d.get("e2e_mean_s"), d.get("throughput_tok_s")))
    sys.exit(0 if (n_req > 0 and n_ok > 0) else 1)
except SystemExit:
    raise
except Exception as e:
    print(f"  [warn] 解析 latency.json 失败: {e}")
    sys.exit(2)
PY
        local check_rc=$?
        if [ "$check_rc" -ne 0 ]; then
            log "  [ERROR] case $case_idx 请求全部失败 (或 JSON 解析异常), bench 日志尾部:"
            tail -n 40 "$bench_log" >&2 2>/dev/null || true
            return 1
        fi
    elif [ "$rc" -ne 0 ]; then
        log "  [ERROR] latency 脚本退出码 $rc, bench 日志尾部:"
        tail -n 40 "$bench_log" >&2 2>/dev/null || true
        return 1
    fi

    return 0
}

cleanup() {
    # CLEANED 标志防重复执行(主流程显式调 + trap EXIT 兜底,可能触发两次)
    if [ "${CLEANED:-0}" -eq 0 ]; then
        CLEANED=1
        log "清理资源..."
        stop_server
    fi
}
trap cleanup EXIT

# ============ 主流程 ============

log "=========================================="
log "TEST_ID:           $TEST_ID"
log "CONTAINER:         $CONTAINER"
log "MODEL_PATH:        $MODEL_PATH (容器内)"
log "SERVED_MODEL:      $SERVED_MODEL"
log "GPU_LIST:          $GPU_LIST  TP_SIZE: $TP_SIZE"
log "PORT:              $PORT"
log "NUM_PROMPTS:       $NUM_PROMPTS"
log "MAX_CONCURRENCY:   $MAX_CONCURRENCY"
log "PROFILE_EVERY:     $PROFILE_EVERY (1=全采集, N=每 N 个 case 一次)"
log "RUN_DIR:           $RUN_DIR"
log "PROFILE_ROOT_DIR:  $PROFILE_ROOT_DIR"
log "=========================================="
log "测试用例 ($((${#CASES[@]})) 个):"
for case_idx in "${!CASES[@]}"; do
    read -r input_len output_len <<< "${CASES[$case_idx]}"
    log "  case $case_idx: in=$input_len out=$output_len"
done

# 探测容器是否能写宿主挂载目录(决定 server 日志落点)
probe_host_mount

FAILED_CASES=()
log "MODES: $MODES"

for mode in ${MODES}; do
    log "=========================="
    log ">>> 模式: $mode <<<"
    log "=========================="

    if ! start_server "$mode"; then
        log "FATAL: server 启动失败(mode=$mode), 跳过该模式"
        FAILED_CASES+=("mode_${mode}_server_start")
        continue
    fi

    for idx in "${!CASES[@]}"; do
        read -r input_len output_len <<< "${CASES[$idx]}"

        # 每个 case 自己的时延输出目录(按 case x mode 分开)
        PROFILE_DIRS="${PROFILE_ROOT_DIR}/case${idx}_in${input_len}_out${output_len}_${mode}"
        mkdir -p "$PROFILE_DIRS"
        container_exec "mkdir -p '$PROFILE_DIRS'" >/dev/null 2>&1 || true

        if ! run_case "$idx" "$input_len" "$output_len" "$mode"; then
            FAILED_CASES+=("in${input_len}_out${output_len}_${mode}")
        fi

        if [ "$idx" -lt "$(( ${#CASES[@]} - 1 ))" ]; then
            log "等待 ${CASE_INTERVAL}s..."
            sleep "$CASE_INTERVAL"
        fi
    done

    # 该模式跑完,停 server 释放资源再切下一个模式
    log "模式 $mode 完成, 停 server..."
    stop_server
    sleep "$CASE_INTERVAL"
done

# off/on 对比摘要
python3 - "$PROFILE_ROOT_DIR" <<'PY'
import sys, json, os, glob
root = sys.argv[1]
by = {}
for d in sorted(glob.glob(os.path.join(root, "case*"))):
    if not os.path.isdir(d):
        continue
    base = d.rsplit("_", 1)[0]
    mode = os.path.basename(d).rsplit("_", 1)[-1]
    if mode not in ("off", "on"):
        continue
    p = os.path.join(d, "latency.json")
    if not os.path.exists(p):
        continue
    try:
        a = json.load(open(p)).get("aggregate", {})
    except Exception:
        continue
    by.setdefault(base, {})[mode] = a
print()
print("===== off/on 时延对比 =====")
def pct(o, n):
    if o is None or n is None or o == 0:
        return "--"
    return "%+.1f%%" % ((n / o - 1) * 100)
for base in sorted(by):
    a_off = by[base].get("off", {})
    a_on = by[base].get("on", {})
    print(base)
    print("  TTFT p50: off=%s ms  on=%s ms  (%s)" % (
        a_off.get("ttft_p50_ms"), a_on.get("ttft_p50_ms"), pct(a_off.get("ttft_p50_ms"), a_on.get("ttft_p50_ms"))))
    print("  TPOT p50: off=%s ms  on=%s ms  (%s)" % (
        a_off.get("tpot_p50_ms"), a_on.get("tpot_p50_ms"), pct(a_off.get("tpot_p50_ms"), a_on.get("tpot_p50_ms"))))
    print("  吞吐:     off=%s tok/s  on=%s tok/s  (%s)" % (
        a_off.get("throughput_tok_s"), a_on.get("throughput_tok_s"), pct(a_off.get("throughput_tok_s"), a_on.get("throughput_tok_s"))))
PY

# 测试完成,显式调用 cleanup 清理 server
log "测试完成,调用 cleanup 清理资源..."
cleanup

# 列出产物
echo ""
echo "===== 产物 ====="
find "$PROFILE_ROOT_DIR" -type f 2>/dev/null | head -50
echo ""
if [ ${#FAILED_CASES[@]} -gt 0 ]; then
    log "WARN: 以下 case 跑失败: ${FAILED_CASES[*]}"
fi
echo "全部完成。日志: $LOG_DIR"
echo "产物目录: $PROFILE_ROOT_DIR"
