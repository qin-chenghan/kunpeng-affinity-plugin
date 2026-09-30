# Active Validation Assignments

This file contains only validation work that is still open. Completed inventory,
dummy-spawn, TP=1, and Iluvatar fallback tasks have been removed from the active
brief. Do not repeat them except where this brief explicitly requires a fresh
baseline for the authoritative TP=4 result.

Return factual reports with commands, return codes, relevant raw evidence, and
artifact paths. Do not modify plugin or framework source to make a test pass.

## Common Safety Boundary

- Start with a read-only inventory of the host, containers, GPU occupancy,
  listening ports, and existing model services.
- Never stop, restart, kill, or reconfigure a process or container that the test
  Agent did not create for this validation run.
- Real-service and performance stages may run only in a dedicated validation
  container on explicitly allocated idle devices. If isolation is unavailable,
  mark the affected stage `BLOCKED`.
- Do not install drivers, kernel modules, system packages, or unrelated Python
  dependencies. Installing the plugin source in the dedicated validation
  container is allowed.
- Do not edit plugin source, tests, validation scripts, package metadata, vLLM,
  SGLang, or model files. Do not commit or push.
- If the checkout is clean and behind its configured remote branch, update it
  only with `git pull --ff-only`. If it is dirty, do not pull; record the state
  and stop before running a result that could be attributed to the wrong code.
- Keep result artifacts outside the Git checkout. Preserve commands, return
  codes, stdout/stderr logs, process IDs, and machine-readable raw results.
- Redact credentials and sensitive command arguments. Model identity and the
  fact that it is locally available must remain visible in the report; a
  sensitive absolute model path may be redacted.
- Report every stage as `PASS`, `FAIL`, `BLOCKED`, or `SKIP`. A missing log is
  not a pass, and a blocked stage must not be summarized as successful.

## Task 1: Authoritative 184 vLLM Self-Validation

### Objective and Authority

Use the model already available locally on 184:

```text
model: Qwen3-32B
tensor parallel size: 4
framework: vLLM
accelerator: Iluvatar
```

This is the authoritative acceptance scenario for the vLLM side of the
plugin. It is not an additional scenario with the same status as the previous
Qwen3-8B TP=1 run. The new report must use Qwen3-32B TP=4 as the basis of its
main conclusions. The TP=1 report may be cited only as historical supporting
evidence and must not be used to fill missing TP=4 evidence.

The environment exposes four runtime logical devices backed by two physical
Iluvatar cards. Do not call them four physical cards. Confirm the relationship
from current runtime evidence before testing; do not copy the old inventory
without rechecking it.

Use the existing TP=1 validation procedure and request methodology where it is
still applicable, changing the model to local Qwen3-32B and the tensor parallel
size to 4. Do not silently redesign the workload. If a TP=1 parameter cannot be
reused with Qwen3-32B TP=4, document the exact incompatibility and the minimal
change made.

The plugin checkout is expected at:

```text
/home/qch/tools/kunpeng-affinity-plugin
```

Test the current remote commit containing the SGLang Ascend integration
(`de5bd6a`) or a later commit explicitly identified in the report. Do not test
an older checkout and describe it as current.

### Result Directory

Create one unique result directory outside the checkout, for example:

```text
/home/qch/tools/kunpeng-affinity-results/qwen3-32b-tp4-authority-<timestamp>/
```

Use this structure when the corresponding stage runs:

```text
report.md
environment.txt
commands.txt
source-tests/
topology-provider/
hook-dummy/
service-native-control/
service-plugin-forced-generic/
process-correlation/
correctness/
performance/
```

The final response must include the absolute result directory and the path to
`report.md`.

### Stage A: Environment, Model, and Isolation Preflight

Before changing the environment, record:

- exact plugin commit, branch, remote relationship, worktree status, and source
  import location;
- host kernel, architecture, CPU topology, NUMA nodes, online CPUs, process
  affinity, and cgroup CPU/memory restrictions;
- validation container name and image, Python version, complete vLLM version,
  Iluvatar runtime/driver version, `ixsmi`, and `numactl` paths;
- current containers, model services, listening ports, GPU processes, and GPU
  memory occupancy;
- the local Qwen3-32B model identity and enough model metadata to prove that the
  intended model was used, without copying or modifying model files;
- four logical device identities, UUIDs, physical board association, PCI BDFs,
  and whether all required devices are idle and isolated;
- the exact visible-device order selected for TP=4.

Proceed to a real service only when all four selected logical devices and the
container are dedicated to this run. Do not terminate an existing process to
create an idle baseline. If the allocation is not safe, finish the read-only
inventory and mark service stages `BLOCKED`.

### Stage B: Fresh Source and Controlled Baseline

Run the current repository tests and record the measured count rather than
copying a previous result:

```bash
cd /home/qch/tools/kunpeng-affinity-plugin
PYTHON_BIN=python3 ./scripts/test.sh
```

Run the existing Iluvatar validation suite in the dedicated idle container
using a fresh result subdirectory. Preserve evidence for:

- source installation and both framework entry-point metadata records;
- Provider single-device, all-device, and visibility-reorder checks;
- explicit BDF to PCIe path, NUMA node, and target CPU intersection for every
  visible logical device;
- forced-generic dummy spawn with zero native NUMA-query calls;
- controlled native-empty to generic fallback with one native query;
- child CPU affinity and NUMA memory-policy verification.

These checks are prerequisites and regression evidence. They do not replace the
Qwen3-32B TP=4 real-service result.

Configure and run the existing suite through its documented local file:

```bash
cd /home/qch/tools/kunpeng-affinity-plugin
cp validation/iluvatar/config.env.example validation/iluvatar/config.env
```

Set `ENABLE_ENVIRONMENT_CHANGES=1`, a unique absolute `RESULT_DIR`, the actual
Python and `ixsmi` commands, `VISIBILITY_ENV=CUDA_VISIBLE_DEVICES`, and
`KUNPENG_AFFINITY_PROVIDER=iluvatar-runtime-pci`. Populate `AFFINITY_BDFS` from
the current `ixsmi` inventory after normalizing domains for Linux sysfs; do not
copy old BDFs without verifying them. This Git-ignored local configuration is
allowed, but tracked files must remain unchanged. Then run:

```bash
./validation/iluvatar/run.sh --dry-run
./validation/iluvatar/run.sh
```

Record the first failing stage and its individual log. Do not reinterpret a
later skipped stage as passed.

### Stage C: TP=4 Device and Process Contract

Before launching the model, derive and save the expected ordered mapping:

```text
visible logical id
  -> runtime UUID
  -> runtime physical index
  -> physical board
  -> PCI BDF
  -> PCIe parent path
  -> NUMA node
  -> expected CPU intersection
```

Validate the full four-device batch. A missing identity, duplicate UUID/BDF,
unknown NUMA node, empty CPU intersection, or changed visibility fingerprint
invalidates the batch. Never accept a successful subset.

Do not assume the old node layout. If the current inventory still maps logical
devices 0/1 to one NUMA node and 2/3 to another, state that as newly observed
evidence. Also preserve evidence showing that logical devices 0/1 and 2/3 are
two logical devices per physical board.

### Stage D: Real Qwen3-32B TP=4 Service Pair

Run two isolated services sequentially, never concurrently:

```text
run A: native control
  - plugin mode off
  - vLLM --numa-bind enabled
  - vLLM native NUMA behavior retained

run B: authoritative plugin path
  - plugin mode strict
  - KUNPENG_AFFINITY_PROVIDER=iluvatar-runtime-pci
  - KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1
  - vLLM --numa-bind enabled
```

Both runs must use the same:

- local Qwen3-32B model and tokenizer;
- container image and vLLM configuration;
- four logical devices in the same visible order;
- `tensor-parallel-size=4` and unchanged DP size;
- scheduler, memory, quantization, dtype, context-length, and eager/graph
  settings;
- request corpus, seed, concurrency, input/output limits, and warmup policy;
- service-side CPU restrictions other than the binding behavior under test.

Record every difference between the two launch commands. The intended
differences are only the plugin control variables required to select run A or
run B. If another difference is unavoidable, explain why before interpreting
performance.

For each run, preserve:

- complete redacted launch command, environment delta, startup log, readiness
  check, shutdown log, and return code;
- one fixed deterministic request and response proving service availability;
- plugin discovery, Hook installation, decision, Provider, topology, commit,
  and vLLM binding logs where applicable;
- process tree with PID, parent PID, process kind, local rank, and DP local rank;
- clean shutdown of only the service created by this run and proof that no test
  process remains.

Run B is a plugin execution-chain `PASS` only when its logs prove:

```text
vLLM discovers plugin
  -> configure_subprocess Hook is installed
  -> native NUMA query is bypassed
  -> Iluvatar Provider maps all four logical devices to UUID/BDF
  -> Linux topology resolves the complete ordered NUMA list
  -> plugin commits the list atomically
  -> original vLLM numactl path launches the TP=4 processes
```

A healthy service or a generic vLLM binding message without this chain is not
sufficient plugin evidence.

### Stage E: Per-Process Binding Verification

For both runs, correlate every relevant process in one table:

```text
PID -> PPID -> process kind -> TP rank -> local rank/dp_local_rank
    -> logical device -> UUID -> BDF -> physical board -> NUMA node
    -> expected CPUs -> observed Cpus_allowed_list
    -> observed memory policy
```

Use runtime logs and read-only `/proc`/NUMA inspection while the service is
alive. Distinguish EngineCore, TP Workers, API processes, and unrelated
processes. Do not label EngineCore evidence as Worker evidence. If a policy is
inherited, show the parent-child relationship and identify it as inherited.

The TP=4 binding stage passes only if all four TP ranks are accounted for and
each observed Worker policy agrees with its mapped BDF/NUMA result. Verify the
EngineCore policy separately; do not assume it must equal one Worker's CPU set.
If a Worker disappears before inspection or cannot be mapped unambiguously,
mark that part `BLOCKED` rather than inferring the result.

Check CPU affinity and memory policy independently. `Mems_allowed_list` alone
does not prove a `numactl` memory-binding policy.

### Stage F: Inference-Result Consistency

Use the same deterministic request set and ordering for runs A and B. Reuse the
accepted TP=1 method unless the saved assets are unavailable:

- run an explicit warmup set and exclude it from measured comparison;
- run at least three measured rounds;
- use the same number of requests per round on both arms;
- use deterministic generation, a fixed seed, fixed tokenizer, and fixed output
  limit;
- save redacted raw JSONL responses and a machine-readable comparison result.

Compare HTTP status, error/empty/truncated responses, generated token counts,
and exact response text for every measured pair. If exact determinism is not
available, prove and document why before using a normalized comparison.

Call this an inference-result consistency or regression test, not a model
accuracy evaluation. Qwen3-32B results must be compared between run A and run B;
they must not be compared with old Qwen3-8B outputs.

### Stage G: Performance Comparison

After functional and result-consistency stages pass, run the same performance
methodology on run A and run B:

- identical warmup count excluded from measurement;
- at least three independent measured rounds per arm;
- identical request order, concurrency, input/output lengths, and token limits;
- raw per-request records plus per-round and aggregate results;
- affinity and memory-policy evidence sampled during each arm;
- GPU and CPU utilization when an existing read-only collection tool is
  available.

Report at minimum:

- request count and success rate;
- TTFT p50/p95;
- TPOT/ITL p50/p95;
- end-to-end latency p50/p95/p99;
- input and output token throughput;
- aggregate request throughput;
- round-to-round variation;
- absolute and relative differences between run A and run B.

The comparison is between vLLM's native binding control and the plugin's forced
generic path, because both runs keep `--numa-bind` enabled. Do not describe it
as "unbound versus bound." Without a predefined threshold, label performance as
comparison data rather than `PASS` or `FAIL`, and do not guarantee an
improvement from a positive point estimate.

### Required Authoritative Report

Write `report.md` in Chinese with these sections:

1. Executive conclusion and exact tested commit.
2. Authority statement: Qwen3-32B TP=4 is the primary vLLM acceptance case;
   Qwen3-8B TP=1 is historical supporting evidence only.
3. Environment, model, logical-device/physical-board inventory, and isolation
   decision.
4. Stage matrix with command, return code, status, and evidence path.
5. Source, Provider/topology, and controlled Hook regression evidence.
6. TP=4 ordered device mapping and complete PCIe/NUMA table.
7. Real-service plugin execution chain.
8. EngineCore and four-rank Worker process-correlation table.
9. Inference-result consistency result.
10. Performance comparison and limitations.
11. Failures and blockers classified as code, environment, workload, or test
    harness.
12. Confirmed capabilities, unverified capabilities, and recommended next
    action.

The report must clearly separate what is proven by source tests, Provider and
topology probes, controlled dummy spawn, real TP=4 service, result consistency,
and performance data. Do not infer DP, Ray, SGLang, another model, another
device order, or production-wide performance from this case.

Return the completed report and artifacts to the requesting Agent. Do not edit
the repository's existing self-validation document; it will be updated after
the report is reviewed.

## Task 2: Pending Ascend Validation - Keep Until Completed

This is a separate target-environment task and is intentionally retained. Do
not delete or mark it complete based on the 184 Iluvatar result. Do not execute
it on 184 and do not infer Ascend behavior from Iluvatar.

### Current Source Scope to Validate

The current source includes:

- vLLM support for the `ascend-sysfs-pci` Provider;
- Ascend short and long BDF parsing through
  `devdrv_sysfs_bdf_to_devid`;
- supported vLLM base `0.25.1`, including vendor-local versions;
- SGLang Provider selection for explicit `ascend-sysfs-pci` and automatic
  selection when `ASCEND_RT_VISIBLE_DEVICES` is present and no direct runtime
  BDF is available.

These are source-level claims until the corresponding real Ascend environment
tests pass.

### Ascend Safety Boundary

- Begin read-only and inventory the actual Ascend host/container, active TP=8
  or other services, NPU occupancy, ports, and repository state.
- Do not inject the plugin into a shared running service and do not stop or
  reconfigure an existing service.
- Use a dedicated container and isolated devices for Hook, process-binding, or
  real-service validation. If unavailable, run only source/Provider read-only
  checks and mark integration stages `BLOCKED`.
- Do not modify version gates, sysfs data, framework source, or plugin source to
  force a pass.

### Ascend vLLM Validation

On the target vLLM `0.25.1` vendor environment, validate in order:

1. exact plugin commit and clean checkout;
2. source tests and entry-point metadata;
3. `ASCEND_RT_VISIBLE_DEVICES` parsing, including nontrivial ordered visibility;
4. complete logical-device to physical ID to BDF mapping from the real Ascend
   sysfs table;
5. BDF to PCIe path, NUMA node, CPU intersection, and all-or-nothing batch
   result for all selected devices;
6. vLLM Hook installation in an isolated process;
7. forced-generic and native-empty fallback through the original vLLM
   `numactl` execution path;
8. observed child CPU affinity and memory policy;
9. a real isolated service and request only when dedicated resources exist.

The earlier read-only 8/8 Provider mapping result proves mapping only. It does
not prove vLLM Hook installation or real service binding.

### Ascend SGLang Validation

When an isolated SGLang 0.5.18 Ascend environment is available, validate:

1. package discovery through `sglang.srt.plugins`;
2. installation of the around Hook on
   `get_numa_node_if_available`;
3. explicit `ascend-sysfs-pci` and `auto` Provider selection separately;
4. native result precedence and native-empty generic fallback;
5. all visible logical devices mapped through
   `ASCEND_RT_VISIBLE_DEVICES -> sysfs BDF -> Linux NUMA`;
6. original SGLang subprocess/`numactl` launch behavior;
7. Worker CPU affinity and memory policy;
8. one real request for a TP=1 service, then a separate multi-GPU TP case;
9. DP, Ray, or external-launcher paths only as separate tests when they are in
   delivery scope.

Do not reuse the old statement that Ascend is not wired into SGLang; source
commit `de5bd6a` added that path. Real Ascend SGLang lifecycle support remains
unverified until this task produces runtime evidence.

### Ascend Report

Return a separate report containing:

1. exact host/container/runtime/framework/plugin versions and commit;
2. safety and isolation decision;
3. command/status/evidence matrix;
4. logical device to BDF to NUMA mapping table;
5. vLLM source, Hook, child-binding, and real-service results;
6. SGLang source, Hook, child-binding, and real-service results;
7. failures classified as code, environment, or harness;
8. confirmed and unverified boundaries.

Do not merge the Ascend report into the 184 Qwen3-32B TP=4 authority report.
