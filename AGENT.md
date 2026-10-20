# Active Validation Assignments

Return factual reports with commands, return codes, relevant raw evidence, and
artifact paths. Do not modify plugin or framework source to make a test pass.

## Task 1: 184 vLLM Qwen3-32B TP=4 Validation

Use the dedicated 184 environment and the temporary machine-specific helper:

```text
host: 184
container: corex5-v0.23.0
model: /home/model/Qwen3-32B-W8A8
framework: vLLM 0.23.0+corex.5.0.0
tensor parallel size: 4
repository: /home/qch/tools/kunpeng-affinity-plugin
```

This is the authoritative vLLM scenario. The helper under `scripts/` is only a
temporary 184 driver; do not treat it as a reusable or formal repository test.
The environment is dedicated, so no separate idle-resource survey is required.

Run only:

```bash
cd /home/qch/tools/kunpeng-affinity-plugin
git status --short
git pull --ff-only
TEST_ID=vllm_tp4_$(date +%Y%m%d_%H%M%S) bash scripts/184timeline.sh
```

If `git status --short` shows tracked changes, do not pull or test. Report the
state instead. Do not edit source, the helper, vLLM, the model, or the container.
Do not remove `--numa-bind`, clear `numa_bind_nodes`, or otherwise manufacture a
generic-path pass.

The helper runs the existing Iluvatar suite, then fixed TP=4 off/on services,
deterministic requests, a small benchmark, and process-affinity capture. Its on
arm uses a validation-only `sitecustomize.py` because this CoreX vLLM build did
not automatically load the general plugin in the earlier real-service run.

Return a concise Chinese report containing:

1. exact tested commit and absolute result directory;
2. command return code and the final line/classification in `run-summary.txt`;
3. Iluvatar suite result;
4. Hook installation and either real `source=generic` evidence or the exact
   `PLUGIN_LOADED_NATIVE_PRESERVED` blocker;
5. EngineCore and four Worker CPU/memory-policy evidence from both arms;
6. deterministic request comparison and both benchmark summaries;
7. confirmed facts, failures or blockers, and evidence paths.

At minimum preserve and cite these artifacts:

```text
run-summary.txt
environment.txt
iluvatar-suite.log
iluvatar-suite/
service-off/server.log
service-off/processes.txt
service-off/bench.log
service-on/server.log
service-on/affinity.log
service-on/processes.txt
service-on/bench.log
request-comparison.txt
```

## Task 3: 184 vLLM TP=4 Prefill Measurement

### Diagnose the latest missing generic-path evidence first

The 2026-10-12 `prefill1` run confirms `--numa-bind`, strict mode, and forced
generic in the generated startup command. It logged Hook installation and
native EngineCore/Worker binding, but no plugin call-entry or selection
message. The import-only check showed that both vLLM callers reference the
patched `numa_utils` module; the missing evidence was caused by vLLM/CoreX
raising the root logger threshold to `ERROR`. The plugin now installs its own
WARNING-level diagnostic handler during vLLM Hook installation.

After pulling the logging fix, run the complete benchmark directly:

```bash
cd /home/qch/tools/kunpeng-affinity-plugin
git pull --ff-only
TEST_ID=vllm_gain_$(date +%Y%m%d_%H%M%S) bash scripts/184-affinity-gain.sh
```

The expected on-service log must now contain the plugin call-entry and
`source=generic` messages, in addition to the EngineCore and Worker binding
messages. Return the complete report and preserve the result directory. The
temporary `184-hook-check.sh` remains available as an import-only diagnostic if
the complete run still lacks plugin evidence; it does not load a model, start a
service, launch a child, or change an existing inference process.

### Prefill benchmark

Run this as a separate temporary benchmark after Task 1, or when the dedicated
184 environment is available. Keep Task 1 and the pending Ascend task
unchanged. This task uses the same authoritative Qwen3-32B TP=4 service and
measures only the prefill-oriented request shape:

- Random input 2048 tokens, random output 128 tokens.

The `off` and `on` services are started separately. The `off` arm still uses
the same vLLM model, TP, graph, prefill, and memory settings as the `on` arm,
but deliberately omits `--numa-bind`. The `on` arm adds `--numa-bind` to enable
the binding lifecycle and sets `KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1` to select
the plugin's generic discovery path. Neither arm specifies NUMA nodes manually.
It is a temporary plugin-gain measurement, not a comparison against
vLLM's native NUMA implementation.

Both arms must use the same graph configuration: `VLLM_ENFORCE_CUDA_GRAPH=1`,
`--enable-chunked-prefill`, and
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "mode":0}'`.

Run only:

```bash
cd /home/qch/tools/kunpeng-affinity-plugin
git status --short
git pull --ff-only
TEST_ID=vllm_gain_$(date +%Y%m%d_%H%M%S) bash scripts/184-affinity-gain.sh
```

Do not edit the plugin, framework, model, existing `184timeline.sh`, or this
temporary helper to make a benchmark pass. If the command fails, preserve the
result directory and report the first failing stage and its log.

Return a concise Chinese report with the exact commit, command return code,
absolute result directory, Hook/generic-path evidence, EngineCore and four
Worker CPU/memory-policy evidence, and the prefill `off/on` comparison.
Preserve these artifacts:

```text
run-summary.txt
environment.txt
comparison.txt
service-off/server.log
service-off/processes.txt
service-off/bench_prefill.json
service-on/server.log
service-on/affinity.log
service-on/processes.txt
service-on/bench_prefill.json
```

A nonzero exit caused by `PLUGIN_LOADED_GENERIC_NOT_OBSERVED` is a valid blocked
result, not permission to change the test conditions.

## Task 2: Pending Ascend Validation - Keep Until Completed

The 165 prefill helper `scripts/165-affinity-gain.sh` now targets container
`sglang-serve-new` (expected SGLang 0.5.18). Confirm the actual runtime version
in the report; earlier measurements from the old container remain separate.

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

## Task 4: Read-only Environment Inventory for 184 and 165

Collect only the missing environment facts needed by
`docs/test-environment-matrix.md` for hosts 184 and 165. This is an inventory
task, not a plugin or framework test.

Do not modify plugin, framework, container, driver, Runtime, sysfs, or service
configuration. Do not start or stop a service, load a model, run a benchmark,
change device visibility, or inject a Hook into a running process. Do not pull
or change the repository checkout. Do not dump complete process environments;
filter only named plugin/framework variables when needed. Redact credentials,
tokens, and unrelated business configuration from the report.

The report must be a new Markdown artifact outside the source tree, for example:

```text
/home/qch/tools/kunpeng-affinity-results/environment-inventory-184-165-<timestamp>.md
```

The report must contain the exact collection timestamp, host/container, commit
if the repository was inspected, commands, return codes, concise raw evidence,
normalized tables, and an explicit `unknown/not collected` value for every
missing field. Do not infer physical-card count from logical-device count.

### 184 collection

Host facts are already known, but collect the missing device-level evidence for:

```text
0000:45:00.0
0000:48:00.0
0000:af:00.0
0000:b2:00.0
```

For every BDF, collect read-only:

```bash
lspci -D -nnk -s <BDF>
lspci -D -vv -s <BDF> | grep -E 'LnkCap|LnkSta|Physical Slot|NUMA|Kernel driver'
cat /sys/bus/pci/devices/<BDF>/numa_node
readlink -f /sys/bus/pci/devices/<BDF>
```

Also collect:

```bash
lspci -tv
ixsmi -L
ixsmi --query-gpu=index,uuid,pci.bus_id,minor_number,name --format=csv,noheader,nounits
command -v ixsmi
ixsmi --version 2>&1 || true
ls -l /dev/iluvatar* 2>/dev/null || true
```

In the actual CoreX/vLLM container, collect package versions for Python,
PyTorch, vLLM, SGLang if installed, and `kunpeng-affinity`, plus the exact
CoreX/driver version if a read-only Runtime query exposes it. A missing tool or
version is a reportable unknown, not a reason to install anything.

### 165 collection

For the eight known Ascend BDFs:

```text
0000:01:00.0  0000:02:00.0
0000:41:00.0  0000:42:00.0
0000:81:00.0  0000:82:00.0
0000:c1:00.0  0000:c2:00.0
```

For every BDF, collect the same `lspci -D -nnk`, selected `lspci -vv`,
`numa_node`, and resolved sysfs path fields. Also collect:

```bash
lspci -tv
printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "${ASCEND_RT_VISIBLE_DEVICES-<unset>}"
find /sys -name devdrv_sysfs_bdf_to_devid -print -exec sh -c 'echo "--- $1"; cat "$1"' _ {} \;
command -v npu-smi && npu-smi info || true
command -v dcmi-smi && dcmi-smi info || true
ls -l /dev/davinci* 2>/dev/null || true
```

In the actual Ascend/SGLang container, collect Python, PyTorch, torch-npu if
present, vLLM, SGLang, and `kunpeng-affinity` versions. Record the actual
container name and whether it is the `sglang-serve-new` / SGLang 0.5.18 target;
do not infer the version from the container name. Record CANN/driver versions
only from a command or readable version file, and identify the evidence source.

### Report requirements

The normalized report must include:

1. one host/software table for 184 and one for 165;
2. one device table per host with logical/physical identity as actually
   observable, UUID where available, BDF, driver, NUMA node, and link state;
3. the PCIe topology classification supported by `lspci -tv` and sysfs;
4. the exact fields still missing after collection;
5. a separation between static environment facts and transient runtime state;
6. no claim about plugin binding, performance, model accuracy, or service
   compatibility, because this task does not run those validations.

Return the absolute Markdown path and a concise Chinese summary. Preserve the
raw command outputs alongside the report when practical, but do not include
secrets or full unfiltered environment dumps.
