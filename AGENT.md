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

A nonzero exit caused by `PLUGIN_LOADED_NATIVE_PRESERVED` is a valid blocked
result, not permission to change the test conditions.

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
