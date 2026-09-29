# Iluvatar Validation Instructions

This file is a temporary test brief for the Agent running in the Iluvatar
environment. It is not a request to modify the plugin. The final response
must be a factual validation report with commands, return codes, relevant raw
errors, and evidence paths.

## Objective

Validate the latest checkout in this order:

```text
source baseline
  -> Linux PCIe/NUMA topology
  -> Iluvatar Runtime Provider
  -> logical-device visibility reorder
  -> vLLM entry-point installation
  -> vLLM Hook installation
  -> forced-generic dummy spawn
  -> native-to-generic fallback dummy spawn
```

The first four stages validate the Provider and topology core. The last four
stages validate framework integration. Do not merge these conclusions:
Provider success is not plugin binding success.

## Safety and scope

- Do not edit, reformat, or generate files inside the Git checkout.
- Do not commit or push.
- Do not start a model server.
- Do not stop, restart, or reconfigure any existing container or GPU job.
- Before any environment-changing stage, confirm that the selected container
  is a dedicated test container and that no GPU workload is running.
- Do not install drivers, kernel modules, system packages, or Python
  dependencies.
- An editable source install is permitted only in the dedicated test container
  after recording the current Python environment.
- Do not modify plugin source or package metadata to bypass a version check.
- Do not rename or delete existing files, logs, caches, or temporary utilities.
- If a command needs a temporary `ixsmi` wrapper or library path, record the
  exact source and destination. Treat that as a validation prerequisite, not
  as proof that a clean image contains the utility.
- Preserve all failed command output in the result directory.

## 1. Repository and environment

The checkout is expected at:

```text
/home/qch/tools/kunpeng-affinity-plugin
```

Run in the container that has vLLM, `ixsmi`, `/sys`, and `numactl`. First
record, without changing anything:

```bash
cd /home/qch/tools/kunpeng-affinity-plugin
git status --short --branch
git log -3 --oneline --decorate
uname -a
command -v python3
python3 -c 'import sys; print(sys.executable); print(sys.version)'
python3 -c 'import importlib.metadata as m; print(m.version("vllm")); print(m.distribution("vllm").locate_file(""))'
command -v ixsmi || true
command -v numactl || true
ixsmi -L || true
```

If the checkout is behind the requested branch and the worktree is clean, use
`git pull --ff-only`. If it is dirty, do not pull and report the state. The
report must include the exact commit tested.

Record current GPU occupancy. Do not infer idleness from the absence of a
Python process alone; inspect the vendor tool output and the container list.

For the manual commands below, resolve the actual tools in the selected
container instead of assuming a shell variable from `config.env` is exported:

```bash
PYTHON_BIN="$(command -v python3)"
IXSMI_BIN="$(command -v ixsmi)"
```

The version string is important. Record both the complete version and its
base version. A value such as `0.23.0+vendor.suffix` is not the same string as
`0.23.0`, but it may represent the same upstream base version. Do not change
the code to decide this; report whether the current plugin accepts or rejects
the complete value and quote the exact log.

## 2. Configure the validation suite

Use the committed template and a Git-ignored local configuration:

```bash
cd /home/qch/tools/kunpeng-affinity-plugin
cp validation/iluvatar/config.env.example validation/iluvatar/config.env
```

Set the following values in the local file:

```text
ENABLE_ENVIRONMENT_CHANGES=1
RESULT_DIR=/home/qch/tools/kunpeng-affinity-results/<unique-run-name>
PYTHON_BIN=python3
IXSMI_BIN=ixsmi
VISIBILITY_ENV=CUDA_VISIBLE_DEVICES
KUNPENG_AFFINITY_PROVIDER=iluvatar-runtime-pci
AFFINITY_BDFS=0000:45:00.0,0000:48:00.0,0000:af:00.0,0000:b2:00.0
```

Use a unique result directory for this run. The runner prints the normalized
absolute result directory before starting and again in the final summary. Do
not use a result directory inside the Git checkout.

First inspect the selected stages without executing them:

```bash
./validation/iluvatar/run.sh --dry-run
```

## 3. Run the one-command suite once

Run the suite exactly as configured:

```bash
./validation/iluvatar/run.sh
```

Record the first failing stage and its log. Do not reinterpret a failure as a
later-stage result. In particular, on a heterogeneous host the automatic
`probe-host.sh` candidate scan may include display or network PCI functions.
That is a candidate-discovery limitation, not automatically a Provider
failure. Continue with the explicit-BDF procedure below so the target GPU
topology is still tested.

## 4. Explicit target-BDF topology check

Obtain the Iluvatar device inventory without guessing its order:

```bash
ixsmi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader,nounits
```

Normalize each reported PCI domain to the Linux sysfs form if necessary, for
example `00000000:45:00.0` to `0000:45:00.0`. Verify each BDF exists below
`/sys/bus/pci/devices/`, then run the topology probe with only the Iluvatar
BDFs:

```bash
AFFINITY_BDFS=<comma-separated-iluvatar-bdfs> \
  ./scripts/probe-host.sh
```

For every target GPU, preserve the output showing:

- canonical BDF;
- complete PCIe parent path;
- any PCIe Switch ancestors;
- NUMA evidence and resolved node;
- online CPU set;
- current process allowed CPU set;
- target CPU intersection;
- `status` and `bindable`.

The automatic candidate scan and this explicit target-BDF run must be reported
separately. Do not call the automatic scan failure a target-GPU topology
failure when the explicit run succeeds.

The current validation accepts vLLM `0.23.0`, `0.25.1`, and `0.26.0`, plus
vendor-local builds whose PEP 440 version is based on one of these versions,
such as `0.25.1+ascend.8.0`. An upstream post-release or development version
is still outside the validated contract. If a local build based on one of
these three versions is rejected, report it as a regression in the version
gate rather than changing the source to bypass the check.

## 5. Provider and visibility checks

Run these commands if the one-command suite stopped before them, or use the
stage logs when it reached them:

```bash
cd /home/qch/tools/kunpeng-affinity-plugin
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" demo/iluvatar_provider_probe.py \
  --ixsmi "$IXSMI_BIN" --sysfs-root /sys --device 0

PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" demo/iluvatar_provider_probe.py \
  --ixsmi "$IXSMI_BIN" --sysfs-root /sys --json
```

The all-device result must be checked as a batch:

- every visible logical device has a runtime UUID;
- UUIDs are unique;
- BDFs are unique;
- UUID-to-BDF association is complete;
- every topology result is bindable;
- no partial result is treated as successful.

If at least two devices are visible, the one-command suite should run the
reorder check. The runner uses a marker-based numeric capture so vLLM startup
logs on stdout cannot make a four-device result look like fewer than two. The
manual equivalent is:

```bash
CUDA_VISIBLE_DEVICES=1,0 \
  PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" demo/iluvatar_provider_probe.py \
  --ixsmi "$IXSMI_BIN" --sysfs-root /sys --json
```

Compare the default and reordered JSON. The logical ID may change, but the
UUID and its BDF must remain associated. Do not assume
`ILUVATAR_VISIBLE_DEVICES` works unless the runtime demonstrates it.

## 6. Entry point and controlled plugin checks

Only run installation and spawn checks after confirming the dedicated test
container is idle. The suite performs:

```bash
PYTHON_BIN="$PYTHON_BIN" ./scripts/install-source.sh
PYTHON_BIN="$PYTHON_BIN" ./scripts/verify-source.sh
```

`verify-source.sh` proves installed metadata and source import location. It
does not by itself prove that vLLM installed the Hook. The Hook check is in the
spawn diagnostic:

```bash
CUDA_VISIBLE_DEVICES=0 \
KUNPENG_AFFINITY_PROVIDER=iluvatar-runtime-pci \
KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1 \
PYTHON_BIN="$PYTHON_BIN" \
./scripts/verify-vllm-generic-spawn.sh
```

The forced-generic run is successful only if all of these are true:

- vLLM's plugin loader finds the entry point;
- the Hook marker is installed;
- native GPU NUMA discovery is not called;
- the Iluvatar Provider maps the visible device to a BDF;
- a NUMA node is committed to the vLLM config;
- the original vLLM subprocess context is entered;
- the dummy child starts and exits;
- child `Cpus_allowed_list` equals the expected NUMA CPU intersection;
- `numactl --show` reports the expected bind policy and memory node;
- no diagnostic child process remains.

Then run the controlled native-to-generic fallback check:

```bash
CUDA_VISIBLE_DEVICES=0 \
KUNPENG_AFFINITY_PROVIDER=iluvatar-runtime-pci \
KUNPENG_AFFINITY_VERIFY_PATH=auto-fallback \
PYTHON_BIN="$PYTHON_BIN" \
./scripts/verify-vllm-generic-spawn.sh
```

The expected fallback evidence is one controlled native-query call followed
by a generic result and the same child CPU/memory checks.

If a version outside the validated base versions is rejected before installing
the Hook, classify both spawn stages as `BLOCKED` at version gate. Do not call
this a Provider or topology failure, and do not bypass the gate by editing
source or package metadata. Preserve the exact error. A
vendor-local builds based on `0.23.0`, `0.25.1`, or `0.26.0` are expected to
pass this gate; rejection is a regression.

## 7. Report format

Return one report with these sections:

1. Environment and exact commit.
2. Configuration source and normalized result directory.
3. One-command suite summary, including the first stopping stage.
4. Explicit-BDF topology results and raw PCIe paths.
5. Provider device table for default and reordered visibility.
6. Entry-point result versus actual Hook-install result.
7. Forced-generic and auto-fallback spawn results.
8. Failures, blockers, and whether each is code, environment, or test-script behavior.
9. Confirmed capabilities.
10. Capabilities that remain unverified.

For every stage include command, return code, result (`PASS`, `FAIL`,
`BLOCKED`, or `SKIP`), and log path. Do not summarize a skipped or blocked
stage as passed.

The strongest possible conclusion from this procedure is one of:

- Provider/topology validation failed;
- Provider/topology passed, but vLLM Hook or binding validation is blocked;
- Provider and controlled vLLM dummy-spawn binding both passed, so real single-GPU
  service validation may begin.

Do not claim production support, real model-service support, TP/DP support, or
SGLang lifecycle support from this test alone.

## 8. Next task: inventory existing real-workload tests

Before starting any real model-service or performance run, inspect the current
184 environment and report which test assets already exist. This task is
read-only. It is an inventory task, not a request to start a server or run a
benchmark.

### Safety boundary

- Do not edit, reformat, or generate files inside the Git checkout.
- Do not start, stop, restart, or reconfigure any model server, container, GPU
  workload, or background process.
- Do not kill processes, even if they appear idle or unrelated.
- Do not install packages, drivers, Python dependencies, or benchmark tools.
- Do not run a performance benchmark or send inference requests.
- Do not print passwords, tokens, SSH details, IP addresses, or complete
  command lines containing credentials.
- Redact model paths, user data, and other sensitive values when they are not
  needed to identify a reusable test asset.
- Preserve the current checkout and report its exact commit and worktree state.

### Read-only inventory

Inspect, using commands appropriate for the environment:

1. Existing vLLM/SGLang test scripts, benchmark scripts, launch scripts,
   prompt files, request datasets, and result files under the user's working
   directories and the selected test container.
2. Existing model-serving commands and parameters, including model identity,
   TP/DP size, visible devices, concurrency, input/output length controls,
   quantization, and any CPU or NUMA constraints. Report sensitive paths in a
   redacted form.
3. Available benchmark tools and their versions, such as vLLM benchmark
   commands, `genai-perf`, `lm-evaluation-harness`, or local project tools.
   Check availability and versions only; do not execute a workload.
4. Existing baseline results and their measurement fields. Identify whether
   they contain TTFT, TPOT/ITL, end-to-end latency, throughput, GPU
   utilization, CPU utilization, or NUMA/memory-policy observations.
5. Current container/image information needed to reproduce a test, without
   changing container state. Include framework version, Python version,
   accelerator runtime version, and whether a test container is available.
6. Whether there is an approved idle window and isolated GPU allocation for a
   future TP=1 service test. Do not create or reserve one during this task.

Use read-only commands such as `find`, `grep`, `sed`, `command -v`, version
queries, `docker ps`, and `docker inspect` where available. On systems without
`rg`, use `grep` and `find`. Do not infer that a test is reusable merely from
its filename; inspect its parameters and result format.

### Classification required in the report

Classify every discovered asset into one of these categories:

- `FUNCTIONAL`: suitable for checking that a real service starts and answers a
  request;
- `EXECUTION-CHAIN`: suitable for checking real Worker/EngineCore Hook,
  rank-to-device mapping, CPU affinity, and NUMA memory policy;
- `PERFORMANCE-BASELINE`: suitable for comparing plugin-off and plugin-on
  behavior under the same workload;
- `REUSABLE-WITH-CHANGES`: useful but missing a parameter, metric, isolation
  condition, or reproducibility detail;
- `NOT-REUSABLE`: unrelated, incomplete, or unsafe for this validation.

Keep the three conclusions separate:

```text
existing business workload
  -> candidate for performance comparison

plugin-owned validation script
  -> required for execution-chain correctness

real service request
  -> required before claiming model-serving support
```

### Required report format

Return a factual report with:

1. Environment and exact plugin commit inspected.
2. Existing test assets, with redacted paths, commands, framework versions,
   and classification.
3. Existing baseline result files and the metrics they contain.
4. TP=1 and TP=4 coverage, if present; explicitly state what is absent.
5. Candidate assets for future functional, execution-chain, and performance
   validation.
6. Missing information or blockers.
7. A minimal recommended next test matrix, without running it.
8. Commands used and return codes for all meaningful checks.

The report must explicitly answer:

- Can an existing 184 workload be reused for performance comparison?
- Is there an existing real-service functional test?
- Is there an existing test that externally verifies Worker CPU affinity and
  NUMA memory policy?
- Which tests must be supplied by this plugin repository?
- What must be fixed or controlled before a TP=1 run?

## 9. Requested regression after the latest plugin update

Run this regression after updating to the commit that contains the Ascend
short-BDF parser fix and the vLLM `0.25.1` support entry.

### Required safety boundary

- Do not edit plugin source, tests, configuration templates, or package
  metadata to make a stage pass.
- Do not start, stop, restart, or reconfigure an existing model service.
- Do not run the full environment-changing suite while a GPU workload shares
  the selected container. In that case run `./validation/iluvatar/run.sh
  --read-only` and report the dummy-spawn stages as `BLOCKED` or `SKIP`.
- Only run editable installation and dummy-spawn stages after confirming the
  container is dedicated and all target GPUs are idle.

### Regression steps

1. Record the exact commit, worktree state, Python version, complete vLLM
   version, base version, `ixsmi` path, and GPU occupancy.
2. Run the source tests with the environment's Python 3.10+ interpreter:

   ```bash
   PYTHON_BIN=python3 ./scripts/test.sh
   ```

   If `python3` is not 3.10+, resolve the available interpreter and report the
   exact replacement command. Do not install Python.
3. Run the Iluvatar Provider single-device, all-device, and visibility-reorder
   probes. Confirm UUID-to-BDF association remains stable under reordered
   visibility and that every result is bindable.
4. If the actual environment contains vLLM `0.25.1` or a vendor-local build
   based on it, run the source-install, entry-point, forced-generic
   dummy-spawn, and automatic-fallback dummy-spawn stages in the dedicated
   idle container. Confirm the Hook is installed, the generic path commits the
   expected NUMA node, the child CPU set matches the expected CPUs, and the
   memory policy is correct.
5. If the actual environment does not contain vLLM `0.25.1`, do not fabricate
   a version or modify the version gate. Run the available supported-version
   validation and explicitly report that `0.25.1` runtime integration remains
   unverified.

### Required regression report

Return command, return code, result status, and log path for every stage. The
report must separately state:

- whether the three new parser tests pass;
- whether the actual vLLM version is one of the supported bases;
- whether `0.25.1` was tested in a real vLLM environment or only by source
  contract tests;
- whether any existing GPU service was observed and left untouched;
- whether the result proves only Provider/topology behavior or also proves
  controlled vLLM Hook and child-binding behavior.

## 10. Completed Iluvatar auto-fallback regression

The focused Iluvatar auto-fallback regression has been re-run after the fix
following `7e7498a` and is now **PASS** in the dedicated vLLM `0.23.0+corex`
environment. Do not report this stage as pending or repeat it unless a later
change touches the vLLM decision path.

The defect was that an empty native `get_auto_numa_nodes()` result was
preserved when the platform exposed a BDF API, even though vLLM's original
subprocess context then failed to resolve a NUMA node. The fixed behavior is:

```text
valid native node list -> preserve the native vLLM path
empty native result    -> run the generic Provider and Linux topology path
native query exception -> propagate the original error
```

The accepted evidence includes:

- the complete validation suite passing;
- `native_numa_query_calls=1`;
- `generic_fallback_verified=true`;
- generic NUMA node selection;
- expected and observed child CPU sets;
- memory-policy verification;
- forced-generic still passing;
- no model service started and no unrelated GPU process changed.

## 11. Remaining validation matrix

The following items remain open. A test report must distinguish real runtime
evidence from source-level contract tests and must not infer one framework's
result from another framework.

### Required vLLM validation

1. **vLLM 0.25.1 real Hook integration**

   Ascend 0.25.1 Provider mapping has passed, but the complete plugin path has
   not been proven in an isolated environment:

   ```text
   entry point
     -> configure_subprocess Hook
     -> Ascend Provider
     -> generic NUMA fallback
     -> original numactl wrapper
     -> dummy Worker CPU and memory policy
   ```

   Use a dedicated container or isolated GPUs. Do not inject the plugin into a
   shared TP=8 production container.

2. **Real single-GPU vLLM service**

   Start an isolated TP=1 service with the plugin disabled and enabled. Send a
   real request in both cases, then collect:

   - service startup result;
   - request result;
   - Worker CPU affinity;
   - NUMA memory policy;
   - plugin decision logs.

   Dummy spawn success alone does not prove model-service support.

3. **vLLM multi-process and parallel execution**

   In an isolated environment, cover at least one multi-GPU TP case and the
   relevant EngineCore/Worker path. If DP is in scope, also cover a DP shard.
   Verify:

   - each logical device maps to the intended BDF;
   - each Worker receives the correct NUMA node and CPU set;
   - EngineCore receives the intended CPU superset;
   - visibility fingerprints survive process creation and revalidation;
   - a partial or changed mapping does not commit a partial result.

### Required SGLang validation

4. **SGLang 0.5.18 real single-GPU service**

   Source-level SGLang tests pass, but no real SGLang service lifecycle has
   been accepted yet. Validate plugin discovery, the NUMA query Hook, native
   empty-result fallback, original process launch, Worker CPU affinity, memory
   policy, and one real request.

5. **SGLang multi-GPU and launcher coverage**

   After TP=1 passes, validate the SGLang multi-GPU/TP path and DP path if they
   are in scope. Validate Controller/Worker and Ray or external-launcher paths
   separately; do not infer their support from the ordinary Engine path.

6. **Ascend Provider coverage in SGLang**

   The Ascend Provider is currently wired into the vLLM assembly path. The
   SGLang Provider selection path currently covers `auto`,
   `sglang-runtime-pci`, and `iluvatar-runtime-pci`; `ascend-sysfs-pci` has not
   been wired into SGLang. Decide whether Ascend+SGLang is part of the first
   delivery. If yes, implement and validate that adapter path before claiming
   Ascend support for SGLang.

### Optional validation and delivery closure

7. **Performance baseline**

   If the delivery must claim a performance benefit, compare plugin-off and
   plugin-on under the same model, prompts, concurrency, visible devices, and
   CPU constraints. Record TTFT, token latency, end-to-end latency, throughput,
   CPU/GPU utilization, and NUMA policy. Functional tests do not establish a
   performance improvement.

8. **Real PCIe Switch hardware**

   Fixture tests already cover direct, single-level, and multi-level paths. If
   the acceptance target requires physical Switch hardware, preserve one real
   host report containing the complete PCIe parent path, Switch ancestors,
   NUMA evidence, and target CPU set.

9. **Documentation synchronization**

   Before release, update the formal delivery design with the current commit,
   test count, supported vLLM baselines, Ascend Provider status, and the now
   passing Iluvatar auto-fallback result. Do not leave old status tables or old
   commit references as the delivery state.

### Required status wording

Every future report must state separately:

- what is proven by source tests;
- what is proven by Provider/topology probes;
- what is proven by controlled framework dummy spawn;
- what is proven by a real model service;
- what remains unverified;
- whether any existing GPU workload was observed and left untouched.

## 12. Complete 184 self-validation task for acceptance evidence

This section is the current end-to-end task brief. Validate the checkout at
commit `f4591ea` or a later commit explicitly identified in the report. The
goal is to produce an acceptance-oriented self-validation report, not merely a
unit-test transcript. The report must separate source-level evidence,
controlled framework evidence, real-service evidence, correctness evidence,
and performance evidence.

For this section only, the earlier inventory-only restrictions are superseded
after Stage A preflight: an isolated validation service may be started and
stopped by the test Agent, but only on newly allocated idle resources. The
earlier prohibition on touching existing services, shared containers, or busy
GPUs remains absolute.

### 12.1 Safety and authority boundary

- Work only in the user's plugin checkout and a dedicated validation result
  directory outside that checkout.
- Do not modify plugin source, tests, validation scripts, package metadata, or
  framework source to make a stage pass. Do not commit or push.
- Start no test until current containers, processes, GPU occupancy, ports, and
  existing model services have been inventoried.
- Never stop, restart, kill, or reconfigure an existing service or process.
- Real-service and performance stages may run only on GPUs confirmed idle and
  isolated from existing workloads. If isolation is unavailable, mark the
  stage `BLOCKED`; do not borrow a busy GPU.
- Installing the plugin source in a dedicated validation container is allowed.
  Do not install drivers, kernel modules, system packages, or unrelated Python
  dependencies. Record any pre-existing image/container and tool versions.
- Do not expose passwords, tokens, private paths, IP addresses, user prompts,
  model credentials, or complete credential-bearing commands in the report.
- Preserve every command, return code, stdout/stderr log, process PID, and
  artifact path needed to reproduce the result.

### 12.2 Report directory and status vocabulary

Create a unique result directory outside the Git checkout, for example:

```text
/home/qch/kunpeng-affinity-results/self-validation-<timestamp>/
```

Use this layout where the corresponding stage is executed:

```text
report.md
environment.txt
workload-inventory.txt
commands.txt
source-tests/
topology-provider/
hook-dummy/
vllm-service-off/
vllm-service-on/
correctness/
performance/
sglang/
```

Every stage must be reported as exactly one of `PASS`, `FAIL`, `BLOCKED`, or
`SKIP`, with the command, return code, reason, and evidence path. `BLOCKED`
means the stage was required but could not be safely or technically run;
`SKIP` means it is outside the available acceptance scope and the reason is
explicit. A missing log is not a pass.

### 12.3 Stage A: environment and workload preflight

Record read-only facts before any installation or service launch:

- exact plugin commit, branch, worktree state, and source location;
- host kernel, architecture, CPU topology, NUMA nodes, online CPUs, and
  current process cpuset/affinity;
- container name/image, Python version, vLLM/SGLang versions, accelerator
  runtime and driver versions;
- `ixsmi` GPU inventory, UUID, PCI BDF, visible-device environment, and GPU
  occupancy/processes;
- listening ports and existing model services;
- available `numactl`, `taskset`, `nsys`/`perf`/vendor monitoring tools, and
  their versions, without starting a benchmark;
- whether an isolated idle GPU and a dedicated container are available for
  TP=1 and multi-GPU tests.

Classify existing workloads before reuse:

```text
FUNCTIONAL
EXECUTION-CHAIN
PERFORMANCE-BASELINE
REUSABLE-WITH-CHANGES
NOT-REUSABLE
```

Do not start a service or send an inference request in this stage.

### 12.4 Stage B: source and static baseline

Run the repository's source test entry point with the environment's Python:

```bash
PYTHON_BIN=python3 ./scripts/test.sh
```

Also run source compilation and whitespace checks. If `coverage.py` is already
available, collect statement and branch coverage without adding a dependency.
Otherwise use the standard-library `trace` command or report that branch
coverage is unavailable. Record the test count, coverage method, and coverage
by vLLM adapter module. The current local reference is 159 passing tests, but
the result on 184 must be measured rather than copied.

### 12.5 Stage C: topology and Provider evidence

Run the read-only generic topology probe and the Iluvatar Provider probes. Use
explicit target GPU BDFs when automatic candidate discovery sees unrelated PCI
functions. Record for every visible GPU:

```text
logical device -> runtime identity -> physical identity -> canonical BDF
  -> complete PCIe parent path -> Switch ancestors -> NUMA node
  -> online CPUs ∩ allowed CPUs -> target CPUs -> bindable status
```

Verify both default visibility and a safe reordered visibility case. The
reordered case must preserve UUID-to-BDF association and must not use vendor
tool row order as the logical mapping. A partial mapping, duplicate identity,
unknown NUMA node, or empty CPU intersection must be reported as a failed
batch, never as a successful partial result.

### 12.6 Stage D: controlled vLLM Hook and binding chain

Run the complete Iluvatar validation suite for the current commit in a
dedicated idle container. At minimum collect:

1. source tests and source installation;
2. entry-point metadata and source import location;
3. forced-generic dummy spawn;
4. native-empty automatic fallback dummy spawn;
5. expected child CPU affinity and `numactl --show` memory policy;
6. Hook logs proving the entry point ran in the relevant process;
7. child start/exit and absence of residue after completion.

The forced-generic path must prove zero native NUMA query calls. The fallback
path must prove a native empty result, exactly one controlled fallback, and the
same child CPU/memory-policy evidence. Report the full chain rather than only
the runner return code.

### 12.7 Stage E: real vLLM TP=1 functional validation

Only if Stage A confirms an isolated idle GPU and a reproducible model image,
run the same TP=1 service twice:

```text
run A: plugin disabled
run B: plugin enabled, automatic generic path observable
```

Keep model, image, visible device, request parameters, seeds, tokenizer,
maximum output tokens, and service settings identical. Do not alter any
existing service. Collect:

- startup and readiness result;
- one fixed deterministic request and HTTP response;
- EngineCore/Worker PIDs and process kinds;
- plugin discovery and decision logs;
- each relevant PID's `Cpus_allowed_list`;
- `numactl --show`, `Mems_allowed_list`, and memory-policy evidence;
- selected Provider, BDF, NUMA node, and expected/observed CPU set;
- clean service shutdown and no leftover test processes.

The plugin-disabled run is a control only. It does not prove that the plugin
was loaded or that binding occurred. A service response alone does not prove
the execution chain.

### 12.8 Stage F: inference result consistency

Use the exact same fixed request set against the plugin-disabled and
plugin-enabled TP=1 services. Prefer deterministic generation (`temperature=0`
or greedy), fixed seed, fixed tokenizer and fixed output limit. Save redacted
JSONL outputs and a machine-readable comparison.

Compare, at minimum:

- HTTP success and error counts;
- empty/error/truncated responses;
- generated token counts;
- exact output match where the runtime is deterministic;
- normalized or task-specific match only when nondeterminism is documented.

Do not call this a model-accuracy benchmark unless a defined evaluation set
and metric are used. For this plugin, it is primarily an inference-result
regression check proving that NUMA binding does not alter observable output.

### 12.9 Stage G: real vLLM multi-process and parallel validation

If isolated hardware permits, run one multi-GPU TP case after TP=1 passes. If
the environment and acceptance scope include DP, run one DP case separately.
Collect and correlate PIDs, local ranks, logical devices, BDFs, NUMA nodes,
CPU sets, and memory policy. Verify:

- every Worker maps to the intended visible device;
- EngineCore receives the intended process-level policy;
- spawn inheritance marker validation succeeds;
- visibility fingerprints remain valid after process creation;
- a changed or incomplete mapping prevents partial commit;
- service startup and one fixed request succeed.

Do not infer TP/DP support from the dummy-spawn test or from TP=1.

### 12.10 Stage H: performance A/B validation

Only run performance tests after functional and consistency stages pass and an
idle allocation is approved. Use plugin-off and plugin-on under the same
container, model, visible devices, TP/DP, CPU constraints, request set,
concurrency, input/output token limits, warmup count, duration, and scheduler
settings. Prefer an existing reusable workload from Stage A; otherwise report
that no performance baseline is available instead of inventing a business
workload.

Record raw per-request data and aggregate at least:

- request success rate;
- TTFT;
- TPOT/ITL or inter-token latency;
- end-to-end latency p50/p95/p99;
- input/output tokens per second and aggregate throughput;
- CPU and GPU utilization;
- worker affinity and NUMA memory policy during the run.

Use the same warmup and sample counts for both arms. Report absolute values,
relative change, variance or confidence interval where available, and the
measurement limitations. Do not claim a performance improvement without a
predefined acceptance threshold; if no threshold exists, report comparison
data without a PASS/FAIL performance conclusion.

### 12.11 Stage I: SGLang and remaining framework scope

Inspect whether a clean SGLang 0.5.18 environment is available. If it is
available and isolated, validate a real TP=1 service with plugin off/on,
including Hook discovery, native-empty fallback, Worker affinity, memory
policy, one fixed request, and output consistency. Then separately assess
multi-GPU TP/DP and launcher/Ray paths. If SGLang is unavailable on 184, mark
the real-service stages `BLOCKED` and record the missing environment; source
tests must not be presented as real SGLang validation.

Confirm and report that `ascend-sysfs-pci` is not currently wired into the
SGLang assembly path unless the source has changed. Do not silently broaden
the claimed hardware/framework support.

### 12.12 Final self-validation report

Write `report.md` with these sections:

1. Scope, acceptance criteria, and exact commit.
2. Environment, workload inventory, isolation decision, and safety result.
3. Stage matrix with command, return code, status, and evidence path.
4. Source test and coverage result.
5. Provider/topology mapping tables and PCIe paths.
6. Controlled vLLM Hook and dummy binding evidence.
7. Real vLLM service evidence and process-to-device correlation.
8. Inference result consistency comparison.
9. Performance A/B data and limitations, or a precise blocked reason.
10. SGLang results and unsupported boundaries.
11. Failures classified as code, environment, workload, or harness.
12. Confirmed capabilities, unverified capabilities, and recommended next step.

The conclusion must use separate statements for:

```text
source tests prove ...
Provider/topology probes prove ...
controlled dummy spawn proves ...
real vLLM service proves ...
result-consistency test proves ...
performance comparison shows ...
SGLang remains ...
```

Never turn `BLOCKED` into `PASS`, never infer performance from functional
success, and never claim production or multi-process support without the
corresponding evidence.
