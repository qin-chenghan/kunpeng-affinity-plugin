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

The current validation accepts vLLM `0.23.0` and vendor-local builds whose
PEP 440 version is based on it, such as `0.23.0+corex.5.0.0`. An upstream
post-release or development version is still outside the validated contract.
If a `0.23.0+...` build is rejected, report it as a regression in the version
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
`0.23.0+corex.5.0.0` rejection is a regression because vendor-local builds of
the validated base are expected to pass this gate.

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
