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

If at least two devices are visible, run the reorder check using the variable
confirmed in the environment report:

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

If the complete vLLM version is `0.23.0+...` and the current code rejects it
before installing the Hook, classify both spawn stages as `BLOCKED` at version
gate. Do not call this a Provider or topology failure, and do not bypass the
gate by editing source or package metadata. Preserve the exact error.

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
