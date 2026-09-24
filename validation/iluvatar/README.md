# Iluvatar Validation Suite

This directory provides one configuration file and one command for staged
validation in the target vLLM container. It does not start a model server.

## Quick start

Run the safe, read-only stages directly from the checkout:

```bash
cd /home/qch/tools/kunpeng-affinity-plugin
./validation/iluvatar/run.sh
```

When no local configuration exists, the runner uses the committed
`config.env.example`. It enables the source, topology and Provider checks but
keeps environment-changing stages blocked.

In a dedicated test container with no active workload, create the
machine-local configuration once:

```bash
cp validation/iluvatar/config.env.example validation/iluvatar/config.env
```

Then set the following value in `config.env` to run the complete suite:

```text
ENABLE_ENVIRONMENT_CHANGES=1
```

Run the same command again. `config.env` is ignored by Git, so machine-specific
settings do not make the checkout dirty or interfere with later pulls. No
environment-variable prefix is needed on the command line.

## Configuration selection

The runner selects exactly one configuration file in this order:

1. `--config PATH`;
2. `KUNPENG_AFFINITY_VALIDATION_CONFIG`;
3. the Git-ignored `validation/iluvatar/config.env`;
4. the committed `validation/iluvatar/config.env.example`.

The file is sourced by Bash and should contain only `NAME=value` assignments
and comments. Boolean stage settings accept `1/0`, `true/false`, `yes/no`, or
`on/off`.

### Runtime and device settings

| Setting | Default | Meaning |
|---|---|---|
| `PYTHON_BIN` | `python3` | Interpreter from the target vLLM environment. |
| `IXSMI_BIN` | `ixsmi` | Iluvatar inventory command used by Provider probes and exported as `KUNPENG_AFFINITY_IXSMI` during controlled vLLM spawn checks. |
| `SYSFS_ROOT` | `/sys` | Linux sysfs root used for PCIe and NUMA analysis. |
| `PROBE_DEVICE` | `0` | Logical device used by the first single-device probe. |
| `VISIBILITY_ENV` | `CUDA_VISIBLE_DEVICES` | Visibility variable; may also be `ILUVATAR_VISIBLE_DEVICES`. |
| `REORDER_VISIBLE_DEVICES` | `1,0` | Device order used by the reorder probe. |
| `SPAWN_VISIBLE_DEVICES` | `0` | Devices exposed to controlled dummy-spawn checks. |
| `AFFINITY_BDFS` | empty | Trusted comma-separated target BDFs for topology probing; use this on heterogeneous hosts to exclude unrelated PCI functions. |

### Plugin settings

| Setting | Default | Meaning |
|---|---|---|
| `KUNPENG_AFFINITY_MODE` | `auto` | Plugin failure policy used by spawn checks. |
| `KUNPENG_AFFINITY_PROVIDER` | `iluvatar-runtime-pci` | Forces the Iluvatar UUID-to-BDF Provider. |
| `KUNPENG_AFFINITY_CPU_POLICY` | `node` | Uses vLLM's node-level binding path. |
| `KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL` | `detail` | Plugin diagnostic verbosity. |
| `KUNPENG_AFFINITY_VLLM_FORCE_GENERIC` | `1` | Bypasses native GPU NUMA discovery in the forced-generic check. |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Required process method for the vLLM wrapper check. |

### Stage and safety settings

`RUN_UNIT_TESTS`, `RUN_TOPOLOGY_PROBE`, `RUN_PROVIDER_PROBE`,
`RUN_REORDER_PROBE`, `RUN_SOURCE_INSTALL`, `RUN_FORCED_GENERIC_SPAWN`, and
`RUN_AUTO_FALLBACK_SPAWN` enable their corresponding stages. They default to
`1`. The reorder stage is skipped automatically when fewer than two devices
are visible. vLLM startup logs are ignored while reading the device count; the
runner requires an explicit numeric marker from the platform query.

On a heterogeneous host, set `AFFINITY_BDFS` to the target GPU BDFs reported
by the Runtime inventory. If it is empty, the topology stage scans all display
and processing-accelerator PCI candidates; an unrelated device outside the
container CPU set can make that candidate scan fail even when the target GPU
topology is valid.

`ENABLE_ENVIRONMENT_CHANGES=0` is the final safety gate. While it remains
disabled, source installation and both dummy-spawn stages are reported as
`SKIP` even if their `RUN_*` settings are enabled. `--read-only` enforces the
same restriction regardless of the configuration file.

The stages are ordered as follows:

```text
environment preflight
  -> source unit tests
  -> Linux topology probe
  -> Iluvatar Provider single-device probe
  -> Iluvatar Provider all-device probe
  -> visibility reorder probe
  -> editable install and entry-point verification
  -> forced-generic vLLM dummy spawn
  -> automatic-fallback vLLM dummy spawn
```

Each stage stops the suite on failure. Output is written to a new directory
under `/tmp` unless `RESULT_DIR` is configured.

## Output and logs

With the default empty setting, each invocation creates a unique directory:

```text
/tmp/kunpeng-affinity-validation-<timestamp>-<pid>/
```

Set an absolute output directory in `config.env` when logs need to survive
container cleanup:

```text
RESULT_DIR=/home/qch/tools/kunpeng-affinity-results/current
```

A relative value is resolved from the repository root:

```text
RESULT_DIR=validation-results/current
```

The runner prints the normalized absolute result directory before the first
stage and again in the final or failure summary. Existing files with the same
stage names are overwritten, so use a unique directory when preserving
multiple runs.

| Log | Stage |
|---|---|
| `00-preflight.log` | Environment, Git revision, Python, vLLM, `ixsmi`, sysfs and optional `numactl`. |
| `01-unit-tests.log` | Complete Python unit-test output. |
| `02-topology.log` | Framework-independent PCIe/NUMA topology probe. |
| `03-provider-single.log` | One logical device through UUID, BDF and topology. |
| `04-provider-all.log` | JSON result for every visible logical device. |
| `05-provider-reorder.log` | Visibility-reordered Provider result. |
| `06-source-install.log` | Editable install, source path and entry-point verification. |
| `07-forced-generic.log` | Forced generic vLLM dummy-spawn binding result. |
| `08-auto-fallback.log` | Native-to-generic fallback dummy-spawn result. |

The summary marks every selected stage as `PASS`, `FAIL`, `SKIP`, or
`DRY-RUN` and prints the associated log path. On failure, later stages are not
executed.

## Command options

Useful alternatives:

```bash
# Force read-only execution regardless of config.env.
./validation/iluvatar/run.sh --read-only

# Review the selected stages without running them.
./validation/iluvatar/run.sh --dry-run

# Use a machine-local configuration stored outside the checkout.
./validation/iluvatar/run.sh --config /path/to/config.env
```

The Provider probes do not change CPU affinity or memory policy. Editable
installation changes package metadata in the selected Python environment, and
the spawn stages apply CPU and memory binding to short-lived dummy child
processes. The suite does not start a model server.
