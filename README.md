# Kunpeng Affinity Plugin

This repository contains the incremental implementation of the Kunpeng GPU
affinity plugin. It includes the vLLM decision hook, the framework-independent
Linux topology core, provider/batch resolution, and isolated vLLM spawn
diagnostics.

It also contains Demo 2, a framework-independent, read-only Linux topology
analyzer. Given a trusted PCI BDF, it follows the real sysfs parent path,
resolves NUMA evidence, and suggests the intersection of NUMA-node, online, and
currently allowed CPUs. It does not import vLLM/SGLang or execute a binding.

The `demo/` directory contains a portable generic probe for running the same
flow on another Linux machine before framework integration. It accepts trusted
BDFs or reports PCI accelerator candidates, prints the PCIe path, NUMA evidence,
and CPU intersection, and can emit JSON. Automatic PCI enumeration is clearly
reported as candidate discovery rather than logical-GPU mapping. See
`docs/generic-affinity-probe.md`.

The `demo/probe-iluvatar-provider.sh` command validates the Iluvatar-specific
identity path in an environment that has vLLM and `ixsmi` available. It prints
the visible logical device, runtime UUID, physical index, BDF, PCIe path, NUMA
node, and target CPU intersection. It is read-only and does not start vLLM or
change affinity. This probe is distinct from the framework-independent probe:
automatic PCI enumeration cannot prove logical-device identity.

Demo 3 adds the framework-independent identity and batch layer. A
`DeviceContext` is mapped to a canonical PCI BDF by a `DeviceMapper`, then the
ordered batch is resolved through the same topology analyzer. Mapping errors,
duplicate devices, visibility changes, and any per-device topology failure make
the batch non-committable; no binding operation is executed.

## vLLM plugin behavior

The package registers a `vllm.general_plugins` entry point whose callable
installs an idempotent wrapper around
`vllm.utils.numa_utils.configure_subprocess`. The entry point is metadata for
vLLM's general-plugin loader; package installation alone does not execute the
callable. The selected vLLM startup path must call
`load_general_plugins()` before the first relevant subprocess is created. Once
the callable has run, the wrapper runs only when vLLM's own `numa_bind` switch
is enabled and `numa_bind_nodes` is absent. Its normal decision order is:

```text
explicit nodes -> validated vLLM native nodes -> generic BDF/sysfs nodes
```

Explicit nodes always remain under vLLM control. Explicit `numa_bind_cpus` are
preserved while the plugin fills only the missing node list. A successful
automatic decision is passed to vLLM's original `configure_subprocess` and
`numactl` execution chain.

Plugin configuration is loaded once during registration. Environment variables
override an optional UTF-8 JSON file selected with
`KUNPENG_AFFINITY_CONFIG`; omitted values use the defaults below:

| Setting | Environment variable | Values | Default |
|---|---|---|---|
| `mode` | `KUNPENG_AFFINITY_MODE` | `off`, `auto`, `strict` | `auto` |
| `provider` | `KUNPENG_AFFINITY_PROVIDER` | `auto` or a provider name | `auto` |
| `cpu_policy` | `KUNPENG_AFFINITY_CPU_POLICY` | `node`, `exact` | `node` |
| `diagnostic_level` | `KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL` | `error`, `summary`, `detail` | `summary` |

The JSON file accepts the setting names in the first column. Unknown or
duplicate fields, invalid values, unreadable files, non-UTF-8 input, and a
non-object JSON root fail registration. Provider-specific JSON schemas remain
undefined until their runtime contracts are implemented.

The current vLLM adapter supports `cpu_policy=node` and the
`vllm-platform-pci` provider. When vLLM does not expose a usable direct BDF
mapping, it can use `iluvatar-runtime-pci`, which joins the vLLM logical-device
UUID to the read-only Iluvatar `ixsmi` UUID/BDF inventory. It rejects `exact`
or an unregistered explicit provider before installing the Hook; parsing a
stable configuration value does not imply that every framework adapter already
implements it.

The Iluvatar Provider executes `ixsmi` by default. Set
`KUNPENG_AFFINITY_IXSMI` to an executable path when the utility is mounted at a
nonstandard location. The staged validation runner resolves `IXSMI_BIN` once
and passes that path to both Provider probes and controlled vLLM spawn checks.

The `ascend-sysfs-pci` Provider is selected when vLLM has no direct BDF API and
`ASCEND_RT_VISIBLE_DEVICES` is present. It maps the ordered visible physical
device IDs through the read-only Ascend sysfs attribute
`devdrv_sysfs_bdf_to_devid`, then hands the resulting BDFs to the common Linux
PCIe/NUMA analyzer. It does not import `torch_npu`, invoke `npu-smi`, call
DCMI, or infer an identity from `/dev/davinciN`. If the visibility variable or
the sysfs mapping is unavailable or inconsistent, the Provider fails closed.
For a read-only target check, run `demo/probe-ascend-provider.sh` inside the
vLLM Ascend environment.

`KUNPENG_AFFINITY_MODE` controls failure behavior:

| Value | Behavior |
|---|---|
| `auto` | Default. Try native then generic discovery; if both fail, add no binding and allow startup to continue. |
| `strict` | Convert a plugin discovery failure into a startup error with a stable error code. |
| `off` | Return before importing vLLM or installing a Hook. An environment `off` also skips the optional JSON file. |

Invalid plugin configuration always fails. Framework configuration, device
index, and binding-executor errors are not treated as discovery failures and
are not swallowed by `auto` mode. Unsupported vLLM versions or incompatible
Hook signatures leave the Hook uninstalled in `auto`/`off` mode and fail early
in `strict` mode.

For isolated integration testing, setting
`KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1` forces a missing
`numa_bind_nodes` list to be resolved as follows:

```text
vLLM logical device -> selected Provider PCI BDF -> Linux sysfs -> NUMA node
```

The diagnostic override still requires vLLM's `numa_bind=True`, preserves explicit
`numa_bind_nodes`, and reuses vLLM's original `configure_subprocess` and
`numactl` execution. It deliberately bypasses only vLLM's native GPU NUMA
query. Leave the switch unset for normal deployment. It is suitable for isolated
diagnostics or a real-model benchmark that must prove the generic path was used.
A discovery failure on this forced path raises even when the plugin mode is
`auto`; silently proceeding without binding would invalidate that verification.

| Condition with `FORCE_GENERIC=1` | Behavior |
|---|---|
| `KUNPENG_AFFINITY_MODE=off` | No Hook is installed; the override has no effect. Framework binding settings remain independent. |
| vLLM `numa_bind=False` | Preserve the disabled binding switch; no forced discovery. |
| Explicit `numa_bind_nodes` | Preserve the nodes; the override does not replace them. |
| Binding enabled, nodes missing | Bypass native NUMA discovery and resolve through the selected Provider and Linux topology. Discovery failure raises. |

This switch neither enables `--numa-bind` nor selects a Provider. It also does
not make every installation failure strict: version/signature checks still
follow `KUNPENG_AFFINITY_MODE`. To compare binding against no binding, disable
the framework's own binding in the baseline; plugin `off` alone is insufficient.

The supported vLLM base versions are 0.23.0, 0.25.1, and 0.26.0. Vendor-local
builds such as `0.25.1+ascend.8.0` are accepted when their upstream base
version is listed; upstream post/dev releases are not implicitly accepted.
The 0.25.1 baseline is included for Ascend deployment compatibility and still
requires the target environment's end-to-end validation before production use.
The 0.26.0 baseline has additionally passed forced-generic and
automatic-fallback single-device dummy spawn diagnostics.

## SGLang plugin behavior

The `sglang.srt.plugins` entry point installs an around hook on
`sglang.srt.utils.numa_utils.get_numa_node_if_available`. The hook is evaluated
inside SGLang's existing `configure_subprocess` path, so the original
`numactl` wrapper remains responsible for process launch and binding. Its
decision order is:

```text
explicit server_args.numa_node -> SGLang native query -> generic runtime BDF/NUMA query
```

When `ASCEND_RT_VISIBLE_DEVICES` is set, the SGLang adapter uses that ordered
visibility list to build the complete device batch and selects the Ascend sysfs
BDF-to-device mapping without consulting `torch.cuda`. This applies to `auto`,
`sglang-runtime-pci`, and explicit `ascend-sysfs-pci` selection. On other platforms, a
small Torch runtime facade supplies the device count and either a direct PCI
BDF or a runtime UUID that is joined to the Iluvatar `ixsmi` UUID/BDF
inventory. The generic Linux topology analyzer then proves the NUMA node and
CPU intersection. If discovery fails, `auto` returns `None` so SGLang's normal
fallback behavior continues; `strict` raises from the query hook.
`SGLANG_AUTO_NUMA_BIND=0` and explicit `server_args.numa_node` are always
respected.

This first adapter covers SGLang 0.5.18 and the observed 0.5.17.dev386+gc5bd3d7dc
compatibility build on the ordinary Engine subprocess path. It validates the
target NUMA-query signature before registering the Hook. The Data Parallel
controller and Ray actor path have separate launch/binding code; generic
fallback is explicitly skipped there until those paths are validated. Generic
fallback is also skipped when `SGLANG_SET_CPU_AFFINITY=1`, because that feature
would overwrite the NUMA-derived CPU affinity after process launch. The adapter
is covered by fake-runtime contract tests; no real SGLang service is started by
the tests.

SGLang's default `SGLANG_NUMA_BIND_V2=1` path requires the `numactl` command. If
it is absent, `auto` mode logs the missing executor and leaves the process
unmodified. Setting `SGLANG_NUMA_BIND_V2=0` selects SGLang's in-process libnuma
path, which does not require the command but still requires usable libnuma and
memory-policy permissions. This is an explicit execution-mode choice, not an
automatic equivalent replacement for V2.

SGLang's general-plugin loader catches exceptions raised while executing a
plugin entry point. Consequently, a `strict` version or signature failure
during `register()` is reported by SGLang but may not terminate framework
startup. Strict failures raised later by the applied NUMA-query Hook still
propagate. Confirm both this plugin's `registered` message and SGLang's
`Applied hook` message during integration validation.

After source installation, SGLang can restrict discovery to this plugin with:

```bash
SGLANG_PLUGINS=kunpeng_affinity sglang serve /path/to/model
```

The source-level entry point and fake-runtime tests prove packaging and hook
semantics. In addition, an Ascend TP=4 service on the observed 0.5.17.dev386+gc5bd3d7dc
build has completed startup and inference requests with plugin and generic
NUMA-path logs. Per-process rank, CPU-affinity, and memory-policy evidence is
still pending, so this does not yet establish a complete multi-GPU lifecycle
or a formal performance result. See
`docs/sglang-ascend-tp4-validation-progress.md`.

## One-command Iluvatar validation

The staged target-environment checks are consolidated under
`validation/iluvatar/`. The committed `config.env.example` provides safe
read-only defaults. For complete validation in a dedicated container, copy it
once to the Git-ignored `config.env` and change the environment-change gate:

```bash
cp validation/iluvatar/config.env.example validation/iluvatar/config.env
sed -i 's/ENABLE_ENVIRONMENT_CHANGES=0/ENABLE_ENVIRONMENT_CHANGES=1/' \
  validation/iluvatar/config.env
```

Then run:

```bash
./validation/iluvatar/run.sh
```

Without a local file, the runner uses the template and performs only the
read-only source, topology and Runtime Provider checks. Setting
`ENABLE_ENVIRONMENT_CHANGES=1` additionally enables editable installation,
entry-point verification and the two vLLM dummy-spawn binding checks. Every
stage has its own log and the suite stops at the first failed gate. See
`validation/iluvatar/README.md` for the stage order and safety boundary.

## Runtime dependencies and environment checks

Check these requirements **inside the actual service container**, using the
same Python interpreter, user, visibility variables, and resource limits as
the service. Host-installed packages do not supply missing container packages.
Installing this plugin does not install a framework, accelerator Runtime, or
system NUMA tools.

### Common requirements

| Requirement | Scope and purpose |
|---|---|
| Linux with multiple NUMA nodes | The generic automatic-binding precheck requires at least two nodes. A single-node system is skipped rather than treated as an optimization opportunity. |
| Python 3.10+ and `packaging>=21` | Plugin requirements; the selected framework may require a newer Python. |
| Compatible vLLM or SGLang installation | Use the supported versions listed above and the matching vendor extensions. Only the framework being used is required; version acceptance does not certify every hardware/launch path. |
| Installed plugin metadata and active framework loader | Install into the service interpreter. Plugin filters must permit `kunpeng_affinity`, and the loader must run before the relevant subprocess launch. An entry point on disk alone does not prove the Hook was called. |
| Usable `libnuma.so` or `libnuma.so.1` | Required by the framework NUMA probe, including SGLang V2. The library must load, report NUMA available, and allow the framework's memory-policy query. |
| `numactl` on `PATH` | Required by vLLM and SGLang V2. SGLang V1 uses in-process libnuma without the command; its execution/policy semantics differ from V2. |
| Readable Linux sysfs topology | `/sys/bus/pci/devices` links must resolve to the real parent hierarchy under `/sys/devices`. NUMA evidence (`numa_node` or usable `local_cpulist`), node CPU lists under `/sys/devices/system/node`, and `/sys/devices/system/cpu/online` must be visible. |
| Allowed CPUs, memory nodes, and syscalls | Target CPUs must intersect the process's allowed online CPUs. Target memory nodes must be allowed, and policy queries and actual CPU/memory binding must be permitted. Container seccomp and cpuset limits can block these even when tools are installed. |
| Working framework subprocess executor | The wrapper and service Python must be executable, with `/bin/sh` available. vLLM requires `VLLM_WORKER_MULTIPROC_METHOD=spawn`; SGLang's V2 path also uses spawn and executable temporary scripts in writable `/tmp`. A `noexec` mount can prevent launch. |

Existing affinity may cause the framework to protect an external binding and
skip automatic adjustment. Do not expand a container's CPU/memory limits merely
to satisfy a probe. Resolve a permission failure through the deployment's
specific syscall/resource policy; root access or a privileged container is not
a universal requirement or remedy.

### Provider-specific requirements

| Path | Additional requirements |
|---|---|
| Framework PCI-BDF interface | Runtime device count/identity and a BDF mapping covering every visible logical device. A partial mapping is insufficient. |
| Iluvatar Runtime Provider | Logical-device UUIDs from the framework/Torch Runtime, plus executable `ixsmi` and its driver/Runtime shared libraries. Its read-only query must return `index,uuid,pci.bus_id`. Use `KUNPENG_AFFINITY_IXSMI` if the command needs an alternate executable path. Physical listing order is never a logical-device mapping. |
| Ascend sysfs Provider | Explicit ordered `ASCEND_RT_VISIBLE_DEVICES`, and readable driver-exported `/sys/bus/pci/devices/*/devdrv_sysfs_bdf_to_devid` tables covering those physical IDs. Enumeration order cannot replace the visibility list. The Provider itself does not import `torch_npu`, DCMI, or invoke `npu-smi`. |
| SGLang generic resolver | Importable Torch, including on Ascend. The Ascend path bypasses CUDA device enumeration, but the current resolver still imports Torch. |

The accelerator driver, device access, vendor Runtime (for example CoreX or
CANN), and framework extensions must already support model serving in the
container. They are framework/platform prerequisites, not dependencies that
this plugin installs. `pynvml`, `nvidia-smi`, `npu-smi`, `lspci`, and `lscpu`
are not universal plugin requirements; optional diagnostics may use them.
The standalone sysfs topology CLI does not need a model-serving framework,
libnuma, or numactl because it only reads topology and recommends affinity.

### Install missing packages

On Debian/Ubuntu, install the runtime packages inside the service container:

```bash
apt-get update
apt-get install -y numactl libnuma1
```

On openEuler/RHEL-compatible images:

```bash
dnf install -y numactl numactl-libs
```

Prebuilt runtime use does not require `libnuma-dev` or `numactl-devel`. Source
installation additionally needs pip and setuptools 64+ in the selected Python:

```bash
python3 -m pip install 'packaging>=21' 'setuptools>=64'
```

Git is needed to clone the repository and Bash to run its shell helpers.
Ruff, pre-commit, ShellCheck, and wheel-building tools are development/release
tools, not service runtime dependencies. The source installer uses
`--no-deps --no-build-isolation`, so it will not fill in missing Python dependencies.

### Check a new container

Replace `python3` below if the service uses another interpreter. These commands
inspect the current process and do not launch or change an inference service.

```bash
python3 -m pip check
command -v numactl
numactl --hardware
numactl --show
```

For SGLang V1, the `numactl` checks are optional. Use the following check for
either execution mode; set `FRAMEWORK=sglang` for SGLang:

```bash
FRAMEWORK=vllm python3 - <<'PY'
import os
import sys
from importlib import import_module
from importlib.metadata import entry_points, version

framework = os.environ['FRAMEWORK']
group = 'vllm.general_plugins' if framework == 'vllm' else 'sglang.srt.plugins'
module = 'vllm.utils.numa_utils' if framework == 'vllm' else 'sglang.srt.utils.numa_utils'
print('python:', sys.executable)
for package in ('kunpeng-affinity-plugin', 'packaging', framework):
    print(package, version(package))
print('entry_points:', [ep.value for ep in entry_points(group=group) if ep.name == 'kunpeng_affinity'])
print('allowed_cpus:', sorted(os.sched_getaffinity(0)))
print('memory_policy_query_available:', import_module(module)._can_set_mempolicy())
PY
```

The entry-point list must contain this plugin's registration function. A
missing distribution or an empty list indicates installation in the wrong
environment or missing package metadata; correct that before service testing.

`memory_policy_query_available=False` can mean a missing library, unavailable
NUMA, or a blocked syscall. `True` proves only the framework's query succeeded
in this process; it does not prove a later EngineCore/Worker can apply binding.
Likewise, `numactl --show` warnings must be inspected rather than relying solely
on its exit status. `Mems_allowed_list` describes permitted nodes, not the
process's applied memory policy.

If desired, choose a target node allowed for both CPUs and memory, then test
the executor on a disposable child (replace `<node>` with that node):

```text
numactl --cpunodebind=<node> --membind=<node> /bin/true
```

This last check changes only the child's affinity/policy. It is not a read-only
probe and does not modify a running inference process. Complete validation
still requires Hook-call logs and actual child CPU affinity/memory policy.
Common prerequisite failures are `NUMA_NOT_AVAILABLE`,
`MEMPOLICY_UNAVAILABLE`, and `BIND_EXECUTOR_MISSING`; Provider/topology failures
may instead be summarized as a failed batch. `auto` can skip plugin discovery
failures, but it does not suppress errors from the original binding executor.

## Source deployment

Clone the repository on the target machine, enter the Python environment used
by vLLM, and register this checkout as an editable package:

```bash
git clone git@github.com:qin-chenghan/kunpeng-affinity-plugin.git
cd kunpeng-affinity-plugin
./scripts/install-source.sh
./scripts/verify-source.sh
```

If the vLLM environment does not expose its interpreter as `python3`, select it
explicitly for both commands:

```bash
PYTHON_BIN=/path/to/vllm/python ./scripts/install-source.sh
PYTHON_BIN=/path/to/vllm/python ./scripts/verify-source.sh
```

`install-source.sh` requires Python 3.10+ and setuptools 64+, then runs the
selected interpreter with `-m pip install --no-deps --no-build-isolation -e .`.
It does not download dependencies. Editable
installation does not copy or build the project into a separate artifact:
Python imports the checked-out source tree directly, while package metadata
registers the `vllm.general_plugins` entry point needed for automatic vLLM
plugin discovery. Source edits therefore take effect when a new Python/vLLM
process starts.

Use the same Python interpreter/environment that starts vLLM. Installing into a
different virtual environment will not make the plugin visible to vLLM. No
vLLM service or GPU workload is started by either script.

## Local tests

```bash
./scripts/test.sh
```

This runs the tests directly from `src/`. It does not install the package,
build a wheel, register the vLLM plugin entry point, inspect host hardware, or
start a GPU workload. Set `PYTHON_BIN` when the desired interpreter is not
exposed as `python3`.

## Pre-commit checks

The development repository uses the pinned checks in
`.pre-commit-config.yaml`. They cover `src/`, `tests/`, and reusable scripts;
experimental probes, validation artifacts, and documentation are excluded
from this development gate.

```bash
python3 -m pip install pre-commit
pre-commit install --install-hooks
pre-commit run --all-files
```

For a pull request or merge request checkout that contains the target branch,
run the incremental check with:

```bash
TARGET_REF=origin/main ./ci-pre-commit-pr.sh
```

## Live host topology probe

Run the separate read-only probe on a target Linux host:

```bash
./scripts/probe-host.sh
```

It prints the online and allowed CPUs, NUMA-node CPU lists, candidate
accelerator PCI metadata, the real PCIe parent path, NUMA evidence, and the
suggested CPU intersection.

By default, display-controller (PCI base class `0x03`) and processing-
accelerator (`0x12`) functions are treated as candidates. This is convenient
discovery, not a provider-neutral logical-GPU mapping. For a trusted runtime
mapping, provide one or more comma-separated BDFs:

```bash
AFFINITY_BDFS=0000:41:00.0,0000:81:00.0 ./scripts/probe-host.sh
```

The probe returns nonzero if no candidate is found or any topology result is not
`success`. It reads sysfs and the current process affinity but never changes
affinity or starts a GPU workload.

Run the independent topology analyzer directly from a checkout:

```bash
PYTHONPATH=src python3 -m kunpeng_affinity.topology.cli --bdf 0000:ab:00.0
```

Add `--json` for structured output. See `docs/topology-demo.md` for the input
contract, result fields, and test coverage.

The provider and batch APIs are used without a framework:

```python
from kunpeng_affinity.core.models import DeviceContext
from kunpeng_affinity.policy import GenericAffinityProvider
from kunpeng_affinity.providers import StaticMappingProvider

contexts = (
    DeviceContext(framework="example", logical_device_id=0),
    DeviceContext(framework="example", logical_device_id=1),
)
resolver = GenericAffinityProvider(
    StaticMappingProvider({0: "0000:01:00.0", 1: "0000:02:00.0"})
)
batch = resolver.resolve_all(contexts)
assert batch.committable
```

See `docs/provider-batch-demo.md` for the mapping contract and failure
semantics. `StaticMappingProvider` is a configuration/test implementation.
`LinuxContextProvider` accepts only an already-proven BDF carried by the
framework context or a device path; a target GPU runtime provider is still
needed when the framework does not expose either fact.

## vLLM spawn diagnostics

After editable source installation in an isolated vLLM environment with
`numactl`, run:

```bash
./scripts/verify-vllm-generic-spawn.sh
```

Select the vLLM interpreter explicitly when needed:

```bash
PYTHON_BIN=/path/to/vllm/python ./scripts/verify-vllm-generic-spawn.sh
```

The script loads the real vLLM plugin entry point, makes the native GPU NUMA
query fail if called, resolves BDF and topology through the generic path,
launches a dummy multiprocessing child through vLLM's real numactl wrapper,
and compares the child's `Cpus_allowed_list` with the expected NUMA CPUs. It
does not start the vLLM engine, load a model, or run GPU computation. See
`docs/vllm-generic-spawn-demo.md` for the exact boundary.

To exercise the normal `native -> generic` decision, make the controlled native
query return no result and require generic fallback:

```bash
PYTHON_BIN=/path/to/vllm/python ./scripts/verify-vllm-auto-fallback-spawn.sh
```

## Optional package artifact

A wheel is an installable Python package, not an executable file. It is useful
for versioned or offline distribution but is not required for source-based Demo
development. If needed, build one with:

```bash
python3 -m pip wheel --no-deps --wheel-dir dist .
```

The resulting `py3-none-any` wheel can be installed on another Linux machine
with Python 3.10 or newer. The topology CLI and provider/batch core do not
require vLLM, SGLang, CUDA, or a specific CPU architecture. The vLLM entry point
is executed only when a compatible vLLM process invokes its general-plugin
loader; entry-point metadata by itself does not prove that a service loaded the
plugin.

The current Hook implements automatic `native -> generic -> skip/fail`
selection. Real target-framework service lifecycles and multi-process
acceptance remain environment-specific validation work. The formal delivery baseline is
`docs/kunpeng-affinity-plugin-delivery-design.md`; it defines the implementation
status, compatibility boundaries, and remaining development steps.
