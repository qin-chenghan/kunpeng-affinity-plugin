# Kunpeng Affinity Plugin

This repository contains the incremental implementation of the Kunpeng GPU
affinity plugin. It includes the vLLM decision hook, the framework-independent
Linux topology core, provider/batch resolution, and isolated vLLM spawn
diagnostics.

It also contains Demo 2, a framework-independent, read-only Linux topology
analyzer. Given a trusted PCI BDF, it follows the real sysfs parent path,
resolves NUMA evidence, and suggests the intersection of NUMA-node, online, and
currently allowed CPUs. It does not import vLLM/SGLang or execute a binding.

Demo 3 adds the framework-independent identity and batch layer. A
`DeviceContext` is mapped to a canonical PCI BDF by a `DeviceMapper`, then the
ordered batch is resolved through the same topology analyzer. Mapping errors,
duplicate devices, visibility changes, and any per-device topology failure make
the batch non-committable; no binding operation is executed.

## vLLM plugin behavior

The `vllm.general_plugins` entry point installs an idempotent wrapper around
`vllm.utils.numa_utils.configure_subprocess`. The wrapper runs only when vLLM's
own `numa_bind` switch is enabled and `numa_bind_nodes` is absent. Its normal
decision order is:

```text
explicit nodes -> validated vLLM native nodes -> generic BDF/sysfs nodes
```

Explicit nodes always remain under vLLM control. Explicit `numa_bind_cpus` are
preserved while the plugin fills only the missing node list. A successful
automatic decision is passed to vLLM's original `configure_subprocess` and
`numactl` execution chain.

`KUNPENG_AFFINITY_MODE` controls failure behavior:

| Value | Behavior |
|---|---|
| `auto` | Default. Try native then generic discovery; if both fail, add no binding and allow startup to continue. |
| `strict` | Convert a plugin discovery failure into a startup error with a stable error code. |
| `off` | Disable plugin discovery and preserve vLLM's original behavior. |

Invalid plugin configuration always fails. Framework configuration, device
index, and binding-executor errors are not treated as discovery failures and
are not swallowed by `auto` mode. Unsupported vLLM versions or incompatible
Hook signatures leave the Hook uninstalled in `auto`/`off` mode and fail early
in `strict` mode.

For isolated integration testing, setting
`KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1` forces a missing
`numa_bind_nodes` list to be resolved as follows:

```text
vLLM logical device -> vLLM platform PCI BDF -> Linux sysfs -> NUMA node
```

The diagnostic override still requires vLLM's `numa_bind=True`, preserves explicit
`numa_bind_nodes`, and reuses vLLM's original `configure_subprocess` and
`numactl` execution. It deliberately bypasses only vLLM's native GPU NUMA
query. The switch is disabled by default and fails strictly because it is a
verification aid, not a production policy control.

The target contract is vLLM 0.23.0. vLLM 0.26.0 has additionally passed the
single-device dummy spawn diagnostic, but neither version has completed the
full compatibility and binding validation matrix.

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
is activated only when a compatible vLLM process loads general plugins.

The current Hook implements automatic `native -> generic -> skip/fail`
selection, but the full vLLM service lifecycle and target non-native GPU
Runtime Provider are not yet complete. The formal delivery baseline is
`docs/kunpeng-affinity-plugin-delivery-design.md`; it defines the implementation
status, compatibility boundaries, and remaining development steps.
