# Kunpeng Affinity Plugin

This repository contains the incremental demos for the Kunpeng GPU affinity
plugin. The current implementation includes the vLLM hook demo, the
framework-independent Linux topology demo, and the provider/batch resolution
demo. The vLLM hook still only observes and delegates; it does not yet inject
generic affinity results.

It also contains Demo 2, a framework-independent, read-only Linux topology
analyzer. Given a trusted PCI BDF, it follows the real sysfs parent path,
resolves NUMA evidence, and suggests the intersection of NUMA-node, online, and
currently allowed CPUs. It does not import vLLM/SGLang or execute a binding.

Demo 3 adds the framework-independent identity and batch layer. A
`DeviceContext` is mapped to a canonical PCI BDF by a `DeviceMapper`, then the
ordered batch is resolved through the same topology analyzer. Mapping errors,
duplicate devices, visibility changes, and any per-device topology failure make
the batch non-committable; no binding operation is executed.

## Demo 1 behavior

The `vllm.general_plugins` entry point installs an idempotent wrapper around
`vllm.utils.numa_utils.configure_subprocess`. The wrapper logs the process ID,
process kind, ranks, vLLM version, and `numa_bind` state, then delegates to the
original function without changing configuration or binding behavior.

The target contract is vLLM 0.23.0. The hook shape is also covered by unit tests
for vLLM 0.26.0, but that version has not completed the full compatibility and
binding validation matrix.

## Local tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

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
semantics. The static provider is a configuration/test implementation; a
target GPU runtime provider must be added before framework integration.

## Package and transfer

Build a portable pure-Python wheel:

```bash
python3 -m pip wheel --no-deps --wheel-dir dist .
```

The resulting `py3-none-any` wheel can be installed on another Linux machine
with Python 3.10 or newer. The topology CLI and provider/batch core do not
require vLLM, SGLang, CUDA, or a specific CPU architecture. The vLLM entry point
is activated only when a compatible vLLM process loads general plugins.

The current hook observes and delegates; it does not yet inject generic NUMA
results or perform binding. See `docs/design.md` for the implementation status,
compatibility boundaries, and remaining development steps.
