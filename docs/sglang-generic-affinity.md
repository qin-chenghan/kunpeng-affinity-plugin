# SGLang Generic NUMA Adapter

This adapter targets SGLang 0.5.18 and adds a provider-backed fallback without
modifying SGLang source code.

## Insertion point

SGLang discovers the package through the `sglang.srt.plugins` entry-point group.
The package registers an `AROUND` hook for:

```text
sglang.srt.utils.numa_utils.get_numa_node_if_available
```

That function is called by SGLang's ordinary Engine subprocess launcher before
`configure_subprocess` creates the existing `numactl` executable wrapper. The
plugin supplies only the NUMA node decision; SGLang continues to perform the
actual process launch and binding.

## Decision order

```text
server_args.numa_node is set
  -> return SGLang's explicit result
otherwise
  -> call SGLang's native GPU NUMA query
  -> if no node is returned, resolve runtime identity and Linux topology
  -> auto: return None on discovery failure
  -> strict: raise the discovery failure
```

The generic adapter first accepts a direct runtime BDF when Torch exposes one.
If it does not, it obtains a runtime UUID and uses the Iluvatar provider's
UUID-to-`ixsmi`-BDF mapping. The shared topology layer then follows the Linux
PCI parent chain and computes:

```text
NUMA node CPUs ∩ online CPUs ∩ current process allowed CPUs
```

No logical device is associated with a host GPU by enumeration order.

## Source layout

| File | Responsibility |
|---|---|
| `src/kunpeng_affinity/sglang_plugin.py` | SGLang entry point and around-hook decision order |
| `src/kunpeng_affinity/adapters/sglang_generic.py` | Torch runtime facade, direct-BDF path and UUID/BDF fallback |
| `src/kunpeng_affinity/providers/iluvatar_runtime.py` | Iluvatar UUID/BDF runtime provider |
| `src/kunpeng_affinity/policy/batch.py` | Ordered all-or-nothing mapping and topology resolution |
| `src/kunpeng_affinity/topology/analyzer.py` | Linux PCIe, NUMA and CPU-set analysis |
| `tests/test_sglang_adapter.py` | Fake Torch runtime and hook decision contract tests |

## Current boundary

The implementation is source-level and fake-runtime tested. It does not claim
real SGLang service validation, multi-GPU rank validation, Data Parallel
controller coverage, or Ray actor coverage. Those paths use separate SGLang
launch code and require independent contracts before they can be included in
the supported scope.
