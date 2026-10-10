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
  -> if no node is returned, resolve direct BDF, Ascend sysfs, or runtime UUID
     identity and then Linux topology
  -> auto: return None on discovery failure
  -> strict: raise the discovery failure
```

When `ASCEND_RT_VISIBLE_DEVICES` is set, the generic adapter builds the complete
logical-device batch from that ordered list and uses the Ascend sysfs
BDF-to-device mapping without consulting `torch.cuda`, including when the
wrapper Provider `sglang-runtime-pci` is selected explicitly. On other platforms it
first accepts a direct runtime BDF when Torch exposes one, then falls back to a
runtime UUID and the Iluvatar provider's UUID-to-`ixsmi`-BDF mapping. The shared
topology layer then follows the Linux PCI parent chain and computes:

```text
NUMA node CPUs ∩ online CPUs ∩ current process allowed CPUs
```

No logical device is associated with a host GPU by enumeration order.

## Source layout

| File | Responsibility |
|---|---|
| `src/kunpeng_affinity/sglang_plugin.py` | SGLang entry point and around-hook decision order |
| `src/kunpeng_affinity/adapters/sglang_generic.py` | Torch runtime facade, Provider selection and BDF mapping |
| `src/kunpeng_affinity/providers/ascend_sysfs.py` | Ascend visibility and sysfs BDF-to-device provider |
| `src/kunpeng_affinity/providers/iluvatar_runtime.py` | Iluvatar UUID/BDF runtime provider |
| `src/kunpeng_affinity/policy/batch.py` | Ordered all-or-nothing mapping and topology resolution |
| `src/kunpeng_affinity/topology/analyzer.py` | Linux PCIe, NUMA and CPU-set analysis |
| `tests/test_sglang_adapter.py` | Fake Torch runtime and hook decision contract tests |

## Current boundary

The implementation accepts the validated source baseline 0.5.18 and the
observed compatibility build 0.5.17.dev386+gc5bd3d7dc, and checks the target NUMA-query
signature before registering the Hook. An Ascend TP=4 service on the observed
0.5.17.dev386+gc5bd3d7dc build has completed startup and inference requests with plugin
registration and generic NUMA-path logs. This moves the ordinary Engine path
beyond source-only validation, but per-process rank, CPU-affinity, and memory-
policy evidence is still pending. The preliminary performance samples are not
a final performance result; see `sglang-ascend-tp4-validation-progress.md`.

The adapter does not claim Data Parallel controller or Ray actor coverage.
Generic fallback fails closed on those call sites until their separate launch
contracts are validated. It also fails closed when
`SGLANG_SET_CPU_AFFINITY=1` would overwrite the resulting CPU affinity.

The default V2 subprocess path requires `numactl`. Without that executable,
`auto` mode preserves SGLang's unbound behavior. `SGLANG_NUMA_BIND_V2=0` may be
used to select SGLang's existing in-process libnuma path; it does not require
the command, but libnuma availability and memory-policy permission remain
mandatory.

SGLang catches exceptions raised while executing general-plugin entry points.
Therefore, `strict` failures detected during plugin registration may be logged
without aborting framework startup; strict failures raised later by an applied
query Hook still propagate. Integration validation must observe both the
plugin's `registered` message and SGLang's `Applied hook` message.
