# vLLM Generic-Fallback Spawn Diagnostics

## Purpose

This diagnostic validates the narrow integration path needed before a real
vLLM service test:

```text
vLLM plugin discovery
-> configure_subprocess Hook
-> vLLM logical device identity
-> vLLM platform PCI BDF
-> generic Linux PCIe/NUMA analysis
-> numa_bind_nodes injection
-> original vLLM numactl wrapper
-> dummy child CPU affinity
```

It does not start an engine, load a model, run GPU computation, or validate an
unsupported accelerator Provider. One entry forces generic discovery directly;
the other exercises the production native-to-generic decision with a controlled
native no-result response.

## Safety Gates

The forced path is selected only when all of the following are true:

- `KUNPENG_AFFINITY_VLLM_FORCE_GENERIC` is a true value;
- vLLM's `parallel_config.numa_bind` is enabled;
- the user did not provide `numa_bind_nodes`;
- the plugin's vendor-neutral automatic-binding eligibility checks succeed;
- the platform exposes a valid device count and logical-to-physical identity;
- every visible device maps to a unique PCI BDF;
- every BDF produces a bindable Linux topology result.

Explicit nodes and a disabled vLLM NUMA binding switch are always preserved.
Discovery failure raises before the original binding executor is entered. The
diagnostic never guesses a BDF or commits a partial device list.

## Running

First install the checkout into the Python environment that contains vLLM:

```bash
PYTHON_BIN=/path/to/vllm/python ./scripts/install-source.sh
PYTHON_BIN=/path/to/vllm/python ./scripts/verify-source.sh
```

Then run the isolated spawn diagnostic:

```bash
PYTHON_BIN=/path/to/vllm/python ./scripts/verify-vllm-generic-spawn.sh
```

To verify the production `native -> generic` decision instead of directly
forcing the generic branch, run:

```bash
PYTHON_BIN=/path/to/vllm/python ./scripts/verify-vllm-auto-fallback-spawn.sh
```

The second command makes the native query return no result, confirms it was
called exactly once, then requires the generic fallback and child binding to
succeed.

A successful JSON result contains:

- the selected `verification_path` and native query call count;
- `generic_fallback_verified: true` for the auto-fallback diagnostic;
- the generated `generic_nodes` list;
- the expected NUMA CPU list;
- the dummy child's actual `Cpus_allowed_list` and `Mems_allowed_list`;
- selected fields from the child's `numactl --show` output and a separate
  `memory_policy_verified` result.

In forced-generic mode, the script replaces `get_auto_numa_nodes()` with a
function that raises if called; success is direct evidence that native GPU NUMA
discovery was bypassed. In auto-fallback mode, the replacement returns `None`
and the script requires exactly one native call before generic discovery. Before
importing vLLM, both modes set
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, which vLLM requires before it will replace
the multiprocessing executable with its numactl wrapper. The child is started
with Python multiprocessing `spawn` while the real vLLM
`configure_subprocess()` context is active, so its CPU affinity tests the
original vLLM numactl execution path rather than a plugin-owned binding
implementation.

After the child exits, the diagnostic closes its multiprocessing process and
stops the spawn resource tracker explicitly. This prevents orphaned tracker
zombies in minimal test containers whose PID 1 does not reap children.

`Mems_allowed_list` reports the process or cgroup's permitted memory nodes; it
does not prove the active NUMA memory policy. The diagnostic therefore checks
the child's `numactl --show` policy and `membind` fields separately. vLLM may
intentionally fall back to CPU-only binding when the environment rejects
`--membind`; in that case CPU verification can succeed while
`memory_policy_verified` is false.

## Interpretation

Success proves the tested framework package can discover the plugin, map its
visible device through the platform PCI identity API, resolve that BDF through
Linux sysfs, inject the node list, and bind a dummy child through vLLM's
existing wrapper.

Both modes passed in the isolated vLLM 0.26.0 single-GPU environment recorded
by the project test report and were rerun successfully after the Registry,
visibility-fingerprint, and commit-transaction changes in commit `0e8367e`.
That result is compatibility evidence for the exact tested package and topology,
not a blanket vLLM 0.26 compatibility claim.

It does not prove support for other framework versions, multi-device rank
layouts, Ray or external launchers, EngineCore-to-Worker CPU supersets, real
service startup, or target hardware whose platform does not expose the vLLM
PCI identity methods.
