# Framework-independent Linux topology demo

## Boundary

The analyzer accepts one or more trusted PCI BDFs and reads Linux sysfs. It does
not enumerate accelerators or infer runtime GPU ordering because Linux has no
provider-neutral mapping from a framework's logical GPU ID to a PCI function.

```text
provider or deployment configuration: logical GPU -> trusted BDF
this demo: trusted BDF -> PCIe parent path -> NUMA node -> suggested CPUs
```

The module imports neither vLLM nor SGLang. It does not invoke NVML,
`nvidia-smi`, another vendor utility, `numactl`, or `sched_setaffinity`.

## Usage

For a live, human-readable environment probe from a source checkout:

```bash
./scripts/probe-host.sh
```

The probe enumerates PCI base-class `0x03` display controllers and `0x12`
processing accelerators as candidates, then runs the analyzer for each BDF. The
candidate scan is not a replacement for a runtime Provider. Override it when a
trusted mapping is available:

```bash
AFFINITY_BDFS=0000:41:00.0,0000:81:00.0 ./scripts/probe-host.sh
```

For every device, the output includes PCI identity fields, every endpoint and
bridge in the real sysfs parent path, the root bus, selected NUMA node and
evidence source, node/online/allowed CPU sets, and the final suggested set. The
probe exits nonzero when discovery is empty or any result is not `success`.

From the repository checkout:

```bash
PYTHONPATH=src python3 -m kunpeng_affinity.topology.cli \
  --bdf 00000000:AB:00.0
```

Analyze several already-mapped devices by repeating `--bdf`:

```bash
PYTHONPATH=src python3 -m kunpeng_affinity.topology.cli \
  --bdf 0000:ab:00.0 \
  --bdf 0000:cd:00.0 \
  --json
```

`--sysfs-root` points at a fixture instead of `/sys`. `--allowed-cpus` replaces
the current process affinity only for diagnostics and tests; normal use reads
`sched_getaffinity(0)`.

The exit status is zero only when all supplied BDFs produce `success`.

## Result states

- `success`: the BDF, complete parent path, consistent NUMA evidence, and a
  non-empty CPU intersection were established. `bindable` is true.
- `partial`: the PCI path is known but no unique NUMA node can be proven. No
  binding recommendation is emitted.
- `failed`: the BDF/path is invalid, evidence conflicts, required sysfs data is
  invalid, or the final CPU intersection is empty.

Every result separates the input mapping source, NUMA evidence source, PCIe
path, source CPU sets, final target set, and diagnostics. A later framework
adapter can consume `AffinityResult` without duplicating topology logic.

## Topology rules

1. Normalize BDFs to lowercase `dddd:bb:ss.f`. Eight-digit vendor domains are
   accepted only when their high bits are zero.
2. Resolve `/sys/bus/pci/devices/<BDF>` and require the target to remain in the
   sysfs devices tree.
3. Walk only real parent directories until the PCI root-bus representation is
   reached. The same loop handles direct, single-switch, and multi-switch paths.
4. Prefer valid Endpoint NUMA evidence, otherwise use the nearest valid
   ancestor or a uniquely matching Endpoint `local_cpulist`.
5. Reject conflicting Endpoint, ancestor, and local-CPU evidence.
6. Compute `node cpulist intersect online CPUs intersect current affinity`.

## Test coverage

The synthetic sysfs tests cover BDF normalization, CPU-list syntax, direct
attachment, single- and multi-level switch paths, nearest-ancestor fallback,
conflicting and unknown NUMA evidence, a broken parent chain, an empty final CPU
set, and an unrelated memory-only NUMA node.
