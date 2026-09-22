# Iluvatar Runtime Provider

The Iluvatar provider resolves the framework-visible device identity without
using the physical order reported by `ixsmi`:

```text
vLLM logical device
  -> vLLM platform get_device_uuid()
  -> normalized GPU UUID
  -> ixsmi UUID/BDF inventory
  -> canonical PCI BDF
  -> generic Linux PCIe/NUMA analyzer
```

The provider runs `ixsmi` read-only with:

```text
ixsmi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader,nounits
```

The UUID is the join key. `GPU-` prefixes, case differences, and eight-digit
PCI domains are normalized before the mapping is accepted. Duplicate UUIDs or
BDFs, missing runtime UUIDs, command failures, and incomplete inventory cause
the provider to report unsupported; no partial mapping is returned.

The vLLM adapter keeps the direct platform-BDF provider as the preferred path.
It registers this provider automatically only when the platform does not
return a usable direct BDF mapping. It may also be selected explicitly with:

```text
KUNPENG_AFFINITY_PROVIDER=iluvatar-runtime-pci
```

This component does not calculate NUMA locality and does not bind CPUs. After
it returns an ordered BDF mapping, the existing Linux topology and vLLM
configuration layers perform those steps.

The current tests use injected `ixsmi` command output and cover UUID prefix
normalization, visible-device reordering, duplicate identity, missing identity
and command failure. A real vLLM 0.23+ Iluvatar container still needs an
integration run after the runtime is upgraded.
