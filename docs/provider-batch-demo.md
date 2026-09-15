# Provider and Batch Demo

## Boundary

This demo defines the boundary between framework-visible device identity and
the provider-neutral Linux topology analyzer:

```text
DeviceContext -> DeviceMapper -> canonical PCI BDF
              -> GenericAffinityProvider -> AffinityResult
```

The demo does not enumerate GPUs, call a vendor runtime from the generic core,
execute `numactl`, or change process affinity. The concrete mappers are:

- `StaticMappingProvider`: requires an explicit logical-device-to-BDF mapping
  and is intended for deployment configuration and tests.
- `LinuxContextProvider`: accepts only a BDF already carried by
  `DeviceContext.explicit_bdf`, a BDF-valued `runtime_device_id`, or a real
  `device_node` path resolving below the configured sysfs devices tree.
- `VllmPlatformProvider`: consumes the vLLM 0.23 platform methods
  `get_all_gpu_pci_bus_ids()` and `device_id_to_physical_device_id()`; it is
  loaded by an adapter and does not import vLLM in the framework-neutral core.

`LinuxContextProvider` does not infer identity from logical IDs, sysfs ordering,
bus numbers, names, or device model. A target GPU runtime provider is still
required when the target framework or hardware exposes none of these trusted
facts. The vLLM provider is not a replacement for a target provider on
framework platforms that do not implement these APIs.

## Mapping contract

`DeviceMapper.map_all()` must return exactly one `DeviceMapping` for every
input `DeviceContext`, in the same order. The batch layer then:

1. Normalizes and validates every provider BDF.
2. Verifies logical-device order and result count.
3. Rejects duplicate BDFs unless a provider explicitly supports shared device
   instances.
4. Resolves every BDF through the existing sysfs analyzer.
5. Returns `committable=False` if any device is not bindable.

The batch layer never submits a partial framework configuration. It returns
all per-device results when mapping succeeds so callers can inspect diagnostics,
but only `committable=True` may be converted into framework settings.

## Visibility consistency

`DeviceContext.visibility_fingerprint` is optional. If present, all contexts in
one batch must carry the same value. Different fingerprints return
`VISIBILITY_CHANGED` before topology analysis begins.

## Provider selection

`ProviderRegistry.select()` supports an explicitly requested provider or
automatic selection. Automatic selection succeeds only when exactly one
registered provider reports support. Zero providers returns
`PROVIDER_NOT_FOUND`; multiple providers returns `PROVIDER_AMBIGUOUS`.

## Verification

The unit tests cover input-order preservation, canonicalization, explicit and
context-derived BDFs, device-path resolution, provider ambiguity, missing
mappings, duplicate/malformed mapping boundaries, visibility mismatch,
per-device topology failure, and successful multi-device resolution.
