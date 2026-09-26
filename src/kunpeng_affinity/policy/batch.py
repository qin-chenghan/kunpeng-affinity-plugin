"""All-or-nothing batch resolution on top of the Linux topology analyzer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kunpeng_affinity.core.errors import (
    AffinityDiscoveryError,
    DeviceMappingError,
    PluginContractError,
    ProviderSelectionError,
)
from kunpeng_affinity.core.identity import (
    affinity_snapshot_json,
    mapping_fingerprint,
    serialized_snapshot_fingerprint,
)
from kunpeng_affinity.core.models import (
    BatchAffinityResult,
    BatchStatus,
    DeviceContext,
    DeviceMapping,
    DeviceResolution,
)
from kunpeng_affinity.providers.base import DeviceMapper
from kunpeng_affinity.topology.analyzer import analyze_bdf, normalize_bdf


class GenericAffinityProvider:
    """Resolve an ordered device batch without executing any binding."""

    def __init__(
        self,
        mapper: DeviceMapper | None = None,
        *,
        registry: Any | None = None,
        requested_provider: str | None = None,
        sysfs_root: Path | str = Path("/sys"),
        allowed_cpus: frozenset[int] | set[int] | None = None,
        snapshot_metadata: dict[str, str] | None = None,
    ) -> None:
        if mapper is None and registry is None:
            raise PluginContractError(
                "generic affinity provider requires a mapper or registry",
                code="MAPPER_UNAVAILABLE",
            )
        if mapper is not None and registry is not None:
            raise PluginContractError(
                "generic affinity provider accepts either mapper or registry",
                code="MAPPER_AMBIGUOUS",
            )
        self.mapper = mapper
        self.registry = registry
        self.requested_provider = requested_provider
        self.sysfs_root = Path(sysfs_root)
        self.allowed_cpus = allowed_cpus
        self.snapshot_metadata = dict(snapshot_metadata or {})

    def resolve_all(self, contexts: Sequence[DeviceContext]) -> BatchAffinityResult:
        ordered_contexts = tuple(contexts)
        if not ordered_contexts:
            raise PluginContractError(
                "generic affinity resolution requires a non-empty device batch",
                code="EMPTY_DEVICE_BATCH",
            )
        # Reject contexts that already disagree about the visible device set.
        fingerprints = {
            context.visibility_fingerprint
            for context in ordered_contexts
            if context.visibility_fingerprint is not None
        }
        if len(fingerprints) > 1:
            return BatchAffinityResult(
                ordered_results=(),
                expected_device_count=len(ordered_contexts),
                visibility_fingerprint=None,
                committable=False,
                status=BatchStatus.MAPPING_FAILED,
                failure_summary=(
                    "VISIBILITY_CHANGED: contexts contain multiple visibility fingerprints",
                ),
            )
        try:
            mapper = self.mapper
            if mapper is None:
                try:
                    mapper = self.registry.select(
                        ordered_contexts, requested=self.requested_provider
                    )
                except ProviderSelectionError as exc:
                    status_by_code = {
                        "PROVIDER_NOT_FOUND": BatchStatus.UNSUPPORTED_PROVIDER,
                        "PROVIDER_AMBIGUOUS": BatchStatus.AMBIGUOUS_PROVIDER,
                        "PROVIDER_PROBE_FAILED": BatchStatus.PROVIDER_PROBE_FAILED,
                    }
                    status = status_by_code.get(
                        exc.code, BatchStatus.PROVIDER_PROBE_FAILED
                    )
                    return BatchAffinityResult(
                        ordered_results=(),
                        expected_device_count=len(ordered_contexts),
                        visibility_fingerprint=next(iter(fingerprints), None),
                        committable=False,
                        status=status,
                        failure_summary=(f"{exc.code}: {exc}",),
                    )
            assert mapper is not None
            # Obtain and validate one ordered mapping before touching sysfs.
            mappings, fingerprint = self.mapping_snapshot(
                ordered_contexts, mapper=mapper
            )
        except AffinityDiscoveryError as exc:
            return BatchAffinityResult(
                ordered_results=(),
                expected_device_count=len(ordered_contexts),
                visibility_fingerprint=next(iter(fingerprints), None),
                committable=False,
                status=BatchStatus.MAPPING_FAILED,
                failure_summary=(f"{exc.code}: {exc}",),
            )

        expected_fingerprint = next(iter(fingerprints), None)
        result_fingerprint = expected_fingerprint or fingerprint
        snapshot_json: str | None = None

        resolutions: list[DeviceResolution] = []
        failures: list[str] = []
        failure_statuses: list[BatchStatus] = []

        # Analyze each mapped BDF, but commit nothing until every device succeeds.
        for context, mapping in zip(ordered_contexts, mappings, strict=True):
            try:
                affinity = analyze_bdf(
                    mapping.pci_bdf,
                    sysfs_root=self.sysfs_root,
                    allowed_cpus=self.allowed_cpus,
                    mapping_source=mapping.source,
                )
            except Exception as exc:
                raise PluginContractError(
                    "topology analyzer violated its result contract",
                    code="TOPOLOGY_CONTRACT_VIOLATION",
                ) from exc
            resolutions.append(
                DeviceResolution(context=context, mapping=mapping, affinity=affinity)
            )
            if not affinity.bindable:
                failures.append(
                    f"logical device {context.logical_device_id}: "
                    f"{affinity.failure_code or 'TOPOLOGY_FAILED'}: "
                    f"topology result is {affinity.status.value}"
                )
                failure_statuses.append(
                    BatchStatus.CPUSET_FAILED
                    if affinity.failure_code == "CPUSET_INVALID"
                    else BatchStatus.TOPOLOGY_FAILED
                )

        # A single failed or incomplete device invalidates the entire batch.
        committable = len(resolutions) == len(ordered_contexts) and not failures
        if committable:
            status = BatchStatus.COMMITTABLE
        elif failure_statuses and all(
            item is BatchStatus.CPUSET_FAILED for item in failure_statuses
        ):
            status = BatchStatus.CPUSET_FAILED
        else:
            status = BatchStatus.TOPOLOGY_FAILED
        if committable:
            snapshot_json = affinity_snapshot_json(
                mappings,
                tuple(resolutions),
                metadata={
                    "provider": mapper.name,
                    "contract_version": "1",
                    **self.snapshot_metadata,
                },
            )
            result_fingerprint = serialized_snapshot_fingerprint(snapshot_json)
        return BatchAffinityResult(
            ordered_results=tuple(resolutions),
            expected_device_count=len(ordered_contexts),
            visibility_fingerprint=result_fingerprint,
            committable=committable,
            status=status,
            snapshot_json=snapshot_json,
            failure_summary=tuple(failures),
        )

    def mapping_snapshot(
        self,
        contexts: Sequence[DeviceContext],
        *,
        mapper: DeviceMapper | None = None,
    ) -> tuple[tuple[DeviceMapping, ...], str]:
        """Capture and validate one ordered provider visibility snapshot."""
        # Canonicalize and validate the provider output before generating its digest.
        ordered_contexts = tuple(contexts)
        active_mapper = mapper or self.mapper
        if active_mapper is None:
            raise PluginContractError(
                "mapping snapshot requires a selected mapper",
                code="MAPPER_UNAVAILABLE",
            )
        try:
            raw_mappings = active_mapper.map_all(ordered_contexts)
            mappings = tuple(
                self._canonicalize_mapping(mapping) for mapping in raw_mappings
            )
        except (AffinityDiscoveryError, PluginContractError):
            raise
        except Exception as exc:
            raise PluginContractError(
                "provider map_all violated its contract",
                code="PROVIDER_CONTRACT_VIOLATION",
            ) from exc
        self._validate_mappings(ordered_contexts, mappings, mapper=active_mapper)
        return mappings, mapping_fingerprint(mappings)

    @staticmethod
    def _canonicalize_mapping(mapping: DeviceMapping) -> DeviceMapping:
        if not isinstance(mapping, DeviceMapping):
            raise PluginContractError(
                "provider returned a non-DeviceMapping value",
                code="PROVIDER_CONTRACT_VIOLATION",
            )
        try:
            bdf = normalize_bdf(mapping.pci_bdf)
        except (TypeError, ValueError) as exc:
            raise DeviceMappingError(
                f"provider returned an invalid BDF: {mapping.pci_bdf!r}",
                code="BDF_INVALID",
            ) from exc
        if bdf == mapping.pci_bdf:
            return mapping
        return DeviceMapping(
            logical_device_id=mapping.logical_device_id,
            pci_bdf=bdf,
            source=mapping.source,
            physical_device_id=mapping.physical_device_id,
            evidence=mapping.evidence,
            instance_id=mapping.instance_id,
        )

    def _validate_mappings(
        self,
        contexts: Sequence[DeviceContext],
        mappings: Sequence[DeviceMapping],
        *,
        mapper: DeviceMapper | None = None,
    ) -> None:
        if len(mappings) != len(contexts):
            raise DeviceMappingError(
                f"provider returned {len(mappings)} mappings for "
                f"{len(contexts)} devices",
                code="DEVICE_COUNT_MISMATCH",
            )
        expected_ids = tuple(context.logical_device_id for context in contexts)
        actual_ids = tuple(mapping.logical_device_id for mapping in mappings)
        if actual_ids != expected_ids:
            raise DeviceMappingError(
                f"provider returned device order {actual_ids}, expected {expected_ids}",
                code="DEVICE_ORDER_MISMATCH",
            )
        bdfs = [mapping.pci_bdf for mapping in mappings]
        active_mapper = mapper or self.mapper
        if active_mapper is None:
            raise PluginContractError(
                "mapping validation requires a selected mapper",
                code="MAPPER_UNAVAILABLE",
            )
        if not active_mapper.supports_shared_bdf and len(set(bdfs)) != len(bdfs):
            raise DeviceMappingError(
                "provider returned duplicate BDFs without shared-instance support",
                code="DEVICE_MAPPING_CONFLICT",
            )
