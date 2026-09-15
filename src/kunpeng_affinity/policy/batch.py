"""All-or-nothing batch resolution on top of the Linux topology analyzer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from kunpeng_affinity.core.errors import AffinityError, DeviceMappingError
from kunpeng_affinity.core.models import (
    BatchAffinityResult,
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
        mapper: DeviceMapper,
        *,
        sysfs_root: Path | str = Path("/sys"),
        allowed_cpus: frozenset[int] | set[int] | None = None,
    ) -> None:
        self.mapper = mapper
        self.sysfs_root = Path(sysfs_root)
        self.allowed_cpus = allowed_cpus

    def resolve_all(self, contexts: Sequence[DeviceContext]) -> BatchAffinityResult:
        ordered_contexts = tuple(contexts)
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
                failure_summary=(
                    "VISIBILITY_CHANGED: contexts contain multiple visibility fingerprints",
                ),
            )
        fingerprint = next(iter(fingerprints), None)
        try:
            mappings = tuple(
                self._canonicalize_mapping(mapping)
                for mapping in self.mapper.map_all(ordered_contexts)
            )
            self._validate_mappings(ordered_contexts, mappings)
        except AffinityError as exc:
            return BatchAffinityResult(
                ordered_results=(),
                expected_device_count=len(ordered_contexts),
                visibility_fingerprint=fingerprint,
                committable=False,
                failure_summary=(f"{exc.code}: {exc}",),
            )

        resolutions: list[DeviceResolution] = []
        failures: list[str] = []
        for context, mapping in zip(ordered_contexts, mappings, strict=True):
            affinity = analyze_bdf(
                mapping.pci_bdf,
                sysfs_root=self.sysfs_root,
                allowed_cpus=self.allowed_cpus,
                mapping_source=mapping.source,
            )
            resolutions.append(
                DeviceResolution(context=context, mapping=mapping, affinity=affinity)
            )
            if not affinity.bindable:
                failures.append(
                    f"logical device {context.logical_device_id}: "
                    f"topology result is {affinity.status.value}"
                )

        committable = len(resolutions) == len(ordered_contexts) and not failures
        return BatchAffinityResult(
            ordered_results=tuple(resolutions),
            expected_device_count=len(ordered_contexts),
            visibility_fingerprint=fingerprint,
            committable=committable,
            failure_summary=tuple(failures),
        )

    @staticmethod
    def _canonicalize_mapping(mapping: DeviceMapping) -> DeviceMapping:
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
        if not self.mapper.supports_shared_bdf and len(set(bdfs)) != len(bdfs):
            raise DeviceMappingError(
                "provider returned duplicate BDFs without shared-instance support",
                code="DEVICE_MAPPING_CONFLICT",
            )
