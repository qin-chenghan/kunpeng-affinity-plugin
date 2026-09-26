"""Stable errors for the provider and generic affinity layers."""

from __future__ import annotations


class AffinityError(RuntimeError):
    """Base class for errors that are safe for an adapter to classify."""

    code = "AFFINITY_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class PluginConfigError(AffinityError):
    """Plugin configuration is invalid and cannot be interpreted safely."""


class PluginContractError(AffinityError):
    """An internal plugin protocol or value-object contract was violated."""


class AffinityDiscoveryError(AffinityError):
    """A device mapping or topology fact could not be established."""


class DeviceMappingError(AffinityDiscoveryError):
    """The logical-device to PCI-function mapping is invalid or incomplete."""


class ProviderSelectionError(DeviceMappingError):
    """No unique provider can map the requested devices."""


class BatchValidationError(AffinityDiscoveryError):
    """A batch cannot be committed as a complete affinity result."""


class AffinityIntegrationError(AffinityError):
    """A strict framework integration cannot continue safely."""


class NativeContractError(AffinityError):
    """A framework-native query violated its declared return contract."""
