"""Deterministic provider selection."""

from __future__ import annotations

from collections.abc import Sequence

from kunpeng_affinity.core.errors import (
    PluginConfigError,
    PluginContractError,
    ProviderSelectionError,
)
from kunpeng_affinity.core.models import DeviceContext, ProbeResult, ProbeStatus
from kunpeng_affinity.providers.base import DeviceMapper


class ProviderRegistry:
    """Select exactly one mapper without relying on registration order."""

    def __init__(self, providers: Sequence[DeviceMapper] = ()) -> None:
        self._providers: dict[str, DeviceMapper] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: DeviceMapper) -> None:
        if provider.name in self._providers:
            raise ProviderSelectionError(
                f"provider {provider.name!r} is registered more than once",
                code="PROVIDER_AMBIGUOUS",
            )
        self._providers[provider.name] = provider

    def select(
        self,
        contexts: Sequence[DeviceContext],
        requested: str | None = None,
    ) -> DeviceMapper:
        if not contexts:
            raise PluginContractError(
                "provider selection requires a non-empty device batch",
                code="EMPTY_DEVICE_BATCH",
            )

        def probe(provider: DeviceMapper):
            try:
                result = provider.probe(contexts)
            except Exception as exc:
                raise ProviderSelectionError(
                    f"provider {provider.name!r} probe failed: {exc}",
                    code="PROVIDER_PROBE_FAILED",
                ) from exc
            if not isinstance(result, ProbeResult):
                raise PluginContractError(
                    f"provider {provider.name!r} returned an invalid probe result",
                    code="PROVIDER_CONTRACT_VIOLATION",
                )
            if result.status is ProbeStatus.INVALID_CONTEXT:
                raise PluginContractError(
                    result.reason or f"provider {provider.name!r} received invalid context",
                    code="INVALID_PROVIDER_CONTEXT",
                )
            return result

        if requested is not None:
            try:
                provider = self._providers[requested]
            except KeyError as exc:
                raise ProviderSelectionError(
                    f"provider {requested!r} is not registered",
                    code="PROVIDER_NOT_FOUND",
                ) from exc
            result = probe(provider)
            if result.status is ProbeStatus.NO_MATCH:
                raise PluginConfigError(
                    result.reason or f"provider {requested!r} is not supported",
                    code="PROVIDER_NOT_FOUND",
                )
            return provider

        supported = [
            provider
            for provider in self._providers.values()
            if probe(provider).status is ProbeStatus.MATCH
        ]
        if not supported:
            raise ProviderSelectionError(
                "no registered provider supports the requested devices",
                code="PROVIDER_NOT_FOUND",
            )
        if len(supported) > 1:
            names = ", ".join(provider.name for provider in supported)
            raise ProviderSelectionError(
                f"multiple providers support the requested devices: {names}",
                code="PROVIDER_AMBIGUOUS",
            )
        return supported[0]
