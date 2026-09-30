"""vLLM subprocess lifecycle and affinity transaction orchestration."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from kunpeng_affinity.adapters.vllm_contract import TRANSACTION_MARKER
from kunpeng_affinity.adapters.vllm_execution import (
    VllmCall,
    automatic_affinity_context,
    strict_failure,
)
from kunpeng_affinity.config import PluginMode
from kunpeng_affinity.core.errors import (
    AffinityDiscoveryError,
    AffinityIntegrationError,
)


def should_resolve(parallel_config: Any, process_kind: str) -> bool:
    if parallel_config is None or not getattr(parallel_config, "numa_bind", False):
        return False
    if getattr(parallel_config, "numa_bind_nodes", None) is not None:
        return False
    return process_kind in {"worker", "EngineCore"}


@dataclass(frozen=True)
class VllmAffinityAdapter:
    """Run vLLM subprocess calls through the affinity decision pipeline."""

    current: Any
    numa_utils: Any
    mode: PluginMode
    force_generic: bool
    requested_provider: str | None
    detected_version: str
    platform_factory: Callable[[], Any]
    resolver: Callable[..., Any]
    visibility_validator: Callable[..., None]
    commit_nodes: Callable[..., Any]
    inherited_validator: Callable[..., None]
    transaction_releaser: Callable[[Any], None]
    logger: logging.Logger

    @contextmanager
    def configure_subprocess(self, *args: Any, **kwargs: Any) -> Iterator[None]:
        call = VllmCall.from_invocation(args, kwargs)
        self.logger.warning(
            "[kunpeng-affinity] pid=%s vllm=%s process_kind=%s local_rank=%s dp_local_rank=%s numa_bind=%s",
            os.getpid(),
            self.detected_version,
            call.process_kind,
            call.local_rank,
            call.dp_local_rank,
            call.numa_bind,
        )

        if getattr(call.parallel_config, TRANSACTION_MARKER, None) is not None:
            try:
                self.inherited_validator(
                    call.parallel_config,
                    numa_utils=self.numa_utils,
                    platform=self.platform_factory(),
                    local_rank=call.local_rank,
                    dp_local_rank=call.dp_local_rank,
                    process_kind=call.process_kind,
                )
            except AffinityDiscoveryError as exc:
                if exc.code == "TRANSACTION_IN_PROGRESS":
                    raise AffinityIntegrationError(
                        "vLLM affinity transaction is already in progress",
                        code=exc.code,
                    ) from exc
                if self.mode is PluginMode.STRICT:
                    raise strict_failure(exc) from exc
                self.logger.warning(
                    "[kunpeng-affinity] inherited transaction released code=%s: %s",
                    exc.code,
                    exc,
                )
                self.transaction_releaser(call.parallel_config)
            with self.current(*args, **kwargs):
                yield
            return

        if should_resolve(call.parallel_config, call.process_kind):
            with automatic_affinity_context(
                current=self.current,
                numa_utils=self.numa_utils,
                mode=self.mode,
                force_generic=self.force_generic,
                requested_provider=self.requested_provider,
                call=call,
                platform_factory=self.platform_factory,
                resolver=self.resolver,
                visibility_validator=self.visibility_validator,
                commit_nodes=self.commit_nodes,
                logger=self.logger,
            ):
                yield
            return

        with self.current(*args, **kwargs):
            yield
