"""Apply one resolved vLLM affinity decision around the native context."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from kunpeng_affinity.config import PluginMode
from kunpeng_affinity.core.errors import (
    AffinityDiscoveryError,
    AffinityIntegrationError,
)


def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str, default: Any) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


@dataclass(frozen=True)
class VllmCall:
    """Arguments and process metadata for one configure_subprocess call."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    parallel_config: Any
    process_kind: str
    local_rank: int | None
    dp_local_rank: int | None

    @classmethod
    def from_invocation(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> VllmCall:
        vllm_config = _argument(args, kwargs, 0, "vllm_config", None)
        return cls(
            args=args,
            kwargs=kwargs,
            parallel_config=getattr(vllm_config, "parallel_config", None),
            local_rank=_argument(args, kwargs, 1, "local_rank", None),
            dp_local_rank=_argument(args, kwargs, 2, "dp_local_rank", None),
            process_kind=_argument(args, kwargs, 3, "process_kind", "worker"),
        )

    @property
    def numa_bind(self) -> Any:
        return getattr(self.parallel_config, "numa_bind", "unknown")


def strict_failure(error: AffinityDiscoveryError) -> AffinityIntegrationError:
    return AffinityIntegrationError(
        f"strict vLLM affinity discovery failed [{error.code}]: {error}",
        code=error.code,
    )


@contextmanager
def automatic_affinity_context(
    *,
    current: Any,
    numa_utils: Any,
    mode: PluginMode,
    force_generic: bool,
    requested_provider: str | None,
    call: VllmCall,
    platform_factory: Callable[[], Any],
    resolver: Callable[..., Any],
    visibility_validator: Callable[..., None],
    commit_nodes: Callable[..., Any],
    logger: logging.Logger,
) -> Iterator[None]:
    """Resolve, commit, and delegate one automatic-affinity subprocess call."""
    platform = platform_factory()
    try:
        resolution = resolver(
            numa_utils,
            platform,
            force_generic=force_generic,
            requested_provider=requested_provider,
            process_kind=call.process_kind,
            local_rank=call.local_rank,
            dp_local_rank=call.dp_local_rank,
        )
        visibility_validator(
            resolution,
            platform,
            process_kind=call.process_kind,
            local_rank=call.local_rank,
            dp_local_rank=call.dp_local_rank,
        )
        if resolution.source == "native":
            with current(*call.args, **call.kwargs):
                yield
            return
        commit = commit_nodes(
            call.parallel_config,
            list(resolution.nodes),
            visibility_fingerprint=resolution.visibility_fingerprint,
            snapshot_json=resolution.snapshot_json,
            requested_provider=resolution.requested_provider,
        )
    except AffinityDiscoveryError as exc:
        if force_generic or mode is PluginMode.STRICT:
            raise strict_failure(exc) from exc
        logger.warning(
            "[kunpeng-affinity] automatic affinity skipped code=%s: %s; launching without additional binding",
            exc.code,
            exc,
        )
        if getattr(call.parallel_config, "numa_bind_cpus", None) is not None:
            with current(*call.args, **call.kwargs):
                yield
            return
        yield
        return

    logger.warning(
        "[kunpeng-affinity] selected NUMA nodes=%s source=%s%s",
        list(resolution.nodes),
        resolution.source,
        "; native GPU NUMA query bypassed" if force_generic else "",
    )
    try:
        manager = current(*call.args, **call.kwargs)
        manager.__enter__()
    except BaseException:
        commit.rollback()
        raise
    try:
        commit.mark_committed()
    except BaseException:
        error = sys.exc_info()
        try:
            manager.__exit__(*error)
        finally:
            commit.rollback()
        raise

    try:
        yield
    except BaseException:
        if not manager.__exit__(*sys.exc_info()):
            raise
    else:
        manager.__exit__(None, None, None)
