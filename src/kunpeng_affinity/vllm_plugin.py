"""vLLM general-plugin entry point for the subprocess hook demo."""

from __future__ import annotations

import inspect
import logging
import os
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_HOOK_MARKER = "__kunpeng_affinity_original__"
_REQUIRED_PARAMETERS = {
    "vllm_config",
    "local_rank",
    "dp_local_rank",
    "process_kind",
}
_TARGET_VERSION = "0.23.0"
_AUXILIARY_DEMO_VERSION = "0.26.0"


def _vllm_version() -> str:
    try:
        return version("vllm")
    except PackageNotFoundError:
        return "unknown"


def _argument(
    args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str, default: Any
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


def _numa_bind_enabled(vllm_config: Any) -> Any:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return getattr(parallel_config, "numa_bind", "unknown")


def register() -> None:
    """Install the Demo 1 hook in the current vLLM process."""
    from vllm.utils import numa_utils

    current = numa_utils.configure_subprocess
    if hasattr(current, _HOOK_MARKER):
        return

    parameters = set(inspect.signature(current).parameters)
    missing = _REQUIRED_PARAMETERS - parameters
    if missing:
        raise RuntimeError(
            "Unsupported vLLM configure_subprocess signature; missing parameters: "
            + ", ".join(sorted(missing))
        )

    detected_version = _vllm_version()
    if detected_version not in {_TARGET_VERSION, _AUXILIARY_DEMO_VERSION}:
        logger.warning(
            "[kunpeng-affinity-demo] unvalidated vLLM version=%s; "
            "target=%s, auxiliary-demo=%s",
            detected_version,
            _TARGET_VERSION,
            _AUXILIARY_DEMO_VERSION,
        )

    @contextmanager
    def configure_subprocess_wrapper(*args: Any, **kwargs: Any) -> Iterator[None]:
        vllm_config = _argument(args, kwargs, 0, "vllm_config", None)
        local_rank = _argument(args, kwargs, 1, "local_rank", None)
        dp_local_rank = _argument(args, kwargs, 2, "dp_local_rank", None)
        process_kind = _argument(args, kwargs, 3, "process_kind", "worker")
        logger.warning(
            "[kunpeng-affinity-demo] pid=%s vllm=%s process_kind=%s "
            "local_rank=%s dp_local_rank=%s numa_bind=%s",
            os.getpid(),
            detected_version,
            process_kind,
            local_rank,
            dp_local_rank,
            _numa_bind_enabled(vllm_config),
        )
        with current(*args, **kwargs):
            yield

    setattr(configure_subprocess_wrapper, _HOOK_MARKER, current)
    numa_utils.configure_subprocess = configure_subprocess_wrapper
    logger.warning(
        "[kunpeng-affinity-demo] installed vLLM subprocess hook pid=%s vllm=%s",
        os.getpid(),
        detected_version,
    )
