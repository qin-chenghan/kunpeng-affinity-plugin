"""Immutable plugin configuration with environment-over-file precedence."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from kunpeng_affinity.core.errors import PluginConfigError


class PluginMode(str, Enum):
    OFF = "off"
    AUTO = "auto"
    STRICT = "strict"


class CpuPolicy(str, Enum):
    NODE = "node"
    EXACT = "exact"


class DiagnosticLevel(str, Enum):
    ERROR = "error"
    SUMMARY = "summary"
    DETAIL = "detail"


@dataclass(frozen=True)
class PluginConfig:
    mode: PluginMode = PluginMode.AUTO
    provider: str = "auto"
    cpu_policy: CpuPolicy = CpuPolicy.NODE
    diagnostic_level: DiagnosticLevel = DiagnosticLevel.SUMMARY
    config_file: Path | None = None


_CONFIG_FILE_ENV = "KUNPENG_AFFINITY_CONFIG"
_ENV_FIELDS = {
    "mode": "KUNPENG_AFFINITY_MODE",
    "provider": "KUNPENG_AFFINITY_PROVIDER",
    "cpu_policy": "KUNPENG_AFFINITY_CPU_POLICY",
    "diagnostic_level": "KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL",
}
_ALLOWED_FILE_FIELDS = frozenset(_ENV_FIELDS)
_PROVIDER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_EnumT = TypeVar("_EnumT", bound=Enum)


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PluginConfigError(
                f"duplicate configuration field {key!r}",
                code="CONFIG_FIELD_DUPLICATE",
            )
        result[key] = value
    return result


def _config_path(environ: Mapping[str, str]) -> Path | None:
    raw_path = environ.get(_CONFIG_FILE_ENV)
    if raw_path is None or not raw_path.strip():
        return None
    return Path(raw_path).expanduser()


def _read_config_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PluginConfigError(
            f"cannot read plugin configuration file {path}: {exc}",
            code="CONFIG_FILE_READ_FAILED",
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PluginConfigError(
            f"plugin configuration file is not valid UTF-8: {path}",
            code="CONFIG_FILE_ENCODING_INVALID",
        ) from exc
    try:
        data = json.loads(text, object_pairs_hook=_pairs_without_duplicates)
    except PluginConfigError:
        raise
    except json.JSONDecodeError as exc:
        raise PluginConfigError(
            f"plugin configuration file is not valid JSON: {exc}",
            code="CONFIG_FILE_JSON_INVALID",
        ) from exc
    if not isinstance(data, dict):
        raise PluginConfigError(
            "plugin configuration root must be a JSON object",
            code="CONFIG_FILE_ROOT_INVALID",
        )
    return data


def _reject_unknown_fields(data: Mapping[str, Any]) -> None:
    unknown = sorted(set(data) - _ALLOWED_FILE_FIELDS)
    if unknown:
        raise PluginConfigError(
            "unknown plugin configuration fields: " + ", ".join(unknown),
            code="CONFIG_FIELD_UNKNOWN",
        )


def _string_value(field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PluginConfigError(
            f"configuration field {field!r} must be a non-empty string",
            code="CONFIG_VALUE_INVALID",
        )
    return value.strip()


def _enum_value(field: str, value: Any, enum_type: type[_EnumT]) -> _EnumT:
    normalized = _string_value(field, value).lower()
    try:
        return enum_type(normalized)
    except ValueError as exc:
        expected = ", ".join(item.value for item in enum_type)
        raise PluginConfigError(
            f"invalid {field}={normalized!r}; expected {expected}",
            code="CONFIG_VALUE_INVALID",
        ) from exc


def _mode_value(value: Any) -> PluginMode:
    return _enum_value("mode", value, PluginMode)


def load_plugin_mode(environ: Mapping[str, str] | None = None) -> PluginMode:
    """Resolve mode only, allowing an environment ``off`` to avoid file I/O."""
    source = os.environ if environ is None else environ
    if _ENV_FIELDS["mode"] in source:
        return _mode_value(source[_ENV_FIELDS["mode"]])
    data = _read_config_file(_config_path(source))
    _reject_unknown_fields(data)
    return _mode_value(data.get("mode", PluginMode.AUTO.value))


def load_plugin_config(environ: Mapping[str, str] | None = None) -> PluginConfig:
    """Load and validate one immutable configuration snapshot."""
    source = os.environ if environ is None else environ
    path = _config_path(source)
    data = _read_config_file(path)
    _reject_unknown_fields(data)

    values: dict[str, Any] = {
        "mode": PluginMode.AUTO.value,
        "provider": "auto",
        "cpu_policy": CpuPolicy.NODE.value,
        "diagnostic_level": DiagnosticLevel.SUMMARY.value,
    }
    values.update(data)
    for field, variable in _ENV_FIELDS.items():
        if variable in source:
            values[field] = source[variable]

    provider = _string_value("provider", values["provider"])
    if _PROVIDER_RE.fullmatch(provider) is None:
        raise PluginConfigError(
            f"invalid provider={provider!r}",
            code="CONFIG_VALUE_INVALID",
        )

    return PluginConfig(
        mode=_mode_value(values["mode"]),
        provider=provider,
        cpu_policy=_enum_value("cpu_policy", values["cpu_policy"], CpuPolicy),
        diagnostic_level=_enum_value(
            "diagnostic_level", values["diagnostic_level"], DiagnosticLevel
        ),
        config_file=path,
    )
