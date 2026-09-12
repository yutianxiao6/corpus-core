"""Common immutable value types and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | tuple["JSONValue", ...] | Mapping[str, "JSONValue"]


def require_non_empty(value: str, field_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def require_non_negative(value: int | None, field_name: str) -> int | None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def require_positive(value: int, field_name: str) -> int:
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def require_range(
    start: int | None,
    end: int | None,
    start_name: str,
    end_name: str,
) -> None:
    require_non_negative(start, start_name)
    require_non_negative(end, end_name)
    if start is not None and end is not None and end < start:
        raise ValueError(f"{end_name} must be greater than or equal to {start_name}")


def freeze_json(value: object) -> JSONValue:
    """Copy JSON-compatible values into immutable containers.

    The contracts are frozen dataclasses. Freezing nested metadata as well prevents
    callers from mutating an otherwise immutable object through a shared dictionary.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        frozen = {str(key): freeze_json(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(freeze_json(item) for item in value)
    raise TypeError(f"metadata value is not JSON-compatible: {type(value).__name__}")


def freeze_metadata(metadata: Mapping[str, object] | None) -> Mapping[str, JSONValue]:
    if metadata is None:
        return MappingProxyType({})
    frozen = {str(key): freeze_json(value) for key, value in metadata.items()}
    return MappingProxyType(frozen)
