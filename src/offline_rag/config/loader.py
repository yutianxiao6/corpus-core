"""Safe YAML loading and deterministic configuration precedence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from offline_rag.config.models import RagConfig
from offline_rag.exceptions import ConfigurationError


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConfigurationError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge mappings; scalar values and sequences replace lower layers."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.load(stream, Loader=_UniqueKeyLoader)
    except ConfigurationError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            f"cannot read configuration {config_path}: {exc}",
            details={"path": str(config_path)},
        ) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ConfigurationError("the YAML document root must be a string-keyed mapping")
    return loaded


def _parse_scalar(value: str) -> Any:
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid CLI override value {value!r}: {exc}") from exc
    if isinstance(parsed, (dict, list)):
        raise ConfigurationError("CLI override values must be scalar YAML values")
    return parsed


def parse_cli_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    """Convert dotted ``key=value`` CLI options into a nested mapping."""

    result: dict[str, Any] = {}
    for pair in pairs:
        key, separator, raw_value = pair.partition("=")
        parts = key.split(".")
        if not separator or not key or any(not part for part in parts):
            raise ConfigurationError(f"invalid override {pair!r}; expected dotted.key=value")
        cursor = result
        for part in parts[:-1]:
            existing = cursor.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ConfigurationError(f"conflicting CLI override path at {part!r}")
            cursor = existing
        leaf = parts[-1]
        if leaf in cursor:
            raise ConfigurationError(f"duplicate CLI override for {key!r}")
        cursor[leaf] = _parse_scalar(raw_value)
    return result


def load_config(
    path: str | Path | None = None,
    *,
    cli_overrides: Mapping[str, Any] | None = None,
    api_overrides: Mapping[str, Any] | None = None,
) -> RagConfig:
    """Load config using defaults < YAML < CLI < Python API precedence."""

    data = RagConfig().model_dump(mode="python")
    if path is not None:
        data = deep_merge(data, _read_yaml(path))
    if cli_overrides:
        data = deep_merge(data, cli_overrides)
    if api_overrides:
        data = deep_merge(data, api_overrides)
    if path is not None:
        data = _resolve_relative_paths(data, Path(path).expanduser().resolve().parent)
    try:
        return RagConfig.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False)
        raise ConfigurationError(
            "configuration validation failed",
            details={"errors": errors},
        ) from exc


def _resolve_relative_paths(data: Mapping[str, Any], base_directory: Path) -> dict[str, Any]:
    """Resolve filesystem settings relative to the YAML file, not process CWD."""

    resolved = deepcopy(dict(data))

    def rebase(mapping: object, key: str) -> None:
        if not isinstance(mapping, dict):
            return
        value = mapping.get(key)
        if not isinstance(value, str) or not value.strip():
            return
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = base_directory / candidate
        mapping[key] = str(candidate.resolve())

    runtime = resolved.get("runtime")
    rebase(runtime, "work_dir")
    rebase(runtime, "cache_dir")
    rebase(resolved.get("embedding"), "model_path")
    rebase(resolved.get("vector_store"), "path")
    rerankers = resolved.get("rerankers")
    if isinstance(rerankers, dict):
        for reranker in rerankers.values():
            rebase(reranker, "model_path")
    return resolved
