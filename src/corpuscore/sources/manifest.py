"""Strict manifest and per-file sidecar import configuration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from corpuscore.config.loader import _UniqueKeyLoader
from corpuscore.contracts.common import JSONValue, freeze_json
from corpuscore.contracts.documents import SourceDescriptor
from corpuscore.exceptions import ConfigurationError
from corpuscore.sources.filesystem import DiscoveryOptions, FileSystemSourceProvider


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ManifestDocument(_StrictModel):
    path: str
    profile: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ManifestConfig(_StrictModel):
    version: int = Field(default=1, ge=1, le=1)
    namespace: str = "default"
    documents: tuple[ManifestDocument, ...]


class SidecarConfig(_StrictModel):
    version: int = Field(default=1, ge=1, le=1)
    profile: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


_ModelT = TypeVar("_ModelT", bound=_StrictModel)


class ManifestSourceProvider:
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        options: DiscoveryOptions | None = None,
    ) -> None:
        self._path = Path(manifest_path).expanduser().resolve()
        self._config = _load_model(self._path, ManifestConfig)
        self._options = options or DiscoveryOptions()

    def discover(self) -> Iterable[SourceDescriptor]:
        seen: set[str] = set()
        for item in self._config.documents:
            metadata = _json_metadata(item.metadata)
            if item.profile:
                metadata["corpus_profile"] = item.profile
            candidate = self._path.parent / item.path
            provider = FileSystemSourceProvider(
                [candidate],
                root=self._path.parent,
                namespace=self._config.namespace,
                options=self._options,
                metadata=metadata,
            )
            for source in provider.discover():
                if source.source_id in seen:
                    raise ConfigurationError(
                        f"manifest resolves the same source more than once: {source.relative_path}"
                    )
                seen.add(source.source_id)
                yield source


def apply_sidecar(source: SourceDescriptor) -> SourceDescriptor:
    """Apply a CorpusCore sidecar unless manifest metadata already wins."""

    path = _source_path(source.uri)
    sidecar_path = path.with_name(f"{path.name}.corpus.yaml")
    if not sidecar_path.is_file():
        return source
    sidecar = _load_model(sidecar_path, SidecarConfig)
    metadata: dict[str, JSONValue] = _json_metadata(sidecar.metadata)
    if sidecar.profile:
        metadata["corpus_profile"] = sidecar.profile
    metadata.update(source.metadata)
    return replace(source, metadata=metadata)


def _source_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file") or parsed.netloc not in ("", "localhost"):
        raise ConfigurationError("sidecar configuration only supports local file sources")
    raw = unquote(parsed.path) if parsed.scheme else uri
    return Path(url2pathname(raw))


def _load_model(path: Path, model_type: type[_ModelT]) -> _ModelT:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=_UniqueKeyLoader)
        return model_type.model_validate(value)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigurationError(
            f"invalid import configuration: {path.name}", details={"path": str(path)}
        ) from exc


def _json_metadata(metadata: Mapping[str, object]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in metadata.items():
        result[key] = freeze_json(value)
    return result
