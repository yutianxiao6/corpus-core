"""Index specifications and ingestion result contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum

from offline_rag.contracts.common import (
    JSONValue,
    freeze_metadata,
    require_non_empty,
    require_non_negative,
    require_positive,
)


class DistanceMetric(StrEnum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


@dataclass(frozen=True, slots=True)
class EmbeddingSpecification:
    provider: str
    model: str
    revision: str
    dimension: int
    normalized: bool
    distance: DistanceMetric
    max_length: int
    query_instruction: str = ""
    tokenizer_revision: str = ""
    model_checksum: str = ""

    def __post_init__(self) -> None:
        require_non_empty(self.provider, "provider")
        require_non_empty(self.model, "model")
        require_non_empty(self.revision, "revision")
        require_positive(self.dimension, "dimension")
        require_positive(self.max_length, "max_length")

    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["distance"] = self.distance.value
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class IndexSpecification:
    index_format_version: int
    payload_schema_version: int
    embedding: EmbeddingSpecification
    parser_versions: Mapping[str, str]
    chunking_configuration: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        require_positive(self.index_format_version, "index_format_version")
        require_positive(self.payload_schema_version, "payload_schema_version")
        object.__setattr__(self, "parser_versions", freeze_metadata(self.parser_versions))
        object.__setattr__(
            self,
            "chunking_configuration",
            freeze_metadata(self.chunking_configuration),
        )

    def fingerprint(self) -> str:
        payload = {
            "index_format_version": self.index_format_version,
            "payload_schema_version": self.payload_schema_version,
            "embedding": self.embedding.fingerprint(),
            "parser_versions": dict(self.parser_versions),
            "chunking_configuration": _plain_json(self.chunking_configuration),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _plain_json(value: JSONValue) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


class IngestionStage(StrEnum):
    DISCOVERED = "discovered"
    LOADED = "loaded"
    PARSED = "parsed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ItemFailure:
    source_uri: str
    stage: IngestionStage
    error_code: str
    message: str

    def __post_init__(self) -> None:
        require_non_empty(self.source_uri, "source_uri")
        require_non_empty(self.error_code, "error_code")
        require_non_empty(self.message, "message")


@dataclass(frozen=True, slots=True)
class IngestionItemResult:
    source_uri: str
    stage: IngestionStage
    chunk_count: int = 0
    duration_ms: float = 0.0
    failure: ItemFailure | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.source_uri, "source_uri")
        require_non_negative(self.chunk_count, "chunk_count")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        if self.stage is IngestionStage.FAILED and self.failure is None:
            raise ValueError("failed item must include failure details")
        if self.stage is not IngestionStage.FAILED and self.failure is not None:
            raise ValueError("only failed item may include failure details")


@dataclass(frozen=True, slots=True)
class IngestionReport:
    job_id: str
    index_version: str
    items: Sequence[IngestionItemResult]
    duration_ms: float

    def __post_init__(self) -> None:
        require_non_empty(self.job_id, "job_id")
        require_non_empty(self.index_version, "index_version")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        object.__setattr__(self, "items", tuple(self.items))

    @property
    def failed_count(self) -> int:
        return sum(item.stage is IngestionStage.FAILED for item in self.items)

    @property
    def indexed_count(self) -> int:
        return sum(item.stage is IngestionStage.INDEXED for item in self.items)

    @property
    def chunk_count(self) -> int:
        return sum(item.chunk_count for item in self.items)


@dataclass(frozen=True, slots=True)
class VectorRecord:
    chunk_id: str
    dense_vector: Sequence[float]
    payload: Mapping[str, JSONValue]
    sparse_indices: Sequence[int] = ()
    sparse_values: Sequence[float] = ()

    def __post_init__(self) -> None:
        require_non_empty(self.chunk_id, "chunk_id")
        if not self.dense_vector:
            raise ValueError("dense_vector must not be empty")
        if len(self.sparse_indices) != len(self.sparse_values):
            raise ValueError("sparse vector indices and values must have equal length")
        if any(index < 0 for index in self.sparse_indices):
            raise ValueError("sparse vector indices must be non-negative")
        object.__setattr__(self, "dense_vector", tuple(float(value) for value in self.dense_vector))
        object.__setattr__(self, "sparse_indices", tuple(self.sparse_indices))
        object.__setattr__(
            self, "sparse_values", tuple(float(value) for value in self.sparse_values)
        )
        object.__setattr__(self, "payload", freeze_metadata(self.payload))


@dataclass(frozen=True, slots=True)
class UpsertReport:
    requested_count: int
    completed_count: int

    def __post_init__(self) -> None:
        require_non_negative(self.requested_count, "requested_count")
        require_non_negative(self.completed_count, "completed_count")
        if self.completed_count > self.requested_count:
            raise ValueError("completed_count must not exceed requested_count")


@dataclass(frozen=True, slots=True)
class DeleteReport:
    requested_count: int
    deleted_count: int

    def __post_init__(self) -> None:
        require_non_negative(self.requested_count, "requested_count")
        require_non_negative(self.deleted_count, "deleted_count")
        if self.deleted_count > self.requested_count:
            raise ValueError("deleted_count must not exceed requested_count")
