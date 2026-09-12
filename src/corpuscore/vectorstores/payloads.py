"""Stable conversion between Chunk contracts and vector-store JSON payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from corpuscore.contracts.chunks import Chunk
from corpuscore.contracts.common import JSONValue
from corpuscore.exceptions import VectorStoreError


def chunk_to_payload(chunk: Chunk) -> dict[str, JSONValue]:
    return {
        "payload_schema_version": 1,
        "record_type": "chunk",
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "content": chunk.content,
        "embedding_text": chunk.embedding_text,
        "source_uri": chunk.source_uri,
        "chunk_index": chunk.chunk_index,
        "title": chunk.title,
        "heading_path": tuple(chunk.heading_path),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "parent_id": chunk.parent_id,
        "previous_id": chunk.previous_id,
        "next_id": chunk.next_id,
        "token_count": chunk.token_count,
        "metadata": chunk.metadata,
    }


def chunk_from_payload(payload: Mapping[str, JSONValue]) -> Chunk:
    try:
        heading_value = payload.get("heading_path", ())
        heading_path = _string_sequence(heading_value, "heading_path")
        metadata_value = payload.get("metadata", {})
        if not isinstance(metadata_value, Mapping):
            raise TypeError("metadata must be a mapping")
        return Chunk(
            chunk_id=_string(payload, "chunk_id"),
            document_id=_string(payload, "document_id"),
            content=_string(payload, "content"),
            embedding_text=_string(payload, "embedding_text"),
            source_uri=_string(payload, "source_uri"),
            chunk_index=_integer(payload, "chunk_index"),
            title=_optional_string(payload.get("title"), "title"),
            heading_path=heading_path,
            page_start=_optional_integer(payload.get("page_start"), "page_start"),
            page_end=_optional_integer(payload.get("page_end"), "page_end"),
            char_start=_optional_integer(payload.get("char_start"), "char_start"),
            char_end=_optional_integer(payload.get("char_end"), "char_end"),
            parent_id=_optional_string(payload.get("parent_id"), "parent_id"),
            previous_id=_optional_string(payload.get("previous_id"), "previous_id"),
            next_id=_optional_string(payload.get("next_id"), "next_id"),
            token_count=_optional_integer(payload.get("token_count"), "token_count"),
            metadata=metadata_value,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VectorStoreError("vector payload does not match schema version 1") from exc


def _string(payload: Mapping[str, JSONValue], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _integer(payload: Mapping[str, JSONValue], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value


def _optional_string(value: JSONValue, key: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _optional_integer(value: JSONValue, key: str) -> int | None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise TypeError(f"{key} must be an integer or null")
    return value


def _string_sequence(value: JSONValue, key: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{key} must be a sequence")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must contain strings")
    return tuple(item for item in value if isinstance(item, str))
