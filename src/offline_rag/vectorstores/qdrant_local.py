"""Single-process Qdrant Local implementation of VectorStorePort."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Self

from qdrant_client import QdrantClient
from qdrant_client.http import models

from offline_rag.contracts.common import JSONValue
from offline_rag.contracts.indexing import (
    DeleteReport,
    DistanceMetric,
    IndexSpecification,
    UpsertReport,
    VectorRecord,
)
from offline_rag.contracts.retrieval import SearchHit, SearchRequest
from offline_rag.exceptions import IndexCompatibilityError, VectorStoreError

_POINT_NAMESPACE = uuid.UUID("45a42a71-690b-44fb-abee-e44463894cd1")


class QdrantLocalVectorStore:
    """Own one Qdrant Local client and serialize access inside this process."""

    def __init__(self, path: str | Path, *, collection_name: str = "documents") -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        self._path = Path(path).expanduser().resolve()
        self._path.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._client = QdrantClient(path=str(self._path))
        self._lock = RLock()

    def ensure_index(self, specification: IndexSpecification) -> None:
        fingerprint = specification.fingerprint()
        with self._lock:
            try:
                if not self._client.collection_exists(self._collection_name):
                    self._client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=models.VectorParams(
                            size=specification.embedding.dimension,
                            distance=self._distance(specification.embedding.distance),
                        ),
                        metadata={
                            "offline_rag_index_fingerprint": fingerprint,
                            "index_format_version": specification.index_format_version,
                            "payload_schema_version": specification.payload_schema_version,
                        },
                    )
                    return
                info = self._client.get_collection(self._collection_name)
                metadata = info.config.metadata or {}
                actual = metadata.get("offline_rag_index_fingerprint")
                if actual != fingerprint:
                    raise IndexCompatibilityError(
                        "Qdrant collection has an incompatible index fingerprint",
                        details={"expected": fingerprint, "actual": actual},
                    )
            except IndexCompatibilityError:
                raise
            except Exception as exc:
                raise VectorStoreError("cannot ensure Qdrant Local collection") from exc

    def upsert(self, records: Sequence[VectorRecord]) -> UpsertReport:
        if not records:
            return UpsertReport(requested_count=0, completed_count=0)
        points = [
            models.PointStruct(
                id=self._point_id(record.chunk_id),
                vector=list(record.dense_vector),
                payload=self._payload(record),
            )
            for record in records
        ]
        with self._lock:
            try:
                self._client.upsert(collection_name=self._collection_name, points=points, wait=True)
            except Exception as exc:
                raise VectorStoreError("Qdrant Local upsert failed") from exc
        return UpsertReport(requested_count=len(records), completed_count=len(records))

    def delete(self, chunk_ids: Sequence[str]) -> DeleteReport:
        unique_ids = tuple(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return DeleteReport(requested_count=0, deleted_count=0)
        point_ids = [self._point_id(chunk_id) for chunk_id in unique_ids]
        with self._lock:
            try:
                existing = self._client.retrieve(
                    collection_name=self._collection_name,
                    ids=point_ids,
                    with_payload=False,
                    with_vectors=False,
                )
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=models.PointIdsList(points=point_ids),
                    wait=True,
                )
            except Exception as exc:
                raise VectorStoreError("Qdrant Local delete failed") from exc
        return DeleteReport(requested_count=len(unique_ids), deleted_count=len(existing))

    def search(self, request: SearchRequest) -> Sequence[SearchHit]:
        with self._lock:
            try:
                response = self._client.query_points(
                    collection_name=self._collection_name,
                    query=list(request.query_vector),
                    query_filter=self._filter(request.filters),
                    limit=request.limit,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                raise VectorStoreError("Qdrant Local search failed") from exc
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if not isinstance(chunk_id, str):
                raise VectorStoreError("Qdrant point is missing its chunk_id payload")
            hits.append(SearchHit(chunk_id=chunk_id, score=float(point.score), payload=payload))
        return tuple(hits)

    async def asearch(self, request: SearchRequest) -> Sequence[SearchHit]:
        return await asyncio.to_thread(self.search, request)

    def close(self) -> None:
        with self._lock:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))

    @staticmethod
    def _distance(distance: DistanceMetric) -> models.Distance:
        return {
            DistanceMetric.COSINE: models.Distance.COSINE,
            DistanceMetric.DOT: models.Distance.DOT,
            DistanceMetric.EUCLIDEAN: models.Distance.EUCLID,
        }[distance]

    @staticmethod
    def _payload(record: VectorRecord) -> dict[str, object]:
        payload = _plain_mapping(record.payload)
        payload["chunk_id"] = record.chunk_id
        return payload

    @staticmethod
    def _filter(filters: Mapping[str, JSONValue]) -> models.Filter | None:
        if not filters:
            return None
        conditions: list[models.FieldCondition] = []
        for key, value in filters.items():
            if isinstance(value, tuple):
                values = list(value)
                if not all(
                    isinstance(item, (str, int)) and not isinstance(item, bool) for item in values
                ):
                    raise VectorStoreError(f"unsupported filter values for field {key!r}")
                conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))
            elif isinstance(value, (str, int, bool)):
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
            else:
                raise VectorStoreError(f"unsupported filter value for field {key!r}")
        return models.Filter(must=conditions)


def _plain_mapping(mapping: Mapping[str, JSONValue]) -> dict[str, object]:
    return {key: _plain_value(value) for key, value in mapping.items()}


def _plain_value(value: JSONValue) -> object:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return value
