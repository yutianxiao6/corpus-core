"""Single-process Qdrant Local implementation of VectorStorePort."""

from __future__ import annotations

import asyncio
import re
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
from offline_rag.contracts.retrieval import SearchHit, SearchRequest, SearchVector
from offline_rag.exceptions import IndexCompatibilityError, VectorStoreError

_POINT_NAMESPACE = uuid.UUID("45a42a71-690b-44fb-abee-e44463894cd1")
_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"


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
        self._owns_client = True
        self._sparse_enabled = False

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def create_staging(
        self, specification: IndexSpecification, *, version: str
    ) -> QdrantLocalVectorStore:
        normalized_version = re.sub(r"[^a-zA-Z0-9_-]+", "-", version).strip("-")
        if not normalized_version:
            raise ValueError("staging version must contain a letter or number")
        collection_name = f"{self._collection_name}__{normalized_version}"
        with self._lock:
            if self._physical_collection_exists(collection_name):
                raise VectorStoreError(f"staging collection already exists: {collection_name}")
        staging = self.collection(collection_name)
        staging.ensure_index(specification)
        return staging

    def activate_staging(self, staging: QdrantLocalVectorStore | str) -> None:
        collection_name = (
            staging.collection_name if isinstance(staging, QdrantLocalVectorStore) else staging
        )
        with self._lock:
            try:
                if not self._physical_collection_exists(collection_name):
                    raise VectorStoreError(f"staging collection does not exist: {collection_name}")
                if self._physical_collection_exists(self._collection_name):
                    raise VectorStoreError(
                        "active name is a physical collection and cannot be used as an alias"
                    )
                aliases = {
                    item.alias_name: item.collection_name
                    for item in self._client.get_aliases().aliases
                }
                if aliases.get(self._collection_name) == collection_name:
                    return
                operations: list[models.CreateAliasOperation | models.DeleteAliasOperation] = []
                if self._collection_name in aliases:
                    operations.append(
                        models.DeleteAliasOperation(
                            delete_alias=models.DeleteAlias(alias_name=self._collection_name)
                        )
                    )
                operations.append(
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(
                            collection_name=collection_name,
                            alias_name=self._collection_name,
                        )
                    )
                )
                self._client.update_collection_aliases(operations)
            except VectorStoreError:
                raise
            except Exception as exc:
                raise VectorStoreError("cannot atomically activate staging collection") from exc

    def active_collection(self) -> str | None:
        with self._lock:
            aliases = {
                item.alias_name: item.collection_name for item in self._client.get_aliases().aliases
            }
        return aliases.get(self._collection_name)

    def point_count(self) -> int:
        with self._lock:
            try:
                return int(self._client.get_collection(self._collection_name).points_count or 0)
            except Exception as exc:
                raise VectorStoreError("cannot read Qdrant collection size") from exc

    def drop_collection(self, collection_name: str) -> None:
        if collection_name == self._collection_name:
            raise VectorStoreError("refusing to drop the configured active name")
        with self._lock:
            try:
                if self._physical_collection_exists(collection_name):
                    self._client.delete_collection(collection_name)
            except Exception as exc:
                raise VectorStoreError("cannot drop Qdrant collection") from exc

    def collection(self, collection_name: str) -> QdrantLocalVectorStore:
        """Return a non-owning view backed by this instance's single local client."""

        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        view = object.__new__(QdrantLocalVectorStore)
        view._path = self._path
        view._collection_name = collection_name
        view._client = self._client
        view._lock = self._lock
        view._owns_client = False
        view._sparse_enabled = False
        return view

    def _physical_collection_exists(self, collection_name: str) -> bool:
        return any(
            item.name == collection_name for item in self._client.get_collections().collections
        )

    def ensure_index(self, specification: IndexSpecification) -> None:
        fingerprint = specification.fingerprint()
        sparse_enabled = specification.sparse_embedding is not None
        with self._lock:
            try:
                if not self._client.collection_exists(self._collection_name):
                    vectors_config: models.VectorParams | Mapping[str, models.VectorParams]
                    vectors_config = models.VectorParams(
                        size=specification.embedding.dimension,
                        distance=self._distance(specification.embedding.distance),
                    )
                    sparse_vectors_config = None
                    if sparse_enabled:
                        vectors_config = {_DENSE_VECTOR_NAME: vectors_config}
                        sparse_vectors_config = {_SPARSE_VECTOR_NAME: models.SparseVectorParams()}
                    self._client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=vectors_config,
                        sparse_vectors_config=sparse_vectors_config,
                        metadata={
                            "offline_rag_index_fingerprint": fingerprint,
                            "index_format_version": specification.index_format_version,
                            "payload_schema_version": specification.payload_schema_version,
                        },
                    )
                    self._sparse_enabled = sparse_enabled
                    return
                info = self._client.get_collection(self._collection_name)
                metadata = info.config.metadata or {}
                actual = metadata.get("offline_rag_index_fingerprint")
                if actual != fingerprint:
                    raise IndexCompatibilityError(
                        "Qdrant collection has an incompatible index fingerprint",
                        details={"expected": fingerprint, "actual": actual},
                    )
                self._sparse_enabled = sparse_enabled
            except IndexCompatibilityError:
                raise
            except Exception as exc:
                raise VectorStoreError("cannot ensure Qdrant Local collection") from exc

    def upsert(self, records: Sequence[VectorRecord]) -> UpsertReport:
        if not records:
            return UpsertReport(requested_count=0, completed_count=0)
        points: list[models.PointStruct] = []
        for record in records:
            vector: list[float] | dict[str, list[float] | models.SparseVector]
            if self._sparse_enabled:
                if not record.sparse_indices:
                    raise VectorStoreError("sparse-enabled index requires sparse vectors on upsert")
                vector = {
                    _DENSE_VECTOR_NAME: list(record.dense_vector),
                    _SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=list(record.sparse_indices), values=list(record.sparse_values)
                    ),
                }
            else:
                if record.sparse_indices:
                    raise VectorStoreError("cannot upsert sparse vectors into a dense-only index")
                vector = list(record.dense_vector)
            points.append(
                models.PointStruct(
                    id=self._point_id(record.chunk_id),
                    vector=vector,
                    payload=self._payload(record),
                )
            )
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
        if request.vector is SearchVector.SPARSE and not self._sparse_enabled:
            raise VectorStoreError("sparse search requires a sparse-enabled index")
        query: list[float] | models.SparseVector
        using: str | None
        if request.vector is SearchVector.SPARSE:
            query = models.SparseVector(
                indices=list(request.sparse_indices), values=list(request.sparse_values)
            )
            using = _SPARSE_VECTOR_NAME
        else:
            query = list(request.query_vector)
            using = _DENSE_VECTOR_NAME if self._sparse_enabled else None
        with self._lock:
            try:
                response = self._client.query_points(
                    collection_name=self._collection_name,
                    query=query,
                    using=using,
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
        if self._owns_client:
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
