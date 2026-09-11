"""Multi-process Qdrant Server adapter using the shared vector-store contract."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models

from offline_rag.contracts.retrieval import SearchHit, SearchRequest, SearchVector
from offline_rag.exceptions import VectorStoreError
from offline_rag.vectorstores.qdrant_local import QdrantLocalVectorStore

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"


class QdrantServerVectorStore(QdrantLocalVectorStore):
    """Use a remote/self-hosted Qdrant service without serializing client operations."""

    def __init__(
        self,
        url: str,
        *,
        collection_name: str = "documents",
        api_key: str | None = None,
        prefer_grpc: bool = False,
        timeout_seconds: int = 30,
        pool_size: int | None = None,
        client: QdrantClient | None = None,
        async_client: AsyncQdrantClient | None = None,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("Qdrant Server URL must use http:// or https://")
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._url = url
        self._collection_name = collection_name
        self._client = client or QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            timeout=timeout_seconds,
            pool_size=pool_size,
            cloud_inference=False,
            check_compatibility=False,
        )
        self._async_client = async_client or AsyncQdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            timeout=timeout_seconds,
            pool_size=pool_size,
            cloud_inference=False,
            check_compatibility=False,
        )
        self._lock: AbstractContextManager[object] = nullcontext()
        self._owns_client = client is None
        self._owns_async_client = async_client is None
        self._sparse_enabled = False

    def collection(self, collection_name: str) -> QdrantServerVectorStore:
        """Return a non-owning collection view backed by the same server client."""

        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        view = object.__new__(QdrantServerVectorStore)
        view._url = self._url
        view._collection_name = collection_name
        view._client = self._client
        view._async_client = self._async_client
        view._lock = self._lock
        view._owns_client = False
        view._owns_async_client = False
        view._sparse_enabled = False
        return view

    def backup(
        self,
        destination: str | Path,
        *,
        index_fingerprint: str | None = None,
    ) -> dict[str, object]:
        """Request a server-side snapshot and save a portable restore descriptor."""

        try:
            snapshot = self._client.create_snapshot(self._collection_name, wait=True)
            name = getattr(snapshot, "name", None)
            if not isinstance(name, str) or not name:
                raise VectorStoreError("Qdrant Server returned an invalid snapshot name")
            manifest: dict[str, object] = {
                "format_version": 1,
                "mode": "server",
                "collection_name": self._collection_name,
                "snapshot_name": name,
                "index_fingerprint": index_fingerprint,
            }
            destination_path = Path(destination).expanduser().resolve()
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            with destination_path.open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
            return manifest
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("cannot create Qdrant Server snapshot") from exc

    async def asearch(self, request: SearchRequest) -> Sequence[SearchHit]:
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
        try:
            response = await self._async_client.query_points(
                collection_name=self._collection_name,
                query=query,
                using=using,
                query_filter=self._filter(request.filters),
                limit=request.limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant async search failed") from exc
        return self._search_hits(response.points)

    async def afetch(self, chunk_ids: Sequence[str]) -> Sequence[SearchHit]:
        unique_ids = tuple(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return ()
        try:
            points = await self._async_client.retrieve(
                collection_name=self._collection_name,
                ids=[self._point_id(chunk_id) for chunk_id in unique_ids],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant async payload fetch failed") from exc
        by_id = {hit.chunk_id: hit for hit in self._search_hits(points, score=0.0)}
        return tuple(by_id[chunk_id] for chunk_id in unique_ids if chunk_id in by_id)

    async def aclose(self) -> None:
        if self._owns_async_client:
            self._owns_async_client = False
            await self._async_client.close()
        super().close()

    def close(self) -> None:
        super().close()
        if not self._owns_async_client:
            return
        self._owns_async_client = False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._async_client.close())
        else:
            loop.create_task(self._async_client.close())

    @staticmethod
    def _search_hits(
        points: Sequence[models.ScoredPoint | models.Record], *, score: float | None = None
    ) -> tuple[SearchHit, ...]:
        hits: list[SearchHit] = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            chunk_id = payload.get("chunk_id")
            if not isinstance(chunk_id, str):
                raise VectorStoreError("Qdrant point is missing its chunk_id payload")
            if score is None:
                if not isinstance(point, models.ScoredPoint):
                    raise VectorStoreError("Qdrant search point is missing its score")
                point_score = float(point.score)
            else:
                point_score = score
            hits.append(SearchHit(chunk_id=chunk_id, score=point_score, payload=payload))
        return tuple(hits)
