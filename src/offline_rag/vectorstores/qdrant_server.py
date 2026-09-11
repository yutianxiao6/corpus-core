"""Multi-process Qdrant Server adapter using the shared vector-store contract."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

from qdrant_client import QdrantClient

from offline_rag.vectorstores.qdrant_local import QdrantLocalVectorStore


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
        )
        self._lock: AbstractContextManager[object] = nullcontext()
        self._owns_client = client is None
        self._sparse_enabled = False

    def collection(self, collection_name: str) -> QdrantServerVectorStore:
        """Return a non-owning collection view backed by the same server client."""

        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        view = object.__new__(QdrantServerVectorStore)
        view._url = self._url
        view._collection_name = collection_name
        view._client = self._client
        view._lock = self._lock
        view._owns_client = False
        view._sparse_enabled = False
        return view
