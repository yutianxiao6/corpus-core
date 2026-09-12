"""Optional FastAPI reference adapter for the reusable retrieval engine."""

from __future__ import annotations

import hmac
import logging
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, Protocol, cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from corpuscore.config.models import CorpusConfig
from corpuscore.contracts.retrieval import QueryOverrides, RetrievalResult
from corpuscore.exceptions import (
    ConcurrentAccessError,
    ConfigurationError,
    CorpusCoreError,
    IndexCompatibilityError,
    LocalResourceMissingError,
    VectorStoreError,
)

LOGGER = logging.getLogger(__name__)


class HttpEngine(Protocol):
    async def aquery(
        self,
        query: str,
        *,
        profile: str = "fast",
        overrides: QueryOverrides | None = None,
    ) -> RetrievalResult: ...

    async def abatch_retrieve(
        self,
        queries: Sequence[str],
        *,
        profile: str = "fast",
        overrides: QueryOverrides | None = None,
    ) -> tuple[RetrievalResult, ...]: ...

    async def aclose(self) -> None: ...


class _OverridesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str = Field(default="fast", min_length=1, max_length=128)
    filters: dict[str, Any] = Field(default_factory=dict)
    organizer: str | None = Field(default=None, min_length=1, max_length=128)
    final_k: int | None = Field(default=None, gt=0, le=1000)
    score_threshold: float | None = None
    rerank_top_n: int | None = Field(default=None, gt=0, le=1000)
    mmr_lambda: float | None = Field(default=None, ge=0, le=1)
    mmr_fetch_k: int | None = Field(default=None, gt=0, le=1000)
    neighbor_expansion: int | None = Field(default=None, ge=0, le=100)
    maximum_chunks_per_document: int | None = Field(default=None, gt=0, le=1000)

    def to_overrides(self) -> QueryOverrides:
        return QueryOverrides(
            filters=self.filters,
            organizer=self.organizer,
            final_k=self.final_k,
            score_threshold=self.score_threshold,
            rerank_top_n=self.rerank_top_n,
            mmr_lambda=self.mmr_lambda,
            mmr_fetch_k=self.mmr_fetch_k,
            neighbor_expansion=self.neighbor_expansion,
            maximum_chunks_per_document=self.maximum_chunks_per_document,
        )


class QueryRequest(_OverridesRequest):
    query: str = Field(min_length=1, max_length=32_768)


class BatchQueryRequest(_OverridesRequest):
    queries: list[str] = Field(min_length=1)


EngineFactory = Callable[[CorpusConfig], HttpEngine]


def create_app(
    config: CorpusConfig,
    *,
    engine: HttpEngine | None = None,
    engine_factory: EngineFactory | None = None,
    bearer_token: str | None = None,
    bearer_token_env: str | None = None,
    maximum_batch_size: int = 32,
) -> FastAPI:
    """Build an ASGI app while keeping FastAPI outside the core dependency set."""

    if maximum_batch_size <= 0:
        raise ValueError("maximum_batch_size must be positive")
    if bearer_token is not None and bearer_token_env is not None:
        raise ConfigurationError("configure bearer_token or bearer_token_env, not both")
    if bearer_token_env is not None:
        bearer_token = os.environ.get(bearer_token_env)
        if not bearer_token:
            raise ConfigurationError(
                f"HTTP bearer token environment variable is missing: {bearer_token_env}"
            )
    if bearer_token is not None and not bearer_token:
        raise ConfigurationError("HTTP bearer token must not be empty")

    owns_engine = engine is None
    factory = engine_factory or _default_engine_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = engine or factory(config)
        try:
            yield
        finally:
            current = cast(HttpEngine | None, getattr(app.state, "engine", None))
            if owns_engine and current is not None:
                await current.aclose()
            app.state.engine = None

    app = FastAPI(
        title="CorpusCore",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_controls(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = _request_id(request.headers.get("x-request-id"))
        if bearer_token is not None and request.url.path != "/health/live":
            supplied = request.headers.get("authorization", "")
            expected = f"Bearer {bearer_token}"
            if not hmac.compare_digest(supplied.encode(), expected.encode()):
                return JSONResponse(
                    status_code=401,
                    content={"error": "unauthorized", "message": "invalid bearer token"},
                    headers={"x-request-id": request_id, "www-authenticate": "Bearer"},
                )
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(CorpusCoreError)
    async def corpuscore_error(_request: Request, exc: CorpusCoreError) -> JSONResponse:
        return JSONResponse(
            status_code=_status_for(exc),
            content={"error": exc.code, "message": str(exc)},
        )

    @app.exception_handler(ValueError)
    @app.exception_handler(TypeError)
    async def value_error(_request: Request, exc: ValueError | TypeError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "message": str(exc)},
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("unhandled HTTP adapter error on %s", request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "internal retrieval error"},
        )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready(request: Request) -> JSONResponse:
        available = getattr(request.app.state, "engine", None) is not None
        return JSONResponse(
            status_code=200 if available else 503,
            content={"status": "ready" if available else "not_ready"},
        )

    @app.post("/v1/query")
    async def query(payload: QueryRequest, request: Request) -> dict[str, object]:
        current = _engine(request)
        result = await current.aquery(
            payload.query,
            profile=payload.profile,
            overrides=payload.to_overrides(),
        )
        return retrieval_result_to_dict(result)

    @app.post("/v1/query:batch")
    async def batch_query(payload: BatchQueryRequest, request: Request) -> dict[str, object]:
        if len(payload.queries) > maximum_batch_size:
            raise ValueError(f"batch contains more than {maximum_batch_size} queries")
        if any(not query.strip() for query in payload.queries):
            raise ValueError("batch queries must not be blank")
        if any(len(query) > 32_768 for query in payload.queries):
            raise ValueError("each batch query must contain at most 32768 characters")
        current = _engine(request)
        results = await current.abatch_retrieve(
            payload.queries,
            profile=payload.profile,
            overrides=payload.to_overrides(),
        )
        return {"results": [retrieval_result_to_dict(result) for result in results]}

    return app


def retrieval_result_to_dict(result: RetrievalResult) -> dict[str, object]:
    """Serialize the stable retrieval contract without exposing Python-only containers."""

    return {
        "query": result.query,
        "processed_query": result.processed_query,
        "index_version": result.index_version,
        "embedding_fingerprint": result.embedding_fingerprint,
        "hits": [
            {
                "chunk_id": candidate.chunk.chunk_id,
                "document_id": candidate.chunk.document_id,
                "source": candidate.chunk.source_uri,
                "title": candidate.chunk.title,
                "content": candidate.chunk.content,
                "page_start": candidate.chunk.page_start,
                "page_end": candidate.chunk.page_end,
                "score": candidate.final_score,
                "dense_score": candidate.dense_score,
                "sparse_score": candidate.sparse_score,
                "fusion_score": candidate.fusion_score,
                "rerank_score": candidate.rerank_score,
                "rank": candidate.rank,
                "origins": list(candidate.origins),
                "metadata": _plain(candidate.chunk.metadata),
            }
            for candidate in result.hits
        ],
        "context": result.context,
        "citations": [
            {
                "id": citation.citation_id,
                "chunk_ids": list(citation.chunk_ids),
                "source": citation.source_uri,
                "title": citation.title,
                "page_start": citation.page_start,
                "page_end": citation.page_end,
            }
            for citation in result.citations
        ],
        "groups": [
            {
                "group_id": group.group_id,
                "chunk_ids": [candidate.chunk.chunk_id for candidate in group.hits],
            }
            for group in result.groups
        ],
        "debug": _plain(result.debug),
        "timings_ms": dict(result.timings_ms),
        "warnings": list(result.warnings),
    }


def _default_engine_factory(config: CorpusConfig) -> HttpEngine:
    from corpuscore.engine import RetrievalEngine

    return RetrievalEngine.from_config(config)


def _engine(request: Request) -> HttpEngine:
    current = cast(HttpEngine | None, getattr(request.app.state, "engine", None))
    if current is None:
        raise LocalResourceMissingError("retrieval engine is not ready")
    return current


def _status_for(exc: CorpusCoreError) -> int:
    if isinstance(exc, ConcurrentAccessError):
        return 429
    if isinstance(
        exc,
        (LocalResourceMissingError, VectorStoreError, IndexCompatibilityError),
    ):
        return 503
    return 400


def _request_id(value: str | None) -> str:
    if value is not None and 1 <= len(value) <= 128 and value.isascii() and value.isprintable():
        return value
    return uuid.uuid4().hex


def _plain(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value
