from __future__ import annotations

import unittest

import httpx

from corpuscore.cli import _serve, build_parser
from corpuscore.config.models import CorpusConfig
from corpuscore.contracts.chunks import Chunk
from corpuscore.contracts.retrieval import QueryOverrides, RetrievalCandidate, RetrievalResult
from corpuscore.exceptions import ConcurrentAccessError, ConfigurationError
from corpuscore.http_service import create_app


def retrieval_result(query: str) -> RetrievalResult:
    chunk = Chunk(
        chunk_id="chunk-1",
        document_id="document-1",
        content="离线部署说明",
        embedding_text="离线部署说明",
        source_uri="file:///manual.md",
        chunk_index=0,
        metadata={"department": "engineering"},
    )
    candidate = RetrievalCandidate(
        chunk=chunk,
        dense_score=0.9,
        final_score=0.9,
        rank=1,
        origins=["dense"],
    )
    return RetrievalResult(
        query=query,
        processed_query=query,
        hits=(candidate,),
        index_version="index-v1",
        embedding_fingerprint="embedding-v1",
        timings_ms={"total": 1.0},
    )


class EngineFixture:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, QueryOverrides | None]] = []
        self.closed = False
        self.fail = False

    async def aquery(
        self,
        query: str,
        *,
        profile: str = "fast",
        overrides: QueryOverrides | None = None,
    ) -> RetrievalResult:
        if self.fail:
            raise ConcurrentAccessError("embedding queue is full")
        self.requests.append((query, profile, overrides))
        return retrieval_result(query)

    async def abatch_retrieve(
        self,
        queries,
        *,
        profile: str = "fast",
        overrides: QueryOverrides | None = None,
    ):  # type: ignore[no-untyped-def]
        return tuple(
            await self.aquery(query, profile=profile, overrides=overrides) for query in queries
        )

    async def aclose(self) -> None:
        self.closed = True


class HttpServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_factory_engine_is_closed_with_application_lifespan(self) -> None:
        engine = EngineFixture()
        app = create_app(CorpusConfig(), engine_factory=lambda _config: engine)

        async with app.router.lifespan_context(app):
            self.assertFalse(engine.closed)

        self.assertTrue(engine.closed)

    async def test_query_serialization_overrides_and_request_id(self) -> None:
        engine = EngineFixture()
        app = create_app(CorpusConfig(), engine=engine)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/v1/query",
                    headers={"x-request-id": "request-123"},
                    json={
                        "query": "如何部署？",
                        "profile": "fast",
                        "final_k": 3,
                        "filters": {"department": "engineering"},
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-request-id"], "request-123")
        self.assertEqual(response.json()["hits"][0]["chunk_id"], "chunk-1")
        assert engine.requests[0][2] is not None
        self.assertEqual(engine.requests[0][2].final_k, 3)
        self.assertFalse(engine.closed, "injected engines remain owned by the caller")

    async def test_bearer_auth_and_batch_limit(self) -> None:
        engine = EngineFixture()
        app = create_app(CorpusConfig(), engine=engine, bearer_token="secret", maximum_batch_size=2)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                live = await client.get("/health/live")
                denied = await client.post("/v1/query", json={"query": "test"})
                too_many = await client.post(
                    "/v1/query:batch",
                    headers={"authorization": "Bearer secret"},
                    json={"queries": ["one", "two", "three"]},
                )

        self.assertEqual(live.status_code, 200)
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(too_many.status_code, 400)

    async def test_queue_saturation_maps_to_429(self) -> None:
        engine = EngineFixture()
        engine.fail = True
        app = create_app(CorpusConfig(), engine=engine)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/query", json={"query": "busy"})

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["error"], "concurrent_access")

    def test_cli_requires_auth_for_non_loopback_binding(self) -> None:
        args = build_parser().parse_args(["serve", "--host", "0.0.0.0"])
        self.assertEqual(args.command, "serve")
        with self.assertRaises(ConfigurationError):
            _serve(CorpusConfig(), host="0.0.0.0", port=8000, bearer_token_env=None)


if __name__ == "__main__":
    unittest.main()
