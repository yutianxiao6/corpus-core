from __future__ import annotations

import unittest
from collections.abc import Sequence
from dataclasses import replace

from offline_rag.config.models import RagConfig, RerankerConfig, RetrievalProfile
from offline_rag.contracts.chunks import Chunk
from offline_rag.contracts.indexing import (
    DistanceMetric,
    EmbeddingSpecification,
    IndexSpecification,
)
from offline_rag.contracts.retrieval import RetrievalCandidate, SearchHit, SearchRequest
from offline_rag.engine import OfflineRagEngine
from offline_rag.exceptions import RerankerError
from offline_rag.ports import Vector
from offline_rag.vectorstores import chunk_to_payload


class EmbeddingFixture:
    specification = EmbeddingSpecification(
        provider="fixture",
        model="fixture",
        revision="1",
        dimension=2,
        normalized=True,
        distance=DistanceMetric.COSINE,
        max_length=100,
    )

    def embed_query(self, text: str) -> Vector:
        return (1.0, 0.0)

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Vector]:
        return tuple(self.embed_query(text) for text in texts)

    async def aembed_query(self, text: str) -> Vector:
        return self.embed_query(text)


class StoreFixture:
    def __init__(self) -> None:
        self.requests: list[SearchRequest] = []
        self.async_searches = 0

    def ensure_index(self, specification: IndexSpecification) -> None:
        self.specification = specification

    def search(self, request: SearchRequest) -> Sequence[SearchHit]:
        self.requests.append(request)
        return tuple(self._hit(name, score) for name, score in (("a", 0.9), ("b", 0.8), ("c", 0.7)))

    async def asearch(self, request: SearchRequest) -> Sequence[SearchHit]:
        self.async_searches += 1
        return self.search(request)

    def fetch(self, chunk_ids: Sequence[str]) -> Sequence[SearchHit]:
        return ()

    async def afetch(self, chunk_ids: Sequence[str]) -> Sequence[SearchHit]:
        return self.fetch(chunk_ids)

    @staticmethod
    def _hit(chunk_id: str, score: float) -> SearchHit:
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            content=chunk_id,
            embedding_text=chunk_id,
            source_uri=f"file:///{chunk_id}.txt",
            chunk_index=0,
        )
        return SearchHit(chunk_id, score, chunk_to_payload(chunk))


class ReverseReranker:
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int | None = None,
    ) -> Sequence[RetrievalCandidate]:
        return tuple(
            replace(candidate, rerank_score=float(rank), final_score=float(rank), rank=rank)
            for rank, candidate in enumerate(reversed(candidates), start=1)
        )

    async def arerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int | None = None,
    ) -> Sequence[RetrievalCandidate]:
        return self.rerank(query, candidates, top_n=top_n)


class FailingReranker(ReverseReranker):
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int | None = None,
    ) -> Sequence[RetrievalCandidate]:
        raise RerankerError("fixture failure")


class AsyncOnlyReranker(ReverseReranker):
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int | None = None,
    ) -> Sequence[RetrievalCandidate]:
        raise AssertionError("sync reranker must not be called by aquery")

    async def arerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int | None = None,
    ) -> Sequence[RetrievalCandidate]:
        return ReverseReranker.rerank(self, query, candidates, top_n=top_n)


def make_engine(
    *, failure_policy: str = "return_unranked"
) -> tuple[OfflineRagEngine, StoreFixture]:
    config = RagConfig(
        retrieval_profiles={
            "precise": RetrievalProfile(
                strategy="dense",
                fetch_k=3,
                reranker="local",
                rerank_top_n=3,
                final_k=2,
            )
        },
        rerankers={
            "local": RerankerConfig(
                model_path="./model",
                failure_policy=failure_policy,  # type: ignore[arg-type]
            )
        },
    )
    embedding = EmbeddingFixture()
    store = StoreFixture()
    engine = object.__new__(OfflineRagEngine)
    engine.config = config
    engine.embedding = embedding  # type: ignore[assignment]
    engine.query_embedding = embedding  # type: ignore[assignment]
    engine.vector_store = store  # type: ignore[assignment]
    engine.index_specification = IndexSpecification(
        index_format_version=1,
        payload_schema_version=1,
        embedding=embedding.specification,
        parser_versions={"text": "1"},
        chunking_configuration={"size": 100},
    )
    engine._reranker_cache = {}
    return engine, store


class EngineRerankingTests(unittest.TestCase):
    def test_reranker_receives_larger_pool_then_engine_applies_final_k(self) -> None:
        engine, store = make_engine()
        engine._reranker_cache["local"] = ReverseReranker()

        result = engine.query("query", profile="precise")

        self.assertEqual(store.requests[0].limit, 3)
        self.assertEqual([item.chunk.chunk_id for item in result.hits], ["c", "b"])
        self.assertIn("rerank", result.timings_ms)

    def test_default_failure_policy_returns_recall_order_with_warning(self) -> None:
        engine, _store = make_engine()
        engine._reranker_cache["local"] = FailingReranker()

        result = engine.query("query", profile="precise")

        self.assertEqual([item.chunk.chunk_id for item in result.hits], ["a", "b"])
        self.assertIn("failed; returned recall order", result.warnings[0])

    def test_fail_policy_propagates_reranker_error(self) -> None:
        engine, _store = make_engine(failure_policy="fail")
        engine._reranker_cache["local"] = FailingReranker()
        with self.assertRaises(RerankerError):
            engine.query("query", profile="precise")

    def test_per_call_overrides_do_not_mutate_profile(self) -> None:
        engine, store = make_engine()
        engine._reranker_cache["local"] = ReverseReranker()

        result = engine.retrieve(
            "query",
            profile="precise",
            filters={"department": "support"},
            organizer="debug",
            final_k=1,
            score_threshold=0.5,
        )

        self.assertEqual([item.chunk.chunk_id for item in result.hits], ["c"])
        self.assertEqual(store.requests[0].filters["department"], "support")
        self.assertEqual(result.debug["selected_count"], 1)
        self.assertEqual(engine.config.retrieval_profiles["precise"].final_k, 2)

    def test_unknown_per_call_organizer_is_rejected(self) -> None:
        engine, _store = make_engine()
        with self.assertRaisesRegex(ValueError, "unknown organizer"):
            engine.retrieve("query", profile="precise", organizer="missing")


class AsyncEngineRerankingTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_pipeline_uses_async_strategy_and_reranker(self) -> None:
        engine, store = make_engine()
        engine._reranker_cache["local"] = AsyncOnlyReranker()

        result = await engine.aquery("query", profile="precise")

        self.assertEqual(store.async_searches, 1)
        self.assertEqual([item.chunk.chunk_id for item in result.hits], ["c", "b"])

    async def test_sync_and_async_batches_preserve_input_order(self) -> None:
        engine, _store = make_engine()
        engine._reranker_cache["local"] = ReverseReranker()

        sync_results = engine.batch_retrieve(["first", "second"], profile="precise")
        async_results = await engine.abatch_retrieve(["third", "fourth"], profile="precise")

        self.assertEqual([result.query for result in sync_results], ["first", "second"])
        self.assertEqual([result.query for result in async_results], ["third", "fourth"])
        self.assertEqual(engine.batch_retrieve([], profile="precise"), ())
        self.assertEqual(await engine.abatch_retrieve([], profile="precise"), ())


if __name__ == "__main__":
    unittest.main()
