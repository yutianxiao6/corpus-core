from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

from offline_rag.contracts.chunks import Chunk
from offline_rag.contracts.retrieval import (
    Citation,
    QueryOverrides,
    RetrievalCandidate,
    RetrievalResult,
)
from offline_rag.engine import OfflineRagEngine
from offline_rag.langchain import OfflineRagLangChainRetriever


def native_result() -> RetrievalResult:
    candidate = RetrievalCandidate(
        chunk=Chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="退款政策正文",
            embedding_text="退款政策正文",
            source_uri="file:///refund.md",
            chunk_index=2,
            heading_path=("售后", "退款"),
            page_start=3,
            page_end=3,
            metadata={"department": "support", "acl": ["staff"]},
        ),
        dense_score=0.8,
        sparse_score=4.2,
        fusion_score=0.03,
        rerank_score=0.95,
        final_score=0.95,
        rank=1,
        origins=["dense", "sparse"],
    )
    return RetrievalResult(
        query="退款",
        processed_query="退款",
        hits=[candidate],
        citations=[
            Citation(
                citation_id="evidence-1",
                chunk_ids=["chunk-1"],
                source_uri="file:///refund.md",
                page_start=3,
                page_end=3,
            )
        ],
        index_version="index-v1",
        embedding_fingerprint="embedding-v1",
    )


class LangChainAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_and_ainvoke_return_standard_serializable_documents(self) -> None:
        engine = Mock(spec=OfflineRagEngine)
        engine.query.return_value = native_result()
        overrides = QueryOverrides(filters={"metadata.department": "support"})
        retriever = OfflineRagLangChainRetriever(
            engine=engine,
            profile="balanced",
            overrides=overrides,
        )

        documents = retriever.invoke("退款")
        async_documents = await retriever.ainvoke("退款")

        self.assertEqual(documents[0].page_content, "退款政策正文")
        self.assertEqual(documents[0].id, "chunk-1")
        self.assertEqual(documents[0].metadata["source"], "file:///refund.md")
        self.assertEqual(documents[0].metadata["rerank_score"], 0.95)
        self.assertEqual(documents[0].metadata["citation"]["chunk_ids"], ["chunk-1"])
        self.assertEqual(async_documents, documents)
        json.dumps(documents[0].metadata, ensure_ascii=False)
        engine.query.assert_called_with("退款", profile="balanced", overrides=overrides)

    async def test_engine_factory_validates_profile(self) -> None:
        engine = object.__new__(OfflineRagEngine)
        engine.config = Mock(retrieval_profiles={"fast": object()})
        retriever = engine.as_langchain_retriever(profile="fast")
        self.assertIs(retriever.engine, engine)
        with self.assertRaisesRegex(ValueError, "unknown retrieval profile"):
            engine.as_langchain_retriever(profile="missing")


if __name__ == "__main__":
    unittest.main()
