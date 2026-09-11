"""High-level reusable indexing and retrieval module."""

from __future__ import annotations

import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Self

from offline_rag.config.models import OrganizerConfig, RagConfig
from offline_rag.contracts.indexing import IngestionReport
from offline_rag.contracts.retrieval import (
    ContextBudget,
    RetrievalOptions,
    RetrievalRequest,
    RetrievalResult,
)
from offline_rag.embeddings import (
    HashedLexicalSparseEmbedding,
    QwenSentenceTransformerEmbedding,
)
from offline_rag.indexing import IngestionJournal
from offline_rag.ingestion import IngestionService, PreviewReport, build_index_specification
from offline_rag.organizers import ContextOrganizer, FlatOrganizer
from offline_rag.ports import RetrievalStrategy
from offline_rag.retrieval import (
    DenseSimilarityStrategy,
    HybridRetrievalStrategy,
    SparseSimilarityStrategy,
)
from offline_rag.vectorstores import QdrantLocalVectorStore


class OfflineRagEngine:
    """Library facade used by the CLI and downstream company QA applications."""

    def __init__(self, config: RagConfig) -> None:
        if config.vector_store.mode != "local" or config.vector_store.path is None:
            raise ValueError("P1 engine currently supports Qdrant Local mode only")
        self.config = config
        embedding_config = config.embedding
        self.embedding = QwenSentenceTransformerEmbedding(
            embedding_config.model_path,
            revision=embedding_config.model_revision or "pinned-local",
            dimension=embedding_config.dimension,
            normalize=embedding_config.normalize,
            batch_size=embedding_config.batch_size,
            max_length=embedding_config.max_length,
            query_instruction=embedding_config.query_instruction,
            device=embedding_config.device,
        )
        sparse_config = config.sparse_embedding
        self.sparse_embedding = (
            HashedLexicalSparseEmbedding(
                hash_space=sparse_config.hash_space,
                normalize=sparse_config.normalize,
                include_cjk_bigrams=sparse_config.include_cjk_bigrams,
                revision=sparse_config.revision,
            )
            if sparse_config is not None
            else None
        )
        self.vector_store = QdrantLocalVectorStore(
            config.vector_store.path,
            collection_name=config.vector_store.collection_alias,
        )
        self.index_specification = build_index_specification(
            config, self.embedding.specification, self.sparse_embedding
        )
        self.journal = IngestionJournal(Path(config.runtime.work_dir) / "ingestion.sqlite3")
        self.journal.recover_interrupted_jobs()
        self.ingestion = IngestionService(
            config,
            embedding=self.embedding,
            sparse_embedding=self.sparse_embedding,
            vector_store=self.vector_store,
            token_counter=self.embedding.count_tokens,
            journal=self.journal,
        )

    @classmethod
    def from_config(cls, config: RagConfig) -> OfflineRagEngine:
        return cls(config)

    def preview(self, *inputs: str | Path) -> PreviewReport:
        return IngestionService(self.config, token_counter=self.embedding.count_tokens).preview(
            inputs
        )

    def build(self, *inputs: str | Path) -> IngestionReport:
        return self.rebuild(*inputs)

    def rebuild(self, *inputs: str | Path) -> IngestionReport:
        version = f"{self.index_specification.fingerprint()[:8]}-{time.time_ns()}"
        staging = self.vector_store.create_staging(self.index_specification, version=version)
        work_dir = Path(self.config.runtime.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            with (
                tempfile.TemporaryDirectory(prefix="staging-journal-", dir=work_dir) as directory,
                IngestionJournal(Path(directory) / "journal.sqlite3") as staging_journal,
            ):
                service = IngestionService(
                    self.config,
                    embedding=self.embedding,
                    sparse_embedding=self.sparse_embedding,
                    vector_store=staging,
                    token_counter=self.embedding.count_tokens,
                    journal=staging_journal,
                )
                report = service.build(inputs, self.index_specification)
                if report.failed_count:
                    self.vector_store.drop_collection(staging.collection_name)
                    return report
                if staging.point_count() != report.chunk_count:
                    self.vector_store.drop_collection(staging.collection_name)
                    raise RuntimeError(
                        "staging collection point count does not match the ingestion report"
                    )
                snapshot = staging_journal.list_sources()
                self.journal.save_collection_snapshot(staging.collection_name, snapshot)
                self.vector_store.activate_staging(staging)
                job_id = self.journal.start_job(report.index_version)
                try:
                    self.journal.replace_sources(snapshot)
                    self.journal.finish_job(job_id)
                except Exception:
                    self.journal.finish_job(job_id, status="failed")
                    raise
                return replace(report, job_id=job_id)
        except Exception:
            if self.vector_store.active_collection() != staging.collection_name:
                self.vector_store.drop_collection(staging.collection_name)
            raise

    def activate(self, collection_name: str) -> None:
        """Roll back or forward to a retained compatible physical collection."""

        view = self.vector_store.collection(collection_name)
        view.ensure_index(self.index_specification)
        if not self.journal.has_collection_snapshot(collection_name):
            raise ValueError(
                f"no ingestion journal snapshot exists for collection: {collection_name}"
            )
        snapshot = self.journal.collection_snapshot(collection_name)
        self.vector_store.activate_staging(collection_name)
        self.journal.replace_sources(snapshot)

    def sync(self, *inputs: str | Path) -> IngestionReport:
        return self.ingestion.sync(inputs, self.index_specification)

    def query(self, query: str, *, profile: str = "fast") -> RetrievalResult:
        started = time.perf_counter()
        try:
            retrieval_profile = self.config.retrieval_profiles[profile]
        except KeyError as exc:
            raise ValueError(f"unknown retrieval profile: {profile}") from exc
        self.vector_store.ensure_index(self.index_specification)
        options = RetrievalOptions(
            profile=profile,
            final_k=retrieval_profile.final_k,
            organizer=retrieval_profile.organizer,
        )
        request = RetrievalRequest(query, options)
        strategy: RetrievalStrategy
        if retrieval_profile.strategy == "dense":
            strategy = DenseSimilarityStrategy(
                self.embedding,
                self.vector_store,
                fetch_k=retrieval_profile.fetch_k or retrieval_profile.final_k,
            )
        elif retrieval_profile.strategy == "sparse":
            if self.sparse_embedding is None:
                raise ValueError("sparse retrieval requires sparse_embedding configuration")
            strategy = SparseSimilarityStrategy(
                self.sparse_embedding,
                self.vector_store,
                fetch_k=retrieval_profile.sparse_fetch_k
                or retrieval_profile.fetch_k
                or retrieval_profile.final_k,
            )
        elif retrieval_profile.strategy == "hybrid":
            if self.sparse_embedding is None or retrieval_profile.fusion is None:
                raise ValueError("hybrid retrieval requires sparse embedding and fusion")
            fusion = retrieval_profile.fusion
            strategy = HybridRetrievalStrategy(
                self.embedding,
                self.sparse_embedding,
                self.vector_store,
                dense_fetch_k=retrieval_profile.dense_fetch_k
                or retrieval_profile.fetch_k
                or retrieval_profile.final_k,
                sparse_fetch_k=retrieval_profile.sparse_fetch_k
                or retrieval_profile.fetch_k
                or retrieval_profile.final_k,
                fusion=fusion.type,
                rrf_constant=fusion.constant,
                dense_weight=fusion.dense_weight,
                sparse_weight=fusion.sparse_weight,
            )
        else:
            raise ValueError(
                f"retrieval strategy is not implemented yet: {retrieval_profile.strategy}"
            )
        candidates = strategy.retrieve(request)
        retrieved_at = time.perf_counter()
        organizer_config = self.config.organizers.get(retrieval_profile.organizer)
        budget = self._budget(organizer_config)
        organizer = (
            ContextOrganizer()
            if organizer_config is not None and organizer_config.type == "context"
            else FlatOrganizer()
        )
        organized = organizer.organize(query, candidates, budget)
        finished = time.perf_counter()
        return RetrievalResult(
            query=query,
            processed_query=query.strip(),
            hits=organized.hits,
            context=organized.context,
            citations=organized.citations,
            index_version=self.index_specification.fingerprint()[:16],
            embedding_fingerprint=self.embedding.specification.fingerprint(),
            timings_ms={
                "retrieve": (retrieved_at - started) * 1000,
                "organize": (finished - retrieved_at) * 1000,
                "total": (finished - started) * 1000,
            },
            warnings=organized.warnings,
        )

    @staticmethod
    def _budget(config: OrganizerConfig | None) -> ContextBudget:
        if config is None:
            return ContextBudget(max_characters=1_000_000)
        return ContextBudget(
            max_tokens=config.max_context_tokens,
            maximum_chunks_per_document=config.maximum_chunks_per_document,
        )

    def close(self) -> None:
        self.journal.close()
        self.vector_store.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
