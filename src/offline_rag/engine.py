"""High-level reusable indexing and retrieval module."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from offline_rag.langchain import OfflineRagLangChainRetriever

from offline_rag.concurrency import AsyncMicroBatcher, QueuedEmbeddingProvider, QueuedReranker
from offline_rag.config.models import OrganizerConfig, RagConfig, RerankerConfig, RetrievalProfile
from offline_rag.contracts.common import JSONValue
from offline_rag.contracts.indexing import IngestionReport
from offline_rag.contracts.retrieval import (
    ContextBudget,
    OrganizedResult,
    QueryOverrides,
    RetrievalCandidate,
    RetrievalOptions,
    RetrievalRequest,
    RetrievalResult,
)
from offline_rag.embeddings import (
    HashedLexicalSparseEmbedding,
    QwenSentenceTransformerEmbedding,
)
from offline_rag.exceptions import ConfigurationError, OfflineResourceMissingError, RerankerError
from offline_rag.indexing import IngestionJournal
from offline_rag.ingestion import IngestionService, PreviewReport, build_index_specification
from offline_rag.organizers import (
    ContextOrganizer,
    DebugOrganizer,
    DiverseOrganizer,
    FlatOrganizer,
    GroupByDocumentOrganizer,
    MergeNeighborsOrganizer,
    ParentOrganizer,
)
from offline_rag.ports import Reranker, ResultOrganizer, RetrievalStrategy
from offline_rag.rerankers import QwenCrossEncoderReranker
from offline_rag.retrieval import (
    DenseSimilarityStrategy,
    HybridRetrievalStrategy,
    MaximalMarginalRelevanceSelector,
    NeighborExpander,
    SparseSimilarityStrategy,
    limit_per_document,
    score_threshold_filter,
)
from offline_rag.vectorstores import QdrantLocalVectorStore, QdrantServerVectorStore


class OfflineRagEngine:
    """Library facade used by the CLI and downstream company QA applications."""

    def __init__(self, config: RagConfig) -> None:
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
        concurrency = config.query_concurrency
        self.query_embedding = QueuedEmbeddingProvider(
            self.embedding,
            AsyncMicroBatcher(
                self.embedding.embed_queries,
                max_batch_size=concurrency.embedding_microbatch_size,
                max_wait_ms=concurrency.embedding_wait_ms,
                workers=concurrency.embedding_workers,
                queue_capacity=concurrency.queue_capacity,
                enqueue_timeout_seconds=concurrency.enqueue_timeout_seconds,
                execution_timeout_seconds=concurrency.execution_timeout_seconds,
                name="embedding",
            ),
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
        vector_config = config.vector_store
        if vector_config.mode == "local":
            assert vector_config.path is not None
            self.vector_store = QdrantLocalVectorStore(
                vector_config.path,
                collection_name=vector_config.collection_alias,
            )
        else:
            assert vector_config.url is not None
            api_key = None
            if vector_config.api_key_env:
                api_key = os.environ.get(vector_config.api_key_env)
                if not api_key:
                    raise ConfigurationError(
                        f"Qdrant API key environment variable is missing: "
                        f"{vector_config.api_key_env}"
                    )
            self.vector_store = QdrantServerVectorStore(
                vector_config.url,
                collection_name=vector_config.collection_alias,
                api_key=api_key,
                prefer_grpc=vector_config.prefer_grpc,
                timeout_seconds=vector_config.timeout_seconds,
                pool_size=vector_config.pool_size,
            )
        self.index_specification = build_index_specification(
            config, self.embedding.specification, self.sparse_embedding
        )
        self.journal = IngestionJournal(Path(config.runtime.work_dir) / "ingestion.sqlite3")
        self._reranker_cache: dict[str, Reranker] = {}
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

    def as_langchain_retriever(
        self,
        *,
        profile: str = "fast",
        overrides: QueryOverrides | None = None,
    ) -> OfflineRagLangChainRetriever:
        from offline_rag.langchain import OfflineRagLangChainRetriever

        if profile not in self.config.retrieval_profiles:
            raise ValueError(f"unknown retrieval profile: {profile}")
        return OfflineRagLangChainRetriever(
            engine=self,
            profile=profile,
            overrides=overrides or QueryOverrides(),
        )

    def retrieve(
        self,
        query: str,
        *,
        profile: str = "fast",
        filters: Mapping[str, JSONValue] | None = None,
        organizer: str | None = None,
        final_k: int | None = None,
        score_threshold: float | None = None,
        rerank_top_n: int | None = None,
        mmr_lambda: float | None = None,
        mmr_fetch_k: int | None = None,
        neighbor_expansion: int | None = None,
        maximum_chunks_per_document: int | None = None,
    ) -> RetrievalResult:
        """Retrieve with safe per-call overrides that cannot alter the index specification."""

        return self.query(
            query,
            profile=profile,
            overrides=self._make_overrides(
                filters,
                organizer,
                final_k,
                score_threshold,
                rerank_top_n,
                mmr_lambda,
                mmr_fetch_k,
                neighbor_expansion,
                maximum_chunks_per_document,
            ),
        )

    async def aretrieve(
        self,
        query: str,
        *,
        profile: str = "fast",
        filters: Mapping[str, JSONValue] | None = None,
        organizer: str | None = None,
        final_k: int | None = None,
        score_threshold: float | None = None,
        rerank_top_n: int | None = None,
        mmr_lambda: float | None = None,
        mmr_fetch_k: int | None = None,
        neighbor_expansion: int | None = None,
        maximum_chunks_per_document: int | None = None,
    ) -> RetrievalResult:
        """Asynchronously execute the same pipeline and overrides as :meth:`retrieve`."""

        return await self.aquery(
            query,
            profile=profile,
            overrides=self._make_overrides(
                filters,
                organizer,
                final_k,
                score_threshold,
                rerank_top_n,
                mmr_lambda,
                mmr_fetch_k,
                neighbor_expansion,
                maximum_chunks_per_document,
            ),
        )

    def batch_retrieve(
        self,
        queries: Sequence[str],
        *,
        profile: str = "fast",
        overrides: QueryOverrides | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Retrieve a batch in input order with one shared immutable option set."""

        return tuple(self.query(query, profile=profile, overrides=overrides) for query in queries)

    async def abatch_retrieve(
        self,
        queries: Sequence[str],
        *,
        profile: str = "fast",
        overrides: QueryOverrides | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Concurrently retrieve a batch while preserving input order."""

        return tuple(
            await asyncio.gather(
                *(self.aquery(query, profile=profile, overrides=overrides) for query in queries)
            )
        )

    def query(
        self,
        query: str,
        *,
        profile: str = "fast",
        overrides: QueryOverrides | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        retrieval_profile, request, strategy = self._query_plan(query, profile, overrides)
        self.vector_store.ensure_index(self.index_specification)
        candidates = strategy.retrieve(request)
        retrieved_at = time.perf_counter()
        candidates, warnings = self._rerank(query, candidates, retrieval_profile)
        reranked_at = time.perf_counter()
        candidates = self._postprocess(query, candidates, retrieval_profile)
        postprocessed_at = time.perf_counter()
        organized = self._organize(query, candidates, retrieval_profile)
        finished = time.perf_counter()
        return self._retrieval_result(
            query,
            organized,
            warnings,
            started,
            retrieved_at,
            reranked_at,
            postprocessed_at,
            finished,
        )

    async def aquery(
        self,
        query: str,
        *,
        profile: str = "fast",
        overrides: QueryOverrides | None = None,
    ) -> RetrievalResult:
        """Native asynchronous query pipeline with the same result contract as ``query``."""

        started = time.perf_counter()
        retrieval_profile, request, strategy = self._query_plan(query, profile, overrides)
        await asyncio.to_thread(self.vector_store.ensure_index, self.index_specification)
        candidates = await strategy.aretrieve(request)
        retrieved_at = time.perf_counter()
        candidates, warnings = await self._arerank(query, candidates, retrieval_profile)
        reranked_at = time.perf_counter()
        candidates = await self._apostprocess(query, candidates, retrieval_profile)
        postprocessed_at = time.perf_counter()
        organized = await asyncio.to_thread(self._organize, query, candidates, retrieval_profile)
        finished = time.perf_counter()
        return self._retrieval_result(
            query,
            organized,
            warnings,
            started,
            retrieved_at,
            reranked_at,
            postprocessed_at,
            finished,
        )

    def _query_plan(
        self,
        query: str,
        profile: str,
        overrides: QueryOverrides | None,
    ) -> tuple[RetrievalProfile, RetrievalRequest, RetrievalStrategy]:
        try:
            retrieval_profile = self.config.retrieval_profiles[profile]
        except KeyError as exc:
            raise ValueError(f"unknown retrieval profile: {profile}") from exc
        retrieval_profile = self._resolve_profile(profile, retrieval_profile, overrides)
        filters = overrides.filters if overrides is not None else {}
        retrieval_limit = max(
            retrieval_profile.final_k,
            (
                retrieval_profile.rerank_top_n or retrieval_profile.final_k
                if retrieval_profile.reranker
                else retrieval_profile.final_k
            ),
            (
                retrieval_profile.mmr_fetch_k
                or retrieval_profile.fetch_k
                or retrieval_profile.dense_fetch_k
                or retrieval_profile.sparse_fetch_k
                or retrieval_profile.final_k * 4
                if retrieval_profile.mmr_lambda is not None
                else retrieval_profile.final_k
            ),
        )
        options = RetrievalOptions(
            profile=profile,
            final_k=retrieval_limit,
            organizer=retrieval_profile.organizer,
            filters=filters,
        )
        request = RetrievalRequest(query, options)
        strategy: RetrievalStrategy
        if retrieval_profile.strategy == "dense":
            strategy = DenseSimilarityStrategy(
                self.query_embedding,
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
                self.query_embedding,
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
        return retrieval_profile, request, strategy

    def _rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        profile: RetrievalProfile,
    ) -> tuple[Sequence[RetrievalCandidate], tuple[str, ...]]:
        warnings: list[str] = []
        if profile.reranker:
            reranker_config = self.config.rerankers[profile.reranker]
            try:
                candidates = self._reranker_for(profile.reranker, reranker_config).rerank(
                    query,
                    candidates,
                    top_n=profile.rerank_top_n or profile.final_k,
                )
            except (OfflineResourceMissingError, RerankerError):
                if reranker_config.failure_policy == "fail":
                    raise
                warnings.append(f"reranker {profile.reranker!r} failed; returned recall order")
        return candidates, tuple(warnings)

    async def _arerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        profile: RetrievalProfile,
    ) -> tuple[Sequence[RetrievalCandidate], tuple[str, ...]]:
        if not profile.reranker:
            return candidates, ()
        reranker_config = self.config.rerankers[profile.reranker]
        try:
            reranked = await self._reranker_for(profile.reranker, reranker_config).arerank(
                query,
                candidates,
                top_n=profile.rerank_top_n or profile.final_k,
            )
            return reranked, ()
        except (OfflineResourceMissingError, RerankerError):
            if reranker_config.failure_policy == "fail":
                raise
            return candidates, (f"reranker {profile.reranker!r} failed; returned recall order",)

    def _postprocess(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        profile: RetrievalProfile,
    ) -> Sequence[RetrievalCandidate]:
        candidates = score_threshold_filter(candidates, profile.score_threshold)
        candidates = limit_per_document(candidates, profile.maximum_chunks_per_document)
        if profile.mmr_lambda is not None:
            candidates = MaximalMarginalRelevanceSelector(self.query_embedding).select(
                query,
                candidates,
                limit=profile.final_k,
                relevance_weight=profile.mmr_lambda,
            )
        else:
            candidates = tuple(candidates[: profile.final_k])
        if profile.neighbor_expansion:
            candidates = NeighborExpander(self.vector_store).expand(
                candidates, distance=profile.neighbor_expansion
            )
        return candidates

    async def _apostprocess(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        profile: RetrievalProfile,
    ) -> Sequence[RetrievalCandidate]:
        candidates = score_threshold_filter(candidates, profile.score_threshold)
        candidates = limit_per_document(candidates, profile.maximum_chunks_per_document)
        if profile.mmr_lambda is not None:
            candidates = await MaximalMarginalRelevanceSelector(self.query_embedding).aselect(
                query,
                candidates,
                limit=profile.final_k,
                relevance_weight=profile.mmr_lambda,
            )
        else:
            candidates = tuple(candidates[: profile.final_k])
        if profile.neighbor_expansion:
            candidates = await NeighborExpander(self.vector_store).aexpand(
                candidates, distance=profile.neighbor_expansion
            )
        return candidates

    def _organize(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        profile: RetrievalProfile,
    ) -> OrganizedResult:
        organizer_config = self._organizer_config(profile.organizer)
        budget = self._budget(organizer_config)
        organizer = self._organizer(organizer_config)
        return organizer.organize(query, candidates, budget)

    def _retrieval_result(
        self,
        query: str,
        organized: OrganizedResult,
        warnings: Sequence[str],
        started: float,
        retrieved_at: float,
        reranked_at: float,
        postprocessed_at: float,
        finished: float,
    ) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            processed_query=query.strip(),
            hits=organized.hits,
            context=organized.context,
            citations=organized.citations,
            groups=organized.groups,
            debug=organized.debug,
            index_version=self.index_specification.fingerprint()[:16],
            embedding_fingerprint=self.embedding.specification.fingerprint(),
            timings_ms={
                "retrieve": (retrieved_at - started) * 1000,
                "rerank": (reranked_at - retrieved_at) * 1000,
                "postprocess": (postprocessed_at - reranked_at) * 1000,
                "organize": (finished - postprocessed_at) * 1000,
                "total": (finished - started) * 1000,
            },
            warnings=(*warnings, *organized.warnings),
        )

    @staticmethod
    def _make_overrides(
        filters: Mapping[str, JSONValue] | None,
        organizer: str | None,
        final_k: int | None,
        score_threshold: float | None,
        rerank_top_n: int | None,
        mmr_lambda: float | None,
        mmr_fetch_k: int | None,
        neighbor_expansion: int | None,
        maximum_chunks_per_document: int | None,
    ) -> QueryOverrides:
        return QueryOverrides(
            filters=filters or {},
            organizer=organizer,
            final_k=final_k,
            score_threshold=score_threshold,
            rerank_top_n=rerank_top_n,
            mmr_lambda=mmr_lambda,
            mmr_fetch_k=mmr_fetch_k,
            neighbor_expansion=neighbor_expansion,
            maximum_chunks_per_document=maximum_chunks_per_document,
        )

    def _resolve_profile(
        self,
        name: str,
        profile: RetrievalProfile,
        overrides: QueryOverrides | None,
    ) -> RetrievalProfile:
        if overrides is None:
            return profile
        data = profile.model_dump()
        for field_name in (
            "organizer",
            "final_k",
            "score_threshold",
            "rerank_top_n",
            "mmr_lambda",
            "mmr_fetch_k",
            "neighbor_expansion",
            "maximum_chunks_per_document",
        ):
            value = getattr(overrides, field_name)
            if value is not None:
                data[field_name] = value
        resolved = RetrievalProfile.model_validate(data)
        if resolved.mmr_fetch_k is not None and resolved.mmr_lambda is None:
            raise ValueError(f"query profile {name!r} configures mmr_fetch_k without mmr_lambda")
        if resolved.mmr_fetch_k is not None and resolved.mmr_fetch_k < resolved.final_k:
            raise ValueError(f"query profile {name!r} mmr_fetch_k must be at least final_k")
        if resolved.reranker:
            reranker = self.config.rerankers[resolved.reranker]
            rerank_limit = resolved.rerank_top_n or resolved.final_k
            if rerank_limit < resolved.final_k:
                raise ValueError(f"query profile {name!r} rerank_top_n must be at least final_k")
            if rerank_limit > reranker.maximum_candidates:
                raise ValueError(f"query profile {name!r} exceeds reranker maximum_candidates")
        self._organizer_config(resolved.organizer)
        return resolved

    def _reranker_for(self, name: str, config: RerankerConfig) -> Reranker:
        cached = self._reranker_cache.get(name)
        if cached is not None:
            return cached
        if config.provider != QwenCrossEncoderReranker.provider_name:
            raise ValueError(f"unsupported reranker provider: {config.provider}")
        raw_reranker = QwenCrossEncoderReranker(
            config.model_path,
            revision=config.model_revision,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
            maximum_candidates=config.maximum_candidates,
            instruction=config.instruction,
            score_mode=config.score_mode,
        )
        concurrency = self.config.query_concurrency
        reranker = QueuedReranker(
            raw_reranker,
            AsyncMicroBatcher(
                raw_reranker.rerank_batch,
                max_batch_size=concurrency.reranker_microbatch_size,
                max_wait_ms=concurrency.reranker_wait_ms,
                workers=concurrency.reranker_workers,
                queue_capacity=concurrency.queue_capacity,
                enqueue_timeout_seconds=concurrency.enqueue_timeout_seconds,
                execution_timeout_seconds=concurrency.execution_timeout_seconds,
                name=f"reranker-{name}",
            ),
        )
        self._reranker_cache[name] = reranker
        return reranker

    def _organizer(self, config: OrganizerConfig | None) -> ResultOrganizer:
        if config is None or config.type == "flat":
            return FlatOrganizer()
        if config.type == "context":
            if config.merge_neighbors:
                return MergeNeighborsOrganizer(token_counter=self.embedding.count_tokens)
            return ContextOrganizer(token_counter=self.embedding.count_tokens)
        if config.type == "grouped":
            return GroupByDocumentOrganizer()
        if config.type == "merge_neighbors":
            return MergeNeighborsOrganizer(token_counter=self.embedding.count_tokens)
        if config.type == "parent":
            return ParentOrganizer(token_counter=self.embedding.count_tokens)
        if config.type == "diverse":
            return DiverseOrganizer()
        if config.type == "debug":
            return DebugOrganizer()
        raise ValueError(f"unsupported organizer type: {config.type}")

    def _organizer_config(self, name: str) -> OrganizerConfig | None:
        configured = self.config.organizers.get(name)
        if configured is not None:
            return configured
        if name == "flat":
            return None
        if name in {
            "context",
            "grouped",
            "merge_neighbors",
            "parent",
            "diverse",
            "debug",
        }:
            return OrganizerConfig.model_validate({"type": name})
        raise ValueError(f"unknown organizer: {name}")

    @staticmethod
    def _budget(config: OrganizerConfig | None) -> ContextBudget:
        if config is None:
            return ContextBudget(max_characters=1_000_000)
        return ContextBudget(
            max_tokens=config.max_context_tokens,
            maximum_chunks_per_document=config.maximum_chunks_per_document,
        )

    def close(self) -> None:
        async def close_queues() -> None:
            await self._aclose_inference_queues()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(close_queues())
        else:
            loop.create_task(close_queues())
        self.journal.close()
        self.vector_store.close()

    async def aclose(self) -> None:
        await self._aclose_inference_queues()
        self.journal.close()
        await self.vector_store.aclose()

    async def _aclose_inference_queues(self) -> None:
        await self.query_embedding.aclose()
        await asyncio.gather(
            *(
                reranker.aclose()
                for reranker in self._reranker_cache.values()
                if isinstance(reranker, QueuedReranker)
            )
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()
