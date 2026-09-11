"""Pydantic models for configuration schema version 1."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Immutable base model that rejects misspelled and unsupported fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeConfig(StrictModel):
    offline: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    work_dir: str = "./data"
    cache_dir: str = "./data/cache"

    @model_validator(mode="after")
    def require_offline(self) -> RuntimeConfig:
        if not self.offline:
            raise ValueError("this distribution only supports offline=true")
        return self


class EmbeddingConfig(StrictModel):
    provider: str = "qwen_sentence_transformers"
    model_path: str = "./models/Qwen3-Embedding-0.6B"
    model_revision: str | None = None
    device: str = "auto"
    dimension: int = Field(default=1024, gt=0)
    normalize: bool = True
    batch_size: int = Field(default=16, gt=0)
    max_length: int = Field(default=1024, gt=0)
    query_instruction: str = (
        "Given a user question, retrieve relevant passages that answer the question."
    )


class VectorStoreConfig(StrictModel):
    provider: str = "qdrant"
    mode: Literal["local", "server"] = "local"
    path: str | None = "./data/qdrant"
    url: str | None = None
    collection_alias: str = "documents_active"
    distance: Literal["cosine", "dot", "euclid", "manhattan"] = "cosine"
    on_disk: bool = False

    @model_validator(mode="after")
    def validate_location(self) -> VectorStoreConfig:
        if self.mode == "local" and not self.path:
            raise ValueError("vector_store.path is required in local mode")
        if self.mode == "server" and not self.url:
            raise ValueError("vector_store.url is required in server mode")
        return self


class IngestionConfig(StrictModel):
    recursive: bool = True
    follow_symlinks: bool = False
    include_hidden: bool = False
    ignore_patterns: tuple[str, ...] = ("**/~$*", "**/*.tmp", "**/.git/**")
    continue_on_error: bool = True
    delete_missing_on_sync: bool = True
    commit_batch_size: int = Field(default=128, gt=0)


class RecursiveChunkProfile(StrictModel):
    type: Literal["recursive"]
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    length_unit: Literal["character", "token"] = "character"
    minimum_size: int = Field(default=80, ge=0)

    @model_validator(mode="after")
    def validate_sizes(self) -> RecursiveChunkProfile:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.minimum_size > self.chunk_size:
            raise ValueError("minimum_size must not exceed chunk_size")
        return self


class HeadingRecursiveChunkProfile(StrictModel):
    type: Literal["heading_recursive"]
    max_chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    include_heading_in_embedding: bool = True

    @model_validator(mode="after")
    def validate_sizes(self) -> HeadingRecursiveChunkProfile:
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError("chunk_overlap must be smaller than max_chunk_size")
        return self


class TableRowsChunkProfile(StrictModel):
    type: Literal["table_rows"]
    repeat_headers: bool = True
    max_rows_per_chunk: int = Field(default=10, gt=0)


class PageAwareChunkProfile(StrictModel):
    type: Literal["page_aware"]
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def validate_sizes(self) -> PageAwareChunkProfile:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class ParagraphPackingChunkProfile(StrictModel):
    type: Literal["paragraph_packing"]
    target_size: int = Field(default=800, gt=0)
    maximum_size: int = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def validate_sizes(self) -> ParagraphPackingChunkProfile:
        if self.target_size > self.maximum_size:
            raise ValueError("target_size must not exceed maximum_size")
        return self


class SyntaxChunkProfile(StrictModel):
    type: Literal["syntax"]
    fallback_profile: str = "default"


class ParentChildChunkProfile(StrictModel):
    type: Literal["parent_child"]
    parent_size: int = Field(default=1400, gt=0)
    child_size: int = Field(default=350, gt=0)
    child_overlap: int = Field(default=60, ge=0)

    @model_validator(mode="after")
    def validate_sizes(self) -> ParentChildChunkProfile:
        if self.child_size >= self.parent_size:
            raise ValueError("child_size must be smaller than parent_size")
        if self.child_overlap >= self.child_size:
            raise ValueError("child_overlap must be smaller than child_size")
        return self


ChunkProfile = Annotated[
    RecursiveChunkProfile
    | HeadingRecursiveChunkProfile
    | PageAwareChunkProfile
    | ParagraphPackingChunkProfile
    | TableRowsChunkProfile
    | SyntaxChunkProfile
    | ParentChildChunkProfile,
    Field(discriminator="type"),
]


class RouteMatch(StrictModel):
    extensions: tuple[str, ...] = ()
    filename_patterns: tuple[str, ...] = ()
    path_patterns: tuple[str, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_condition(self) -> RouteMatch:
        if not (self.extensions or self.filename_patterns or self.path_patterns or self.metadata):
            raise ValueError("a routing match needs at least one condition")
        return self


class RoutingRule(StrictModel):
    match: RouteMatch
    use: str


class FusionConfig(StrictModel):
    type: Literal["rrf", "weighted"] = "rrf"
    constant: int = Field(default=60, gt=0)
    dense_weight: float = Field(default=0.5, ge=0)
    sparse_weight: float = Field(default=0.5, ge=0)


class RetrievalProfile(StrictModel):
    strategy: Literal["dense", "sparse", "hybrid", "parent_child"] = "dense"
    fetch_k: int | None = Field(default=None, gt=0)
    dense_fetch_k: int | None = Field(default=None, gt=0)
    sparse_fetch_k: int | None = Field(default=None, gt=0)
    fusion: FusionConfig | None = None
    reranker: str | None = None
    rerank_top_n: int | None = Field(default=None, gt=0)
    neighbor_expansion: int = Field(default=0, ge=0)
    final_k: int = Field(default=5, gt=0)
    organizer: str = "flat"


class RerankerConfig(StrictModel):
    provider: str
    model_path: str
    device: str = "auto"
    batch_size: int = Field(default=8, gt=0)


class OrganizerConfig(StrictModel):
    type: Literal["flat", "context", "grouped", "parent", "diverse", "debug"]
    max_context_tokens: int = Field(default=6000, gt=0)
    merge_neighbors: bool = False
    maximum_chunks_per_document: int | None = Field(default=None, gt=0)
    citation_style: Literal["numbered", "inline", "none"] = "numbered"


def _default_chunk_profiles() -> dict[str, ChunkProfile]:
    return {
        "default": RecursiveChunkProfile(type="recursive"),
        "markdown_heading": HeadingRecursiveChunkProfile(type="heading_recursive"),
    }


def _default_retrieval_profiles() -> dict[str, RetrievalProfile]:
    return {
        "fast": RetrievalProfile(strategy="dense", fetch_k=8, final_k=5),
    }


class RagConfig(StrictModel):
    """Root schema for a fully resolved retriever configuration."""

    version: Literal[1] = 1
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    chunk_profiles: dict[str, ChunkProfile] = Field(default_factory=_default_chunk_profiles)
    routing: tuple[RoutingRule, ...] = ()
    retrieval_profiles: dict[str, RetrievalProfile] = Field(
        default_factory=_default_retrieval_profiles
    )
    rerankers: dict[str, RerankerConfig] = Field(default_factory=dict)
    organizers: dict[str, OrganizerConfig] = Field(default_factory=dict)
    enabled_plugins: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> RagConfig:
        for name, chunk_profile in self.chunk_profiles.items():
            if (
                isinstance(chunk_profile, SyntaxChunkProfile)
                and chunk_profile.fallback_profile not in self.chunk_profiles
            ):
                raise ValueError(
                    f"chunk profile {name!r} references missing fallback "
                    f"{chunk_profile.fallback_profile!r}"
                )
        for position, rule in enumerate(self.routing):
            if rule.use not in self.chunk_profiles:
                raise ValueError(
                    f"routing rule {position} references missing chunk profile {rule.use!r}"
                )
        for name, retrieval_profile in self.retrieval_profiles.items():
            if retrieval_profile.reranker and retrieval_profile.reranker not in self.rerankers:
                raise ValueError(
                    f"retrieval profile {name!r} references missing reranker "
                    f"{retrieval_profile.reranker!r}"
                )
            if (
                retrieval_profile.organizer != "flat"
                and retrieval_profile.organizer not in self.organizers
            ):
                raise ValueError(
                    f"retrieval profile {name!r} references missing organizer "
                    f"{retrieval_profile.organizer!r}"
                )
        return self
