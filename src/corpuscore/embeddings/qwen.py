"""Offline Qwen3 embedding adapter with distinct query/document encoding."""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Protocol, cast

from corpuscore.contracts.indexing import DistanceMetric, EmbeddingSpecification
from corpuscore.exceptions import EmbeddingError, LocalResourceMissingError
from corpuscore.ports import Vector


class _SentenceModel(Protocol):
    max_seq_length: int

    def encode(self, inputs: Sequence[str] | str, **kwargs: object) -> object: ...

    def preprocess(self, inputs: list[str], **kwargs: object) -> Mapping[str, object]: ...


class QwenSentenceTransformerEmbedding:
    """SentenceTransformers adapter that is unable to download missing models."""

    provider_name = "qwen_sentence_transformers"

    def __init__(
        self,
        model_path: str | Path,
        *,
        revision: str = "pinned-local",
        dimension: int = 1024,
        normalize: bool = True,
        batch_size: int = 16,
        max_length: int = 1024,
        query_instruction: str = (
            "Given a user question, retrieve relevant passages that answer the question."
        ),
        device: str = "auto",
        trust_remote_code: bool = False,
        model_checksum: str | None = None,
        model: _SentenceModel | None = None,
    ) -> None:
        self._model_path = Path(model_path).expanduser().resolve()
        if not self._model_path.is_dir():
            raise LocalResourceMissingError(
                "local embedding model directory is missing",
                details={"path": str(self._model_path)},
            )
        if dimension <= 0 or batch_size <= 0 or max_length <= 0:
            raise ValueError("dimension, batch_size and max_length must be positive")
        if not revision.strip():
            raise ValueError("revision must not be empty")
        self._dimension = dimension
        self._normalize = normalize
        self._batch_size = batch_size
        self._max_length = max_length
        self._query_instruction = query_instruction.strip()
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._model_checksum = model_checksum or self._checksum_directory(self._model_path)
        self._model: _SentenceModel | None = model
        self._lock = RLock()
        self._specification = EmbeddingSpecification(
            provider=self.provider_name,
            model=self._model_path.name,
            revision=revision,
            dimension=dimension,
            normalized=normalize,
            distance=DistanceMetric.COSINE,
            max_length=max_length,
            query_instruction=self._query_instruction,
            tokenizer_revision=revision,
            model_checksum=self._model_checksum,
        )
        if self._model is not None:
            self._model.max_seq_length = max_length

    @property
    def specification(self) -> EmbeddingSpecification:
        return self._specification

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        if not texts:
            return ()
        if any(not text.strip() for text in texts):
            raise EmbeddingError("document embedding input contains empty text")
        return self._encode(tuple(texts), prompt=None)

    def embed_query(self, text: str) -> Vector:
        return self.embed_queries((text,))[0]

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Vector]:
        if not texts:
            return ()
        if any(not text.strip() for text in texts):
            raise EmbeddingError("query embedding input must not be empty")
        prompt = (
            f"Instruct: {self._query_instruction}\nQuery: "
            if self._query_instruction
            else "Query: "
        )
        return self._encode(tuple(texts), prompt=prompt)

    async def aembed_query(self, text: str) -> Vector:
        return await asyncio.to_thread(self.embed_query, text)

    def count_tokens(self, text: str) -> int:
        """Count tokens with the exact local model tokenizer used for indexing."""

        if not text:
            return 0
        try:
            tokenized = self._get_model().preprocess([text])
            input_ids = tokenized["input_ids"]
            shape = getattr(input_ids, "shape", None)
            if shape is not None:
                return int(shape[-1])
            if isinstance(input_ids, Sequence) and input_ids:
                first = input_ids[0]
                if isinstance(first, Sequence):
                    return len(first)
        except Exception as exc:
            raise EmbeddingError("local tokenizer failed to count input tokens") from exc
        raise EmbeddingError("local tokenizer returned an unsupported input_ids value")

    def _encode(self, texts: Sequence[str], *, prompt: str | None) -> tuple[Vector, ...]:
        model = self._get_model()
        options: dict[str, object] = {
            "batch_size": self._batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": self._normalize,
        }
        if prompt is not None:
            options["prompt"] = prompt
        try:
            raw = model.encode(texts, **options)
            vectors = self._to_vectors(raw)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("local embedding inference failed") from exc
        if len(vectors) != len(texts):
            raise EmbeddingError(
                "embedding model returned an unexpected batch size",
                details={"expected": len(texts), "actual": len(vectors)},
            )
        for vector in vectors:
            if len(vector) != self._dimension:
                raise EmbeddingError(
                    "embedding dimension does not match configuration",
                    details={"expected": self._dimension, "actual": len(vector)},
                )
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingError("embedding model returned a non-finite value")
        return vectors

    def _get_model(self) -> _SentenceModel:
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    loaded = SentenceTransformer(
                        str(self._model_path),
                        device=None if self._device == "auto" else self._device,
                        local_files_only=True,
                        trust_remote_code=self._trust_remote_code,
                    )
                    self._model = cast(_SentenceModel, loaded)
                    self._model.max_seq_length = self._max_length
                except Exception as exc:
                    raise LocalResourceMissingError(
                        "cannot load the local embedding model; no download was attempted",
                        details={"path": str(self._model_path)},
                    ) from exc
            return self._model

    @staticmethod
    def _to_vectors(raw: object) -> tuple[Vector, ...]:
        converted = raw.tolist() if hasattr(raw, "tolist") else raw
        if not isinstance(converted, Sequence) or isinstance(converted, (str, bytes)):
            raise EmbeddingError("embedding model returned an unsupported value")
        vectors: list[Vector] = []
        for row in converted:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise EmbeddingError("embedding model returned an invalid vector")
            try:
                vectors.append(tuple(float(value) for value in row))
            except (TypeError, ValueError) as exc:
                raise EmbeddingError("embedding vector contains a non-numeric value") from exc
        return tuple(vectors)

    @staticmethod
    def _checksum_directory(path: Path) -> str:
        digest = hashlib.sha256()
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            raise LocalResourceMissingError(
                "local embedding model directory is empty", details={"path": str(path)}
            )
        for file_path in files:
            relative = file_path.relative_to(path).as_posix().encode()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            try:
                with file_path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
            except OSError as exc:
                raise LocalResourceMissingError(
                    "cannot checksum local embedding model",
                    details={"path": str(file_path)},
                ) from exc
        return digest.hexdigest()
