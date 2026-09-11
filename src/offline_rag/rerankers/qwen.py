"""Strictly local Qwen3 cross-encoder reranker."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Protocol, cast

from offline_rag.contracts.retrieval import RetrievalCandidate
from offline_rag.exceptions import OfflineResourceMissingError, RerankerError


class _CrossEncoder(Protocol):
    def predict(self, inputs: Sequence[tuple[str, str]], **kwargs: object) -> object: ...


class QwenCrossEncoderReranker:
    """Rerank candidates with a local Qwen3-Reranker model directory."""

    provider_name = "qwen_cross_encoder"

    def __init__(
        self,
        model_path: str | Path,
        *,
        revision: str | None = None,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 2048,
        maximum_candidates: int = 100,
        instruction: str = (
            "Given a user question, retrieve relevant passages that answer the question."
        ),
        score_mode: str = "sigmoid",
        model: _CrossEncoder | None = None,
    ) -> None:
        self._model_path = Path(model_path).expanduser().resolve()
        if not self._model_path.is_dir():
            raise OfflineResourceMissingError(
                "local reranker model directory is missing",
                details={"path": str(self._model_path)},
            )
        if batch_size <= 0 or max_length <= 0 or maximum_candidates <= 0:
            raise ValueError("batch_size, max_length and maximum_candidates must be positive")
        if score_mode not in ("sigmoid", "raw"):
            raise ValueError("score_mode must be sigmoid or raw")
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        self._revision = revision
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._maximum_candidates = maximum_candidates
        self._instruction = instruction.strip()
        self._score_mode = score_mode
        self._model = model
        self._lock = RLock()

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int | None = None,
    ) -> Sequence[RetrievalCandidate]:
        return self.rerank_batch(((query, candidates, top_n),))[0]

    def rerank_batch(
        self,
        requests: Sequence[tuple[str, Sequence[RetrievalCandidate], int | None]],
    ) -> Sequence[Sequence[RetrievalCandidate]]:
        if not requests:
            return ()
        selected_batches: list[tuple[RetrievalCandidate, ...]] = []
        pairs: list[tuple[str, str]] = []
        for query, candidates, top_n in requests:
            if not query.strip():
                raise RerankerError("reranker query must not be empty")
            if not candidates:
                selected_batches.append(())
                continue
            limit = len(candidates) if top_n is None else top_n
            if limit <= 0:
                raise ValueError("top_n must be positive")
            selected = tuple(candidates[: min(limit, self._maximum_candidates)])
            selected_batches.append(selected)
            pairs.extend((query, candidate.chunk.content) for candidate in selected)
        if not pairs:
            return tuple(() for _request in requests)
        try:
            raw = self._get_model().predict(
                tuple(pairs),
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scores = self._scores(raw)
        except RerankerError:
            raise
        except Exception as exc:
            raise RerankerError("local reranker inference failed") from exc
        if len(scores) != len(pairs):
            raise RerankerError(
                "reranker returned an unexpected score count",
                details={"expected": len(pairs), "actual": len(scores)},
            )
        results: list[Sequence[RetrievalCandidate]] = []
        offset = 0
        for selected in selected_batches:
            end = offset + len(selected)
            results.append(self._rank(selected, scores[offset:end]))
            offset = end
        return tuple(results)

    @staticmethod
    def _rank(
        selected: Sequence[RetrievalCandidate], scores: Sequence[float]
    ) -> tuple[RetrievalCandidate, ...]:
        reranked = [
            replace(
                candidate,
                rerank_score=score,
                final_score=score,
                rank=None,
                origins=list(candidate.origins),
            )
            for candidate, score in zip(selected, scores, strict=True)
        ]
        reranked.sort(
            key=lambda candidate: (
                -(candidate.rerank_score if candidate.rerank_score is not None else -math.inf),
                candidate.chunk.chunk_id,
            )
        )
        return tuple(
            replace(candidate, rank=rank, origins=list(candidate.origins))
            for rank, candidate in enumerate(reranked, start=1)
        )

    async def arerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int | None = None,
    ) -> Sequence[RetrievalCandidate]:
        return await asyncio.to_thread(self.rerank, query, candidates, top_n=top_n)

    def _get_model(self) -> _CrossEncoder:
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder

                    loaded = CrossEncoder(
                        str(self._model_path),
                        device=None if self._device == "auto" else self._device,
                        prompts={"retrieval": self._instruction},
                        default_prompt_name="retrieval",
                        revision=self._revision,
                        local_files_only=True,
                        trust_remote_code=False,
                        max_length=self._max_length,
                    )
                    self._model = cast(_CrossEncoder, loaded)
                except Exception as exc:
                    raise OfflineResourceMissingError(
                        "cannot load the local reranker model; no download was attempted",
                        details={"path": str(self._model_path)},
                    ) from exc
            return self._model

    def _scores(self, raw: object) -> tuple[float, ...]:
        converted = raw.tolist() if hasattr(raw, "tolist") else raw
        if not isinstance(converted, Sequence) or isinstance(converted, (str, bytes)):
            raise RerankerError("reranker returned an unsupported score value")
        scores: list[float] = []
        for value in converted:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) != 1:
                    raise RerankerError("reranker returned a non-scalar score")
                value = value[0]
            try:
                score = float(value)
            except (TypeError, ValueError) as exc:
                raise RerankerError("reranker returned a non-numeric score") from exc
            if not math.isfinite(score):
                raise RerankerError("reranker returned a non-finite score")
            scores.append(self._sigmoid(score) if self._score_mode == "sigmoid" else score)
        return tuple(scores)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            exponent = math.exp(-value)
            return 1.0 / (1.0 + exponent)
        exponent = math.exp(value)
        return exponent / (1.0 + exponent)
