"""Bounded asynchronous micro-batching for local model inference."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from corpuscore.contracts.indexing import EmbeddingSpecification
from corpuscore.contracts.retrieval import RetrievalCandidate
from corpuscore.exceptions import ConcurrentAccessError
from corpuscore.ports import EmbeddingProvider, Reranker, Vector

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(slots=True)
class _Pending(Generic[InputT, OutputT]):
    value: InputT
    future: asyncio.Future[OutputT]


class AsyncMicroBatcher(Generic[InputT, OutputT]):
    """Combine nearby submissions into bounded batches processed by worker threads."""

    def __init__(
        self,
        processor: Callable[[Sequence[InputT]], Sequence[OutputT]],
        *,
        max_batch_size: int,
        max_wait_ms: int,
        workers: int,
        queue_capacity: int,
        enqueue_timeout_seconds: float,
        execution_timeout_seconds: float,
        name: str,
    ) -> None:
        if min(max_batch_size, workers, queue_capacity) <= 0:
            raise ValueError("batch size, workers and queue capacity must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        if enqueue_timeout_seconds <= 0 or execution_timeout_seconds <= 0:
            raise ValueError("queue timeouts must be positive")
        self._processor = processor
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_ms / 1000
        self._worker_count = workers
        self._queue_capacity = queue_capacity
        self._enqueue_timeout = enqueue_timeout_seconds
        self._execution_timeout = execution_timeout_seconds
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[_Pending[InputT, OutputT]] | None = None
        self._workers: list[asyncio.Task[None]] = []
        self._closed = False

    async def submit(self, value: InputT) -> OutputT:
        queue = self._ensure_started()
        assert self._loop is not None
        future: asyncio.Future[OutputT] = self._loop.create_future()
        pending = _Pending(value, future)
        try:
            await asyncio.wait_for(queue.put(pending), timeout=self._enqueue_timeout)
        except TimeoutError as exc:
            raise ConcurrentAccessError(
                f"{self._name} inference queue is full",
                details={"capacity": self._queue_capacity},
            ) from exc
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=self._execution_timeout)
        except asyncio.CancelledError:
            future.cancel()
            raise
        except TimeoutError as exc:
            future.cancel()
            raise ConcurrentAccessError(
                f"{self._name} inference timed out",
                details={"timeout_seconds": self._execution_timeout},
            ) from exc

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._queue is None:
            return
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            if self._loop is not None and self._loop.is_closed():
                self._workers.clear()
                return
            raise ConcurrentAccessError(
                f"{self._name} inference queue must be closed on its owning event loop"
            )
        await self._queue.join()
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    def _ensure_started(self) -> asyncio.Queue[_Pending[InputT, OutputT]]:
        if self._closed:
            raise ConcurrentAccessError(f"{self._name} inference queue is closed")
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise ConcurrentAccessError(
                f"{self._name} inference queue cannot be shared across event loops"
            )
        if self._queue is None:
            self._loop = loop
            self._queue = asyncio.Queue(maxsize=self._queue_capacity)
            self._workers = [
                loop.create_task(self._worker(), name=f"corpuscore-{self._name}-{index}")
                for index in range(self._worker_count)
            ]
        return self._queue

    async def _worker(self) -> None:
        assert self._queue is not None
        while True:
            first = await self._queue.get()
            batch = [first]
            try:
                await self._fill_batch(batch)
                outputs = await asyncio.to_thread(
                    self._processor, tuple(item.value for item in batch)
                )
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"{self._name} processor returned {len(outputs)} results "
                        f"for {len(batch)} requests"
                    )
                for item, output in zip(batch, outputs, strict=True):
                    if not item.future.done():
                        item.future.set_result(output)
            except asyncio.CancelledError:
                for item in batch:
                    if not item.future.done():
                        item.future.cancel()
                raise
            except Exception as exc:  # noqa: BLE001 - fan out arbitrary processor failures
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
            finally:
                for _item in batch:
                    self._queue.task_done()

    async def _fill_batch(self, batch: list[_Pending[InputT, OutputT]]) -> None:
        assert self._queue is not None
        if self._max_wait_seconds == 0:
            while len(batch) < self._max_batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_wait_seconds
        while len(batch) < self._max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break


class QueuedEmbeddingProvider:
    """Use one bounded micro-batcher for asynchronous query embeddings."""

    def __init__(
        self,
        embedding: EmbeddingProvider,
        batcher: AsyncMicroBatcher[str, Vector],
    ) -> None:
        self._embedding = embedding
        self._batcher = batcher

    @property
    def specification(self) -> EmbeddingSpecification:
        return self._embedding.specification

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return self._embedding.embed_documents(texts)

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Vector]:
        return self._embedding.embed_queries(texts)

    def embed_query(self, text: str) -> Vector:
        return self._embedding.embed_query(text)

    async def aembed_query(self, text: str) -> Vector:
        if not text.strip():
            return await self._embedding.aembed_query(text)
        return await self._batcher.submit(text)

    async def aclose(self) -> None:
        await self._batcher.aclose()


RerankBatchInput = tuple[str, Sequence[RetrievalCandidate], int | None]


class QueuedReranker:
    """Use one bounded micro-batcher for asynchronous reranking requests."""

    def __init__(
        self,
        reranker: Reranker,
        batcher: AsyncMicroBatcher[RerankBatchInput, Sequence[RetrievalCandidate]],
    ) -> None:
        self._reranker = reranker
        self._batcher = batcher

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int | None = None,
    ) -> Sequence[RetrievalCandidate]:
        return self._reranker.rerank(query, candidates, top_n=top_n)

    def rerank_batch(
        self,
        requests: Sequence[RerankBatchInput],
    ) -> Sequence[Sequence[RetrievalCandidate]]:
        return self._reranker.rerank_batch(requests)

    async def arerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int | None = None,
    ) -> Sequence[RetrievalCandidate]:
        return await self._batcher.submit((query, candidates, top_n))

    async def aclose(self) -> None:
        await self._batcher.aclose()
