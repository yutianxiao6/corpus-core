from __future__ import annotations

import asyncio
import threading
import time
import unittest
from collections.abc import Sequence

from corpuscore.concurrency import AsyncMicroBatcher
from corpuscore.exceptions import ConcurrentAccessError


class AsyncMicroBatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_nearby_requests_are_processed_as_one_ordered_batch(self) -> None:
        batches: list[tuple[str, ...]] = []

        def process(values: Sequence[str]) -> Sequence[str]:
            batches.append(tuple(values))
            return tuple(value.upper() for value in values)

        batcher = AsyncMicroBatcher(
            process,
            max_batch_size=4,
            max_wait_ms=20,
            workers=1,
            queue_capacity=8,
            enqueue_timeout_seconds=1,
            execution_timeout_seconds=1,
            name="test",
        )
        try:
            results = await asyncio.gather(
                batcher.submit("one"), batcher.submit("two"), batcher.submit("three")
            )
        finally:
            await batcher.aclose()

        self.assertEqual(results, ["ONE", "TWO", "THREE"])
        self.assertEqual(batches, [("one", "two", "three")])

    async def test_processor_failure_is_delivered_to_every_request(self) -> None:
        def fail(values: Sequence[str]) -> Sequence[str]:
            raise RuntimeError(f"failed {len(values)}")

        batcher = AsyncMicroBatcher(
            fail,
            max_batch_size=4,
            max_wait_ms=10,
            workers=1,
            queue_capacity=8,
            enqueue_timeout_seconds=1,
            execution_timeout_seconds=1,
            name="test",
        )
        try:
            results = await asyncio.gather(
                batcher.submit("one"), batcher.submit("two"), return_exceptions=True
            )
        finally:
            await batcher.aclose()

        self.assertTrue(all(isinstance(result, RuntimeError) for result in results))

    async def test_execution_timeout_has_stable_concurrency_error(self) -> None:
        def slow(values: Sequence[str]) -> Sequence[str]:
            time.sleep(0.03)
            return values

        batcher = AsyncMicroBatcher(
            slow,
            max_batch_size=1,
            max_wait_ms=0,
            workers=1,
            queue_capacity=1,
            enqueue_timeout_seconds=1,
            execution_timeout_seconds=0.005,
            name="embedding",
        )
        try:
            with self.assertRaisesRegex(ConcurrentAccessError, "timed out"):
                await batcher.submit("query")
        finally:
            await batcher.aclose()

    async def test_full_queue_applies_bounded_backpressure(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def blocked(values: Sequence[str]) -> Sequence[str]:
            started.set()
            release.wait(timeout=1)
            return values

        batcher = AsyncMicroBatcher(
            blocked,
            max_batch_size=1,
            max_wait_ms=0,
            workers=1,
            queue_capacity=1,
            enqueue_timeout_seconds=0.005,
            execution_timeout_seconds=1,
            name="reranker",
        )
        first = asyncio.create_task(batcher.submit("first"))
        await asyncio.to_thread(started.wait, 1)
        second = asyncio.create_task(batcher.submit("second"))
        await asyncio.sleep(0.001)
        try:
            with self.assertRaisesRegex(ConcurrentAccessError, "queue is full"):
                await batcher.submit("third")
        finally:
            release.set()
            await asyncio.gather(first, second)
            await batcher.aclose()


if __name__ == "__main__":
    unittest.main()
