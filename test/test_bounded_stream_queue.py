import asyncio

import pytest

from uni_api.streaming.bounded_queue import (
    ByteBoundedQueue,
    RetainedByteBudget,
    ReservedChunkBuffer,
    StreamQueueClosed,
    StreamQueueItemTooLarge,
    StreamQueuePutTimeout,
)


async def _consume_for_test(queue: ByteBoundedQueue):
    """Tests that do not model a network send consume an explicit lease."""

    lease = await queue.get_lease()
    item = lease.item
    await lease.release()
    return item


def test_queue_applies_item_backpressure_without_dropping():
    async def run():
        queue = ByteBoundedQueue(max_items=1, max_bytes=16)
        await queue.put(b"first")
        second_put = asyncio.create_task(queue.put(b"second"))
        await asyncio.sleep(0)

        assert not second_put.done()
        assert queue.snapshot().waiting_putters == 1
        assert await _consume_for_test(queue) == b"first"
        assert await second_put >= 0
        assert await _consume_for_test(queue) == b"second"
        assert queue.snapshot().blocked_puts == 1

    asyncio.run(run())


def test_queue_applies_byte_backpressure_and_tracks_peaks():
    async def run():
        queue = ByteBoundedQueue(max_items=4, max_bytes=6)
        await queue.put(b"abcd")
        blocked = asyncio.create_task(queue.put(b"efg"))
        await asyncio.sleep(0)

        assert not blocked.done()
        assert queue.snapshot().bytes == 4
        assert await _consume_for_test(queue) == b"abcd"
        await blocked
        snapshot = queue.snapshot()
        assert snapshot.bytes == 3
        assert snapshot.peak_bytes == 4
        assert snapshot.blocked_puts == 1
        assert snapshot.put_wait_ms >= 0

    asyncio.run(run())


def test_queue_rejects_single_item_larger_than_budget():
    async def run():
        queue = ByteBoundedQueue(max_items=2, max_bytes=3)
        with pytest.raises(StreamQueueItemTooLarge):
            await queue.put(b"four")

    asyncio.run(run())


def test_queue_put_timeout_does_not_mutate_queue():
    async def run():
        queue = ByteBoundedQueue(
            max_items=1,
            max_bytes=16,
            put_timeout_seconds=0.01,
        )
        await queue.put(b"first")
        with pytest.raises(StreamQueuePutTimeout):
            await queue.put(b"second")
        assert queue.snapshot().items == 1
        assert queue.snapshot().put_timeouts == 1
        assert queue.snapshot().waiting_putters == 0
        assert await _consume_for_test(queue) == b"first"

    asyncio.run(run())


def test_close_drains_existing_items_then_signals_end():
    async def run():
        queue = ByteBoundedQueue(max_items=2, max_bytes=16)
        await queue.put(b"one")
        await queue.close()

        assert await _consume_for_test(queue) == b"one"
        with pytest.raises(StreamQueueClosed):
            await _consume_for_test(queue)

    asyncio.run(run())


def test_close_with_error_propagates_after_drain():
    async def run():
        queue = ByteBoundedQueue(max_items=2, max_bytes=16)
        error = RuntimeError("upstream failed")
        await queue.put(b"one")
        await queue.close(error=error)

        assert await _consume_for_test(queue) == b"one"
        with pytest.raises(RuntimeError, match="upstream failed"):
            await _consume_for_test(queue)

    asyncio.run(run())


def test_discard_close_releases_queued_bytes_and_wakes_producer():
    async def run():
        queue = ByteBoundedQueue(max_items=1, max_bytes=16)
        await queue.put(b"first")
        blocked = asyncio.create_task(queue.put(b"second"))
        await asyncio.sleep(0)

        await queue.close(discard=True)

        with pytest.raises(StreamQueueClosed):
            await blocked
        assert queue.snapshot().items == 0
        assert queue.snapshot().bytes == 0
        assert queue.snapshot().waiting_putters == 0

    asyncio.run(run())


def test_inflight_item_remains_counted_until_downstream_send_releases_lease():
    async def run():
        budget = RetainedByteBudget(capacity_bytes=8, wait_timeout_seconds=1)
        queue = ByteBoundedQueue(
            max_items=1,
            max_bytes=8,
            retained_byte_budget=budget,
        )
        await queue.put(b"first")
        lease = await queue.get_lease()

        snapshot = queue.snapshot()
        assert snapshot.items == 1
        assert snapshot.bytes == 5
        assert snapshot.queued_items == 0
        assert snapshot.inflight_items == 1
        assert budget.snapshot().used_bytes == 5

        blocked = asyncio.create_task(queue.put(b"next"))
        await asyncio.sleep(0)
        assert not blocked.done()
        await lease.release()
        await blocked
        assert budget.snapshot().used_bytes == 4
        assert await _consume_for_test(queue) == b"next"
        assert budget.snapshot().used_bytes == 0

    asyncio.run(run())


def test_process_wide_byte_budget_backpressures_independent_queues():
    async def run():
        budget = RetainedByteBudget(capacity_bytes=6, wait_timeout_seconds=1)
        first = ByteBoundedQueue(
            max_items=2,
            max_bytes=6,
            retained_byte_budget=budget,
        )
        second = ByteBoundedQueue(
            max_items=2,
            max_bytes=6,
            retained_byte_budget=budget,
        )
        await first.put(b"four")
        blocked = asyncio.create_task(second.put(b"tri"))
        await asyncio.sleep(0)

        assert not blocked.done()
        assert budget.snapshot().waiting_reservations == 1
        assert await _consume_for_test(first) == b"four"
        await blocked
        assert budget.snapshot().used_bytes == 3
        assert await _consume_for_test(second) == b"tri"
        assert budget.snapshot().used_bytes == 0

    asyncio.run(run())


def test_precommit_reservation_transfers_to_queue_without_self_deadlock():
    async def run():
        budget = RetainedByteBudget(
            capacity_bytes=6,
            wait_timeout_seconds=0.01,
        )
        precommit = ReservedChunkBuffer(
            max_items=1,
            max_bytes=6,
            retained_byte_budget=budget,
        )
        queue = ByteBoundedQueue(
            max_items=1,
            max_bytes=6,
            retained_byte_budget=budget,
        )
        await precommit.append(b"123456")
        chunk, reservation = precommit.popleft()
        transferred = reservation.split(len(chunk))

        await queue.put(chunk, retained_byte_lease=transferred)
        await reservation.release()

        assert budget.snapshot().used_bytes == 6
        assert await _consume_for_test(queue) == b"123456"
        assert budget.snapshot().used_bytes == 0

    asyncio.run(run())


def test_queue_rejects_negative_explicit_item_size():
    async def run():
        queue = ByteBoundedQueue(max_items=1, max_bytes=8)
        with pytest.raises(ValueError, match="negative"):
            await queue.put(b"x", size=-1)

    asyncio.run(run())
