import asyncio

import pytest

from uni_api.admission import (
    AdmissionRejected,
    BoundedAdmissionGate,
    RequestAdmissionController,
    RequestBodyBudgetExhausted,
    RequestBodyTooLarge,
    UpstreamResponseBudgetExhausted,
)


async def _wait_until(predicate, *, timeout=1.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


def test_gate_immediate_acquire_and_idempotent_release():
    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=1,
            wait_timeout_seconds=1,
        )

        lease = await gate.acquire()
        assert lease.wait_ms >= 0
        assert gate.snapshot() == {
            "active": 1,
            "waiters": 0,
            "capacity": 1,
            "waiter_limit": 1,
            "acquired_total": 1,
            "cancelled_total": 0,
            "rejected": {},
        }

        await asyncio.gather(lease.release(), lease.release(), lease.release())
        assert lease.released is True
        assert gate.snapshot()["active"] == 0

    asyncio.run(run())


def test_gate_grants_waiters_in_fifo_order():
    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=2,
            wait_timeout_seconds=1,
        )
        holder = await gate.acquire()
        acquired_order = []
        release_first = asyncio.Event()

        async def wait_for_lease(name, release_event=None):
            lease = await gate.acquire()
            acquired_order.append(name)
            if release_event is not None:
                await release_event.wait()
            await lease.release()

        first = asyncio.create_task(wait_for_lease("first", release_first))
        await _wait_until(lambda: gate.snapshot()["waiters"] == 1)
        second = asyncio.create_task(wait_for_lease("second"))
        await _wait_until(lambda: gate.snapshot()["waiters"] == 2)

        await holder.release()
        await _wait_until(lambda: acquired_order == ["first"])
        assert gate.snapshot()["waiters"] == 1

        release_first.set()
        await asyncio.gather(first, second)
        assert acquired_order == ["first", "second"]
        assert gate.snapshot()["active"] == 0

    asyncio.run(run())


def test_gate_cancellation_before_waiter_task_first_step_returns_queue_position(
    monkeypatch,
):
    """The outer begin_acquire coroutine owns a not-yet-started child task."""

    class CancelBeforeChildStarts:
        def set(self):
            return None

        async def wait(self):
            raise asyncio.CancelledError

    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=1,
            wait_timeout_seconds=1,
        )
        holder = await gate.acquire()

        with monkeypatch.context() as patch:
            # core.py and this test refer to the same asyncio module object.
            # Raising before Event.wait suspends guarantees the child task has
            # not taken its first coroutine step.
            patch.setattr(asyncio, "Event", CancelBeforeChildStarts)
            with pytest.raises(asyncio.CancelledError):
                await gate.begin_acquire()

        assert gate.snapshot()["waiters"] == 0
        assert gate.snapshot()["cancelled_total"] == 1
        await holder.release()
        assert gate.snapshot()["active"] == 0
        assert gate.snapshot()["acquired_total"] == 1

    asyncio.run(run())


def test_gate_acquire_cancellation_same_turn_as_grant_releases_result(monkeypatch):
    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=1,
            wait_timeout_seconds=1,
        )
        holder = await gate.acquire()
        original_begin_acquire = gate.begin_acquire

        async def begin_and_cancel_owner_on_grant(*, timeout_seconds=None):
            acquisition = await original_begin_acquire(
                timeout_seconds=timeout_seconds
            )
            owner = asyncio.current_task()
            assert owner is not None
            acquisition.add_done_callback(lambda _done: owner.cancel())
            return acquisition

        monkeypatch.setattr(gate, "begin_acquire", begin_and_cancel_owner_on_grant)
        owner = asyncio.create_task(gate.acquire())
        await _wait_until(lambda: gate.snapshot()["waiters"] == 1)
        await holder.release()

        with pytest.raises(asyncio.CancelledError):
            await owner
        assert gate.snapshot()["active"] == 0
        assert gate.snapshot()["waiters"] == 0

    asyncio.run(run())


def test_request_acquire_cancellation_same_turn_as_grant_releases_result(
    monkeypatch,
):
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=1,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
        )
        holder = await controller.acquire()
        original_begin_acquire = controller.begin_acquire

        async def begin_and_cancel_owner_on_grant(*, timeout_seconds=None):
            acquisition = await original_begin_acquire(
                timeout_seconds=timeout_seconds
            )
            owner = asyncio.current_task()
            assert owner is not None
            acquisition.add_done_callback(lambda _done: owner.cancel())
            return acquisition

        monkeypatch.setattr(
            controller,
            "begin_acquire",
            begin_and_cancel_owner_on_grant,
        )
        owner = asyncio.create_task(controller.acquire())
        await _wait_until(lambda: controller.snapshot()["waiters"] == 1)
        await holder.release()

        with pytest.raises(asyncio.CancelledError):
            await owner
        assert controller.snapshot()["active"] == 0
        assert controller.snapshot()["waiters"] == 0

    asyncio.run(run())


def test_gate_rejects_when_waiter_queue_is_full():
    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=1,
            wait_timeout_seconds=1,
        )
        holder = await gate.acquire()
        queued = asyncio.create_task(gate.acquire())
        await _wait_until(lambda: gate.snapshot()["waiters"] == 1)

        with pytest.raises(AdmissionRejected) as exc_info:
            await gate.acquire()
        assert exc_info.value.reason == "queue_full"
        assert exc_info.value.status_code == 503
        assert gate.snapshot()["rejected"] == {"queue_full": 1}

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        await holder.release()
        assert gate.snapshot()["active"] == 0

    asyncio.run(run())


def test_gate_times_out_without_leaking_a_waiter_or_slot():
    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=1,
            wait_timeout_seconds=0.02,
        )
        holder = await gate.acquire()

        with pytest.raises(AdmissionRejected) as exc_info:
            await gate.acquire()
        assert exc_info.value.reason == "wait_timeout"
        assert gate.snapshot()["active"] == 1
        assert gate.snapshot()["waiters"] == 0
        assert gate.snapshot()["rejected"] == {"wait_timeout": 1}

        await holder.release()
        followup = await gate.acquire()
        await followup.release()
        assert gate.snapshot()["active"] == 0

    asyncio.run(run())


def test_gate_cancellation_while_queued_does_not_leak_capacity():
    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=1,
            wait_timeout_seconds=1,
        )
        holder = await gate.acquire()
        queued = asyncio.create_task(gate.acquire())
        await _wait_until(lambda: gate.snapshot()["waiters"] == 1)

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert gate.snapshot()["waiters"] == 0
        assert gate.snapshot()["cancelled_total"] == 1

        await holder.release()
        followup = await gate.acquire()
        await followup.release()
        assert gate.snapshot()["active"] == 0

    asyncio.run(run())


def test_gate_grant_cancellation_race_transfers_capacity_or_returns_a_lease():
    async def run():
        # Repeat the handoff race. If cancellation arrives before acquire()
        # returns, the gate must reclaim the grant. If it arrives afterwards,
        # the returned lease remains explicit ownership and is released here.
        for _ in range(50):
            gate = BoundedAdmissionGate(
                1,
                waiter_limit=1,
                wait_timeout_seconds=1,
            )
            holder = await gate.acquire()
            queued = asyncio.create_task(gate.acquire())
            await _wait_until(lambda: gate.snapshot()["waiters"] == 1)

            release = asyncio.create_task(holder.release())
            queued.cancel()
            await release
            try:
                delivered_lease = await queued
            except asyncio.CancelledError:
                delivered_lease = None
            if delivered_lease is not None:
                await delivered_lease.release()

            await _wait_until(lambda: gate.snapshot()["active"] == 0)
            followup = await gate.acquire()
            await followup.release()

    asyncio.run(run())


def test_production_admission_envelope_is_exactly_64_active_plus_936_waiters():
    async def run():
        gate = BoundedAdmissionGate(
            64,
            waiter_limit=936,
            wait_timeout_seconds=2,
        )
        holders = await asyncio.gather(*(gate.acquire() for _ in range(64)))
        waiters = [asyncio.create_task(gate.acquire()) for _ in range(936)]
        await _wait_until(lambda: gate.snapshot()["waiters"] == 936)

        snapshot = gate.snapshot()
        assert snapshot["active"] == 64
        assert snapshot["waiters"] == 936
        with pytest.raises(AdmissionRejected) as rejected:
            await gate.acquire()
        assert rejected.value.reason == "queue_full"

        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        await asyncio.gather(*(holder.release() for holder in holders))
        snapshot = gate.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["waiters"] == 0
        assert snapshot["cancelled_total"] == 936

    asyncio.run(run())


def test_request_controller_enforces_per_request_body_limit():
    async def run():
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=1,
            wait_timeout_seconds=1,
            max_body_bytes=10,
            body_budget_bytes=20,
        )
        lease = await controller.acquire(initial_body_bytes=4)
        assert await lease.reserve_body_bytes(2) == 6

        with pytest.raises(RequestBodyTooLarge) as exc_info:
            await lease.reserve_body_bytes(5)
        assert exc_info.value.status_code == 413
        assert exc_info.value.reason == "body_too_large"
        assert lease.reserved_body_bytes == 6
        assert controller.snapshot()["reserved_body_bytes"] == 6

        await lease.release()
        assert lease.reserved_body_bytes == 0
        assert controller.snapshot()["reserved_body_bytes"] == 0
        assert controller.snapshot()["rejected"] == {"body_too_large": 1}

    asyncio.run(run())


def test_request_controller_enforces_global_body_budget_and_recovers_on_release():
    async def run():
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=1,
            wait_timeout_seconds=1,
            max_body_bytes=10,
            body_budget_bytes=10,
        )
        first = await controller.acquire(initial_body_bytes=6)
        second = await controller.acquire(initial_body_bytes=4)

        with pytest.raises(RequestBodyBudgetExhausted) as exc_info:
            await second.reserve_body_bytes(1)
        assert exc_info.value.status_code == 503
        assert exc_info.value.reason == "body_budget_exhausted"
        assert controller.snapshot()["reserved_body_bytes"] == 10

        await first.release()
        assert await second.reserve_body_bytes(1) == 5
        assert controller.snapshot()["reserved_body_bytes"] == 5

        await asyncio.gather(second.release(), second.release())
        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["body_budget"] == 10
        assert snapshot["rejected"] == {"body_budget_exhausted": 1}

    asyncio.run(run())


def test_body_and_buffered_response_share_one_weighted_memory_budget():
    async def run():
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=20,
            body_budget_bytes=20,
            max_response_bytes=20,
        )
        body_owner = await controller.acquire(initial_body_bytes=8)
        response_owner = await controller.acquire()
        assert await response_owner.reserve_response_bytes(12) == 12
        assert controller.snapshot()["reserved_retained_bytes"] == 20

        with pytest.raises(UpstreamResponseBudgetExhausted):
            await response_owner.reserve_response_bytes(1)
        assert controller.snapshot()["reserved_response_bytes"] == 12

        await body_owner.release()
        assert await response_owner.reserve_response_bytes(1) == 13
        await response_owner.release()

        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["reserved_response_bytes"] == 0
        assert snapshot["reserved_retained_bytes"] == 0

    asyncio.run(run())


def test_request_controller_failed_initial_reservations_release_active_slot():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=5,
            body_budget_bytes=4,
        )

        with pytest.raises(RequestBodyTooLarge):
            await controller.acquire(initial_body_bytes=6)
        assert controller.snapshot()["active"] == 0

        with pytest.raises(RequestBodyBudgetExhausted):
            await controller.acquire(initial_body_bytes=5)
        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["rejected"] == {
            "body_too_large": 1,
            "body_budget_exhausted": 1,
        }

    asyncio.run(run())


def test_request_lease_context_manager_releases_bytes_on_cancellation():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=1,
            wait_timeout_seconds=1,
            max_body_bytes=10,
            body_budget_bytes=10,
        )
        entered = asyncio.Event()

        async def request_task():
            async with await controller.acquire(initial_body_bytes=7):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(request_task())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_body_bytes"] == 0

    asyncio.run(run())


def test_temporary_response_reservation_releases_before_request_end():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
            max_response_bytes=1024,
        )
        request_lease = await controller.acquire()
        temporary = await request_lease.reserve_temporary_response_bytes(400)
        assert controller.snapshot()["reserved_response_bytes"] == 400

        await temporary.release()
        assert controller.snapshot()["reserved_response_bytes"] == 0
        assert request_lease.reserved_response_bytes == 0

        await request_lease.release()
        assert controller.snapshot()["active"] == 0

    asyncio.run(run())


def test_request_releases_active_but_defers_memory_for_live_child_owner():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
            max_response_bytes=1024,
        )
        request_lease = await controller.acquire(initial_body_bytes=200)
        temporary = await request_lease.reserve_temporary_response_bytes(300)

        await request_lease.release()
        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_retained_bytes"] == 500
        assert snapshot["deferred_memory_requests"] == 1
        assert snapshot["deferred_memory_bytes"] == 500

        await temporary.release()
        snapshot = controller.snapshot()
        assert snapshot["reserved_retained_bytes"] == 0
        assert snapshot["deferred_memory_requests"] == 0
        assert snapshot["deferred_memory_bytes"] == 0

    asyncio.run(run())


def test_explicit_cleanup_deferral_keeps_base_request_memory_accounted():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
        )
        request_lease = await controller.acquire(initial_body_bytes=700)
        deferral = await request_lease.defer_memory_release()

        await request_lease.release()
        assert controller.snapshot()["deferred_memory_bytes"] == 700

        await deferral.release()
        assert controller.snapshot()["reserved_retained_bytes"] == 0

    asyncio.run(run())


def test_admission_configuration_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        BoundedAdmissionGate(0, waiter_limit=1, wait_timeout_seconds=1)
    with pytest.raises(ValueError):
        BoundedAdmissionGate(1, waiter_limit=-1, wait_timeout_seconds=1)
    with pytest.raises(ValueError):
        BoundedAdmissionGate(1, waiter_limit=1, wait_timeout_seconds=0)
    with pytest.raises(ValueError):
        RequestAdmissionController(
            capacity=1,
            waiter_limit=1,
            wait_timeout_seconds=1,
            max_body_bytes=-1,
            body_budget_bytes=1,
        )
