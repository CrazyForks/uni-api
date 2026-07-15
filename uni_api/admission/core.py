from __future__ import annotations

import asyncio
import contextlib
import contextvars
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any

from uni_api.admission.memory import AdaptiveMemoryGovernor


class AdmissionRejected(RuntimeError):
    """A bounded admission decision that callers can translate to HTTP."""

    def __init__(self, reason: str, *, status_code: int = 503) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


class RequestBodyTooLarge(AdmissionRejected):
    def __init__(self) -> None:
        super().__init__("body_too_large", status_code=413)


class RequestBodyBudgetExhausted(AdmissionRejected):
    def __init__(self) -> None:
        super().__init__("body_budget_exhausted", status_code=503)


class LargeBodyCapacityExhausted(AdmissionRejected):
    def __init__(self) -> None:
        super().__init__("large_body_capacity_exhausted", status_code=503)


class UpstreamResponseBudgetExhausted(AdmissionRejected):
    local_admission_rejection = True

    def __init__(self) -> None:
        super().__init__("upstream_response_budget_exhausted", status_code=503)


@dataclass(slots=True)
class _Waiter:
    future: asyncio.Future[float]
    state: str = "queued"


async def _finish_cleanup_despite_cancellation(task: asyncio.Task[None]) -> None:
    """Wait for a tiny ownership cleanup even if cancellation is repeated."""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    task.result()


async def _cancel_and_release_acquisition(
    acquisition: asyncio.Future[Any],
) -> None:
    """Consume an acquisition whose awaiting owner was cancelled.

    A Future may become successful in the same loop turn that its waiter is
    cancelled.  Merely propagating cancellation can therefore strand the
    lease stored in the Future's result.  This helper owns that result until it
    is either cancelled/rejected or explicitly released.
    """

    if not acquisition.done():
        acquisition.cancel()
    try:
        abandoned_lease = await acquisition
    except (asyncio.CancelledError, AdmissionRejected):
        return
    await abandoned_lease.release()


async def _await_owned_acquisition(
    acquisition: asyncio.Future[Any],
) -> Any:
    """Await a lease Future without losing a same-turn successful result."""

    try:
        return await asyncio.shield(acquisition)
    except asyncio.CancelledError:
        cleanup_task = asyncio.create_task(
            _cancel_and_release_acquisition(acquisition)
        )
        await _finish_cleanup_despite_cancellation(cleanup_task)
        raise


class AdmissionLease:
    """One active-capacity slot.

    Releasing a lease is idempotent. Cleanup runs in its own shielded task so
    cancellation of the request task cannot strand the slot.
    """

    def __init__(self, gate: BoundedAdmissionGate, *, wait_ms: float) -> None:
        self._gate = gate
        self.wait_ms = max(0.0, wait_ms)
        self._released = False
        self._release_task: asyncio.Task[None] | None = None

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> None:
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._release_once())
        release_task = self._release_task
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError:
            # Do not return control until ownership has actually been released,
            # even if shutdown or disconnect sends cancellation more than once.
            await _finish_cleanup_despite_cancellation(release_task)
            raise

    async def _release_once(self) -> None:
        if self._released:
            return
        await self._gate._release_slot()
        self._released = True

    async def __aenter__(self) -> AdmissionLease:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()


class BoundedAdmissionGate:
    """A FIFO active-capacity gate with a hard bound on queued tasks."""

    def __init__(
        self,
        capacity: int,
        *,
        waiter_limit: int,
        wait_timeout_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        if waiter_limit < 0:
            raise ValueError("waiter_limit cannot be negative")
        if wait_timeout_seconds <= 0:
            raise ValueError("wait_timeout_seconds must be greater than zero")
        self.capacity = capacity
        self.waiter_limit = waiter_limit
        self.wait_timeout_seconds = wait_timeout_seconds
        self._clock = clock
        self._lock = asyncio.Lock()
        self._active = 0
        self._waiters: deque[_Waiter] = deque()
        self._acquired_total = 0
        self._cancelled_total = 0
        self._rejected: Counter[str] = Counter()

    async def begin_acquire(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> asyncio.Future[AdmissionLease]:
        """Atomically claim an active slot or bounded waiter position.

        The returned future is already running (or already resolved), so a
        caller may safely start ancillary work only after this method returns.
        Queue-full rejection happens before any such work can begin.
        """

        timeout = self.wait_timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        started_at = self._clock()
        loop = asyncio.get_running_loop()
        waiter: _Waiter | None = None

        async with self._lock:
            # A new caller must not jump ahead of already queued callers.
            if self._active < self.capacity and not self._waiters:
                self._active += 1
                self._acquired_total += 1
                resolved: asyncio.Future[AdmissionLease] = loop.create_future()
                resolved.set_result(
                    AdmissionLease(
                        self,
                        wait_ms=(self._clock() - started_at) * 1000.0,
                    )
                )
                return resolved

            if len(self._waiters) >= self.waiter_limit:
                self._rejected["queue_full"] += 1
                raise AdmissionRejected("queue_full")

            waiter = _Waiter(loop.create_future())
            self._waiters.append(waiter)

        started = asyncio.Event()

        async def wait_for_waiter() -> AdmissionLease:
            started.set()
            return await self._wait_for_waiter(
                waiter,
                started_at=started_at,
                timeout=timeout,
            )

        task = asyncio.create_task(wait_for_waiter())
        try:
            await started.wait()
        except BaseException:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, AdmissionRejected):
                await task
            # A task cancelled before its coroutine's first step never enters
            # _wait_for_waiter(), so it cannot run that coroutine's cleanup.
            # Explicitly abandon the waiter here as the outer coroutine still
            # owns the queue position.  _abandon_waiter is idempotent for the
            # case where the child did start and already cleaned itself up.
            cleanup_task = asyncio.create_task(
                self._abandon_waiter(waiter, cancelled=True)
            )
            await _finish_cleanup_despite_cancellation(cleanup_task)
            raise
        return task

    async def _wait_for_waiter(
        self,
        waiter: _Waiter,
        *,
        started_at: float,
        timeout: float,
    ) -> AdmissionLease:
        try:
            granted_at = await asyncio.wait_for(
                asyncio.shield(waiter.future),
                timeout=timeout,
            )
        except TimeoutError:
            cleanup_task = asyncio.create_task(
                self._abandon_waiter(
                    waiter,
                    rejection_reason="wait_timeout",
                )
            )
            await _finish_cleanup_despite_cancellation(cleanup_task)
            raise AdmissionRejected("wait_timeout") from None
        except asyncio.CancelledError:
            cleanup_task = asyncio.create_task(
                self._abandon_waiter(waiter, cancelled=True)
            )
            await _finish_cleanup_despite_cancellation(cleanup_task)
            raise

        return AdmissionLease(
            self,
            wait_ms=(granted_at - started_at) * 1000.0,
        )

    async def acquire(self, *, timeout_seconds: float | None = None) -> AdmissionLease:
        acquisition = await self.begin_acquire(timeout_seconds=timeout_seconds)
        return await _await_owned_acquisition(acquisition)

    async def try_acquire(self) -> AdmissionLease | None:
        """Acquire immediately without joining the waiter queue."""

        started_at = self._clock()
        async with self._lock:
            if self._active >= self.capacity or self._waiters:
                return None
            self._active += 1
            self._acquired_total += 1
        return AdmissionLease(
            self,
            wait_ms=(self._clock() - started_at) * 1000.0,
        )

    async def _abandon_waiter(
        self,
        waiter: _Waiter,
        *,
        rejection_reason: str | None = None,
        cancelled: bool = False,
    ) -> None:
        async with self._lock:
            if waiter.state == "queued":
                waiter.state = "abandoned"
                self._waiters.remove(waiter)
                waiter.future.cancel()
            elif waiter.state == "granted":
                # Capacity was transferred to this waiter, but cancellation or
                # timeout won the race before a lease reached the caller.
                waiter.state = "abandoned"
                self._active -= 1
            else:
                return

            if rejection_reason is not None:
                self._rejected[rejection_reason] += 1
            if cancelled:
                self._cancelled_total += 1

            self._promote_waiters_locked()

    async def _release_slot(self) -> None:
        async with self._lock:
            if self._active <= 0:
                raise RuntimeError("admission gate active count underflow")
            self._active -= 1
            self._promote_waiters_locked()

    def _promote_waiters_locked(self) -> None:
        while self._active < self.capacity and self._waiters:
            waiter = self._waiters.popleft()
            if waiter.state != "queued" or waiter.future.done():
                continue
            waiter.state = "granted"
            self._active += 1
            self._acquired_total += 1
            waiter.future.set_result(self._clock())

    def snapshot(self) -> dict[str, Any]:
        """Return an event-loop-consistent metrics snapshot without awaiting."""

        return {
            "active": self._active,
            "waiters": sum(waiter.state == "queued" for waiter in self._waiters),
            "capacity": self.capacity,
            "waiter_limit": self.waiter_limit,
            "wait_timeout_seconds": self.wait_timeout_seconds,
            "acquired_total": self._acquired_total,
            "cancelled_total": self._cancelled_total,
            "rejected": dict(self._rejected),
        }


class PendingBodyReservation:
    """Bytes consumed while a request is still waiting for an active slot."""

    def __init__(self, controller: RequestAdmissionController) -> None:
        self._controller = controller
        self._reserved_bytes = 0
        self._transferred = False
        self._released = False
        self._large_body_slot = False

    @property
    def reserved_bytes(self) -> int:
        return self._reserved_bytes

    async def reserve(self, additional_bytes: int) -> int:
        if additional_bytes < 0:
            raise ValueError("additional_bytes cannot be negative")
        if self._transferred or self._released:
            raise RuntimeError("pending body reservation is no longer active")
        await self._controller._reserve_pending_body_additional(
            self,
            additional_bytes,
        )
        return self._reserved_bytes

    async def transfer_to(self, lease: RequestAdmissionLease) -> int:
        if self._released:
            raise RuntimeError("pending body reservation is released")
        if self._transferred:
            return 0
        transferred = await self._controller._transfer_pending_body(
            self,
            lease,
        )
        self._transferred = True
        return transferred

    async def release(self) -> None:
        if self._transferred or self._released:
            return
        cleanup_task = asyncio.create_task(
            self._controller._release_pending_body(self)
        )
        await _finish_cleanup_despite_cancellation(cleanup_task)
        self._released = True


class RequestAdmissionLease:
    """An active request slot plus its current body-byte reservation."""

    def __init__(
        self,
        controller: RequestAdmissionController,
        active_lease: AdmissionLease,
        *,
        reserved_body_bytes: int,
    ) -> None:
        self._controller = controller
        self._active_lease = active_lease
        self._reserved_body_bytes = reserved_body_bytes
        self._reserved_response_bytes = 0
        self._release_requested = False
        self._released = False
        self._memory_owner_count = 0
        self._memory_finalized = False
        self._large_body_slot = False
        self._release_task: asyncio.Task[None] | None = None

    @property
    def wait_ms(self) -> float:
        return self._active_lease.wait_ms

    @property
    def reserved_body_bytes(self) -> int:
        return self._reserved_body_bytes

    @property
    def reserved_response_bytes(self) -> int:
        return self._reserved_response_bytes

    @property
    def released(self) -> bool:
        return self._released

    async def reserve_body_bytes(self, additional_bytes: int) -> int:
        """Atomically grow this request's reservation by an observed chunk."""

        if additional_bytes < 0:
            raise ValueError("additional_bytes cannot be negative")
        if self._release_requested:
            raise RuntimeError("cannot reserve bytes on a released request lease")
        await self._controller._reserve_additional(self, additional_bytes)
        return self._reserved_body_bytes

    async def reserve_response_bytes(self, additional_bytes: int) -> int:
        """Reserve weighted retained bytes for a buffered upstream response."""

        if additional_bytes < 0:
            raise ValueError("additional_bytes cannot be negative")
        if self._release_requested:
            raise RuntimeError("cannot reserve bytes on a released request lease")
        await self._controller._reserve_response_additional(
            self,
            additional_bytes,
        )
        return self._reserved_response_bytes

    async def reserve_temporary_response_bytes(
        self,
        additional_bytes: int,
    ) -> TemporaryResponseBytesReservation:
        """Reserve live parse/materialization memory with explicit lifetime."""

        if additional_bytes < 0:
            raise ValueError("additional_bytes cannot be negative")
        if self._release_requested:
            raise RuntimeError("cannot reserve bytes on a released request lease")
        reservation = TemporaryResponseBytesReservation(self, 0)
        activate_task = asyncio.create_task(
            self._controller._activate_temporary_response(
                reservation,
                additional_bytes,
            )
        )
        try:
            await asyncio.shield(activate_task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(activate_task)
            if reservation._active:
                cleanup_task = asyncio.create_task(reservation.release())
                await _finish_cleanup_despite_cancellation(cleanup_task)
            raise
        return reservation

    async def defer_memory_release(self) -> MemoryReleaseDeferral:
        """Keep request memory accounted after its active slot is released."""

        deferral = MemoryReleaseDeferral(self)
        activate_task = asyncio.create_task(
            self._controller._activate_memory_deferral(deferral)
        )
        try:
            await asyncio.shield(activate_task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(activate_task)
            if deferral._active:
                cleanup_task = asyncio.create_task(deferral.release())
                await _finish_cleanup_despite_cancellation(cleanup_task)
            raise
        return deferral

    async def release(self) -> None:
        self._release_requested = True
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._release_once())
        release_task = self._release_task
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(release_task)
            raise

    async def _release_once(self) -> None:
        if self._released:
            return
        await self._controller._release_request(self)
        self._released = True

    async def __aenter__(self) -> RequestAdmissionLease:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()


class TemporaryResponseBytesReservation:
    """A response-memory charge released as soon as its object graph dies."""

    def __init__(self, lease: RequestAdmissionLease, size: int) -> None:
        self._lease = lease
        self.size = int(size)
        self._active = False
        self._released = False
        self._committed = False
        self._closing = False
        self._state_lock = asyncio.Lock()
        self._release_task: asyncio.Task[None] | None = None

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> None:
        if self._committed or self._released:
            return
        if self._release_task is None:
            # Establish exact-once finalization before the first await.  The
            # coordinator itself waits for any in-flight grow/commit holder.
            self._closing = True
            self._release_task = asyncio.create_task(
                self._release_coordinated()
            )
        task = self._release_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(task)
            raise

    async def _release_coordinated(self) -> None:
        async with self._state_lock:
            if self._committed or self._released:
                return
            await self._release_once()

    async def _release_once(self) -> None:
        if self._released:
            return
        await self._lease._controller._release_response_bytes(
            self._lease,
            self.size,
        )
        self._released = True

    async def reserve(self, additional_bytes: int) -> int:
        if additional_bytes < 0:
            raise ValueError("additional_bytes cannot be negative")
        async with self._state_lock:
            if self._released or self._committed or self._closing:
                raise RuntimeError("temporary response reservation is closed")
            await self._lease._controller._grow_temporary_response(
                self,
                additional_bytes,
            )
            return self.size

    async def commit(self) -> int:
        """Transfer ownership to the surrounding request lease."""

        async with self._state_lock:
            if self._released or self._closing:
                raise RuntimeError("cannot commit a released reservation")
            if self._committed:
                return self.size
            await self._lease._controller._commit_temporary_response(self)
            return self.size

    async def __aenter__(self) -> TemporaryResponseBytesReservation:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()


class MemoryReleaseDeferral:
    """Child ownership keeping base request memory charged after active exit."""

    def __init__(self, lease: RequestAdmissionLease) -> None:
        self._lease = lease
        self._active = False
        self._released = False
        self._release_task: asyncio.Task[None] | None = None

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> None:
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._release_once())
        task = self._release_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(task)
            raise

    async def _release_once(self) -> None:
        if self._released:
            return
        await self._lease._controller._finish_memory_owner(self._lease)
        self._released = True


class RequestAdmissionController:
    """Bound active requests and all in-memory request-body reservations."""

    def __init__(
        self,
        *,
        capacity: int,
        waiter_limit: int,
        wait_timeout_seconds: float,
        max_body_bytes: int,
        body_budget_bytes: int,
        max_response_bytes: int | None = None,
        max_retained_bytes_per_request: int | None = None,
        large_body_threshold_weighted_bytes: int = 0,
        large_body_limit: int = 0,
        memory_governor: AdaptiveMemoryGovernor | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_body_bytes < 0:
            raise ValueError("max_body_bytes cannot be negative")
        if body_budget_bytes < 0:
            raise ValueError("body_budget_bytes cannot be negative")
        if max_response_bytes is not None and max_response_bytes < 0:
            raise ValueError("max_response_bytes cannot be negative")
        if large_body_threshold_weighted_bytes < 0 or large_body_limit < 0:
            raise ValueError("large body admission settings cannot be negative")
        if bool(large_body_threshold_weighted_bytes) != bool(large_body_limit):
            raise ValueError(
                "large body threshold and limit must both be enabled or disabled"
            )

        self.max_body_bytes = max_body_bytes
        self.body_budget_bytes = body_budget_bytes
        self.max_response_bytes = (
            max_body_bytes if max_response_bytes is None else max_response_bytes
        )
        self.max_retained_bytes_per_request = (
            max(self.max_body_bytes, self.max_response_bytes)
            if max_retained_bytes_per_request is None
            else int(max_retained_bytes_per_request)
        )
        self.large_body_threshold_weighted_bytes = int(
            large_body_threshold_weighted_bytes
        )
        self.large_body_limit = int(large_body_limit)
        self._large_body_active = 0
        if self.max_retained_bytes_per_request < 0:
            raise ValueError("max_retained_bytes_per_request cannot be negative")
        self._memory_governor = memory_governor
        self._active_gate = BoundedAdmissionGate(
            capacity,
            waiter_limit=waiter_limit,
            wait_timeout_seconds=wait_timeout_seconds,
            clock=clock,
        )
        self._body_lock = asyncio.Lock()
        self._reserved_body_bytes = 0
        self._pending_body_reserved_bytes = 0
        self._reserved_response_bytes = 0
        self._deferred_memory_leases: set[RequestAdmissionLease] = set()
        self._body_rejected: Counter[str] = Counter()
        self._response_rejected: Counter[str] = Counter()

    async def acquire(
        self,
        *,
        initial_body_bytes: int = 0,
        timeout_seconds: float | None = None,
    ) -> RequestAdmissionLease:
        if initial_body_bytes < 0:
            raise ValueError("initial_body_bytes cannot be negative")

        acquisition = await self.begin_acquire(timeout_seconds=timeout_seconds)
        lease = await _await_owned_acquisition(acquisition)
        if initial_body_bytes == 0:
            return lease
        try:
            await lease.reserve_body_bytes(initial_body_bytes)
        except BaseException:
            await lease.release()
            raise
        return lease

    async def begin_acquire(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> asyncio.Future[RequestAdmissionLease]:
        active_acquisition = await self._active_gate.begin_acquire(
            timeout_seconds=timeout_seconds
        )
        loop = asyncio.get_running_loop()
        if active_acquisition.done():
            active_lease = active_acquisition.result()
            resolved: asyncio.Future[RequestAdmissionLease] = loop.create_future()
            resolved.set_result(
                RequestAdmissionLease(
                    self,
                    active_lease,
                    reserved_body_bytes=0,
                )
            )
            return resolved

        started = asyncio.Event()

        async def convert() -> RequestAdmissionLease:
            started.set()
            active_lease = await _await_owned_acquisition(active_acquisition)
            return RequestAdmissionLease(
                self,
                active_lease,
                reserved_body_bytes=0,
            )

        task = asyncio.create_task(convert())
        try:
            await started.wait()
        except BaseException:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, AdmissionRejected):
                await task
            # As above, cancellation may happen before convert() executes its
            # first line.  In that window only this outer coroutine can return
            # the inner waiter/lease, so never rely solely on convert()'s
            # cancellation handler.
            cleanup_task = asyncio.create_task(
                _cancel_and_release_acquisition(active_acquisition)
            )
            await _finish_cleanup_despite_cancellation(cleanup_task)
            raise
        return task

    async def try_acquire(self) -> RequestAdmissionLease | None:
        active_lease = await self._active_gate.try_acquire()
        if active_lease is None:
            return None
        return RequestAdmissionLease(
            self,
            active_lease,
            reserved_body_bytes=0,
        )

    def pending_body_reservation(self) -> PendingBodyReservation:
        return PendingBodyReservation(self)

    async def _reserve_pending_body_additional(
        self,
        reservation: PendingBodyReservation,
        additional_bytes: int,
    ) -> None:
        async with self._body_lock:
            next_request_bytes = reservation._reserved_bytes + additional_bytes
            if next_request_bytes > self.max_body_bytes:
                self._body_rejected["body_too_large"] += 1
                raise RequestBodyTooLarge()
            if (
                self._reserved_body_bytes
                + self._reserved_response_bytes
                + additional_bytes
                > self.body_budget_bytes
            ):
                self._body_rejected["body_budget_exhausted"] += 1
                raise RequestBodyBudgetExhausted()
            self._ensure_large_body_slot_available_locked(
                reservation,
                next_request_bytes,
            )
            if not self._reserve_parent_memory("request_body", additional_bytes):
                self._body_rejected["body_budget_exhausted"] += 1
                raise RequestBodyBudgetExhausted()
            self._claim_large_body_slot_locked(
                reservation,
                next_request_bytes,
            )
            reservation._reserved_bytes = next_request_bytes
            self._reserved_body_bytes += additional_bytes
            self._pending_body_reserved_bytes += additional_bytes

    async def _transfer_pending_body(
        self,
        reservation: PendingBodyReservation,
        lease: RequestAdmissionLease,
    ) -> int:
        async with self._body_lock:
            if lease._release_requested or lease._released:
                raise RuntimeError("cannot transfer bytes to a released request lease")
            transferred = reservation._reserved_bytes
            next_request_bytes = lease._reserved_body_bytes + transferred
            if next_request_bytes > self.max_body_bytes:
                raise RuntimeError("pending body transfer exceeds request limit")
            if transferred > self._pending_body_reserved_bytes:
                raise RuntimeError("pending body reservation underflow")
            if reservation._large_body_slot:
                if lease._large_body_slot:
                    raise RuntimeError("large request admission double transfer")
                lease._large_body_slot = True
                reservation._large_body_slot = False
            else:
                self._claim_large_body_slot_locked(lease, next_request_bytes)
            lease._reserved_body_bytes = next_request_bytes
            reservation._reserved_bytes = 0
            self._pending_body_reserved_bytes -= transferred
            return transferred

    async def _release_pending_body(
        self,
        reservation: PendingBodyReservation,
    ) -> None:
        async with self._body_lock:
            released = reservation._reserved_bytes
            reservation._reserved_bytes = 0
            if released > self._pending_body_reserved_bytes:
                raise RuntimeError("pending body reservation underflow")
            if released > self._reserved_body_bytes:
                raise RuntimeError("request body reservation underflow")
            self._pending_body_reserved_bytes -= released
            self._reserved_body_bytes -= released
            self._release_parent_memory("request_body", released)
            if reservation._large_body_slot:
                if self._large_body_active <= 0:
                    raise RuntimeError("large request admission underflow")
                self._large_body_active -= 1
                reservation._large_body_slot = False

    async def _reserve_additional(
        self,
        lease: RequestAdmissionLease,
        additional_bytes: int,
    ) -> None:
        async with self._body_lock:
            if lease._release_requested or lease._released:
                raise RuntimeError("cannot reserve bytes on a released request lease")
            next_request_bytes = lease._reserved_body_bytes + additional_bytes
            if next_request_bytes > self.max_body_bytes:
                self._body_rejected["body_too_large"] += 1
                raise RequestBodyTooLarge()
            if (
                self._reserved_body_bytes
                + self._reserved_response_bytes
                + additional_bytes
                > self.body_budget_bytes
            ):
                self._body_rejected["body_budget_exhausted"] += 1
                raise RequestBodyBudgetExhausted()
            if (
                next_request_bytes + lease._reserved_response_bytes
                > self.max_retained_bytes_per_request
            ):
                self._body_rejected["body_budget_exhausted"] += 1
                raise RequestBodyBudgetExhausted()
            self._ensure_large_body_slot_available_locked(
                lease,
                next_request_bytes,
            )
            if not self._reserve_parent_memory("request_body", additional_bytes):
                self._body_rejected["body_budget_exhausted"] += 1
                raise RequestBodyBudgetExhausted()
            self._claim_large_body_slot_locked(lease, next_request_bytes)
            lease._reserved_body_bytes = next_request_bytes
            self._reserved_body_bytes += additional_bytes

    async def _reserve_response_additional(
        self,
        lease: RequestAdmissionLease,
        additional_bytes: int,
    ) -> None:
        async with self._body_lock:
            if lease._release_requested or lease._released:
                raise RuntimeError("cannot reserve bytes on a released request lease")
            next_request_bytes = lease._reserved_response_bytes + additional_bytes
            if next_request_bytes > self.max_response_bytes:
                self._response_rejected["upstream_response_too_large"] += 1
                raise UpstreamResponseBudgetExhausted()
            if (
                lease._reserved_body_bytes + next_request_bytes
                > self.max_retained_bytes_per_request
            ):
                self._response_rejected[
                    "upstream_response_budget_exhausted"
                ] += 1
                raise UpstreamResponseBudgetExhausted()
            if (
                self._reserved_body_bytes
                + self._reserved_response_bytes
                + additional_bytes
                > self.body_budget_bytes
            ):
                self._response_rejected[
                    "upstream_response_budget_exhausted"
                ] += 1
                raise UpstreamResponseBudgetExhausted()
            if not self._reserve_parent_memory("buffered_response", additional_bytes):
                self._response_rejected[
                    "upstream_response_budget_exhausted"
                ] += 1
                raise UpstreamResponseBudgetExhausted()
            lease._reserved_response_bytes = next_request_bytes
            self._reserved_response_bytes += additional_bytes

    async def _activate_temporary_response(
        self,
        reservation: TemporaryResponseBytesReservation,
        additional_bytes: int,
    ) -> None:
        lease = reservation._lease
        async with self._body_lock:
            if lease._release_requested or lease._released:
                raise RuntimeError("cannot reserve bytes on a released request lease")
            next_request_bytes = lease._reserved_response_bytes + additional_bytes
            if next_request_bytes > self.max_response_bytes:
                self._response_rejected["upstream_response_too_large"] += 1
                raise UpstreamResponseBudgetExhausted()
            if (
                lease._reserved_body_bytes + next_request_bytes
                > self.max_retained_bytes_per_request
            ):
                self._response_rejected[
                    "upstream_response_budget_exhausted"
                ] += 1
                raise UpstreamResponseBudgetExhausted()
            if (
                self._reserved_body_bytes
                + self._reserved_response_bytes
                + additional_bytes
                > self.body_budget_bytes
            ):
                self._response_rejected[
                    "upstream_response_budget_exhausted"
                ] += 1
                raise UpstreamResponseBudgetExhausted()
            if not self._reserve_parent_memory("buffered_response", additional_bytes):
                self._response_rejected[
                    "upstream_response_budget_exhausted"
                ] += 1
                raise UpstreamResponseBudgetExhausted()
            lease._reserved_response_bytes = next_request_bytes
            self._reserved_response_bytes += additional_bytes
            lease._memory_owner_count += 1
            reservation.size = additional_bytes
            reservation._active = True

    async def _grow_temporary_response(
        self,
        reservation: TemporaryResponseBytesReservation,
        additional_bytes: int,
    ) -> None:
        lease = reservation._lease
        async with self._body_lock:
            if not reservation._active or reservation._released or reservation._committed:
                raise RuntimeError("temporary response reservation is closed")
            next_request_bytes = lease._reserved_response_bytes + additional_bytes
            if next_request_bytes > self.max_response_bytes:
                self._response_rejected["upstream_response_too_large"] += 1
                raise UpstreamResponseBudgetExhausted()
            if (
                lease._reserved_body_bytes + next_request_bytes
                > self.max_retained_bytes_per_request
            ):
                self._response_rejected[
                    "upstream_response_budget_exhausted"
                ] += 1
                raise UpstreamResponseBudgetExhausted()
            if (
                self._reserved_body_bytes
                + self._reserved_response_bytes
                + additional_bytes
                > self.body_budget_bytes
            ):
                self._response_rejected[
                    "upstream_response_budget_exhausted"
                ] += 1
                raise UpstreamResponseBudgetExhausted()
            if not self._reserve_parent_memory("buffered_response", additional_bytes):
                self._response_rejected[
                    "upstream_response_budget_exhausted"
                ] += 1
                raise UpstreamResponseBudgetExhausted()
            lease._reserved_response_bytes = next_request_bytes
            self._reserved_response_bytes += additional_bytes
            reservation.size += additional_bytes

    async def _activate_memory_deferral(
        self,
        deferral: MemoryReleaseDeferral,
    ) -> None:
        lease = deferral._lease
        async with self._body_lock:
            if lease._release_requested or lease._memory_finalized:
                raise RuntimeError("cannot defer memory on a released request lease")
            lease._memory_owner_count += 1
            deferral._active = True

    async def _release_request(self, lease: RequestAdmissionLease) -> None:
        async with self._body_lock:
            if lease._memory_owner_count:
                self._deferred_memory_leases.add(lease)
            else:
                self._finalize_request_memory_locked(lease)
        await lease._active_lease.release()

    async def _release_response_bytes(
        self,
        lease: RequestAdmissionLease,
        released_bytes: int,
    ) -> None:
        async with self._body_lock:
            if released_bytes < 0:
                raise ValueError("released_bytes cannot be negative")
            if released_bytes > lease._reserved_response_bytes:
                raise RuntimeError("request response reservation underflow")
            if released_bytes > self._reserved_response_bytes:
                raise RuntimeError("upstream response reservation underflow")
            lease._reserved_response_bytes -= released_bytes
            self._reserved_response_bytes -= released_bytes
            self._release_parent_memory("buffered_response", released_bytes)
            self._finish_memory_owner_locked(lease)

    async def _commit_temporary_response(
        self,
        reservation: TemporaryResponseBytesReservation,
    ) -> None:
        lease = reservation._lease
        async with self._body_lock:
            if not reservation._active or reservation._released:
                raise RuntimeError("temporary response reservation is closed")
            if reservation._committed:
                return
            reservation._committed = True
            self._finish_memory_owner_locked(lease)

    async def _finish_memory_owner(
        self,
        lease: RequestAdmissionLease,
    ) -> None:
        async with self._body_lock:
            self._finish_memory_owner_locked(lease)

    def _finish_memory_owner_locked(self, lease: RequestAdmissionLease) -> None:
        if lease._memory_owner_count <= 0:
            raise RuntimeError("request memory owner underflow")
        lease._memory_owner_count -= 1
        if lease._release_requested and lease._memory_owner_count == 0:
            self._finalize_request_memory_locked(lease)

    def _finalize_request_memory_locked(
        self,
        lease: RequestAdmissionLease,
    ) -> None:
        if lease._memory_finalized:
            return
        reserved_body_bytes = lease._reserved_body_bytes
        reserved_response_bytes = lease._reserved_response_bytes
        if reserved_body_bytes > self._reserved_body_bytes:
            raise RuntimeError("request body reservation underflow")
        if reserved_response_bytes > self._reserved_response_bytes:
            raise RuntimeError("upstream response reservation underflow")
        self._reserved_body_bytes -= reserved_body_bytes
        self._reserved_response_bytes -= reserved_response_bytes
        self._release_parent_memory("request_body", reserved_body_bytes)
        self._release_parent_memory("buffered_response", reserved_response_bytes)
        if lease._large_body_slot:
            if self._large_body_active <= 0:
                raise RuntimeError("large request admission underflow")
            self._large_body_active -= 1
            lease._large_body_slot = False
        lease._reserved_body_bytes = 0
        lease._reserved_response_bytes = 0
        lease._memory_finalized = True
        self._deferred_memory_leases.discard(lease)

    def _claim_large_body_slot_locked(
        self,
        lease: RequestAdmissionLease | PendingBodyReservation,
        next_request_bytes: int,
    ) -> None:
        if (
            not self.large_body_limit
            or lease._large_body_slot
            or next_request_bytes <= self.large_body_threshold_weighted_bytes
        ):
            return
        self._ensure_large_body_slot_available_locked(
            lease,
            next_request_bytes,
        )
        self._large_body_active += 1
        lease._large_body_slot = True

    def _ensure_large_body_slot_available_locked(
        self,
        lease: RequestAdmissionLease | PendingBodyReservation,
        next_request_bytes: int,
    ) -> None:
        if (
            not self.large_body_limit
            or lease._large_body_slot
            or next_request_bytes <= self.large_body_threshold_weighted_bytes
        ):
            return
        if self._large_body_active >= self.large_body_limit:
            self._body_rejected["large_body_capacity_exhausted"] += 1
            raise LargeBodyCapacityExhausted()

    def record_rejection(self, reason: str) -> None:
        normalized = str(reason or "").strip()
        if normalized:
            self._body_rejected[normalized] += 1

    def _reserve_parent_memory(self, category: str, size: int) -> bool:
        if self._memory_governor is None:
            return True
        return self._memory_governor.reserve_nowait(category, size)

    def _release_parent_memory(self, category: str, size: int) -> None:
        if self._memory_governor is not None and size:
            self._memory_governor.release(category, size)

    def snapshot(self) -> dict[str, Any]:
        gate_snapshot = self._active_gate.snapshot()
        rejected = Counter(gate_snapshot["rejected"])
        rejected.update(self._body_rejected)
        rejected.update(self._response_rejected)
        parent_snapshot = (
            self._memory_governor.snapshot()
            if self._memory_governor is not None
            else None
        )
        effective_body_budget = self.body_budget_bytes
        if parent_snapshot is not None:
            effective_body_budget = min(
                effective_body_budget,
                parent_snapshot.capacity_bytes,
            )
        return {
            **gate_snapshot,
            "reserved_body_bytes": self._reserved_body_bytes,
            "pending_body_reserved_bytes": self._pending_body_reserved_bytes,
            "reserved_response_bytes": self._reserved_response_bytes,
            "reserved_retained_bytes": (
                self._reserved_body_bytes + self._reserved_response_bytes
            ),
            "deferred_memory_requests": len(self._deferred_memory_leases),
            "deferred_memory_bytes": sum(
                lease._reserved_body_bytes + lease._reserved_response_bytes
                for lease in self._deferred_memory_leases
            ),
            "body_budget": effective_body_budget,
            "body_budget_hard": self.body_budget_bytes,
            "max_body_bytes": self.max_body_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_retained_bytes_per_request": self.max_retained_bytes_per_request,
            "large_body_threshold_weighted_bytes": (
                self.large_body_threshold_weighted_bytes
            ),
            "large_body_limit": self.large_body_limit,
            "large_body_active": self._large_body_active,
            "memory_parent": parent_snapshot,
            "rejected": dict(rejected),
        }


_current_request_admission_lease: contextvars.ContextVar[
    RequestAdmissionLease | None
] = contextvars.ContextVar("uni_api_request_admission_lease", default=None)


def bind_request_admission_lease(
    lease: RequestAdmissionLease | None,
) -> contextvars.Token[RequestAdmissionLease | None]:
    return _current_request_admission_lease.set(lease)


def reset_request_admission_lease(
    token: contextvars.Token[RequestAdmissionLease | None],
) -> None:
    _current_request_admission_lease.reset(token)


def get_request_admission_lease() -> RequestAdmissionLease | None:
    return _current_request_admission_lease.get()
