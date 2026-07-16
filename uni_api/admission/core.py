from __future__ import annotations

import asyncio
import contextlib
import contextvars
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from time import monotonic, time
from typing import Any
from uuid import uuid4

from uni_api.admission.memory import AdaptiveMemoryGovernor
from uni_api.admission.observability import (
    LargeBodyAdmissionDecision,
    LargeBodyHolderSnapshot,
    RequestBodyObservation,
)


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


@dataclass(frozen=True, slots=True)
class _LargeBodyHolder:
    claim_id: str
    lease_id: str
    observation: RequestBodyObservation
    claimed_at_monotonic: float
    claimed_at_unix_ms: int
    reserved_weighted_bytes: int


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
        self._large_body_claim_id: str | None = None
        self._lease_id = controller._new_lease_id()
        self._body_observation = RequestBodyObservation()
        self._release_reason = "pending_released"

    @property
    def reserved_bytes(self) -> int:
        return self._reserved_bytes

    @property
    def lease_id(self) -> str:
        return self._lease_id

    def observe_body(self, observation: RequestBodyObservation) -> None:
        if self._transferred or self._released:
            return
        self._body_observation = observation

    async def reserve(
        self,
        additional_bytes: int,
        *,
        observation: RequestBodyObservation | None = None,
    ) -> int:
        if additional_bytes < 0:
            raise ValueError("additional_bytes cannot be negative")
        if self._transferred or self._released:
            raise RuntimeError("pending body reservation is no longer active")
        await self._controller._reserve_pending_body_additional(
            self,
            additional_bytes,
            observation=observation,
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

    async def release(self, *, reason: str = "pending_released") -> None:
        if self._transferred or self._released:
            return
        self._release_reason = str(reason or "pending_released")
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
        self._large_body_claim_id: str | None = None
        self._lease_id = controller._new_lease_id()
        self._body_observation = RequestBodyObservation()
        self._release_reason = "request_completed"
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

    @property
    def lease_id(self) -> str:
        return self._lease_id

    def observe_body(self, observation: RequestBodyObservation) -> None:
        if self._release_requested or self._released:
            return
        self._body_observation = observation

    async def reserve_body_bytes(
        self,
        additional_bytes: int,
        *,
        observation: RequestBodyObservation | None = None,
    ) -> int:
        """Atomically grow this request's reservation by an observed chunk."""

        if additional_bytes < 0:
            raise ValueError("additional_bytes cannot be negative")
        if self._release_requested:
            raise RuntimeError("cannot reserve bytes on a released request lease")
        await self._controller._reserve_additional(
            self,
            additional_bytes,
            observation=observation,
        )
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

    async def release(self, *, reason: str = "request_completed") -> None:
        self._release_requested = True
        if self._release_task is None:
            self._release_reason = str(reason or "request_completed")
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
        wall_clock: Callable[[], float] = time,
        decision_observer: Callable[[LargeBodyAdmissionDecision], bool | None]
        | None = None,
        decision_history_limit: int = 64,
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
        if decision_history_limit <= 0:
            raise ValueError("decision_history_limit must be greater than zero")

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
        self._large_body_holders: dict[str, _LargeBodyHolder] = {}
        if self.max_retained_bytes_per_request < 0:
            raise ValueError("max_retained_bytes_per_request cannot be negative")
        self._memory_governor = memory_governor
        self._clock = clock
        self._wall_clock = wall_clock
        self._decision_observer = decision_observer
        self._decision_history: deque[LargeBodyAdmissionDecision] = deque(
            maxlen=int(decision_history_limit)
        )
        self._decision_sequence = 0
        self._decision_history_overwritten = 0
        self._decision_record_failures = 0
        self._decision_observer_errors = 0
        self._decision_observer_enqueue_failures = 0
        self._identity_nonce = uuid4().hex
        self._lease_sequence = 0
        self._claim_sequence = 0
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

    def _new_lease_id(self) -> str:
        self._lease_sequence += 1
        return f"{self._identity_nonce}-lease-{self._lease_sequence:x}"

    def _new_claim_id_locked(self) -> str:
        self._claim_sequence += 1
        return f"{self._identity_nonce}-claim-{self._claim_sequence:x}"

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
        *,
        observation: RequestBodyObservation | None = None,
    ) -> None:
        decision_events: list[LargeBodyAdmissionDecision] = []
        try:
            async with self._body_lock:
                request_before = reservation._reserved_bytes
                global_body_before = self._reserved_body_bytes
                global_response_before = self._reserved_response_bytes
                if observation is not None:
                    reservation._body_observation = observation
                next_request_bytes = request_before + additional_bytes
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
                    request_before=request_before,
                    global_body_before=global_body_before,
                    global_response_before=global_response_before,
                    decision_events=decision_events,
                )
                if not self._reserve_parent_memory("request_body", additional_bytes):
                    self._body_rejected["body_budget_exhausted"] += 1
                    raise RequestBodyBudgetExhausted()
                reservation._reserved_bytes = next_request_bytes
                self._reserved_body_bytes += additional_bytes
                self._pending_body_reserved_bytes += additional_bytes
                self._claim_large_body_slot_locked(
                    reservation,
                    next_request_bytes,
                    request_before=request_before,
                    global_body_before=global_body_before,
                    global_response_before=global_response_before,
                    decision_events=decision_events,
                )
                self._update_large_body_holder_locked(
                    reservation,
                    next_request_bytes,
                )
        finally:
            self._publish_decision_events(decision_events)

    async def _transfer_pending_body(
        self,
        reservation: PendingBodyReservation,
        lease: RequestAdmissionLease,
    ) -> int:
        decision_events: list[LargeBodyAdmissionDecision] = []
        try:
            async with self._body_lock:
                if lease._release_requested or lease._released:
                    raise RuntimeError("cannot transfer bytes to a released request lease")
                transferred = reservation._reserved_bytes
                request_before = lease._reserved_body_bytes
                next_request_bytes = request_before + transferred
                global_body_before = self._reserved_body_bytes
                global_response_before = self._reserved_response_bytes
                if next_request_bytes > self.max_body_bytes:
                    raise RuntimeError("pending body transfer exceeds request limit")
                if transferred > self._pending_body_reserved_bytes:
                    raise RuntimeError("pending body reservation underflow")
                lease._body_observation = reservation._body_observation
                if reservation._large_body_slot:
                    if lease._large_body_slot:
                        raise RuntimeError("large request admission double transfer")
                    claim_id = reservation._large_body_claim_id
                    if not claim_id or claim_id not in self._large_body_holders:
                        raise RuntimeError("large request holder missing during transfer")
                    lease._large_body_slot = True
                    lease._large_body_claim_id = claim_id
                    reservation._large_body_slot = False
                    reservation._large_body_claim_id = None
                    holder = self._large_body_holders[claim_id]
                    self._large_body_holders[claim_id] = replace(
                        holder,
                        lease_id=lease._lease_id,
                        observation=lease._body_observation,
                        reserved_weighted_bytes=next_request_bytes,
                    )
                else:
                    self._ensure_large_body_slot_available_locked(
                        lease,
                        next_request_bytes,
                        request_before=request_before,
                        global_body_before=global_body_before,
                        global_response_before=global_response_before,
                        decision_events=decision_events,
                    )
                lease._reserved_body_bytes = next_request_bytes
                reservation._reserved_bytes = 0
                self._pending_body_reserved_bytes -= transferred
                self._claim_large_body_slot_locked(
                    lease,
                    next_request_bytes,
                    request_before=request_before,
                    global_body_before=global_body_before,
                    global_response_before=global_response_before,
                    decision_events=decision_events,
                )
                self._update_large_body_holder_locked(lease, next_request_bytes)
                return transferred
        finally:
            self._publish_decision_events(decision_events)

    async def _release_pending_body(
        self,
        reservation: PendingBodyReservation,
    ) -> None:
        decision_events: list[LargeBodyAdmissionDecision] = []
        try:
            async with self._body_lock:
                released = reservation._reserved_bytes
                request_before = released
                global_body_before = self._reserved_body_bytes
                global_response_before = self._reserved_response_bytes
                reservation._reserved_bytes = 0
                if released > self._pending_body_reserved_bytes:
                    raise RuntimeError("pending body reservation underflow")
                if released > self._reserved_body_bytes:
                    raise RuntimeError("request body reservation underflow")
                self._pending_body_reserved_bytes -= released
                self._reserved_body_bytes -= released
                self._release_parent_memory("request_body", released)
                self._release_large_body_slot_locked(
                    reservation,
                    request_before=request_before,
                    global_body_before=global_body_before,
                    global_response_before=global_response_before,
                    release_reason=reservation._release_reason,
                    release_finalizer="pending_release",
                    decision_events=decision_events,
                )
        finally:
            self._publish_decision_events(decision_events)

    async def _reserve_additional(
        self,
        lease: RequestAdmissionLease,
        additional_bytes: int,
        *,
        observation: RequestBodyObservation | None = None,
    ) -> None:
        decision_events: list[LargeBodyAdmissionDecision] = []
        try:
            async with self._body_lock:
                if lease._release_requested or lease._released:
                    raise RuntimeError("cannot reserve bytes on a released request lease")
                request_before = lease._reserved_body_bytes
                global_body_before = self._reserved_body_bytes
                global_response_before = self._reserved_response_bytes
                if observation is not None:
                    lease._body_observation = observation
                next_request_bytes = request_before + additional_bytes
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
                    request_before=request_before,
                    global_body_before=global_body_before,
                    global_response_before=global_response_before,
                    decision_events=decision_events,
                )
                if not self._reserve_parent_memory("request_body", additional_bytes):
                    self._body_rejected["body_budget_exhausted"] += 1
                    raise RequestBodyBudgetExhausted()
                lease._reserved_body_bytes = next_request_bytes
                self._reserved_body_bytes += additional_bytes
                self._claim_large_body_slot_locked(
                    lease,
                    next_request_bytes,
                    request_before=request_before,
                    global_body_before=global_body_before,
                    global_response_before=global_response_before,
                    decision_events=decision_events,
                )
                self._update_large_body_holder_locked(lease, next_request_bytes)
        finally:
            self._publish_decision_events(decision_events)

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
        decision_events: list[LargeBodyAdmissionDecision] = []
        ownership_cleanup_complete = False
        try:
            async with self._body_lock:
                if lease._memory_owner_count:
                    self._deferred_memory_leases.add(lease)
                else:
                    self._finalize_request_memory_locked(
                        lease,
                        decision_events=decision_events,
                        release_finalizer="request_release",
                    )
            ownership_cleanup_complete = True
        finally:
            if ownership_cleanup_complete:
                try:
                    # Return business capacity before doing any exporter work.
                    await lease._active_lease.release()
                finally:
                    self._publish_decision_events(decision_events)
            else:
                self._publish_decision_events(decision_events)

    async def _release_response_bytes(
        self,
        lease: RequestAdmissionLease,
        released_bytes: int,
    ) -> None:
        decision_events: list[LargeBodyAdmissionDecision] = []
        try:
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
                self._finish_memory_owner_locked(
                    lease,
                    decision_events=decision_events,
                    release_finalizer="temporary_response_release",
                )
        finally:
            self._publish_decision_events(decision_events)

    async def _commit_temporary_response(
        self,
        reservation: TemporaryResponseBytesReservation,
    ) -> None:
        lease = reservation._lease
        decision_events: list[LargeBodyAdmissionDecision] = []
        try:
            async with self._body_lock:
                if not reservation._active or reservation._released:
                    raise RuntimeError("temporary response reservation is closed")
                if reservation._committed:
                    return
                reservation._committed = True
                self._finish_memory_owner_locked(
                    lease,
                    decision_events=decision_events,
                    release_finalizer="temporary_response_commit",
                )
        finally:
            self._publish_decision_events(decision_events)

    async def _finish_memory_owner(
        self,
        lease: RequestAdmissionLease,
    ) -> None:
        decision_events: list[LargeBodyAdmissionDecision] = []
        try:
            async with self._body_lock:
                self._finish_memory_owner_locked(
                    lease,
                    decision_events=decision_events,
                    release_finalizer="deferred_memory_release",
                )
        finally:
            self._publish_decision_events(decision_events)

    def _finish_memory_owner_locked(
        self,
        lease: RequestAdmissionLease,
        *,
        decision_events: list[LargeBodyAdmissionDecision],
        release_finalizer: str,
    ) -> None:
        if lease._memory_owner_count <= 0:
            raise RuntimeError("request memory owner underflow")
        lease._memory_owner_count -= 1
        if lease._release_requested and lease._memory_owner_count == 0:
            self._finalize_request_memory_locked(
                lease,
                decision_events=decision_events,
                release_finalizer=release_finalizer,
            )

    def _finalize_request_memory_locked(
        self,
        lease: RequestAdmissionLease,
        *,
        decision_events: list[LargeBodyAdmissionDecision],
        release_finalizer: str,
    ) -> None:
        if lease._memory_finalized:
            return
        reserved_body_bytes = lease._reserved_body_bytes
        reserved_response_bytes = lease._reserved_response_bytes
        global_body_before = self._reserved_body_bytes
        global_response_before = self._reserved_response_bytes
        if reserved_body_bytes > self._reserved_body_bytes:
            raise RuntimeError("request body reservation underflow")
        if reserved_response_bytes > self._reserved_response_bytes:
            raise RuntimeError("upstream response reservation underflow")
        self._reserved_body_bytes -= reserved_body_bytes
        self._reserved_response_bytes -= reserved_response_bytes
        self._release_parent_memory("request_body", reserved_body_bytes)
        self._release_parent_memory("buffered_response", reserved_response_bytes)
        self._release_large_body_slot_locked(
            lease,
            request_before=reserved_body_bytes,
            global_body_before=global_body_before,
            global_response_before=global_response_before,
            release_reason=lease._release_reason,
            release_finalizer=release_finalizer,
            decision_events=decision_events,
        )
        lease._reserved_body_bytes = 0
        lease._reserved_response_bytes = 0
        lease._memory_finalized = True
        self._deferred_memory_leases.discard(lease)

    def _claim_large_body_slot_locked(
        self,
        lease: RequestAdmissionLease | PendingBodyReservation,
        next_request_bytes: int,
        *,
        request_before: int,
        global_body_before: int,
        global_response_before: int,
        decision_events: list[LargeBodyAdmissionDecision],
    ) -> None:
        if (
            not self.large_body_limit
            or lease._large_body_slot
            or next_request_bytes <= self.large_body_threshold_weighted_bytes
        ):
            return
        if len(self._large_body_holders) >= self.large_body_limit:
            raise RuntimeError("large request capacity changed after availability check")
        active_before = len(self._large_body_holders)
        claim_id = self._new_claim_id_locked()
        holder = _LargeBodyHolder(
            claim_id=claim_id,
            lease_id=lease._lease_id,
            observation=lease._body_observation,
            claimed_at_monotonic=self._safe_decision_monotonic(),
            claimed_at_unix_ms=self._safe_decision_wall_time_ms(),
            reserved_weighted_bytes=next_request_bytes,
        )
        self._large_body_holders[claim_id] = holder
        lease._large_body_slot = True
        lease._large_body_claim_id = claim_id
        self._append_large_body_decision_locked(
            decision_events,
            decision="claim",
            reason="threshold_crossed",
            owner=lease,
            request_before=request_before,
            attempted_after=next_request_bytes,
            committed_after=next_request_bytes,
            active_before=active_before,
            global_body_before=global_body_before,
            global_response_before=global_response_before,
            holder=holder,
        )

    def _ensure_large_body_slot_available_locked(
        self,
        lease: RequestAdmissionLease | PendingBodyReservation,
        next_request_bytes: int,
        *,
        request_before: int,
        global_body_before: int,
        global_response_before: int,
        decision_events: list[LargeBodyAdmissionDecision],
    ) -> None:
        if (
            not self.large_body_limit
            or lease._large_body_slot
            or next_request_bytes <= self.large_body_threshold_weighted_bytes
        ):
            return
        if len(self._large_body_holders) >= self.large_body_limit:
            self._body_rejected["large_body_capacity_exhausted"] += 1
            self._append_large_body_decision_locked(
                decision_events,
                decision="reject",
                reason="large_body_capacity_exhausted",
                owner=lease,
                request_before=request_before,
                attempted_after=next_request_bytes,
                committed_after=request_before,
                active_before=len(self._large_body_holders),
                global_body_before=global_body_before,
                global_response_before=global_response_before,
                blocking_holders=tuple(self._large_body_holders.values()),
            )
            raise LargeBodyCapacityExhausted()

    def _update_large_body_holder_locked(
        self,
        owner: RequestAdmissionLease | PendingBodyReservation,
        reserved_weighted_bytes: int,
    ) -> None:
        if not owner._large_body_slot:
            return
        claim_id = owner._large_body_claim_id
        if not claim_id or claim_id not in self._large_body_holders:
            raise RuntimeError("large request holder missing")
        holder = self._large_body_holders[claim_id]
        self._large_body_holders[claim_id] = replace(
            holder,
            lease_id=owner._lease_id,
            observation=owner._body_observation,
            reserved_weighted_bytes=reserved_weighted_bytes,
        )

    def _release_large_body_slot_locked(
        self,
        owner: RequestAdmissionLease | PendingBodyReservation,
        *,
        request_before: int,
        global_body_before: int,
        global_response_before: int,
        release_reason: str,
        release_finalizer: str,
        decision_events: list[LargeBodyAdmissionDecision],
    ) -> None:
        if not owner._large_body_slot:
            return
        claim_id = owner._large_body_claim_id
        if not claim_id:
            raise RuntimeError("large request claim id missing")
        holder = self._large_body_holders.get(claim_id)
        if holder is None:
            raise RuntimeError("large request holder missing during release")
        active_before = len(self._large_body_holders)
        del self._large_body_holders[claim_id]
        owner._large_body_slot = False
        owner._large_body_claim_id = None
        self._append_large_body_decision_locked(
            decision_events,
            decision="release",
            reason="slot_released",
            owner=owner,
            request_before=request_before,
            attempted_after=request_before,
            committed_after=0,
            active_before=active_before,
            global_body_before=global_body_before,
            global_response_before=global_response_before,
            release_reason=release_reason,
            release_finalizer=release_finalizer,
            holder=holder,
        )

    def _safe_decision_monotonic(self) -> float:
        try:
            return float(self._clock())
        except Exception:
            self._decision_record_failures += 1
            return monotonic()

    def _safe_decision_wall_time_ms(self) -> int:
        try:
            return int(float(self._wall_clock()) * 1000.0)
        except Exception:
            self._decision_record_failures += 1
            return int(time() * 1000.0)

    def _append_large_body_decision_locked(
        self,
        decision_events: list[LargeBodyAdmissionDecision],
        **kwargs: Any,
    ) -> None:
        """Record observability without changing admission ownership semantics."""

        try:
            event = self._record_large_body_decision_locked(**kwargs)
        except Exception:
            self._decision_record_failures += 1
            return
        decision_events.append(event)

    def _holder_snapshot_locked(
        self,
        holder: _LargeBodyHolder,
        *,
        now_monotonic: float,
    ) -> LargeBodyHolderSnapshot:
        return LargeBodyHolderSnapshot(
            claim_id=holder.claim_id,
            lease_id=holder.lease_id,
            request_id=holder.observation.request_id,
            trace_id=holder.observation.trace_id,
            claimed_at_unix_ms=holder.claimed_at_unix_ms,
            held_ms=max(
                0,
                int(round((now_monotonic - holder.claimed_at_monotonic) * 1000.0)),
            ),
            request_self_body_reserved_weighted_bytes=(
                holder.reserved_weighted_bytes
            ),
        )

    def _record_large_body_decision_locked(
        self,
        *,
        decision: str,
        reason: str,
        owner: RequestAdmissionLease | PendingBodyReservation,
        request_before: int,
        attempted_after: int,
        committed_after: int,
        active_before: int,
        global_body_before: int,
        global_response_before: int,
        release_reason: str | None = None,
        release_finalizer: str | None = None,
        holder: _LargeBodyHolder | None = None,
        blocking_holders: tuple[_LargeBodyHolder, ...] = (),
    ) -> LargeBodyAdmissionDecision:
        now_monotonic = self._safe_decision_monotonic()
        try:
            parent = (
                self._memory_governor.snapshot_cached()
                if self._memory_governor is not None
                else None
            )
        except Exception:
            self._decision_record_failures += 1
            parent = None
        effective_body_budget = self.body_budget_bytes
        if parent is not None:
            effective_body_budget = min(
                effective_body_budget,
                parent.capacity_bytes,
            )
        observation = owner._body_observation
        self._decision_sequence += 1
        event = LargeBodyAdmissionDecision(
            schema_version=1,
            sequence=self._decision_sequence,
            decision=decision,
            reason=reason,
            occurred_at_unix_ms=self._safe_decision_wall_time_ms(),
            release_reason=release_reason,
            release_finalizer=release_finalizer,
            request_self_lease_id=owner._lease_id,
            request_self_request_id=observation.request_id,
            request_self_trace_id=observation.trace_id,
            request_self_method=observation.method,
            request_self_path=observation.path,
            request_self_declared_content_length_bytes=(
                observation.declared_content_length_bytes
            ),
            request_self_wire_bytes=observation.wire_bytes,
            request_self_decoded_bytes=observation.decoded_bytes,
            request_self_decoder_workspace_bytes=(
                observation.decoder_workspace_bytes
            ),
            request_self_json_raw_bytes=observation.json_raw_bytes,
            request_self_json_structural_item_count=(
                observation.json_structural_item_count
            ),
            request_self_json_depth=observation.json_depth,
            request_self_json_peak_depth=observation.json_peak_depth,
            request_self_json_scalar_bytes=observation.json_scalar_bytes,
            request_self_json_estimated_bytes=observation.json_estimated_bytes,
            request_self_json_raw_memory_multiplier=(
                observation.json_raw_memory_multiplier
            ),
            request_self_json_structural_item_memory_bytes=(
                observation.json_structural_item_memory_bytes
            ),
            request_self_body_reserved_weighted_before_bytes=request_before,
            request_self_body_reserved_weighted_attempted_after_bytes=(
                attempted_after
            ),
            request_self_body_reserved_weighted_committed_after_bytes=(
                committed_after
            ),
            runtime_global_large_body_threshold_weighted_bytes=(
                self.large_body_threshold_weighted_bytes
            ),
            runtime_global_large_body_active_before=active_before,
            runtime_global_large_body_active_after=len(self._large_body_holders),
            runtime_global_large_body_limit=self.large_body_limit,
            runtime_global_request_body_reserved_weighted_before_bytes=(
                global_body_before
            ),
            runtime_global_request_body_reserved_weighted_after_bytes=(
                self._reserved_body_bytes
            ),
            runtime_global_upstream_response_reserved_weighted_before_bytes=(
                global_response_before
            ),
            runtime_global_upstream_response_reserved_weighted_after_bytes=(
                self._reserved_response_bytes
            ),
            runtime_global_retained_reserved_weighted_before_bytes=(
                global_body_before + global_response_before
            ),
            runtime_global_retained_reserved_weighted_after_bytes=(
                self._reserved_body_bytes + self._reserved_response_bytes
            ),
            runtime_global_request_body_budget_weighted_bytes=effective_body_budget,
            runtime_global_request_body_budget_hard_weighted_bytes=self.body_budget_bytes,
            runtime_global_cgroup_memory_source=getattr(parent, "source", None),
            runtime_global_cgroup_memory_current_bytes_sampled=getattr(
                parent, "current_bytes", None
            ),
            runtime_global_cgroup_memory_limit_bytes_sampled=getattr(parent, "limit_bytes", None),
            runtime_global_cgroup_memory_high_bytes_sampled=getattr(parent, "high_bytes", None),
            runtime_global_cgroup_memory_soft_limit_bytes_sampled=getattr(
                parent, "soft_limit_bytes", None
            ),
            runtime_global_cgroup_memory_guard_bytes_sampled=getattr(parent, "guard_bytes", None),
            runtime_global_cgroup_memory_capacity_bytes_sampled=getattr(
                parent, "capacity_bytes", None
            ),
            runtime_global_cgroup_memory_available_bytes_sampled=getattr(
                parent, "available_bytes", None
            ),
            runtime_global_cgroup_memory_reserved_bytes_sampled=getattr(
                parent, "reserved_bytes", None
            ),
            runtime_global_cgroup_memory_sample_sequence=getattr(
                parent, "sample_sequence", None
            ),
            runtime_global_cgroup_memory_sample_age_ms_at_decision=getattr(
                parent, "sample_age_ms", None
            ),
            runtime_global_cgroup_memory_sample_error=getattr(
                parent, "sample_error", None
            ),
            holder=(
                self._holder_snapshot_locked(holder, now_monotonic=now_monotonic)
                if holder is not None
                else None
            ),
            blocking_holders=tuple(
                self._holder_snapshot_locked(
                    blocking_holder,
                    now_monotonic=now_monotonic,
                )
                for blocking_holder in blocking_holders
            ),
        )
        if len(self._decision_history) == self._decision_history.maxlen:
            self._decision_history_overwritten += 1
        self._decision_history.append(event)
        return event

    def _publish_decision_events(
        self,
        events: list[LargeBodyAdmissionDecision],
    ) -> None:
        observer = self._decision_observer
        if observer is None:
            return
        for event in events:
            try:
                accepted = observer(event)
            except Exception:
                self._decision_observer_errors += 1
                continue
            if accepted is False:
                self._decision_observer_enqueue_failures += 1

    def recent_large_body_decisions(self) -> tuple[LargeBodyAdmissionDecision, ...]:
        return tuple(self._decision_history)

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
        now_monotonic = self._clock()
        holder_ages_ms = [
            max(
                0,
                int(round((now_monotonic - holder.claimed_at_monotonic) * 1000.0)),
            )
            for holder in self._large_body_holders.values()
        ]
        rejection_decision_total = sum(int(value or 0) for value in rejected.values())
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
            "large_body_active": len(self._large_body_holders),
            "large_body_oldest_holder_age_ms": (
                max(holder_ages_ms) if holder_ages_ms else 0
            ),
            "large_body_decision_events_recorded_total": self._decision_sequence,
            "large_body_decision_history_overwritten_total": (
                self._decision_history_overwritten
            ),
            "large_body_decision_record_failures_total": (
                self._decision_record_failures
            ),
            "large_body_decision_observer_errors_total": (
                self._decision_observer_errors
            ),
            "large_body_decision_observer_enqueue_failures_total": (
                self._decision_observer_enqueue_failures
            ),
            "memory_parent": parent_snapshot,
            "rejected": dict(rejected),
            "rejection_decisions": dict(rejected),
            "rejection_decision_total": rejection_decision_total,
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
