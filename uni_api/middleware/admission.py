from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from inspect import isawaitable
from time import monotonic
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from uni_api.admission import (
    AdmissionRejected,
    RequestAdmissionController,
    bind_request_admission_lease,
    reset_request_admission_lease,
)
from core.log_config import logger
from uni_api.disconnect import DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY
from uni_api.middleware.request_decompression import (
    BODY_BYTES_RESERVATION_SCOPE_KEY,
    BODY_EARLY_RESPONSE_OBSERVER_SCOPE_KEY,
    BODY_REJECTION_RECORDER_SCOPE_KEY,
)


RESERVE_BODY_BYTES_STATE_KEY = BODY_BYTES_RESERVATION_SCOPE_KEY
ADMISSION_LEASE_STATE_KEY = "uni_api_admission_lease"
ADMISSION_WAIT_MS_STATE_KEY = "uni_api_admission_wait_ms"
ADMISSION_PREBUFFER_MESSAGE_HIGH_WATERMARK = 16


async def _finish_ownership_cleanup(
    cleanup_task: asyncio.Task[None],
) -> None:
    """Run a complete ownership transaction despite repeated cancellation."""

    pending_cancel: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
            pending_cancel = pending_cancel or exc
    cleanup_task.result()
    if pending_cancel is not None:
        raise pending_cancel


class RequestAdmissionMiddleware:
    """Bound requests before they can allocate application-owned resources.

    This is deliberately a pure ASGI middleware. Awaiting the downstream app
    therefore covers the complete response body lifecycle, including a
    streaming response and its disconnect cleanup.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        controller: RequestAdmissionController,
        bypass: Callable[[Scope], bool] | None = None,
        retry_after_seconds: int = 1,
        on_rejection: Callable[[Scope, AdmissionRejected, float], Any] | None = None,
        on_early_response: Callable[[Scope, int, str], Any] | None = None,
    ) -> None:
        if retry_after_seconds <= 0:
            raise ValueError("retry_after_seconds must be greater than zero")
        self.app = app
        self.controller = controller
        self.bypass = bypass or (lambda scope: False)
        self.retry_after_seconds = retry_after_seconds
        self.on_rejection = on_rejection
        self.on_early_response = on_early_response

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if self.bypass(scope):
            # A bypassed request owns no admission lease.  Explicitly mask an
            # inherited ContextVar value so health/observability work spawned
            # from another request cannot charge a lease that is already in
            # release.  Reset afterwards to preserve the caller's context.
            bypass_context_token = bind_request_admission_lease(None)
            try:
                await self.app(scope, receive, send)
            finally:
                reset_request_admission_lease(bypass_context_token)
            return

        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        lease = None
        lease_context_token = None
        replayed_messages: deque[tuple[Message, int]] = deque()
        handed_off_receive_task: asyncio.Task[Message] | None = None
        available_replayed_credit = 0
        admission_started_at = monotonic()
        try:
            lease = await self.controller.try_acquire()
            if lease is None:
                (
                    lease,
                    replayed_messages,
                    handed_off_receive_task,
                    disconnected_while_queued,
                ) = await self._acquire_queued_request(scope, receive)
                if disconnected_while_queued:
                    state = scope.setdefault("state", {})
                    state[ADMISSION_WAIT_MS_STATE_KEY] = (
                        monotonic() - admission_started_at
                    ) * 1000.0
                    await self._notify_early_response(
                        scope,
                        499,
                        "disconnected_while_queued",
                    )
                    return
                assert lease is not None
            lease_context_token = bind_request_admission_lease(lease)
            state: dict[str, Any] = scope.setdefault("state", {})
            state[ADMISSION_LEASE_STATE_KEY] = lease
            state[ADMISSION_WAIT_MS_STATE_KEY] = lease.wait_ms
            async def reserve_body_bytes(additional_bytes: int) -> int:
                nonlocal available_replayed_credit
                if int(additional_bytes) < 0:
                    raise ValueError("additional_bytes cannot be negative")
                credited = min(
                    max(0, int(additional_bytes)),
                    available_replayed_credit,
                )
                available_replayed_credit -= credited
                remaining = int(additional_bytes) - credited
                if remaining > 0:
                    return await lease.reserve_body_bytes(remaining)
                return lease.reserved_body_bytes

            state[RESERVE_BODY_BYTES_STATE_KEY] = reserve_body_bytes
            state[BODY_REJECTION_RECORDER_SCOPE_KEY] = (
                self.controller.record_rejection
            )

            async def observe_body_early_response(
                status_code: int,
                reason: str,
            ) -> None:
                await self._notify_early_response(scope, status_code, reason)

            state[BODY_EARLY_RESPONSE_OBSERVER_SCOPE_KEY] = (
                observe_body_early_response
            )
            async def replay_receive() -> Message:
                nonlocal available_replayed_credit
                nonlocal handed_off_receive_task
                if replayed_messages:
                    message, credited_bytes = replayed_messages.popleft()
                    available_replayed_credit += credited_bytes
                    return message
                if handed_off_receive_task is not None:
                    task = handed_off_receive_task
                    handed_off_receive_task = None
                    message = await task
                else:
                    message = await receive()
                if message.get("type") == "http.request":
                    body = message.get("body", b"") or b""
                    if body:
                        await lease.reserve_body_bytes(len(body))
                        available_replayed_credit += len(body)
                return message

            await self.app(scope, replay_receive, tracking_send)
        except AdmissionRejected as exc:
            if response_started:
                raise
            wait_ms = (
                lease.wait_ms
                if lease is not None
                else (monotonic() - admission_started_at) * 1000.0
            )
            await self._notify_rejection(scope, exc, wait_ms)
            await self._send_rejection(scope, receive, send, exc)
        finally:
            abandoned_receive = handed_off_receive_task
            handed_off_receive_task = None

            async def cleanup_request_ownership() -> None:
                try:
                    if abandoned_receive is not None:
                        if not abandoned_receive.done():
                            abandoned_receive.cancel()
                        try:
                            abandoned_message = await abandoned_receive
                            abandoned_message = None
                        except (asyncio.CancelledError, Exception):
                            pass
                finally:
                    # Drop body aliases and callbacks before returning their
                    # byte/accounting owner.  The state mapping can outlive the
                    # downstream app until the ASGI scope itself is discarded.
                    replayed_messages.clear()
                    request_state = scope.get("state")
                    if isinstance(request_state, dict):
                        request_state.pop(ADMISSION_LEASE_STATE_KEY, None)
                        request_state.pop(RESERVE_BODY_BYTES_STATE_KEY, None)
                        request_state.pop(BODY_REJECTION_RECORDER_SCOPE_KEY, None)
                        request_state.pop(
                            BODY_EARLY_RESPONSE_OBSERVER_SCOPE_KEY,
                            None,
                        )
                    if lease is not None:
                        await lease.release()

            cleanup_task = asyncio.create_task(cleanup_request_ownership())
            try:
                await _finish_ownership_cleanup(cleanup_task)
            finally:
                if lease_context_token is not None:
                    reset_request_admission_lease(lease_context_token)

    async def _acquire_queued_request(
        self,
        scope: Scope,
        receive: Receive,
    ) -> tuple[
        Any,
        deque[tuple[Message, int]],
        asyncio.Task[Message] | None,
        bool,
    ]:
        """Observe a queued peer without allowing an unbounded body prebuffer."""

        # Claim a bounded waiter position before starting any receive.  A
        # queue-full request must not be allowed to allocate even one body
        # chunk outside the 64-active/936-waiter envelope.
        acquire_task = await self.controller.begin_acquire()
        receive_task: asyncio.Task[Message] | None = (
            None if acquire_task.done() else asyncio.create_task(receive())
        )
        state = scope.get("state")
        disconnect_event = (
            state.get(DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY)
            if isinstance(state, dict)
            else None
        )
        disconnect_task: asyncio.Task[bool] | None = (
            asyncio.create_task(disconnect_event.wait())
            if isinstance(disconnect_event, asyncio.Event)
            and not acquire_task.done()
            else None
        )
        buffered: deque[tuple[Message, int]] = deque()
        pending = self.controller.pending_body_reservation()
        lease = None
        queued_body_complete = False

        async def cancel_acquire_and_release_result() -> None:
            # Cancellation can lose the race with a just-granted lease.  Always
            # consume the task result and release any transferred ownership;
            # otherwise a same-turn peer disconnect strands an active slot.
            if not acquire_task.done():
                acquire_task.cancel()
            try:
                granted_lease = await acquire_task
            except (asyncio.CancelledError, AdmissionRejected):
                return
            await granted_lease.release()

        async def cancel_disconnect_observer(
            task: asyncio.Task[bool],
        ) -> None:
            async def cancel_and_consume() -> None:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            cleanup_task = asyncio.create_task(cancel_and_consume())
            await _finish_ownership_cleanup(cleanup_task)

        async def cleanup_queued_ownership(
            receive_to_cleanup: asyncio.Task[Message] | None,
            disconnect_to_cleanup: asyncio.Task[bool] | None,
            known_lease: Any,
        ) -> None:
            """Release every queued owner even if an earlier step fails."""

            try:
                if receive_to_cleanup is not None:
                    if not receive_to_cleanup.done():
                        receive_to_cleanup.cancel()
                    try:
                        abandoned_message = await receive_to_cleanup
                        abandoned_message = None
                    except (asyncio.CancelledError, Exception):
                        pass
            finally:
                try:
                    if disconnect_to_cleanup is not None:
                        if not disconnect_to_cleanup.done():
                            disconnect_to_cleanup.cancel()
                        try:
                            await disconnect_to_cleanup
                        except (asyncio.CancelledError, Exception):
                            pass
                finally:
                    try:
                        await cancel_acquire_and_release_result()
                    finally:
                        try:
                            if known_lease is not None:
                                await known_lease.release()
                        finally:
                            buffered.clear()
                            await pending.release()

        async def finish_queued_ownership_cleanup(
            receive_to_cleanup: asyncio.Task[Message] | None,
            disconnect_to_cleanup: asyncio.Task[bool] | None,
            known_lease: Any,
        ) -> None:
            cleanup_task = asyncio.create_task(
                cleanup_queued_ownership(
                    receive_to_cleanup,
                    disconnect_to_cleanup,
                    known_lease,
                )
            )
            await _finish_ownership_cleanup(cleanup_task)

        try:
            while True:
                wait_tasks: set[asyncio.Task[Any]] = {acquire_task}
                if receive_task is not None:
                    wait_tasks.add(receive_task)
                if disconnect_task is not None:
                    wait_tasks.add(disconnect_task)
                done, _ = await asyncio.wait(
                    wait_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # A transport close owns a same-turn race with admission.  If
                # the gate already granted, cleanup consumes and releases that
                # lease before returning the disconnected outcome.
                if disconnect_task is not None and disconnect_task in done:
                    disconnected_wait = disconnect_task
                    disconnect_task = None
                    abandoned_receive = receive_task
                    receive_task = None
                    await finish_queued_ownership_cleanup(
                        abandoned_receive,
                        disconnected_wait,
                        lease,
                    )
                    return None, buffered, None, True

                if acquire_task in done:
                    lease = acquire_task.result()
                    if receive_task is not None:
                        if receive_task.done():
                            message = await receive_task
                            receive_task = None
                            if message.get("type") == "http.disconnect":
                                message = None
                                await finish_queued_ownership_cleanup(
                                    None,
                                    disconnect_task,
                                    lease,
                                )
                                disconnect_task = None
                                lease = None
                                return None, buffered, None, True
                            if not queued_body_complete:
                                await self._buffer_pending_message(
                                    pending,
                                    buffered,
                                    message,
                                )
                            receive_task = None
                    await pending.transfer_to(lease)
                    if (
                        isinstance(disconnect_event, asyncio.Event)
                        and disconnect_event.is_set()
                    ):
                        abandoned_receive = receive_task
                        receive_task = None
                        await finish_queued_ownership_cleanup(
                            abandoned_receive,
                            disconnect_task,
                            lease,
                        )
                        disconnect_task = None
                        lease = None
                        return None, buffered, None, True
                    if disconnect_task is not None:
                        await cancel_disconnect_observer(disconnect_task)
                        disconnect_task = None
                    # Cancellation of event.wait() yields once.  Recheck the
                    # transport-owned event so a close in that turn still owns
                    # the admission race.
                    if (
                        isinstance(disconnect_event, asyncio.Event)
                        and disconnect_event.is_set()
                    ):
                        abandoned_receive = receive_task
                        receive_task = None
                        await finish_queued_ownership_cleanup(
                            abandoned_receive,
                            None,
                            lease,
                        )
                        lease = None
                        return None, buffered, None, True
                    return lease, buffered, receive_task, False

                assert receive_task is not None and receive_task in done
                message = receive_task.result()
                receive_task = None
                if message.get("type") == "http.disconnect":
                    message = None
                    await finish_queued_ownership_cleanup(
                        None,
                        disconnect_task,
                        lease,
                    )
                    disconnect_task = None
                    return None, buffered, None, True

                if not queued_body_complete:
                    await self._buffer_pending_message(pending, buffered, message)
                if (
                    not queued_body_complete
                    and
                    message.get("type") == "http.request"
                    and not message.get("more_body", False)
                ):
                    queued_body_complete = True
                if (
                    not queued_body_complete
                    and len(buffered)
                    >= ADMISSION_PREBUFFER_MESSAGE_HIGH_WATERMARK
                ):
                    # ASGI body frame boundaries are transport details, not a
                    # measure of request size or memory pressure.  Pause the
                    # receive side at a bounded message high-water mark instead
                    # of rejecting an otherwise valid request.  The queued
                    # acquisition task still enforces its normal wait timeout;
                    # once admitted, replay_receive drains these frames and
                    # resumes the remaining body under the active lease.
                    continue
                # Even after the final body frame, keep exactly one receive
                # pending so a peer that disconnects while still queued is
                # observed before the application is admitted.
                receive_task = asyncio.create_task(receive())
        except BaseException:
            abandoned_receive = receive_task
            receive_task = None
            abandoned_disconnect = disconnect_task
            disconnect_task = None
            await finish_queued_ownership_cleanup(
                abandoned_receive,
                abandoned_disconnect,
                lease,
            )
            lease = None
            raise

    @staticmethod
    async def _buffer_pending_message(
        pending: Any,
        buffered: deque[tuple[Message, int]],
        message: Message,
    ) -> None:
        if message.get("type") != "http.request":
            return
        body = message.get("body", b"") or b""
        await pending.reserve(len(body))
        buffered.append((message, len(body)))

    async def _notify_rejection(
        self,
        scope: Scope,
        rejection: AdmissionRejected,
        wait_ms: float,
    ) -> None:
        if self.on_rejection is None:
            return
        try:
            result = self.on_rejection(scope, rejection, max(0.0, wait_ms))
            if isawaitable(result):
                await result
        except Exception:
            # Telemetry must never turn a deliberate 413/503 into a 500.
            logger.exception("Failed to record request admission rejection")

    async def _notify_early_response(
        self,
        scope: Scope,
        status_code: int,
        reason: str,
    ) -> None:
        if self.on_early_response is None:
            return
        try:
            result = self.on_early_response(scope, int(status_code), str(reason))
            if isawaitable(result):
                await result
        except Exception:
            logger.exception("Failed to record early request-body response")

    async def _send_rejection(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        rejection: AdmissionRejected,
    ) -> None:
        status_code = int(rejection.status_code)
        reason = rejection.reason
        if status_code == 413:
            message = "Request body exceeds the configured local limit"
            error_type = "request_too_large"
        else:
            message = "Service is at its bounded local capacity; retry later"
            error_type = "local_overload"

        headers = {"x-uni-api-admission-reason": reason}
        if status_code == 503:
            headers["retry-after"] = str(self.retry_after_seconds)
        response = JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": reason,
                }
            },
            headers=headers,
        )
        await response(scope, receive, send)
