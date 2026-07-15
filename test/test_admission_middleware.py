import asyncio
import json

import pytest

from uni_api.admission import AdmissionRejected, RequestAdmissionController
from uni_api.disconnect import DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY
from uni_api.middleware.admission import (
    ADMISSION_WAIT_MS_STATE_KEY,
    RESERVE_BODY_BYTES_STATE_KEY,
    RequestAdmissionMiddleware,
)


def _scope(path: str = "/v1/responses") -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": {},
    }


async def _receive():
    return {"type": "http.disconnect"}


async def _wait_until(predicate, *, timeout=1.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


def _response(sent: list[dict]) -> tuple[int, dict, dict[bytes, bytes]]:
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return start["status"], json.loads(body), dict(start["headers"])


def _controller(**overrides) -> RequestAdmissionController:
    settings = {
        "capacity": 1,
        "waiter_limit": 1,
        "wait_timeout_seconds": 0.05,
        "max_body_bytes": 8,
        "body_budget_bytes": 8,
    }
    settings.update(overrides)
    return RequestAdmissionController(**settings)


def test_admission_lease_covers_the_complete_streaming_asgi_lifecycle():
    async def run():
        controller = _controller()
        body_may_finish = asyncio.Event()
        response_started = asyncio.Event()
        sent: list[dict] = []

        async def streaming_app(scope, receive, send):
            assert callable(scope["state"][RESERVE_BODY_BYTES_STATE_KEY])
            assert scope["state"][ADMISSION_WAIT_MS_STATE_KEY] >= 0
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"first", "more_body": True})
            response_started.set()
            await body_may_finish.wait()
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        middleware = RequestAdmissionMiddleware(streaming_app, controller=controller)

        async def send(message):
            sent.append(message)

        task = asyncio.create_task(middleware(_scope(), _receive, send))
        await response_started.wait()
        assert controller.snapshot()["active"] == 1

        body_may_finish.set()
        await task
        assert controller.snapshot()["active"] == 0
        assert sent[-1]["more_body"] is False

    asyncio.run(run())


def test_admission_queue_full_returns_structured_retryable_503():
    async def run():
        controller = _controller(waiter_limit=0)
        holder = await controller.acquire()
        called = False
        sent: list[dict] = []

        async def app(scope, receive, send):
            nonlocal called
            called = True

        async def send(message):
            sent.append(message)

        middleware = RequestAdmissionMiddleware(app, controller=controller)
        await middleware(_scope(), _receive, send)
        await holder.release()

        status, payload, headers = _response(sent)
        assert status == 503
        assert payload["error"]["code"] == "queue_full"
        assert payload["error"]["type"] == "local_overload"
        assert headers[b"retry-after"] == b"1"
        assert headers[b"x-uni-api-admission-reason"] == b"queue_full"
        assert called is False

    asyncio.run(run())


def test_admission_rejection_callback_observes_pre_app_overload():
    async def run():
        controller = _controller(waiter_limit=0)
        holder = await controller.acquire()
        observed = []
        sent: list[dict] = []

        async def on_rejection(scope, rejection, wait_ms):
            observed.append((scope["path"], rejection.reason, wait_ms))

        middleware = RequestAdmissionMiddleware(
            lambda scope, receive, send: None,
            controller=controller,
            on_rejection=on_rejection,
        )
        async def send(message):
            sent.append(message)

        await middleware(_scope(), _receive, send)
        await holder.release()

        assert len(observed) == 1
        assert observed[0][0:2] == ("/v1/responses", "queue_full")
        assert observed[0][2] >= 0
        assert _response(sent)[0] == 503

    asyncio.run(run())


@pytest.mark.parametrize(
    ("reserve_bytes", "expected_status", "expected_reason"),
    [(9, 413, "body_too_large"), (5, 503, "body_budget_exhausted")],
)
def test_body_reservation_rejection_is_mapped_before_response_start(
    reserve_bytes,
    expected_status,
    expected_reason,
):
    async def run():
        controller = _controller(
            max_body_bytes=8,
            body_budget_bytes=4 if expected_status == 503 else 8,
        )
        sent: list[dict] = []

        async def app(scope, receive, send):
            reserve = scope["state"][RESERVE_BODY_BYTES_STATE_KEY]
            await reserve(reserve_bytes)
            raise AssertionError("reservation rejection should stop the app")

        async def send(message):
            sent.append(message)

        middleware = RequestAdmissionMiddleware(app, controller=controller)
        await middleware(_scope(), _receive, send)

        status, payload, headers = _response(sent)
        assert status == expected_status
        assert payload["error"]["code"] == expected_reason
        assert headers[b"x-uni-api-admission-reason"] == expected_reason.encode()
        assert controller.snapshot()["active"] == 0
        assert controller.snapshot()["reserved_body_bytes"] == 0

    asyncio.run(run())


def test_large_body_slot_is_bounded_and_released_with_request():
    async def run():
        controller = _controller(
            capacity=3,
            max_body_bytes=64,
            body_budget_bytes=192,
            large_body_threshold_weighted_bytes=16,
            large_body_limit=1,
        )
        first = await controller.acquire(initial_body_bytes=17)
        second = await controller.acquire()
        with pytest.raises(AdmissionRejected) as exc_info:
            await second.reserve_body_bytes(17)
        assert exc_info.value.reason == "large_body_capacity_exhausted"
        assert controller.snapshot()["large_body_active"] == 1

        await first.release()
        await second.reserve_body_bytes(17)
        assert controller.snapshot()["large_body_active"] == 1
        await second.release()
        assert controller.snapshot()["large_body_active"] == 0

    asyncio.run(run())


def test_pending_body_owns_and_transfers_large_body_slot():
    async def run():
        controller = _controller(
            capacity=2,
            max_body_bytes=64,
            body_budget_bytes=192,
            large_body_threshold_weighted_bytes=16,
            large_body_limit=1,
        )
        pending = controller.pending_body_reservation()
        await pending.reserve(17)
        assert controller.snapshot()["large_body_active"] == 1

        blocked = controller.pending_body_reservation()
        with pytest.raises(AdmissionRejected) as exc_info:
            await blocked.reserve(17)
        assert exc_info.value.reason == "large_body_capacity_exhausted"

        lease = await controller.acquire()
        await pending.transfer_to(lease)
        assert controller.snapshot()["large_body_active"] == 1
        await lease.release()
        assert controller.snapshot()["large_body_active"] == 0
        await blocked.release()

    asyncio.run(run())


def test_admission_cancellation_releases_slot_and_body_reservation():
    async def run():
        controller = _controller()
        entered = asyncio.Event()

        async def app(scope, receive, send):
            await scope["state"][RESERVE_BODY_BYTES_STATE_KEY](4)
            entered.set()
            await asyncio.Event().wait()

        middleware = RequestAdmissionMiddleware(app, controller=controller)
        task = asyncio.create_task(middleware(_scope(), _receive, lambda message: None))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_body_bytes"] == 0

    asyncio.run(run())


def test_admission_bypass_does_not_consume_capacity():
    async def run():
        controller = _controller()
        observed_active = None

        async def app(scope, receive, send):
            nonlocal observed_active
            observed_active = controller.snapshot()["active"]

        middleware = RequestAdmissionMiddleware(
            app,
            controller=controller,
            bypass=lambda scope: scope["path"] == "/healthz",
        )
        await middleware(_scope("/healthz"), _receive, lambda message: None)
        assert observed_active == 0

    asyncio.run(run())


def test_queued_disconnect_cancels_waiter_and_releases_pending_body_bytes():
    async def run():
        controller = _controller(max_body_bytes=32, body_budget_bytes=32)
        holder = await controller.acquire()
        messages = iter(
            [
                {"type": "http.request", "body": b"queued", "more_body": True},
                {"type": "http.disconnect"},
            ]
        )
        app_called = False
        observed = []

        async def receive():
            return next(messages)

        async def app(scope, receive, send):
            nonlocal app_called
            app_called = True

        async def on_early(scope, status_code, reason):
            observed.append((scope["path"], status_code, reason))

        middleware = RequestAdmissionMiddleware(
            app,
            controller=controller,
            on_early_response=on_early,
        )
        await middleware(_scope(), receive, lambda message: None)

        snapshot = controller.snapshot()
        assert app_called is False
        assert observed == [
            ("/v1/responses", 499, "disconnected_while_queued")
        ]
        assert snapshot["active"] == 1
        assert snapshot["waiters"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["pending_body_reserved_bytes"] == 0
        await holder.release()

    asyncio.run(run())


def test_same_turn_admission_and_disconnect_race_is_owned_by_disconnect():
    async def run():
        controller = _controller()
        holder = await controller.acquire()
        release_peer = asyncio.Event()
        app_called = False

        async def receive():
            await release_peer.wait()
            return {"type": "http.disconnect"}

        async def app(scope, receive, send):
            nonlocal app_called
            app_called = True

        middleware = RequestAdmissionMiddleware(app, controller=controller)
        task = asyncio.create_task(
            middleware(_scope(), receive, lambda message: None)
        )
        while controller.snapshot()["waiters"] != 1:
            await asyncio.sleep(0)

        release_task = asyncio.create_task(holder.release())
        release_peer.set()
        await asyncio.gather(release_task, task)

        snapshot = controller.snapshot()
        assert app_called is False
        assert snapshot["active"] == 0
        assert snapshot["waiters"] == 0
        assert snapshot["reserved_body_bytes"] == 0

    asyncio.run(run())


def test_queued_prebuffer_high_watermark_backpressures_then_replays_all_frames():
    async def run():
        controller = _controller(
            max_body_bytes=64,
            body_budget_bytes=64,
            wait_timeout_seconds=1,
        )
        holder = await controller.acquire()
        sent = []
        delivered = 0
        app_started = asyncio.Event()
        received_body = bytearray()

        async def receive():
            nonlocal delivered
            delivered += 1
            assert app_started.is_set() or delivered <= 16
            return {
                "type": "http.request",
                "body": b"x",
                "more_body": delivered < 32,
            }

        async def app(scope, replay_receive, send):
            app_started.set()
            reserve_body_bytes = scope["state"][RESERVE_BODY_BYTES_STATE_KEY]
            while True:
                message = await replay_receive()
                body = message.get("body", b"") or b""
                await reserve_body_bytes(len(body))
                received_body.extend(body)
                if not message.get("more_body", False):
                    break
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def send(message):
            sent.append(message)

        middleware = RequestAdmissionMiddleware(
            app,
            controller=controller,
        )
        request = asyncio.create_task(middleware(_scope(), receive, send))
        await _wait_until(
            lambda: controller.snapshot()["pending_body_reserved_bytes"] == 16
        )

        assert delivered == 16
        assert request.done() is False
        assert sent == []
        assert controller.snapshot()["rejected"] == {}

        await holder.release()
        await request

        snapshot = controller.snapshot()
        assert delivered == 32
        assert received_body == b"x" * 32
        assert sent[0]["status"] == 204
        assert snapshot["active"] == 0
        assert snapshot["waiters"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["pending_body_reserved_bytes"] == 0
        assert snapshot["rejected"] == {}

    asyncio.run(run())


def test_wait_timeout_while_prebuffer_backpressured_releases_pending_bytes():
    async def run():
        controller = _controller(
            max_body_bytes=64,
            body_budget_bytes=64,
            wait_timeout_seconds=0.05,
        )
        holder = await controller.acquire()
        sent = []
        delivered = 0

        async def receive():
            nonlocal delivered
            delivered += 1
            return {
                "type": "http.request",
                "body": b"x",
                "more_body": True,
            }

        async def send(message):
            sent.append(message)

        middleware = RequestAdmissionMiddleware(
            lambda scope, receive, send: None,
            controller=controller,
        )
        await middleware(_scope(), receive, send)

        status, payload, headers = _response(sent)
        snapshot = controller.snapshot()
        assert delivered == 16
        assert status == 503
        assert payload["error"]["code"] == "wait_timeout"
        assert headers[b"x-uni-api-admission-reason"] == b"wait_timeout"
        assert snapshot["active"] == 1
        assert snapshot["waiters"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["pending_body_reserved_bytes"] == 0
        assert snapshot["rejected"] == {"wait_timeout": 1}
        await holder.release()

    asyncio.run(run())


def test_queued_body_byte_budget_remains_the_rejection_boundary():
    async def run():
        controller = _controller(
            max_body_bytes=64,
            body_budget_bytes=8,
            wait_timeout_seconds=1,
        )
        holder = await controller.acquire()
        sent = []
        messages = iter(
            [
                {
                    "type": "http.request",
                    "body": b"12345",
                    "more_body": True,
                },
                {
                    "type": "http.request",
                    "body": b"6789",
                    "more_body": True,
                },
            ]
        )

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        middleware = RequestAdmissionMiddleware(
            lambda scope, receive, send: None,
            controller=controller,
        )
        await middleware(_scope(), receive, send)

        status, payload, headers = _response(sent)
        snapshot = controller.snapshot()
        assert status == 503
        assert payload["error"]["code"] == "body_budget_exhausted"
        assert (
            headers[b"x-uni-api-admission-reason"]
            == b"body_budget_exhausted"
        )
        assert snapshot["active"] == 1
        assert snapshot["waiters"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["pending_body_reserved_bytes"] == 0
        assert snapshot["rejected"] == {"body_budget_exhausted": 1}
        await holder.release()

    asyncio.run(run())


def test_cancellation_while_prebuffer_backpressured_releases_queued_ownership():
    async def run():
        controller = _controller(
            max_body_bytes=64,
            body_budget_bytes=64,
            wait_timeout_seconds=1,
        )
        holder = await controller.acquire()
        delivered = 0

        async def receive():
            nonlocal delivered
            delivered += 1
            return {
                "type": "http.request",
                "body": b"x",
                "more_body": True,
            }

        middleware = RequestAdmissionMiddleware(
            lambda scope, receive, send: None,
            controller=controller,
        )
        request = asyncio.create_task(
            middleware(_scope(), receive, lambda message: None)
        )
        await _wait_until(
            lambda: controller.snapshot()["pending_body_reserved_bytes"] == 16
        )

        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        snapshot = controller.snapshot()
        assert delivered == 16
        assert snapshot["active"] == 1
        assert snapshot["waiters"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["pending_body_reserved_bytes"] == 0
        assert snapshot["rejected"] == {}
        await holder.release()

    asyncio.run(run())


def test_transport_disconnect_while_prebuffer_backpressured_releases_waiter():
    async def run():
        controller = _controller(
            max_body_bytes=64,
            body_budget_bytes=64,
            wait_timeout_seconds=1,
        )
        holder = await controller.acquire()
        delivered = 0
        app_called = False
        observed = []
        disconnect_event = asyncio.Event()
        scope = _scope()
        scope["state"][DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY] = disconnect_event

        async def receive():
            nonlocal delivered
            delivered += 1
            return {
                "type": "http.request",
                "body": b"x",
                "more_body": True,
            }

        async def app(scope, receive, send):
            nonlocal app_called
            app_called = True

        async def on_early(scope, status_code, reason):
            observed.append((status_code, reason))

        middleware = RequestAdmissionMiddleware(
            app,
            controller=controller,
            on_early_response=on_early,
        )
        request = asyncio.create_task(
            middleware(scope, receive, lambda message: None)
        )
        await _wait_until(
            lambda: controller.snapshot()["pending_body_reserved_bytes"] == 16
        )

        disconnect_event.set()
        await request

        snapshot = controller.snapshot()
        assert delivered == 16
        assert app_called is False
        assert observed == [(499, "disconnected_while_queued")]
        assert snapshot["active"] == 1
        assert snapshot["waiters"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["pending_body_reserved_bytes"] == 0
        assert snapshot["rejected"] == {}
        await holder.release()

    asyncio.run(run())


def test_same_turn_transport_disconnect_owns_a_queued_admission_grant():
    async def run():
        controller = _controller(
            max_body_bytes=64,
            body_budget_bytes=64,
            wait_timeout_seconds=1,
        )
        holder = await controller.acquire()
        disconnect_event = asyncio.Event()
        scope = _scope()
        scope["state"][DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY] = disconnect_event
        app_called = False

        async def receive():
            return {
                "type": "http.request",
                "body": b"x",
                "more_body": True,
            }

        async def app(scope, receive, send):
            nonlocal app_called
            app_called = True

        middleware = RequestAdmissionMiddleware(app, controller=controller)
        request = asyncio.create_task(
            middleware(scope, receive, lambda message: None)
        )
        await _wait_until(
            lambda: controller.snapshot()["pending_body_reserved_bytes"] == 16
        )

        disconnect_event.set()
        await holder.release()
        await request

        snapshot = controller.snapshot()
        assert app_called is False
        assert snapshot["active"] == 0
        assert snapshot["waiters"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["pending_body_reserved_bytes"] == 0

    asyncio.run(run())


def test_final_body_then_disconnect_is_observed_while_still_queued():
    async def run():
        controller = _controller(
            max_body_bytes=64,
            body_budget_bytes=64,
            wait_timeout_seconds=1,
        )
        holder = await controller.acquire()
        messages = iter(
            [
                {
                    "type": "http.request",
                    "body": b"done",
                    "more_body": False,
                },
                {"type": "http.disconnect"},
            ]
        )
        app_called = False
        observed = []

        async def receive():
            return next(messages)

        async def app(scope, replay_receive, send):
            nonlocal app_called
            app_called = True

        async def on_early(scope, status_code, reason):
            observed.append((status_code, reason))

        middleware = RequestAdmissionMiddleware(
            app,
            controller=controller,
            on_early_response=on_early,
        )
        await middleware(_scope(), receive, lambda _message: None)

        snapshot = controller.snapshot()
        assert app_called is False
        assert observed == [(499, "disconnected_while_queued")]
        assert snapshot["active"] == 1
        assert snapshot["waiters"] == 0
        assert snapshot["pending_body_reserved_bytes"] == 0
        await holder.release()

    asyncio.run(run())


def test_queued_body_reservation_transfers_without_double_charging_on_replay():
    async def run():
        controller = _controller(
            max_body_bytes=32,
            body_budget_bytes=32,
            wait_timeout_seconds=1,
        )
        holder = await controller.acquire()
        two_messages_buffered = asyncio.Event()
        allow_third = asyncio.Event()
        messages = [
            {"type": "http.request", "body": b"ab", "more_body": True},
            {"type": "http.request", "body": b"cd", "more_body": True},
            {"type": "http.request", "body": b"ef", "more_body": False},
        ]
        received = 0

        async def receive():
            nonlocal received
            if received == 2:
                two_messages_buffered.set()
                await allow_third.wait()
            message = messages[received]
            received += 1
            return message

        observed_during_app = []

        async def app(scope, replay_receive, send):
            reserve = scope["state"][RESERVE_BODY_BYTES_STATE_KEY]
            while True:
                message = await replay_receive()
                body = message.get("body", b"")
                if body:
                    await reserve(len(body))
                    observed_during_app.append(
                        controller.snapshot()["reserved_body_bytes"]
                    )
                if not message.get("more_body", False):
                    break

        middleware = RequestAdmissionMiddleware(app, controller=controller)
        task = asyncio.create_task(
            middleware(_scope(), receive, lambda message: None)
        )
        await two_messages_buffered.wait()
        await holder.release()
        allow_third.set()
        await task

        # The third frame can complete in the same turn as the grant; all
        # three raw chunks are then already retained and must remain charged.
        assert observed_during_app == [6, 6, 6]
        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["pending_body_reserved_bytes"] == 0

    asyncio.run(run())


def test_grant_hands_off_receive_that_consumed_body_before_return():
    async def run():
        controller = _controller(
            max_body_bytes=64,
            body_budget_bytes=64,
            wait_timeout_seconds=1,
        )
        holder = await controller.acquire()
        consumed = asyncio.Event()
        allow_return = asyncio.Event()
        calls = 0
        received_by_app = None

        async def receive():
            nonlocal calls
            calls += 1
            if calls == 1:
                message = {
                    "type": "http.request",
                    "body": b"important",
                    "more_body": False,
                }
                consumed.set()
                await allow_return.wait()
                return message
            return {"type": "http.disconnect"}

        async def app(scope, replay_receive, send):
            nonlocal received_by_app
            received_by_app = await replay_receive()

        middleware = RequestAdmissionMiddleware(app, controller=controller)
        task = asyncio.create_task(
            middleware(_scope(), receive, lambda _message: None)
        )
        await consumed.wait()
        await holder.release()
        allow_return.set()
        await task

        assert calls == 1
        assert received_by_app["body"] == b"important"
        assert controller.snapshot()["active"] == 0
        assert controller.snapshot()["reserved_body_bytes"] == 0

    asyncio.run(run())


def test_replay_credit_is_released_per_message_not_all_at_once():
    async def run():
        controller = _controller(
            max_body_bytes=100,
            body_budget_bytes=100,
            wait_timeout_seconds=1,
        )
        holder = await controller.acquire()
        two_buffered = asyncio.Event()
        messages = [
            {"type": "http.request", "body": b"a" * 10, "more_body": True},
            {"type": "http.request", "body": b"b" * 10, "more_body": True},
        ]
        calls = 0

        async def receive():
            nonlocal calls
            if calls < len(messages):
                message = messages[calls]
                calls += 1
                if calls == len(messages):
                    two_buffered.set()
                return message
            await asyncio.Event().wait()

        observed = None

        async def app(scope, replay_receive, send):
            nonlocal observed
            first = await replay_receive()
            assert first["body"] == b"a" * 10
            # Simulate decompression/materialization weighting the delivered
            # first chunk to 40 bytes.  The second raw 10-byte message remains
            # independently charged in the replay deque.
            await scope["state"][RESERVE_BODY_BYTES_STATE_KEY](40)
            observed = controller.snapshot()["reserved_body_bytes"]

        middleware = RequestAdmissionMiddleware(app, controller=controller)
        task = asyncio.create_task(
            middleware(_scope(), receive, lambda _message: None)
        )
        await two_buffered.wait()
        await holder.release()
        await task

        assert observed == 50
        assert controller.snapshot()["reserved_body_bytes"] == 0

    asyncio.run(run())


def test_exactly_sixteen_queued_messages_with_final_body_are_accepted():
    async def run():
        controller = _controller(
            max_body_bytes=64,
            body_budget_bytes=64,
            wait_timeout_seconds=1,
        )
        holder = await controller.acquire()
        all_buffered = asyncio.Event()
        delivered = 0

        async def receive():
            nonlocal delivered
            if delivered >= 16:
                await asyncio.Event().wait()
            delivered += 1
            if delivered == 16:
                all_buffered.set()
            return {
                "type": "http.request",
                "body": b"x",
                "more_body": delivered < 16,
            }

        observed = []

        async def app(scope, replay_receive, send):
            while True:
                message = await replay_receive()
                observed.append(message["body"])
                await scope["state"][RESERVE_BODY_BYTES_STATE_KEY](1)
                if not message.get("more_body", False):
                    break

        middleware = RequestAdmissionMiddleware(app, controller=controller)
        task = asyncio.create_task(
            middleware(_scope(), receive, lambda _message: None)
        )
        await all_buffered.wait()
        await holder.release()
        await task

        assert delivered == 16
        assert observed == [b"x"] * 16
        assert controller.snapshot()["active"] == 0

    asyncio.run(run())


def test_queue_full_rejection_never_starts_request_receive():
    async def run():
        controller = _controller(waiter_limit=0)
        holder = await controller.acquire()
        receive_calls = 0
        sent = []

        async def receive():
            nonlocal receive_calls
            receive_calls += 1
            raise AssertionError("queue-full request must not read a body")

        async def send(message):
            sent.append(message)

        middleware = RequestAdmissionMiddleware(
            lambda scope, receive, send: None,
            controller=controller,
        )
        await middleware(_scope(), receive, send)

        assert receive_calls == 0
        assert _response(sent)[0] == 503
        assert controller.snapshot()["reserved_body_bytes"] == 0
        await holder.release()

    asyncio.run(run())


def test_repeated_cancel_cannot_interrupt_queued_ownership_transaction(
    monkeypatch,
):
    async def run():
        controller = _controller(
            max_body_bytes=64,
            body_budget_bytes=64,
            wait_timeout_seconds=1,
        )
        holder = await controller.acquire()
        captured_acquisition = {}
        original_begin = controller.begin_acquire
        second_receive = asyncio.Event()
        receive_calls = 0

        async def capture_begin(*, timeout_seconds=None):
            acquisition = await original_begin(timeout_seconds=timeout_seconds)
            captured_acquisition["future"] = acquisition
            return acquisition

        monkeypatch.setattr(controller, "begin_acquire", capture_begin)

        async def receive():
            nonlocal receive_calls
            receive_calls += 1
            if receive_calls == 1:
                return {
                    "type": "http.request",
                    "body": b"12345678",
                    "more_body": True,
                }
            await second_receive.wait()
            return {"type": "http.disconnect"}

        middleware = RequestAdmissionMiddleware(
            lambda scope, receive, send: None,
            controller=controller,
        )
        queued = asyncio.create_task(
            middleware._acquire_queued_request(_scope(), receive)
        )
        await _wait_until(
            lambda: controller.snapshot()["pending_body_reserved_bytes"] == 8
            and "future" in captured_acquisition
        )

        release_started = asyncio.Event()
        allow_release = asyncio.Event()

        def cancel_owner_on_grant(done):
            granted = done.result()
            original_release = granted.release

            async def delayed_release():
                release_started.set()
                await allow_release.wait()
                await original_release()

            granted.release = delayed_release
            queued.cancel()

        captured_acquisition["future"].add_done_callback(cancel_owner_on_grant)
        await holder.release()
        await asyncio.wait_for(release_started.wait(), timeout=1)
        queued.cancel()
        queued.cancel()
        allow_release.set()
        with pytest.raises(asyncio.CancelledError):
            await queued

        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["waiters"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["pending_body_reserved_bytes"] == 0

    asyncio.run(run())
