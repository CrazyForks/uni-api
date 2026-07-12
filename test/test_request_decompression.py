import asyncio
import io
import json
import threading

import httpx
import pytest
import zstandard as zstd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.types import Message, Scope

import main
import uni_api.middleware.request_decompression as request_decompression
from uni_api.middleware.request_decompression import (
    DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY,
    RequestBodyDecompressionMiddleware,
    RequestBodyReadTimeout,
)


def _zstd_compress(body: bytes) -> bytes:
    return zstd.ZstdCompressor(level=3).compress(body)


async def _echo_asgi(scope: Scope, receive, send) -> None:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break

    response = JSONResponse(
        {
            "body": b"".join(chunks).decode("utf-8"),
            "content_encoding": _scope_header(scope, b"content-encoding"),
            "content_length": _scope_header(scope, b"content-length"),
        }
    )
    await response(scope, receive, send)


def _scope_header(scope: Scope, wanted: bytes) -> str | None:
    for name, value in scope.get("headers") or []:
        if name.lower() == wanted:
            return value.decode("latin-1")
    return None


async def _run_asgi(
    app,
    messages: list[Message],
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    state: dict | None = None,
    path: str = "/echo",
    http_version: str = "1.1",
) -> list[Message]:
    pending = list(messages)
    sent: list[Message] = []

    async def receive() -> Message:
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": http_version,
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": state or {},
    }
    await app(scope, receive, send)
    return sent


def _asgi_response(messages: list[Message]) -> tuple[int, dict]:
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return start["status"], json.loads(body)


def test_identity_body_timeout_stops_after_complete_body():
    async def scenario():
        observed = []
        receive_queue = asyncio.Queue()
        await receive_queue.put(
            {"type": "http.request", "body": b"{}", "more_body": False}
        )

        async def inner(_scope, receive, _send):
            observed.append(await receive())
            observed.append(await receive())

        middleware = RequestBodyDecompressionMiddleware(
            inner,
            body_idle_timeout_seconds=0.01,
            body_total_timeout_seconds=0.02,
        )

        async def receive():
            return await receive_queue.get()

        async def delayed_disconnect():
            await asyncio.sleep(0.03)
            await receive_queue.put({"type": "http.disconnect"})

        disconnect_task = asyncio.create_task(delayed_disconnect())
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/echo",
                "headers": [],
                "state": {},
            },
            receive,
            lambda _message: None,
        )
        await disconnect_task

        assert observed[0]["type"] == "http.request"
        assert observed[1]["type"] == "http.disconnect"

    asyncio.run(scenario())


def test_zstd_middleware_decodes_body_and_strips_encoding_headers():
    app = FastAPI()
    app.add_middleware(RequestBodyDecompressionMiddleware)

    @app.post("/echo")
    async def echo(request: Request):
        return {
            "body": (await request.body()).decode("utf-8"),
            "content_encoding": request.headers.get("content-encoding"),
            "content_length": request.headers.get("content-length"),
        }

    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/echo",
                content=_zstd_compress(b'{"ok":true}'),
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "zstd",
                    "Content-Length": "999",
                },
            )

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert response.json() == {
        "body": '{"ok":true}',
        "content_encoding": None,
        "content_length": None,
    }


def test_identity_body_at_exact_limit_is_accepted_across_chunks():
    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_identity_body_bytes=4,
    )

    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {"type": "http.request", "body": b"ab", "more_body": True},
                {"type": "http.request", "body": b"cd", "more_body": False},
            ],
        )
    )

    assert _asgi_response(messages) == (
        200,
        {
            "body": "abcd",
            "content_encoding": None,
            "content_length": None,
        },
    )


def test_identity_content_length_over_limit_is_rejected_before_receive():
    app_called = False
    receive_called = False
    sent: list[Message] = []

    async def downstream(scope, receive, send):
        nonlocal app_called
        app_called = True

    async def receive() -> Message:
        nonlocal receive_called
        receive_called = True
        raise AssertionError("oversized body must be rejected before receive")

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/echo",
        "headers": [(b"content-length", b"5")],
    }
    middleware = RequestBodyDecompressionMiddleware(
        downstream,
        max_identity_body_bytes=4,
    )

    asyncio.run(middleware(scope, receive, send))

    assert _asgi_response(sent) == (413, {"detail": "request body too large"})
    assert app_called is False
    assert receive_called is False


def test_identity_chunked_body_over_limit_is_rejected_cumulatively():
    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_identity_body_bytes=4,
    )

    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {"type": "http.request", "body": b"abc", "more_body": True},
                {"type": "http.request", "body": b"de", "more_body": False},
            ],
            headers=[(b"transfer-encoding", b"chunked")],
        )
    )

    assert _asgi_response(messages) == (413, {"detail": "request body too large"})
    start = next(message for message in messages if message["type"] == "http.response.start")
    assert dict(start["headers"])[b"connection"] == b"close"


def test_body_rejection_does_not_emit_illegal_http2_connection_header():
    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_identity_body_bytes=4,
    )

    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {"type": "http.request", "body": b"abc", "more_body": True},
                {"type": "http.request", "body": b"de", "more_body": False},
            ],
            headers=[(b"transfer-encoding", b"chunked")],
            http_version="2",
        )
    )

    start = next(message for message in messages if message["type"] == "http.response.start")
    assert b"connection" not in dict(start["headers"])


def test_identity_body_reservation_callback_receives_each_chunk():
    reserved: list[int] = []

    async def reserve_body_bytes(size: int) -> None:
        reserved.append(size)

    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_identity_body_bytes=4,
    )
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {"type": "http.request", "body": b"ab", "more_body": True},
                {"type": "http.request", "body": b"cd", "more_body": False},
            ],
            state={"uni_api_reserve_body_bytes": reserve_body_bytes},
        )
    )

    assert _asgi_response(messages)[0] == 200
    assert reserved == [8, 8]


def test_json_body_reservation_charges_conservative_memory_weight():
    reserved: list[int] = []

    async def reserve_body_bytes(size: int) -> None:
        reserved.append(size)

    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_identity_body_bytes=8,
    )
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {"type": "http.request", "body": b'{"a"', "more_body": True},
                {"type": "http.request", "body": b":1}", "more_body": False},
            ],
            headers=[(b"content-type", b"application/json")],
            state={"uni_api_reserve_body_bytes": reserve_body_bytes},
        )
    )

    assert _asgi_response(messages)[0] == 200
    assert reserved == [2068, 1039]


def test_known_json_route_without_content_type_cannot_bypass_structure_charge():
    reserved: list[int] = []

    async def reserve_body_bytes(size: int) -> None:
        reserved.append(size)

    middleware = RequestBodyDecompressionMiddleware(_echo_asgi)
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [{"type": "http.request", "body": b'{"a":1}', "more_body": False}],
            state={"uni_api_reserve_body_bytes": reserve_body_bytes},
            path="/v1/chat/completions",
        )
    )

    assert _asgi_response(messages)[0] == 200
    assert sum(reserved) == 3107


def test_image_edit_json_route_without_content_type_uses_structure_charge():
    reserved: list[int] = []

    async def reserve_body_bytes(size: int) -> None:
        reserved.append(size)

    middleware = RequestBodyDecompressionMiddleware(_echo_asgi)
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [{"type": "http.request", "body": b'{"prompt":"x"}', "more_body": False}],
            state={"uni_api_reserve_body_bytes": reserve_body_bytes},
            path="/v1/images/edits",
        )
    )

    assert _asgi_response(messages)[0] == 200
    assert sum(reserved) > len(b'{"prompt":"x"}') * 4


def test_duplicate_content_type_is_rejected_before_body_dispatch():
    middleware = RequestBodyDecompressionMiddleware(_echo_asgi)
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [{"type": "http.request", "body": b'{"prompt":"x"}', "more_body": False}],
            headers=[
                (b"content-type", b"application/json"),
                (b"content-type", b"text/plain"),
            ],
            path="/v1/images/edits",
        )
    )

    status, body = _asgi_response(messages)
    assert status == 400
    assert "multiple content-type" in body["detail"]


def test_decompression_receive_failure_does_not_fabricate_disconnect():
    async def scenario():
        event = asyncio.Event()

        async def receive():
            raise RuntimeError("adapter failed")

        await request_decompression._monitor_disconnect(receive, event)
        assert not event.is_set()

    asyncio.run(scenario())


def test_body_cpu_cancellation_wins_over_late_worker_error():
    async def scenario():
        started = threading.Event()
        release = threading.Event()

        def fail_later():
            started.set()
            release.wait(timeout=2)
            raise ValueError("late decoder error")

        task = asyncio.create_task(request_decompression._run_body_cpu(fail_later))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_zstd_known_json_route_without_content_type_charges_decoded_structure():
    reserved: list[int] = []
    payload = b'{"a":1}'
    compressed = _zstd_compress(payload)

    async def reserve_body_bytes(size: int) -> None:
        reserved.append(size)

    middleware = RequestBodyDecompressionMiddleware(_echo_asgi)
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [{"type": "http.request", "body": compressed, "more_body": False}],
            headers=[(b"content-encoding", b"zstd")],
            state={"uni_api_reserve_body_bytes": reserve_body_bytes},
            path="/v1/responses",
        )
    )

    assert _asgi_response(messages)[0] == 200
    assert reserved == [
        len(compressed),
        zstd.get_frame_parameters(compressed).window_size,
        3107,
    ]


def test_explicit_non_json_content_type_keeps_finite_raw_body_charge():
    reserved: list[int] = []

    async def reserve_body_bytes(size: int) -> None:
        reserved.append(size)

    middleware = RequestBodyDecompressionMiddleware(_echo_asgi)
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [{"type": "http.request", "body": b'{"a":1}', "more_body": False}],
            headers=[(b"content-type", b"text/plain")],
            state={"uni_api_reserve_body_bytes": reserve_body_bytes},
            path="/v1/chat/completions",
        )
    )

    assert _asgi_response(messages)[0] == 200
    assert reserved == [28]


@pytest.mark.parametrize(
    "content_type",
    ["application/jsonp", "text/plain; x=application/json"],
)
def test_json_substrings_do_not_select_json_parsing_or_structure_charge(
    content_type,
):
    reserved: list[int] = []
    payload = b"[{},{}]"

    async def reserve_body_bytes(size: int) -> None:
        reserved.append(size)

    middleware = RequestBodyDecompressionMiddleware(_echo_asgi)
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [{"type": "http.request", "body": payload, "more_body": False}],
            headers=[(b"content-type", content_type.encode())],
            state={"uni_api_reserve_body_bytes": reserve_body_bytes},
            path="/v1/chat/completions",
        )
    )

    assert _asgi_response(messages)[0] == 200
    assert reserved == [len(payload) * 4]


def test_zstd_json_substring_cannot_enter_runtime_json_parser():
    reserved: list[int] = []
    payload = b"[{},{}]"
    compressed = _zstd_compress(payload)

    async def reserve_body_bytes(size: int) -> None:
        reserved.append(size)

    middleware = RequestBodyDecompressionMiddleware(_echo_asgi)
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [{"type": "http.request", "body": compressed, "more_body": False}],
            headers=[
                (b"content-encoding", b"zstd"),
                (b"content-type", b"application/jsonp"),
            ],
            state={"uni_api_reserve_body_bytes": reserve_body_bytes},
            path="/v1/responses",
        )
    )

    assert _asgi_response(messages)[0] == 200
    assert reserved == [
        len(compressed),
        zstd.get_frame_parameters(compressed).window_size,
        len(payload) * 4,
    ]


def test_complete_identity_body_starts_one_sticky_disconnect_monitor():
    async def scenario():
        receive_queue = asyncio.Queue()
        await receive_queue.put(
            {"type": "http.request", "body": b"{}", "more_body": False}
        )
        await receive_queue.put({"type": "http.disconnect"})
        observed_event = None

        async def inner(scope, receive, send):
            nonlocal observed_event
            assert (await receive())["type"] == "http.request"
            observed_event = scope["state"][DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY]
            await asyncio.wait_for(observed_event.wait(), timeout=1)

        middleware = RequestBodyDecompressionMiddleware(inner)

        async def receive():
            return await receive_queue.get()

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/echo",
            "headers": [(b"content-type", b"application/json")],
            "state": {},
        }
        await middleware(scope, receive, lambda message: None)
        assert observed_event is not None and observed_event.is_set()

    asyncio.run(scenario())


def test_bodyless_get_starts_sticky_disconnect_monitor_without_app_receive():
    async def scenario():
        disconnected = asyncio.Event()

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def inner(scope, _receive, _send):
            event = scope["state"][DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY]
            disconnected.set()
            await asyncio.wait_for(event.wait(), timeout=1)

        middleware = RequestBodyDecompressionMiddleware(inner)
        await middleware(
            {
                "type": "http",
                "method": "GET",
                "path": "/v1/models",
                "headers": [],
                "state": {},
            },
            receive,
            lambda message: None,
        )

    asyncio.run(scenario())


def test_declared_bodyless_requests_still_deliver_one_empty_asgi_request():
    async def scenario(method, headers):
        raw_messages = asyncio.Queue()
        await raw_messages.put(
            {"type": "http.request", "body": b"", "more_body": False}
        )
        await raw_messages.put({"type": "http.disconnect"})
        observed = []

        async def receive():
            return await raw_messages.get()

        async def inner(scope, downstream_receive, _send):
            observed.append(await asyncio.wait_for(downstream_receive(), timeout=1))
            event = scope["state"][DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY]
            await asyncio.wait_for(event.wait(), timeout=1)

        middleware = RequestBodyDecompressionMiddleware(inner)
        await middleware(
            {
                "type": "http",
                "method": method,
                "path": "/empty",
                "headers": headers,
                "state": {},
            },
            receive,
            lambda message: None,
        )
        assert observed == [
            {"type": "http.request", "body": b"", "more_body": False}
        ]

    for method, headers in [
        ("GET", []),
        ("HEAD", []),
        ("OPTIONS", []),
        ("POST", [(b"content-length", b"0")]),
    ]:
        asyncio.run(scenario(method, headers))


def test_body_timeout_after_response_start_never_sends_a_second_response_start():
    async def scenario():
        sent = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)

        async def inner(_scope, downstream_receive, downstream_send):
            await downstream_send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await downstream_receive()

        middleware = RequestBodyDecompressionMiddleware(
            inner,
            body_idle_timeout_seconds=0.01,
            body_total_timeout_seconds=0.02,
        )
        with pytest.raises(RequestBodyReadTimeout):
            await middleware(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/timeout",
                    "headers": [(b"transfer-encoding", b"chunked")],
                    "state": {},
                },
                receive,
                send,
            )
        assert [
            message
            for message in sent
            if message["type"] == "http.response.start"
        ] == [{"type": "http.response.start", "status": 200, "headers": []}]

    asyncio.run(scenario())


def test_identity_disconnect_is_forwarded_without_a_synthetic_response():
    reserved: list[int] = []

    async def reserve_body_bytes(size: int) -> None:
        reserved.append(size)

    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_identity_body_bytes=4,
    )
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {"type": "http.request", "body": b"ab", "more_body": True},
                {"type": "http.disconnect"},
            ],
            state={"uni_api_reserve_body_bytes": reserve_body_bytes},
        )
    )

    assert messages == []
    assert reserved == [8]


def test_body_reservation_callback_rejection_propagates_before_downstream():
    class ReservationRejected(Exception):
        pass

    app_called = False

    async def downstream(scope, receive, send):
        nonlocal app_called
        app_called = True
        await receive()

    async def reject_reservation(size: int) -> None:
        raise ReservationRejected(size)

    middleware = RequestBodyDecompressionMiddleware(
        downstream,
        max_identity_body_bytes=4,
    )

    async def run_request() -> None:
        await _run_asgi(
            middleware,
            [{"type": "http.request", "body": b"a", "more_body": False}],
            state={"uni_api_reserve_body_bytes": reject_reservation},
        )

    try:
        asyncio.run(run_request())
    except ReservationRejected as exc:
        assert exc.args == (4,)
    else:
        raise AssertionError("reservation rejection must propagate")
    assert app_called is True


def test_invalid_content_length_is_rejected_for_identity_and_zstd():
    for content_length, content_encoding in [
        (b"not-a-number", None),
        (b"-1", None),
        (b"3, 4", None),
        (b"9" * 5000, None),
        (b"not-a-number", b"zstd"),
    ]:
        headers = [(b"content-length", content_length)]
        if content_encoding is not None:
            headers.append((b"content-encoding", content_encoding))
        middleware = RequestBodyDecompressionMiddleware(
            _echo_asgi,
            max_identity_body_bytes=10,
            max_zstd_compressed_body_bytes=10,
        )

        messages = asyncio.run(
            _run_asgi(
                middleware,
                [{"type": "http.request", "body": b"", "more_body": False}],
                headers=headers,
            )
        )

        assert _asgi_response(messages) == (
            400,
            {"detail": "invalid content-length"},
        )


def test_zstd_body_at_exact_decoded_limit_is_accepted_across_wire_chunks():
    body = b"abcd"
    compressed = _zstd_compress(body)
    split_at = max(1, len(compressed) // 2)
    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_zstd_compressed_body_bytes=len(compressed),
        max_zstd_decompressed_body_bytes=len(body),
    )

    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {
                    "type": "http.request",
                    "body": compressed[:split_at],
                    "more_body": True,
                },
                {
                    "type": "http.request",
                    "body": compressed[split_at:],
                    "more_body": False,
                },
            ],
            headers=[
                (b"content-encoding", b"zstd"),
                (b"content-length", str(len(compressed)).encode("ascii")),
            ],
        )
    )

    assert _asgi_response(messages) == (
        200,
        {
            "body": "abcd",
            "content_encoding": None,
            "content_length": None,
        },
    )


def test_zstd_reserves_wire_chunks_and_decoded_bytes():
    body = b"decoded body"
    compressed = _zstd_compress(body)
    split_at = max(1, len(compressed) // 2)
    reserved: list[int] = []

    async def reserve_body_bytes(size: int) -> None:
        reserved.append(size)

    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_zstd_compressed_body_bytes=len(compressed),
        max_zstd_decompressed_body_bytes=len(body),
    )
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {
                    "type": "http.request",
                    "body": compressed[:split_at],
                    "more_body": True,
                },
                {
                    "type": "http.request",
                    "body": compressed[split_at:],
                    "more_body": False,
                },
            ],
            headers=[(b"content-encoding", b"zstd")],
            state={"uni_api_reserve_body_bytes": reserve_body_bytes},
        )
    )

    assert _asgi_response(messages)[0] == 200
    assert reserved == [
        split_at,
        len(compressed) - split_at,
        zstd.get_frame_parameters(compressed).window_size,
        len(body) * 4,
    ]


def test_zstd_decoded_reservation_rejection_propagates_unchanged():
    class ReservationRejected(Exception):
        pass

    body = b"decoded body"
    compressed = _zstd_compress(body)
    rejection = ReservationRejected("decoded budget exhausted")
    calls = 0

    async def reserve_body_bytes(size: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise rejection

    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_zstd_compressed_body_bytes=len(compressed),
        max_zstd_decompressed_body_bytes=len(body),
    )

    async def run_request() -> None:
        await _run_asgi(
            middleware,
            [{"type": "http.request", "body": compressed, "more_body": False}],
            headers=[(b"content-encoding", b"zstd")],
            state={"uni_api_reserve_body_bytes": reserve_body_bytes},
        )

    try:
        asyncio.run(run_request())
    except ReservationRejected as exc:
        assert exc is rejection
    else:
        raise AssertionError("decoded reservation rejection must propagate")
    assert calls == 2


def test_zstd_compressed_wire_limit_is_enforced_across_chunks():
    compressed = _zstd_compress(b"abcd")
    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_zstd_compressed_body_bytes=len(compressed) - 1,
        max_zstd_decompressed_body_bytes=100,
    )

    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {
                    "type": "http.request",
                    "body": compressed[:-1],
                    "more_body": True,
                },
                {
                    "type": "http.request",
                    "body": compressed[-1:],
                    "more_body": False,
                },
            ],
            headers=[(b"content-encoding", b"zstd")],
        )
    )

    assert _asgi_response(messages) == (413, {"detail": "request body too large"})


def test_zstd_decoder_rejects_tiny_frame_with_oversized_window():
    sink = io.BytesIO()
    parameters = zstd.ZstdCompressionParameters(window_log=27)
    with zstd.ZstdCompressor(compression_params=parameters).stream_writer(
        sink,
        closefd=False,
        size=-1,
    ) as writer:
        writer.write(b"x")
    compressed = sink.getvalue()
    assert zstd.get_frame_parameters(compressed).window_size == 128 * 1024 * 1024

    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_zstd_compressed_body_bytes=1024,
        max_zstd_decompressed_body_bytes=64 * 1024 * 1024,
    )
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [{"type": "http.request", "body": compressed, "more_body": False}],
            headers=[(b"content-encoding", b"zstd")],
        )
    )

    assert _asgi_response(messages) == (400, {"detail": "invalid zstd body"})


def test_zstd_decompressed_body_limit_rejects_compression_bomb():
    body = b"a" * 1024 * 1024
    compressed = _zstd_compress(body)
    assert len(compressed) < 1024
    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_zstd_compressed_body_bytes=len(compressed),
        max_zstd_decompressed_body_bytes=1024,
    )

    messages = asyncio.run(
        _run_asgi(
            middleware,
            [{"type": "http.request", "body": compressed, "more_body": False}],
            headers=[(b"content-encoding", b"zstd")],
        )
    )

    assert _asgi_response(messages) == (413, {"detail": "request body too large"})


def test_zstd_disconnect_does_not_decode_partial_body_or_call_app():
    app_called = False
    reserved: list[int] = []

    async def downstream(scope, receive, send):
        nonlocal app_called
        app_called = True

    async def reserve_body_bytes(size: int) -> None:
        reserved.append(size)

    compressed = _zstd_compress(b"body")
    middleware = RequestBodyDecompressionMiddleware(
        downstream,
        max_zstd_compressed_body_bytes=len(compressed),
        max_zstd_decompressed_body_bytes=4,
    )

    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {
                    "type": "http.request",
                    "body": compressed[:2],
                    "more_body": True,
                },
                {"type": "http.disconnect"},
            ],
            headers=[(b"content-encoding", b"zstd")],
            state={"uni_api_reserve_body_bytes": reserve_body_bytes},
        )
    )

    assert messages == []
    assert app_called is False
    assert reserved == [2]


def test_zstd_limit_environment_is_backwards_compatible(monkeypatch):
    monkeypatch.setenv("ZSTD_REQUEST_MAX_BODY_BYTES", "17")
    monkeypatch.delenv("ZSTD_REQUEST_MAX_COMPRESSED_BODY_BYTES", raising=False)
    monkeypatch.delenv("ZSTD_REQUEST_MAX_DECOMPRESSED_BODY_BYTES", raising=False)

    middleware = RequestBodyDecompressionMiddleware(_echo_asgi)

    assert middleware.max_body_bytes == 17
    assert middleware.max_zstd_compressed_body_bytes == 17
    assert middleware.max_zstd_decompressed_body_bytes == 17


def test_specific_body_limit_environment_overrides_legacy_limit(monkeypatch):
    monkeypatch.setenv("REQUEST_MAX_BODY_BYTES", "11")
    monkeypatch.setenv("ZSTD_REQUEST_MAX_BODY_BYTES", "17")
    monkeypatch.setenv("ZSTD_REQUEST_MAX_COMPRESSED_BODY_BYTES", "13")
    monkeypatch.setenv("ZSTD_REQUEST_MAX_DECOMPRESSED_BODY_BYTES", "19")

    middleware = RequestBodyDecompressionMiddleware(_echo_asgi)

    assert middleware.max_identity_body_bytes == 11
    assert middleware.max_zstd_compressed_body_bytes == 13
    assert middleware.max_zstd_decompressed_body_bytes == 19


def test_zstd_middleware_rejects_invalid_zstd_body():
    app = FastAPI()
    app.add_middleware(RequestBodyDecompressionMiddleware)

    @app.post("/echo")
    async def echo():
        return {"ok": True}

    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/echo",
                content=b"not-zstd",
                headers={"Content-Encoding": "zstd"},
            )

    response = asyncio.run(run_request())

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid zstd body"}


def test_zstd_middleware_rejects_truncated_frame():
    app = FastAPI()
    app.add_middleware(RequestBodyDecompressionMiddleware)

    @app.post("/echo")
    async def echo():
        return {"ok": True}

    compressed = _zstd_compress(b"truncated body")

    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/echo",
                content=compressed[:-1],
                headers={"Content-Encoding": "zstd"},
            )

    response = asyncio.run(run_request())

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid zstd body"}


def test_zstd_middleware_accepts_concatenated_frames_within_total_limit():
    first = _zstd_compress(b"abc")
    second = _zstd_compress(b"defg")
    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_zstd_compressed_body_bytes=len(first) + len(second),
        max_zstd_decompressed_body_bytes=7,
    )

    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {
                    "type": "http.request",
                    "body": first + second,
                    "more_body": False,
                }
            ],
            headers=[(b"content-encoding", b"zstd")],
        )
    )

    assert _asgi_response(messages) == (
        200,
        {
            "body": "abcdefg",
            "content_encoding": None,
            "content_length": None,
        },
    )


def test_zstd_middleware_enforces_decoded_limit_across_concatenated_frames():
    first = _zstd_compress(b"abc")
    second = _zstd_compress(b"defg")
    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_zstd_compressed_body_bytes=len(first) + len(second),
        max_zstd_decompressed_body_bytes=6,
    )

    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {
                    "type": "http.request",
                    "body": first + second,
                    "more_body": False,
                }
            ],
            headers=[(b"content-encoding", b"zstd")],
        )
    )

    assert _asgi_response(messages) == (413, {"detail": "request body too large"})


def test_zstd_middleware_rejects_truncated_unknown_content_size_frame():
    compressed = zstd.ZstdCompressor(
        level=3,
        write_content_size=False,
    ).compress(b"unknown-size body" * 100)
    assert zstd.frame_content_size(compressed) < 0

    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_zstd_compressed_body_bytes=len(compressed),
        max_zstd_decompressed_body_bytes=4096,
    )
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {
                    "type": "http.request",
                    "body": compressed[:-1],
                    "more_body": False,
                }
            ],
            headers=[(b"content-encoding", b"zstd")],
        )
    )

    assert _asgi_response(messages) == (400, {"detail": "invalid zstd body"})


def test_zstd_middleware_accepts_complete_unknown_content_size_frame():
    body = b"unknown-size body" * 100
    compressed = zstd.ZstdCompressor(
        level=3,
        write_content_size=False,
    ).compress(body)
    assert zstd.frame_content_size(compressed) < 0

    middleware = RequestBodyDecompressionMiddleware(
        _echo_asgi,
        max_zstd_compressed_body_bytes=len(compressed),
        max_zstd_decompressed_body_bytes=len(body),
    )
    messages = asyncio.run(
        _run_asgi(
            middleware,
            [
                {
                    "type": "http.request",
                    "body": compressed,
                    "more_body": False,
                }
            ],
            headers=[(b"content-encoding", b"zstd")],
        )
    )

    assert _asgi_response(messages)[0] == 200


def test_zstd_middleware_rejects_unsupported_content_encoding():
    app = FastAPI()
    app.add_middleware(RequestBodyDecompressionMiddleware)

    @app.post("/echo")
    async def echo():
        return {"ok": True}

    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/echo",
                content=b"body",
                headers={"Content-Encoding": "gzip"},
            )

    response = asyncio.run(run_request())

    assert response.status_code == 415
    assert response.json() == {"detail": "unsupported content encoding: gzip"}


def test_main_app_accepts_zstd_chat_completion_request(monkeypatch):
    monkeypatch.setattr(main, "DISABLE_DATABASE", True)
    main.app.state.config = {
        "api_keys": [{"api": "sk-test", "model": ["all"]}],
        "preferences": {"rate_limit": "999999/min"},
    }
    main.app.state.api_list = ["sk-test"]
    main.app.state.api_keys_db = [{"api": "sk-test"}]
    main.app.state.user_api_keys_rate_limit = main._build_user_api_keys_rate_limit(
        main.app.state.config,
        main.app.state.api_list,
    )

    async def fake_request_model(request, api_index, background_tasks, endpoint=None, current_info=None, http_request=None):
        _ = http_request
        assert api_index == 0
        assert endpoint is None
        assert current_info["model"] == "gpt-5.5"
        return JSONResponse({"model": request.model, "message": request.messages[0].content})

    monkeypatch.setattr(main.model_handler, "request_model", fake_request_model)
    payload = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "zstd request"}],
        "stream": False,
    }

    async def run_request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/v1/chat/completions",
                content=_zstd_compress(json.dumps(payload).encode("utf-8")),
                headers={
                    "Authorization": "Bearer sk-test",
                    "Content-Type": "application/json",
                    "Content-Encoding": "zstd",
                },
            )

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert response.json() == {"model": "gpt-5.5", "message": "zstd request"}


def test_main_app_rejects_overly_nested_identity_json_as_413(monkeypatch):
    monkeypatch.setattr(main, "DISABLE_DATABASE", True)
    main.app.state.config = {
        "api_keys": [{"api": "sk-test", "model": ["all"]}],
        "preferences": {"rate_limit": "999999/min"},
    }
    main.app.state.api_list = ["sk-test"]
    main.app.state.api_keys_db = [{"api": "sk-test"}]
    main.app.state.user_api_keys_rate_limit = main._build_user_api_keys_rate_limit(
        main.app.state.config,
        main.app.state.api_list,
    )
    handler_called = False
    emitted = []

    def capture_observability(current_info, runtime_metrics):
        emitted.append((dict(current_info), dict(runtime_metrics)))

    monkeypatch.setattr(main, "emit_request_observability", capture_observability)

    async def fake_request_model(*_args, **_kwargs):
        nonlocal handler_called
        handler_called = True
        return JSONResponse({"unexpected": True})

    monkeypatch.setattr(main.model_handler, "request_model", fake_request_model)
    nested_value = "[" * 129 + "0" + "]" * 129
    body = (
        '{"model":"gpt-5.5","messages":'
        + nested_value
        + ",\"stream\":false}"
    )

    async def run_request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/chat/completions",
                content=body,
                headers={
                    "Authorization": "Bearer sk-test",
                    "Content-Type": "application/json",
                },
            )

    response = asyncio.run(run_request())

    assert response.status_code == 413
    assert response.json() == {"detail": "request body too complex"}
    assert response.headers["x-uni-api-admission-reason"] == "body_too_complex"
    assert handler_called is False
    assert len(emitted) == 1
    assert emitted[0][0]["status_code"] == 413
    assert emitted[0][0]["admission_reason"] == "body_too_complex"


def test_main_app_accepts_zstd_responses_request(monkeypatch):
    monkeypatch.setattr(main, "DISABLE_DATABASE", True)
    main.app.state.config = {
        "api_keys": [{"api": "sk-test", "model": ["all"]}],
        "preferences": {"rate_limit": "999999/min"},
    }
    main.app.state.api_list = ["sk-test"]
    main.app.state.api_keys_db = [{"api": "sk-test"}]
    main.app.state.user_api_keys_rate_limit = main._build_user_api_keys_rate_limit(
        main.app.state.config,
        main.app.state.api_list,
    )

    async def fake_request_responses(
        http_request,
        request,
        api_index,
        background_tasks,
        endpoint="/v1/responses",
    ):
        assert api_index == 0
        assert endpoint == "/v1/responses"
        assert http_request.headers.get("content-encoding") is None
        assert request.model == "gpt-5.5"
        assert request.input == "zstd responses request"
        return JSONResponse({"model": request.model, "input": request.input})

    monkeypatch.setattr(main.responses_handler, "request_responses", fake_request_responses)
    payload = {
        "model": "gpt-5.5",
        "input": "zstd responses request",
        "stream": False,
    }

    async def run_request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/v1/responses",
                content=_zstd_compress(json.dumps(payload).encode("utf-8")),
                headers={
                    "Authorization": "Bearer sk-test",
                    "Content-Type": "application/json",
                    "Content-Encoding": "zstd",
                },
            )

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert response.json() == {"model": "gpt-5.5", "input": "zstd responses request"}
