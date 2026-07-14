import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

import main
import uni_api.providers.responses as response_module
from core.utils import collect_openai_chat_completion_from_streaming_sse
from uni_api.streaming.cleanup import (
    await_stream_cleanup_safely,
    background_stream_cleanup_snapshot,
    force_close_response_httpcore_stream_chain_safely,
    track_background_stream_cleanup_task,
    wait_background_stream_cleanup_tasks,
)
import uni_api.streaming.cleanup as stream_cleanup


async def _mark_first_byte_wrapper_closes_inner_generator():
    closed = False

    async def gen():
        nonlocal closed
        try:
            yield ": keepalive\n\n"
            await asyncio.Event().wait()
        finally:
            closed = True

    wrapped = main._mark_first_byte_on_stream(gen(), {}, skip_keepalive=True)
    first = await wrapped.__anext__()
    assert first == ": keepalive\n\n"

    await wrapped.aclose()
    assert closed


def test_mark_first_byte_wrapper_closes_inner_generator():
    asyncio.run(_mark_first_byte_wrapper_closes_inner_generator())


async def _stream_collector_closes_generator_on_done():
    closed = False

    async def gen():
        nonlocal closed
        try:
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            yield "data: [DONE]\n\n"
            await asyncio.Event().wait()
        finally:
            closed = True

    body = await collect_openai_chat_completion_from_streaming_sse(
        gen(),
        model="gpt-5.5",
    )

    assert '"content": "ok"' in body
    assert closed


def test_stream_collector_closes_generator_on_done():
    asyncio.run(_stream_collector_closes_generator_on_done())


async def _logging_response_closes_body_when_final_send_is_cancelled():
    closed = False

    async def body():
        nonlocal closed
        try:
            yield "data: ok\n\n"
        finally:
            closed = True

    async def send(message):
        if message["type"] == "http.response.body" and not message.get("body"):
            raise asyncio.CancelledError()

    response = main.LoggingStreamingResponse(
        body(),
        media_type="text/event-stream",
        current_info={
            "start_time": 0,
            "endpoint": "POST /v1/chat/completions",
            "request_id": "request",
            "trace_id": "trace",
        },
    )

    try:
        await response({}, None, send)
    except asyncio.CancelledError:
        pass

    assert closed


def test_logging_response_closes_body_when_final_send_is_cancelled():
    asyncio.run(_logging_response_closes_body_when_final_send_is_cancelled())


async def _logging_response_records_stats_after_stream_finishes():
    recorded = []

    async def body():
        yield 'data: {"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}\n\n'

    async def update_stats(current_info):
        recorded.append(dict(current_info))

    async def send(message):
        return None

    response = main.LoggingStreamingResponse(
        body(),
        media_type="text/event-stream",
        current_info={
            "start_time": 0,
            "endpoint": "POST /v1/chat/completions",
            "request_id": "request",
            "trace_id": "trace",
        },
        update_stats=update_stats,
    )

    await response({}, None, send)

    assert len(recorded) == 1
    assert recorded[0]["prompt_tokens"] == 2
    assert recorded[0]["completion_tokens"] == 3
    assert recorded[0]["total_tokens"] == 5


def test_logging_response_records_stats_after_stream_finishes():
    asyncio.run(_logging_response_records_stats_after_stream_finishes())


async def _logging_response_records_stats_from_json_without_newline():
    recorded = []

    async def body():
        yield '{"usage":{"prompt_tokens":7,"completion_tokens":11,"total_tokens":18}}'

    async def update_stats(current_info):
        recorded.append(dict(current_info))

    async def send(message):
        return None

    response = main.LoggingStreamingResponse(
        body(),
        media_type="application/json",
        current_info={
            "start_time": 0,
            "endpoint": "POST /v1/chat/completions",
            "request_id": "request",
            "trace_id": "trace",
        },
        update_stats=update_stats,
    )

    await response({}, None, send)

    assert len(recorded) == 1
    assert recorded[0]["prompt_tokens"] == 7
    assert recorded[0]["completion_tokens"] == 11
    assert recorded[0]["total_tokens"] == 18


def test_logging_response_records_stats_from_json_without_newline():
    asyncio.run(_logging_response_records_stats_from_json_without_newline())


async def _logging_response_records_stats_from_nested_input_output_usage_without_newline():
    recorded = []

    async def body():
        yield '{"response":{"usage":{"input_tokens":13,"output_tokens":17}}}'

    async def update_stats(current_info):
        recorded.append(dict(current_info))

    async def send(message):
        return None

    response = main.LoggingStreamingResponse(
        body(),
        media_type="application/json",
        current_info={
            "start_time": 0,
            "endpoint": "POST /v1/responses",
            "request_id": "request",
            "trace_id": "trace",
        },
        update_stats=update_stats,
    )

    await response({}, None, send)

    assert len(recorded) == 1
    assert recorded[0]["prompt_tokens"] == 13
    assert recorded[0]["completion_tokens"] == 17
    assert recorded[0]["total_tokens"] == 30


def test_logging_response_records_stats_from_nested_input_output_usage_without_newline():
    asyncio.run(_logging_response_records_stats_from_nested_input_output_usage_without_newline())


def test_logging_response_bounds_detachable_stats_snapshot():
    async def scenario():
        recorded = []

        async def body():
            yield b"ok"

        async def update_stats(info):
            recorded.append(info)
            return True

        async def send(_message):
            return None

        response = main.LoggingStreamingResponse(
            body(),
            media_type="application/octet-stream",
            current_info={
                "start_time": 0,
                "text": "测" * (1024 * 1024),
                "request_owned_graph": {"payload": "x" * (1024 * 1024)},
            },
            update_stats=update_stats,
        )
        await response({}, None, send)

        assert len(recorded) == 1
        assert len(recorded[0]["text"].encode("utf-8")) <= 64 * 1024
        assert "request_owned_graph" not in recorded[0]

    asyncio.run(scenario())


async def _fetch_response_stream_closes_selected_provider_stream(monkeypatch):
    closed = False

    async def provider_stream(*args, **kwargs):
        nonlocal closed
        try:
            yield "data: ok\n\n"
            await asyncio.Event().wait()
        finally:
            closed = True

    monkeypatch.setattr(response_module, "fetch_gpt_response_stream", provider_stream)

    stream = response_module.fetch_response_stream(
        client=None,
        url="http://example.test",
        headers={},
        payload={},
        engine="codex",
        model="gpt-5.5",
        timeout=None,
    )

    first = await stream.__anext__()
    assert first == "data: ok\n\n"

    await stream.aclose()
    for _ in range(10):
        if closed:
            break
        await asyncio.sleep(0)
    assert closed


def test_fetch_response_stream_closes_selected_provider_stream(monkeypatch):
    asyncio.run(_fetch_response_stream_closes_selected_provider_stream(monkeypatch))


async def _await_stream_cleanup_logs_cancel_without_traceback(caplog):
    started = asyncio.Event()
    release = asyncio.Event()

    async def cleanup():
        started.set()
        await release.wait()

    task = asyncio.create_task(await_stream_cleanup_safely(cleanup(), label="test cleanup"))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()

    assert await task is True
    cancellation_records = [
        record
        for record in caplog.records
        if "test cleanup cleanup owner was cancelled" in record.message
    ]
    assert cancellation_records
    assert all(record.exc_info is None for record in cancellation_records)


def test_await_stream_cleanup_logs_cancel_without_traceback(caplog):
    with caplog.at_level(logging.WARNING):
        asyncio.run(_await_stream_cleanup_logs_cancel_without_traceback(caplog))


class _FakeConnection:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


class _FakePoolRequest:
    def __init__(self, connection):
        self.connection = connection


class _FakePool:
    def __init__(self, request, connection):
        self._requests = [request]
        self._connections = [connection]
        self._optional_thread_lock = None

    def _assign_requests_to_connections(self):
        return []

    async def _close_connections(self, closing):
        for connection in closing:
            await connection.aclose()


class _FakePoolStream:
    def __init__(self, pool, request):
        self._pool = pool
        self._pool_request = request


async def _force_release_closes_assigned_connection(force_release):
    connection = _FakeConnection()
    request = _FakePoolRequest(connection)
    pool = _FakePool(request, connection)
    stream = _FakePoolStream(pool, request)

    result = await force_release(stream)

    assert result is True
    assert request not in pool._requests
    assert connection not in pool._connections
    assert connection.closed


def test_main_force_release_closes_assigned_connection():
    asyncio.run(_force_release_closes_assigned_connection(main._force_release_httpcore_pool_request_safely))


async def _wait_background_stream_cleanup_tasks_observes_and_clears_detached_task():
    completed = False

    async def cleanup():
        nonlocal completed
        await asyncio.sleep(0)
        completed = True

    task = asyncio.create_task(cleanup())
    track_background_stream_cleanup_task(task, label="test")

    assert background_stream_cleanup_snapshot()["pending"] >= 1
    snapshot = await wait_background_stream_cleanup_tasks(timeout=1.0)

    assert completed is True
    assert snapshot["pending"] == 0
    assert snapshot["completed_during_wait"] >= 1


def test_wait_background_stream_cleanup_tasks_observes_and_clears_detached_task():
    asyncio.run(_wait_background_stream_cleanup_tasks_observes_and_clears_detached_task())


def test_core_force_release_closes_assigned_connection():
    async def force_release(stream):
        return await response_module._force_release_httpcore_pool_request_safely(
            stream,
            label="test upstream stream",
        )

    asyncio.run(_force_release_closes_assigned_connection(force_release))


def test_force_close_evicts_transport_before_waiting_on_stuck_response(
    monkeypatch,
):
    async def scenario():
        monkeypatch.setattr(
            stream_cleanup,
            "STREAM_CLEANUP_TIMEOUT_SECONDS",
            0.01,
        )
        transport_released = asyncio.Event()

        class Connection(_FakeConnection):
            async def aclose(self):
                self.closed = True
                transport_released.set()

        connection = Connection()
        request = _FakePoolRequest(connection)
        pool = _FakePool(request, connection)
        pool_stream = _FakePoolStream(pool, request)
        pool_stream._closed = False

        class BoundClose:
            def __init__(self):
                self.aborted = False
                self.close_finished = False

            async def aclose(self):
                try:
                    await transport_released.wait()
                except asyncio.CancelledError:
                    await transport_released.wait()
                self.close_finished = True

            async def abort_transport(self):
                assert request not in pool._requests
                assert connection not in pool._connections
                self.aborted = True

        bound = BoundClose()
        response = type(
            "Response",
            (),
            {"stream": pool_stream, "aclose": bound.aclose},
        )()

        result = await asyncio.wait_for(
            force_close_response_httpcore_stream_chain_safely(
                response,
                label="stuck response",
            ),
            timeout=1,
        )

        assert result is True
        assert bound.aborted is True
        assert bound.close_finished is True
        assert connection.closed is True
        assert request not in pool._requests
        assert connection not in pool._connections

    asyncio.run(scenario())


class _FakeSweepSocket:
    def __init__(self, fd):
        self._fd = fd

    def fileno(self):
        return self._fd


class _FakeSweepNetworkStream:
    def __init__(self, socket):
        self._socket = socket

    def get_extra_info(self, name):
        if name == "socket":
            return self._socket
        return None


class _FakeSweepInnerConnection:
    def __init__(self, socket):
        self._network_stream = _FakeSweepNetworkStream(socket)


class _FakeSweepConnection:
    def __init__(self, *, closed=False, expired=False, socket_fd=None):
        self._closed = closed
        self._expired = expired
        self._connection = (
            _FakeSweepInnerConnection(_FakeSweepSocket(socket_fd))
            if socket_fd is not None
            else None
        )
        self.aclose_called = False

    def is_closed(self):
        return self._closed

    def has_expired(self):
        return self._expired

    async def aclose(self):
        self.aclose_called = True


class _FakeSweepPool:
    def __init__(self, connections):
        self._connections = list(connections)
        self._optional_thread_lock = None

    def _assign_requests_to_connections(self):
        return []

    async def _close_connections(self, closing):
        for connection in closing:
            await connection.aclose()


class _FakeSweepTransport:
    def __init__(self, pool):
        self._pool = pool


class _FakeSweepClient:
    def __init__(self, pool):
        self._transport = _FakeSweepTransport(pool)


async def _sweep_closes_connections_that_httpcore_assign_would_drop():
    closed = _FakeSweepConnection(closed=True)
    expired = _FakeSweepConnection(expired=True)
    healthy = _FakeSweepConnection()
    pool = _FakeSweepPool([closed, expired, healthy])
    client = _FakeSweepClient(pool)

    result = await main._sweep_httpx_client_idle_connections(client)

    assert result == 2
    assert closed.aclose_called
    assert expired.aclose_called
    assert not healthy.aclose_called
    assert pool._connections == [healthy]


def test_sweep_httpx_client_idle_connections_closes_closed_connections():
    asyncio.run(_sweep_closes_connections_that_httpcore_assign_would_drop())


async def _sweep_preserves_active_kernel_close_wait_connections():
    close_wait = _FakeSweepConnection(socket_fd=123)
    healthy = _FakeSweepConnection(socket_fd=456)
    pool = _FakeSweepPool([close_wait, healthy])
    client = _FakeSweepClient(pool)

    result = await main._sweep_httpx_client_idle_connections(client)

    assert result == 0
    assert not close_wait.aclose_called
    assert not healthy.aclose_called
    assert pool._connections == [close_wait, healthy]


def test_sweep_httpx_client_idle_connections_preserves_active_kernel_close_wait_connections(monkeypatch):
    # Recreate the production condition that triggered the regression.  The
    # removed helper is installed with raising=False so this test also fails
    # against the old implementation, which consulted it and closed the active
    # response solely because its socket had received FIN.
    monkeypatch.setattr(
        main,
        "_tcp_close_wait_socket_inodes",
        lambda: {"inode-close-wait"},
        raising=False,
    )
    monkeypatch.setattr(
        main,
        "_socket_inode_for_fd",
        lambda fd: "inode-close-wait" if fd == 123 else "inode-established",
    )

    asyncio.run(_sweep_preserves_active_kernel_close_wait_connections())


def test_sweep_preserves_active_http11_body_after_peer_fin(monkeypatch):
    async def scenario():
        body_allowed = asyncio.Event()
        body_sent = asyncio.Event()
        deltas = [
            (
                'event: response.output_text.delta\n'
                f'data: {{"type":"response.output_text.delta","sequence_number":{index},'
                '"delta":"x"}}\n\n'
            ).encode()
            for index in range(256)
        ]
        terminal = (
            'event: response.completed\n'
            'data: {"type":"response.completed","sequence_number":256,'
            '"response":{"usage":{"input_tokens":1,"output_tokens":2,'
            '"total_tokens":3}}}\n\n'
        ).encode()
        expected_body = b"".join((*deltas, terminal))

        async def handle(reader, writer):
            try:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/event-stream\r\n"
                    b"Transfer-Encoding: chunked\r\n"
                    b"Connection: close\r\n\r\n"
                )
                await writer.drain()
                await body_allowed.wait()
                for chunk in (*deltas, terminal):
                    writer.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                writer.write(b"0\r\n\r\n")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                body_sent.set()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server:
            async with httpx.AsyncClient(trust_env=False, timeout=5.0) as client:
                request = client.build_request("GET", f"http://127.0.0.1:{port}/stream")
                response = await client.send(request, stream=True)
                body_allowed.set()
                await asyncio.wait_for(body_sent.wait(), timeout=1.0)

                pool = client._transport._pool
                assert len(pool._connections) == 1
                connection = pool._connections[0]
                assert connection.is_idle() is False
                assert len(pool._requests) == 1

                # Model the exact kernel signal from the incident.  Against the
                # old implementation this closes the live transport and makes
                # response.aread() fail before the terminal event is consumed.
                monkeypatch.setattr(
                    main,
                    "_tcp_close_wait_socket_inodes",
                    lambda: {"test-close-wait-inode"},
                    raising=False,
                )
                monkeypatch.setattr(
                    main,
                    "_httpcore_connection_socket_inode",
                    lambda candidate: (
                        "test-close-wait-inode" if candidate is connection else None
                    ),
                )
                assert (
                    main._httpcore_connection_socket_inode(connection)
                    == "test-close-wait-inode"
                )

                assert await main._sweep_httpx_client_idle_connections(client) == 0
                received = await response.aread()
                assert received == expected_body
                assert terminal in received
                await response.aclose()
                assert len(pool._requests) == 0
                assert connection not in pool._connections
                assert connection.is_closed() is True

    asyncio.run(scenario())
