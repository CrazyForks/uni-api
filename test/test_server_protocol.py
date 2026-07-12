from __future__ import annotations

import asyncio
import socket

import uvicorn

from uni_api.server import build_bounded_h11_protocol


async def _instant_app(scope, receive, send):
    assert scope["type"] == "http"
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-length", b"2")],
        }
    )
    await send({"type": "http.response.body", "body": b"ok"})


async def _start_server(*, connection_limit: int, header_timeout: float):
    protocol, stats = build_bounded_h11_protocol(
        connection_limit=connection_limit,
        header_timeout_seconds=header_timeout,
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    config = uvicorn.Config(
        _instant_app,
        http=protocol,
        lifespan="off",
        limit_concurrency=None,
        log_level="error",
        timeout_keep_alive=10,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[listener]))
    while not server.started:
        await asyncio.sleep(0.001)
    return server, task, listener, port, stats


async def _stop_server(server, task, listener):
    server.should_exit = True
    await asyncio.wait_for(task, timeout=2)
    listener.close()


def test_preconnected_keepalive_sockets_do_not_trigger_uvicorn_false_503s():
    async def scenario():
        server, task, listener, port, stats = await _start_server(
            connection_limit=10,
            header_timeout=1,
        )
        connections = []
        overflow_writer = None
        try:
            for _ in range(10):
                connections.append(
                    await asyncio.open_connection("127.0.0.1", port)
                )
            while stats.accepted_connections < 10:
                await asyncio.sleep(0.001)

            overflow_reader, overflow_writer = await asyncio.open_connection(
                "127.0.0.1",
                port,
            )
            assert await asyncio.wait_for(overflow_reader.read(), timeout=1) == b""
            assert stats.rejected_connections == 1

            for _reader, writer in connections:
                writer.write(
                    b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
                )
            await asyncio.gather(
                *(writer.drain() for _reader, writer in connections)
            )
            responses = await asyncio.gather(
                *(reader.read() for reader, _writer in connections)
            )
            assert all(b" 200 OK\r\n" in response for response in responses)
            assert all(b" 503 Service Unavailable\r\n" not in response for response in responses)
        finally:
            if overflow_writer is not None:
                overflow_writer.close()
            for _reader, writer in connections:
                writer.close()
            await _stop_server(server, task, listener)

    asyncio.run(scenario())


def test_incomplete_request_header_is_closed_at_absolute_deadline():
    async def scenario():
        server, task, listener, port, stats = await _start_server(
            connection_limit=2,
            header_timeout=0.05,
        )
        writer = None
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET / HTTP/1.1\r\nHost:")
            await writer.drain()
            assert await asyncio.wait_for(reader.read(), timeout=1) == b""
            assert stats.header_timeouts == 1
        finally:
            if writer is not None:
                writer.close()
            await _stop_server(server, task, listener)

    asyncio.run(scenario())
