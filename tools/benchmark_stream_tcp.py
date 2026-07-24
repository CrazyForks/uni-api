"""A/B benchmark the legacy and persistent SSE write paths over localhost TCP."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from urllib.parse import parse_qs

import httpx
import uvicorn
from starlette.responses import Response

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from uni_api.streaming import logging_response


class _LegacyDeadlineWriter:
    """Recreate the pre-optimization task-plus-wait behavior for A/B runs."""

    def __init__(self, send, *, timeout: float, label: str) -> None:
        self._send = send
        self._timeout = timeout
        self._label = label

    async def write(self, message) -> None:
        await logging_response._await_with_hard_deadline(
            self._send(message),
            timeout=self._timeout,
            label=self._label,
        )

    async def close(self) -> asyncio.CancelledError | None:
        return None


async def _app(scope, receive, send) -> None:
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    if scope["path"] == "/cpu":
        body = json.dumps({"process_time": time.process_time()}).encode()
        await Response(body, media_type="application/json")(scope, receive, send)
        return
    if scope["path"] != "/stream":
        await Response(status_code=404)(scope, receive, send)
        return

    query = parse_qs(bytes(scope.get("query_string", b"")).decode("ascii"))
    events = int(query["events"][0])
    payload_bytes = int(query["payload_bytes"][0])
    chunk = b"data: " + (b"x" * (payload_bytes - 8)) + b"\n\n"

    async def body():
        for _ in range(events):
            yield chunk

    response = logging_response.LoggingStreamingResponse(
        body(),
        media_type="text/event-stream",
        current_info={"start_time": time.time()},
        observe_usage=False,
    )
    await response(scope, receive, send)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_ready(base_url: str, process: subprocess.Popen) -> None:
    async with httpx.AsyncClient(timeout=0.2) as client:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError(f"benchmark server exited: {process.returncode}")
            try:
                (await client.get(f"{base_url}/cpu")).raise_for_status()
                return
            except (httpx.HTTPError, OSError):
                await asyncio.sleep(0.05)
    raise TimeoutError("benchmark server did not become ready")


async def _server_cpu(client: httpx.AsyncClient, base_url: str) -> float:
    response = await client.get(f"{base_url}/cpu")
    response.raise_for_status()
    return float(response.json()["process_time"])


async def _consume(client: httpx.AsyncClient, url: str, expected: int) -> int:
    received = 0
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        async for chunk in response.aiter_raw():
            received += len(chunk)
    if received != expected:
        raise AssertionError(f"received {received} bytes; expected {expected}")
    return received


async def _measure(base_url: str, args) -> dict[str, float | int]:
    limits = httpx.Limits(
        max_connections=args.concurrency + 2,
        max_keepalive_connections=args.concurrency + 2,
    )
    async with httpx.AsyncClient(limits=limits, timeout=120.0) as client:
        url = (
            f"{base_url}/stream?events={args.events_per_stream}"
            f"&payload_bytes={args.payload_bytes}"
        )
        before_cpu = await _server_cpu(client, base_url)
        started = time.perf_counter()
        received = await asyncio.gather(
            *(
                _consume(
                    client,
                    url,
                    args.events_per_stream * args.payload_bytes,
                )
                for _ in range(args.concurrency)
            )
        )
        wall_seconds = time.perf_counter() - started
        await asyncio.sleep(0.05)
        cpu_seconds = (await _server_cpu(client, base_url)) - before_cpu

    total_events = args.concurrency * args.events_per_stream
    return {
        "total_events": total_events,
        "total_bytes": sum(received),
        "wall_seconds": wall_seconds,
        "server_cpu_seconds": cpu_seconds,
        "events_per_wall_second": total_events / wall_seconds,
        "server_cpu_us_per_event": cpu_seconds * 1_000_000 / total_events,
    }


async def _variant(loop_name: str, writer_name: str, args) -> dict[str, object]:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--serve",
            "--loop",
            loop_name,
            "--writer",
            writer_name,
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        await _wait_ready(base_url, process)
        measurements = [await _measure(base_url, args) for _ in range(args.rounds)]
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    return {
        "loop": loop_name,
        "writer": writer_name,
        "measurements": measurements,
        "best_events_per_wall_second": max(
            item["events_per_wall_second"] for item in measurements
        ),
        "best_server_cpu_us_per_event": min(
            item["server_cpu_us_per_event"] for item in measurements
        ),
    }


async def _benchmark(args) -> None:
    results = []
    for writer_name in args.writers:
        for loop_name in args.loops:
            results.append(await _variant(loop_name, writer_name, args))
    print(
        json.dumps(
            {
                "concurrency": args.concurrency,
                "events_per_stream": args.events_per_stream,
                "payload_bytes": args.payload_bytes,
                "rounds": args.rounds,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--loop", choices=("asyncio", "uvloop"), default="asyncio")
    parser.add_argument(
        "--writer", choices=("legacy", "persistent"), default="persistent"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--loops", nargs="+", choices=("asyncio", "uvloop"), default=("asyncio", "uvloop")
    )
    parser.add_argument(
        "--writers",
        nargs="+",
        choices=("legacy", "persistent"),
        default=("legacy", "persistent"),
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--events-per-stream", type=int, default=1500)
    parser.add_argument("--payload-bytes", type=int, default=256)
    args = parser.parse_args()

    if args.serve:
        if args.writer == "legacy":
            logging_response._HardDeadlineASGIWriter = _LegacyDeadlineWriter
        uvicorn.run(
            _app,
            host="127.0.0.1",
            port=args.port,
            loop=args.loop,
            http="h11",
            access_log=False,
            log_level="warning",
        )
        return
    if min(
        args.rounds,
        args.concurrency,
        args.events_per_stream,
        args.payload_bytes,
    ) <= 0:
        parser.error("benchmark dimensions must be positive")
    if args.payload_bytes < 8:
        parser.error("payload-bytes must be at least 8")
    asyncio.run(_benchmark(args))


if __name__ == "__main__":
    main()
