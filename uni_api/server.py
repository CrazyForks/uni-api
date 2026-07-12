from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from uvicorn.protocols.http.h11_impl import H11Protocol


@dataclass(slots=True)
class BoundedHTTPProtocolStats:
    connection_limit: int
    header_timeout_seconds: float
    accepted_connections: int = 0
    rejected_connections: int = 0
    header_timeouts: int = 0

    def snapshot(self) -> dict[str, int | float]:
        return {
            "connection_limit": self.connection_limit,
            "header_timeout_seconds": self.header_timeout_seconds,
            "accepted_connections": self.accepted_connections,
            "rejected_connections": self.rejected_connections,
            "header_timeouts": self.header_timeouts,
        }


def build_bounded_h11_protocol(
    *,
    connection_limit: int,
    header_timeout_seconds: float,
) -> tuple[type[H11Protocol], BoundedHTTPProtocolStats]:
    """Build a Uvicorn h11 protocol with an accept-time connection bound.

    Uvicorn's ``limit_concurrency`` is checked only after a complete HTTP
    request arrives and uses the number of open keep-alive connections. It can
    therefore emit false 503s while still allowing incomplete-header sockets
    to consume file descriptors. This protocol bounds accepted sockets before
    adding them to Uvicorn's shared connection set and applies an absolute
    timeout to each incoming request header. Application request concurrency
    remains the responsibility of ``RequestAdmissionMiddleware``.
    """

    if connection_limit <= 0:
        raise ValueError("connection_limit must be positive")
    if header_timeout_seconds <= 0:
        raise ValueError("header_timeout_seconds must be positive")

    stats = BoundedHTTPProtocolStats(
        connection_limit=int(connection_limit),
        header_timeout_seconds=float(header_timeout_seconds),
    )

    class BoundedH11Protocol(H11Protocol):
        _header_timeout_handle: asyncio.TimerHandle | None
        _rejected_before_registration: bool

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._header_timeout_handle = None
            self._rejected_before_registration = False

        def connection_made(self, transport: asyncio.Transport) -> None:
            if len(self.connections) >= stats.connection_limit:
                self._rejected_before_registration = True
                stats.rejected_connections += 1
                transport.close()
                return
            super().connection_made(transport)
            stats.accepted_connections += 1
            self._set_header_timeout()

        def connection_lost(self, exc: Exception | None) -> None:
            self._unset_header_timeout()
            if self._rejected_before_registration:
                return
            super().connection_lost(exc)

        def data_received(self, data: bytes) -> None:
            previous_cycle = self.cycle
            if (
                self._header_timeout_handle is None
                and (
                    previous_cycle is None
                    or bool(previous_cycle.response_complete)
                )
            ):
                self._set_header_timeout()
            super().data_received(data)
            if self.cycle is not previous_cycle:
                self._unset_header_timeout()

        def _set_header_timeout(self) -> None:
            if self._header_timeout_handle is not None:
                return
            self._header_timeout_handle = self.loop.call_later(
                stats.header_timeout_seconds,
                self._on_header_timeout,
            )

        def _unset_header_timeout(self) -> None:
            handle = self._header_timeout_handle
            self._header_timeout_handle = None
            if handle is not None:
                handle.cancel()

        def _on_header_timeout(self) -> None:
            self._header_timeout_handle = None
            transport = self.transport
            if transport is None or transport.is_closing():
                return
            stats.header_timeouts += 1
            transport.close()

    BoundedH11Protocol.__name__ = "BoundedH11Protocol"
    return BoundedH11Protocol, stats
