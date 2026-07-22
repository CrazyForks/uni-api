from __future__ import annotations

import hashlib
import os
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from time import monotonic
from typing import Callable


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class _RouteState:
    failures: deque[tuple[float, int]] = field(default_factory=deque)
    open_until: float = 0.0
    opened_total: int = 0
    last_status_code: int | None = None


class ProviderModelCircuitBreaker:
    """Bound deterministic 403/404 retries for one provider-model route."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        failure_window_seconds: float = 120.0,
        open_seconds: float = 300.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if failure_window_seconds <= 0 or open_seconds <= 0:
            raise ValueError("circuit durations must be positive")
        self.failure_threshold = int(failure_threshold)
        self.failure_window_seconds = float(failure_window_seconds)
        self.open_seconds = float(open_seconds)
        self._clock = clock
        self._lock = threading.RLock()
        self._states: dict[tuple[str, str], _RouteState] = {}
        self._failure_total: Counter[int] = Counter()
        self._opened_total = 0
        self._success_reset_total = 0

    @classmethod
    def from_environment(cls) -> ProviderModelCircuitBreaker:
        return cls(
            failure_threshold=max(
                1,
                _env_int("PROVIDER_MODEL_CIRCUIT_FAILURE_THRESHOLD", 3),
            ),
            failure_window_seconds=max(
                1.0,
                _env_float("PROVIDER_MODEL_CIRCUIT_WINDOW_SECONDS", 120.0),
            ),
            open_seconds=max(
                1.0,
                _env_float("PROVIDER_MODEL_CIRCUIT_OPEN_SECONDS", 300.0),
            ),
        )

    @staticmethod
    def _key(provider: str, model: str) -> tuple[str, str]:
        return (str(provider or "").strip(), str(model or "").strip())

    def _prune_locked(self, state: _RouteState, now: float) -> None:
        cutoff = now - self.failure_window_seconds
        while state.failures and state.failures[0][0] < cutoff:
            state.failures.popleft()
        if state.open_until and state.open_until <= now:
            state.open_until = 0.0
            state.failures.clear()

    def record_failure(self, provider: str, model: str, status_code: int) -> bool:
        status_code = int(status_code)
        if status_code not in (403, 404):
            return False
        key = self._key(provider, model)
        if not all(key):
            return False
        now = self._clock()
        with self._lock:
            state = self._states.setdefault(key, _RouteState())
            self._prune_locked(state, now)
            self._failure_total[status_code] += 1
            state.last_status_code = status_code
            state.failures.append((now, status_code))
            if state.open_until > now:
                return False
            if len(state.failures) < self.failure_threshold:
                return False
            state.open_until = now + self.open_seconds
            state.opened_total += 1
            self._opened_total += 1
            return True

    def record_success(self, provider: str, model: str) -> bool:
        key = self._key(provider, model)
        if not all(key):
            return False
        with self._lock:
            state = self._states.pop(key, None)
            if state is None:
                return False
            self._success_reset_total += 1
            return True

    def is_open(self, provider: str, model: str) -> bool:
        key = self._key(provider, model)
        if not all(key):
            return False
        now = self._clock()
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return False
            self._prune_locked(state, now)
            if not state.failures and not state.open_until:
                self._states.pop(key, None)
                return False
            return state.open_until > now

    def snapshot(self) -> dict[str, object]:
        now = self._clock()
        open_routes: list[dict[str, object]] = []
        with self._lock:
            for key, state in tuple(self._states.items()):
                self._prune_locked(state, now)
                if not state.failures and not state.open_until:
                    self._states.pop(key, None)
                    continue
                if state.open_until > now:
                    fingerprint = hashlib.sha256(
                        f"{key[0]}\x00{key[1]}".encode("utf-8")
                    ).hexdigest()[:16]
                    open_routes.append(
                        {
                            "route_fingerprint": fingerprint,
                            "remaining_open_ms": max(
                                0,
                                int((state.open_until - now) * 1000),
                            ),
                            "last_status_code": state.last_status_code,
                            "failures_in_window": len(state.failures),
                            "opened_total": state.opened_total,
                        }
                    )
            return {
                "enabled": True,
                "failure_threshold": self.failure_threshold,
                "failure_window_seconds": self.failure_window_seconds,
                "open_seconds": self.open_seconds,
                "open_routes": open_routes[:64],
                "open_route_count": len(open_routes),
                "failure_total_by_status": dict(self._failure_total),
                "opened_total": self._opened_total,
                "success_reset_total": self._success_reset_total,
            }
