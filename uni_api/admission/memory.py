from __future__ import annotations

import asyncio
import os
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable

from uni_api.admission.resources import (
    current_cgroup_v1_root,
    current_cgroup_v2_root,
)


_MIB = 1024 * 1024
_DEFAULT_FALLBACK_BUDGET_BYTES = 256 * _MIB
_DEFAULT_GUARD_BYTES = 512 * _MIB
_DEFAULT_GUARD_RATIO = 0.25
_DEFAULT_SAMPLE_CACHE_SECONDS = 0.05


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


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _read_events(path: Path) -> dict[str, int]:
    events: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError:
        return events
    for line in lines:
        name, separator, value = line.partition(" ")
        if not separator:
            continue
        try:
            events[name] = int(value)
        except ValueError:
            continue
    return events


def _read_failcnt(path: Path) -> dict[str, int]:
    value = _read_int(path)
    return {"max": value} if value is not None else {}


def _proc_rss_bytes() -> int | None:
    try:
        lines = Path("/proc/self/status").read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) < 2:
            return None
        try:
            return int(fields[1]) * 1024
        except ValueError:
            return None
    return None


@dataclass(frozen=True, slots=True)
class ProcessMemorySample:
    current_bytes: int | None
    limit_bytes: int | None
    high_bytes: int | None = None
    events: dict[str, int] | None = None
    source: str = "unknown"


class CgroupMemorySource:
    """Read the current process cgroup without depending on Kubernetes APIs."""

    def __init__(
        self,
        root: str | Path = "/sys/fs/cgroup",
        proc_cgroup: str | Path = "/proc/self/cgroup",
    ) -> None:
        self.root = Path(root)
        self.proc_cgroup = Path(proc_cgroup)

    def sample(self) -> ProcessMemorySample:
        v2_root = current_cgroup_v2_root(self.root, self.proc_cgroup)
        current = _read_int(v2_root / "memory.current")
        limit = _read_int(v2_root / "memory.max")
        high = _read_int(v2_root / "memory.high")
        if current is not None or limit is not None:
            return ProcessMemorySample(
                current_bytes=current,
                limit_bytes=limit,
                high_bytes=high,
                events=_read_events(v2_root / "memory.events"),
                source="cgroup-v2",
            )

        v1_root = current_cgroup_v1_root(
            "memory",
            self.root,
            self.proc_cgroup,
        )
        current = _read_int(v1_root / "memory.usage_in_bytes")
        limit = _read_int(v1_root / "memory.limit_in_bytes")
        if current is not None or limit is not None:
            # cgroup v1 commonly reports an enormous sentinel for unlimited.
            if limit is not None and limit >= (1 << 60):
                limit = None
            return ProcessMemorySample(
                current_bytes=current,
                limit_bytes=limit,
                events=_read_failcnt(v1_root / "memory.failcnt"),
                source="cgroup-v1",
            )

        return ProcessMemorySample(
            current_bytes=_proc_rss_bytes(),
            limit_bytes=None,
            source="procfs",
        )


@dataclass(frozen=True, slots=True)
class AdaptiveMemorySnapshot:
    source: str
    current_bytes: int | None
    limit_bytes: int | None
    high_bytes: int | None
    soft_limit_bytes: int | None
    guard_bytes: int
    capacity_bytes: int
    available_bytes: int
    reserved_bytes: int
    peak_reserved_bytes: int
    reservations: dict[str, int]
    rejected: dict[str, int]
    blocked_reservations: int
    waiting_reservations: int
    wait_timeouts: int
    events: dict[str, int]
    sample_error: str | None
    sample_sequence: int
    sampled_at_monotonic: float | None
    sample_age_ms: int | None


@dataclass(frozen=True, slots=True)
class AdaptiveMemoryReservationDecision:
    """Atomic result and cgroup facts used by one parent reservation."""

    allowed: bool
    category: str
    requested_bytes: int
    reserved_before_bytes: int
    projected_reserved_bytes: int
    reserved_after_bytes: int
    source: str
    current_bytes: int | None
    limit_bytes: int | None
    high_bytes: int | None
    soft_limit_bytes: int | None
    guard_bytes: int
    capacity_bytes: int
    available_before_bytes: int
    available_after_bytes: int
    sample_error: str | None
    sample_sequence: int
    sampled_at_monotonic: float | None
    sample_age_ms: int | None


class AdaptiveMemoryGovernor:
    """One atomic parent budget for every process-owned retained byte.

    Existing reservations deliberately remain part of the projected memory
    even though some bytes are already reflected by ``memory.current``.  This
    conservative double accounting leaves room for JSON/Pydantic
    materialization, allocator fragmentation, and allocations that occur
    between cgroup samples.
    """

    def __init__(
        self,
        *,
        source: Callable[[], ProcessMemorySample] | None = None,
        soft_limit_bytes: int | None = None,
        guard_bytes: int = _DEFAULT_GUARD_BYTES,
        guard_ratio: float = _DEFAULT_GUARD_RATIO,
        fallback_budget_bytes: int = _DEFAULT_FALLBACK_BUDGET_BYTES,
        sample_cache_seconds: float = _DEFAULT_SAMPLE_CACHE_SECONDS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if soft_limit_bytes is not None and soft_limit_bytes <= 0:
            raise ValueError("soft_limit_bytes must be positive when provided")
        if guard_bytes < 0 or not 0 <= guard_ratio < 1:
            raise ValueError("memory guard must be non-negative and below one")
        if fallback_budget_bytes <= 0:
            raise ValueError("fallback_budget_bytes must be positive")
        if sample_cache_seconds < 0:
            raise ValueError("sample_cache_seconds cannot be negative")

        cgroup_source = CgroupMemorySource()
        self._source = source or cgroup_source.sample
        self._configured_soft_limit_bytes = soft_limit_bytes
        self.guard_bytes = int(guard_bytes)
        self.guard_ratio = float(guard_ratio)
        self.fallback_budget_bytes = int(fallback_budget_bytes)
        self.sample_cache_seconds = float(sample_cache_seconds)
        self._clock = clock

        self._lock = threading.RLock()
        self._sample: ProcessMemorySample | None = None
        self._sampled_at = float("-inf")
        self._sample_attempted_at = float("-inf")
        self._sample_sequence = 0
        self._sample_error: str | None = None
        self._reserved: Counter[str] = Counter()
        self._rejected: Counter[str] = Counter()
        self._peak_reserved_bytes = 0
        self._blocked_reservations = 0
        self._waiting_reservations = 0
        self._wait_timeouts = 0
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = set()

    @classmethod
    def from_environment(cls) -> AdaptiveMemoryGovernor:
        configured_soft_limit = _env_int("MEMORY_SOFT_LIMIT_BYTES", 0)
        return cls(
            soft_limit_bytes=(
                configured_soft_limit if configured_soft_limit > 0 else None
            ),
            guard_bytes=max(0, _env_int("MEMORY_GUARD_BYTES", _DEFAULT_GUARD_BYTES)),
            guard_ratio=min(
                0.95,
                max(0.0, _env_float("MEMORY_GUARD_RATIO", _DEFAULT_GUARD_RATIO)),
            ),
            fallback_budget_bytes=max(
                1,
                _env_int(
                    "MEMORY_FALLBACK_BUDGET_BYTES",
                    _DEFAULT_FALLBACK_BUDGET_BYTES,
                ),
            ),
            sample_cache_seconds=max(
                0.0,
                _env_float(
                    "MEMORY_SAMPLE_CACHE_SECONDS",
                    _DEFAULT_SAMPLE_CACHE_SECONDS,
                ),
            ),
        )

    def _refresh_sample_locked(self, *, force: bool = False) -> ProcessMemorySample:
        now = self._clock()
        if (
            not force
            and self._sample is not None
            and now - self._sample_attempted_at < self.sample_cache_seconds
        ):
            return self._sample
        self._sample_attempted_at = now
        try:
            sample = self._source()
            if not isinstance(sample, ProcessMemorySample):
                raise TypeError("memory source returned an invalid sample")
            self._sample = sample
            self._sample_error = None
            self._sampled_at = now
            self._sample_sequence += 1
        except Exception as exc:
            self._sample_error = f"{type(exc).__name__}: {exc}"[:512]
            if self._sample is None:
                self._sample = ProcessMemorySample(None, None, source="unavailable")
        return self._sample

    def _limits_locked(
        self,
        sample: ProcessMemorySample,
    ) -> tuple[int | None, int, int, int]:
        limit = sample.limit_bytes
        if sample.high_bytes is not None:
            limit = min(limit, sample.high_bytes) if limit is not None else sample.high_bytes
        if self._configured_soft_limit_bytes is not None:
            soft_limit = self._configured_soft_limit_bytes
            if limit is not None:
                soft_limit = min(soft_limit, limit)
            guard = max(0, (limit or soft_limit) - soft_limit)
        elif limit is not None:
            # A fixed 512 MiB safety margin is useful for the production-sized
            # Pod, but it must not consume an entire small container.  Cap only
            # the absolute component at half the effective cgroup limit; an
            # explicitly configured ratio remains authoritative.
            absolute_guard = min(self.guard_bytes, limit // 2)
            guard = max(absolute_guard, int(limit * self.guard_ratio))
            soft_limit = max(1, limit - min(guard, max(0, limit - 1)))
        else:
            guard = self.guard_bytes
            soft_limit = None

        reserved = sum(self._reserved.values())
        if soft_limit is None or sample.current_bytes is None:
            capacity = max(reserved, self.fallback_budget_bytes)
        else:
            capacity = max(reserved, soft_limit - sample.current_bytes)
        if self._sample_error is not None:
            # A stale last-good cgroup sample must never authorize expansion.
            # Preserve existing ownership, but contract all new allocations to
            # the portable finite fallback until sampling recovers.
            capacity = max(reserved, min(capacity, self.fallback_budget_bytes))
        available = max(0, capacity - reserved)
        return soft_limit, guard, capacity, available

    def maximum_capacity_bytes(self) -> int:
        with self._lock:
            sample = self._refresh_sample_locked(force=True)
            soft_limit, _guard, _capacity, _available = self._limits_locked(sample)
            return soft_limit or self.fallback_budget_bytes

    def reserve_nowait(self, category: str, size: int) -> bool:
        return self.reserve_nowait_decision(category, size).allowed

    def reserve_nowait_decision(
        self,
        category: str,
        size: int,
    ) -> AdaptiveMemoryReservationDecision:
        """Reserve immediately and return the exact sample used to decide."""

        size = int(size)
        if size < 0:
            raise ValueError("memory reservation cannot be negative")
        normalized = str(category or "unknown").strip() or "unknown"
        with self._lock:
            sample = self._refresh_sample_locked()
            soft_limit, guard, capacity, available_before = self._limits_locked(
                sample
            )
            reserved_before = sum(self._reserved.values())
            allowed = size <= available_before
            if allowed and size:
                self._reserved[normalized] += size
                self._peak_reserved_bytes = max(
                    self._peak_reserved_bytes,
                    reserved_before + size,
                )
            elif not allowed:
                self._rejected[normalized] += 1
            sampled_at = (
                self._sampled_at
                if self._sample_sequence > 0
                and self._sampled_at != float("-inf")
                else None
            )
            sample_age_ms = (
                max(
                    0,
                    int(round((self._clock() - sampled_at) * 1000.0)),
                )
                if sampled_at is not None
                else None
            )
            reserved_after = reserved_before + size if allowed else reserved_before
            return AdaptiveMemoryReservationDecision(
                allowed=allowed,
                category=normalized,
                requested_bytes=size,
                reserved_before_bytes=reserved_before,
                projected_reserved_bytes=reserved_before + size,
                reserved_after_bytes=reserved_after,
                source=sample.source,
                current_bytes=sample.current_bytes,
                limit_bytes=sample.limit_bytes,
                high_bytes=sample.high_bytes,
                soft_limit_bytes=soft_limit,
                guard_bytes=guard,
                capacity_bytes=capacity,
                available_before_bytes=available_before,
                available_after_bytes=max(
                    0,
                    available_before - size if allowed else available_before,
                ),
                sample_error=self._sample_error,
                sample_sequence=self._sample_sequence,
                sampled_at_monotonic=sampled_at,
                sample_age_ms=sample_age_ms,
            )

    def _try_reserve_locked(
        self,
        category: str,
        size: int,
        *,
        record_rejection: bool,
    ) -> bool:
        sample = self._refresh_sample_locked()
        _soft_limit, _guard, _capacity, available = self._limits_locked(sample)
        if size > available:
            if record_rejection:
                self._rejected[category] += 1
            return False
        self._reserved[category] += size
        self._peak_reserved_bytes = max(
            self._peak_reserved_bytes,
            sum(self._reserved.values()),
        )
        return True

    async def reserve(
        self,
        category: str,
        size: int,
        *,
        timeout_seconds: float,
    ) -> bool:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        normalized = str(category or "unknown").strip() or "unknown"
        size = int(size)
        if size < 0:
            raise ValueError("memory reservation cannot be negative")
        if size == 0:
            return True
        with self._lock:
            if self._try_reserve_locked(
                normalized,
                size,
                record_rejection=False,
            ):
                return True

        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        waiter = (loop, event)
        started_at = loop.time()
        with self._lock:
            self._blocked_reservations += 1
            self._waiting_reservations += 1
            self._waiters.add(waiter)
        try:
            while True:
                remaining = timeout_seconds - (loop.time() - started_at)
                if remaining <= 0:
                    with self._lock:
                        self._wait_timeouts += 1
                    return False
                event.clear()
                # Recheck after registration/clear so a release in either race
                # window cannot leave this waiter asleep with available space.
                with self._lock:
                    if self._try_reserve_locked(
                        normalized,
                        size,
                        record_rejection=False,
                    ):
                        return True
                try:
                    async with asyncio.timeout(min(0.1, remaining)):
                        await event.wait()
                except TimeoutError:
                    # ``memory.current`` may fall after GC without any tracked
                    # lease release. Poll the cgroup until the caller's real
                    # deadline so that newly available headroom is observed.
                    continue
        finally:
            with self._lock:
                self._waiters.discard(waiter)
                self._waiting_reservations = max(0, self._waiting_reservations - 1)

    def release(self, category: str, size: int) -> None:
        size = int(size)
        if size < 0:
            raise ValueError("memory release cannot be negative")
        if size == 0:
            return
        normalized = str(category or "unknown").strip() or "unknown"
        with self._lock:
            if size > self._reserved[normalized]:
                raise RuntimeError("adaptive memory reservation underflow")
            self._reserved[normalized] -= size
            if self._reserved[normalized] == 0:
                del self._reserved[normalized]
            waiters = tuple(self._waiters)
        for loop, event in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                continue

    def _snapshot_locked(
        self,
        sample: ProcessMemorySample,
    ) -> AdaptiveMemorySnapshot:
        soft_limit, guard, capacity, available = self._limits_locked(sample)
        sampled_at = (
            self._sampled_at
            if self._sample_sequence > 0 and self._sampled_at != float("-inf")
            else None
        )
        sample_age_ms = (
            max(0, int(round((self._clock() - sampled_at) * 1000.0)))
            if sampled_at is not None
            else None
        )
        return AdaptiveMemorySnapshot(
            source=sample.source,
            current_bytes=sample.current_bytes,
            limit_bytes=sample.limit_bytes,
            high_bytes=sample.high_bytes,
            soft_limit_bytes=soft_limit,
            guard_bytes=guard,
            capacity_bytes=capacity,
            available_bytes=available,
            reserved_bytes=sum(self._reserved.values()),
            peak_reserved_bytes=self._peak_reserved_bytes,
            reservations=dict(self._reserved),
            rejected=dict(self._rejected),
            blocked_reservations=self._blocked_reservations,
            waiting_reservations=self._waiting_reservations,
            wait_timeouts=self._wait_timeouts,
            events=dict(sample.events or {}),
            sample_error=self._sample_error,
            sample_sequence=self._sample_sequence,
            sampled_at_monotonic=sampled_at,
            sample_age_ms=sample_age_ms,
        )

    def snapshot_cached(self) -> AdaptiveMemorySnapshot:
        """Return the latest sample without filesystem I/O.

        Admission decision recording calls this while its asyncio lock is held.
        The sample sequence and age make the cache boundary explicit instead of
        presenting a cached cgroup value as an atomic kernel measurement.
        """

        with self._lock:
            sample = self._sample or ProcessMemorySample(
                None,
                None,
                source="unavailable",
            )
            return self._snapshot_locked(sample)

    def snapshot(self, *, force: bool = False) -> AdaptiveMemorySnapshot:
        with self._lock:
            sample = self._refresh_sample_locked(force=force)
            return self._snapshot_locked(sample)


process_memory_governor = AdaptiveMemoryGovernor.from_environment()
