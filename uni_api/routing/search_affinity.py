from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable


DEFAULT_SEARCH_AFFINITY_TTL_SECONDS = 24 * 60 * 60
DEFAULT_SEARCH_AFFINITY_MAX_ENTRIES = 10_000


@dataclass(frozen=True, slots=True)
class SearchAffinityBinding:
    provider_fingerprint: str
    request_model: str
    original_model: str
    credential_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class _StoredBinding:
    binding: SearchAffinityBinding
    expires_at: float


@dataclass(slots=True)
class _SessionLock:
    lock: asyncio.Lock
    users: int = 0


class SearchAffinityStore:
    """Process-local, bounded affinity for unary alpha/search sessions."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_SEARCH_AFFINITY_TTL_SECONDS,
        max_entries: int = DEFAULT_SEARCH_AFFINITY_MAX_ENTRIES,
        pepper: bytes | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._ttl_seconds = float(ttl_seconds)
        self._max_entries = int(max_entries)
        self._pepper = bytes(pepper or secrets.token_bytes(32))
        self._now = now or time.monotonic
        self._guard = asyncio.Lock()
        self._bindings: OrderedDict[str, _StoredBinding] = OrderedDict()
        self._session_locks: dict[str, _SessionLock] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def session_key(self, api_key_scope: str, search_id: str) -> str:
        digest = hmac.new(self._pepper, digestmod=hashlib.sha256)
        digest.update(b"uni-api-alpha-search-session-v1\0")
        digest.update(str(api_key_scope).encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(str(search_id).encode("utf-8", errors="replace"))
        return digest.hexdigest()

    def credential_fingerprint(self, credential: str | None) -> str | None:
        if credential is None:
            return None
        return self._fingerprint(
            b"uni-api-alpha-search-credential-v1\0",
            credential,
        )

    def provider_fingerprint(self, provider_name: str) -> str:
        return self._fingerprint(
            b"uni-api-alpha-search-provider-v1\0",
            provider_name,
        )

    @asynccontextmanager
    async def session(self, key: str) -> AsyncIterator[None]:
        async with self._guard:
            entry = self._session_locks.get(key)
            if entry is None:
                entry = _SessionLock(lock=asyncio.Lock())
                self._session_locks[key] = entry
            entry.users += 1

        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            cleanup_task = asyncio.create_task(
                self._release_session_entry(key, entry)
            )
            await self._finish_cleanup_despite_cancellation(cleanup_task)

    async def get(self, key: str) -> SearchAffinityBinding | None:
        async with self._guard:
            now = self._now()
            stored = self._bindings.get(key)
            if stored is None:
                self._misses += 1
                return None
            if stored.expires_at <= now:
                self._bindings.pop(key, None)
                self._misses += 1
                return None
            self._bindings.move_to_end(key)
            self._hits += 1
            return stored.binding

    async def bind_if_absent(
        self,
        key: str,
        binding: SearchAffinityBinding,
    ) -> SearchAffinityBinding:
        async with self._guard:
            now = self._now()
            stored = self._bindings.get(key)
            if stored is not None and stored.expires_at > now:
                self._bindings.move_to_end(key)
                return stored.binding
            if stored is not None:
                self._bindings.pop(key, None)

            self._purge_expired_locked(now)
            while len(self._bindings) >= self._max_entries:
                self._bindings.popitem(last=False)
                self._evictions += 1
            self._bindings[key] = _StoredBinding(
                binding=binding,
                expires_at=now + self._ttl_seconds,
            )
            return binding

    async def snapshot(self) -> dict[str, int]:
        async with self._guard:
            self._purge_expired_locked(self._now())
            return {
                "entries": len(self._bindings),
                "active_session_locks": len(self._session_locks),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "max_entries": self._max_entries,
            }

    def _purge_expired_locked(self, now: float) -> None:
        expired = [
            key
            for key, stored in self._bindings.items()
            if stored.expires_at <= now
        ]
        for key in expired:
            self._bindings.pop(key, None)

    def _fingerprint(self, domain: bytes, value: str) -> str:
        digest = hmac.new(self._pepper, digestmod=hashlib.sha256)
        digest.update(domain)
        digest.update(str(value).encode("utf-8", errors="replace"))
        return digest.hexdigest()

    async def _release_session_entry(
        self,
        key: str,
        entry: _SessionLock,
    ) -> None:
        async with self._guard:
            entry.users = max(0, entry.users - 1)
            if entry.users == 0 and self._session_locks.get(key) is entry:
                self._session_locks.pop(key, None)

    @staticmethod
    async def _finish_cleanup_despite_cancellation(
        task: asyncio.Task[None],
    ) -> None:
        pending_cancel: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                pending_cancel = pending_cancel or exc
        task.result()
        if pending_cancel is not None:
            raise pending_cancel
