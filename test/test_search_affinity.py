import asyncio

import pytest

from uni_api.rate_limit.key_pool import ProviderKeyPool
from uni_api.routing.search_affinity import (
    SearchAffinityBinding,
    SearchAffinityStore,
)


def _binding(store: SearchAffinityStore, provider: str, credential: str):
    return SearchAffinityBinding(
        provider_fingerprint=store.provider_fingerprint(provider),
        request_model="gpt-5.4",
        original_model="gpt-5.4-upstream",
        credential_fingerprint=store.credential_fingerprint(credential),
    )


def test_search_affinity_keys_are_scoped_and_do_not_retain_raw_values():
    store = SearchAffinityStore(pepper=b"p" * 32)
    first = store.session_key("client-secret-a", "session-secret")
    second = store.session_key("client-secret-b", "session-secret")

    assert first != second
    assert "client-secret" not in first
    assert "session-secret" not in first
    assert store.provider_fingerprint("provider-a") != "provider-a"
    assert store.credential_fingerprint("provider-key") != "provider-key"


def test_search_affinity_session_serializes_concurrent_first_bindings():
    async def run():
        store = SearchAffinityStore(pepper=b"p" * 32)
        key = store.session_key("client", "session")
        binding = _binding(store, "provider-a", "key-a")
        active = 0
        peak = 0

        async def worker():
            nonlocal active, peak
            async with store.session(key):
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0)
                current = await store.get(key)
                if current is None:
                    current = await store.bind_if_absent(key, binding)
                active -= 1
                return current

        results = await asyncio.gather(*(worker() for _ in range(50)))
        assert peak == 1
        assert results == [binding] * 50
        snapshot = await store.snapshot()
        assert snapshot["entries"] == 1
        assert snapshot["active_session_locks"] == 0

    asyncio.run(run())


def test_search_affinity_fixed_ttl_and_lru_bound():
    async def run():
        clock = [100.0]
        store = SearchAffinityStore(
            ttl_seconds=10,
            max_entries=2,
            pepper=b"p" * 32,
            now=lambda: clock[0],
        )
        keys = [store.session_key("client", f"session-{index}") for index in range(3)]
        bindings = [
            _binding(store, f"provider-{index}", f"key-{index}")
            for index in range(3)
        ]
        await store.bind_if_absent(keys[0], bindings[0])
        await store.bind_if_absent(keys[1], bindings[1])
        assert await store.get(keys[0]) == bindings[0]
        await store.bind_if_absent(keys[2], bindings[2])
        assert await store.get(keys[1]) is None
        assert await store.get(keys[0]) == bindings[0]

        clock[0] = 111.0
        assert await store.get(keys[0]) is None
        assert await store.get(keys[2]) is None

    asyncio.run(run())


def test_search_affinity_owner_cancellation_releases_waiter_and_lock_entry():
    async def run():
        store = SearchAffinityStore(pepper=b"p" * 32)
        key = store.session_key("client", "cancelled-session")
        owner_started = asyncio.Event()
        release_owner = asyncio.Event()

        async def owner():
            async with store.session(key):
                owner_started.set()
                await release_owner.wait()

        async def waiter():
            async with store.session(key):
                return "acquired"

        owner_task = asyncio.create_task(owner())
        await owner_started.wait()
        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        owner_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner_task
        assert await asyncio.wait_for(waiter_task, timeout=1) == "acquired"
        assert (await store.snapshot())["active_session_locks"] == 0

    asyncio.run(run())


def test_provider_key_pool_claims_only_the_bound_fingerprint():
    async def run():
        store = SearchAffinityStore(pepper=b"p" * 32)
        pool = ProviderKeyPool(
            ["key-a", "key-b"],
            rate_limit={"default": "999999/min"},
        )
        claimed = await pool.claim_by_fingerprint(
            store.credential_fingerprint("key-b"),
            store.credential_fingerprint,
            "gpt-5.4",
        )
        missing = await pool.claim_by_fingerprint(
            store.credential_fingerprint("removed-key"),
            store.credential_fingerprint,
            "gpt-5.4",
        )
        assert claimed == "key-b"
        assert missing is None

    asyncio.run(run())


def test_search_affinity_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        SearchAffinityStore(ttl_seconds=0)
    with pytest.raises(ValueError):
        SearchAffinityStore(max_entries=0)
