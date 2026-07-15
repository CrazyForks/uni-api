import asyncio

import pytest

import uni_api.runtime as runtime
from uni_api.runtime import RuntimeGauges
from uni_api.streaming.bounded_queue import ByteBoundedQueue


def test_runtime_snapshot_uses_authoritative_admission_and_cached_network_state(monkeypatch):
    gauges = RuntimeGauges()
    gauges.inflight_requests = 99
    gauges.attach_request_admission(
        lambda: {
            "active": 7,
            "waiters": 11,
            "capacity": 256,
            "waiter_limit": 1024,
            "reserved_body_bytes": 1234,
            "reserved_response_bytes": 4321,
            "reserved_retained_bytes": 5555,
            "deferred_memory_requests": 2,
            "deferred_memory_bytes": 3333,
            "body_budget": 5678,
            "large_body_threshold_weighted_bytes": 1024,
            "large_body_limit": 2,
            "large_body_active": 1,
            "rejected": {"queue_full": 3},
        }
    )
    gauges.open_sockets = 4
    gauges.tcp_states = {"ESTABLISHED": 2, "CLOSE_WAIT": 1}
    gauges.attach_stream_parser_budget(
        lambda: {
            "used_bytes": 777,
            "peak_bytes": 999,
            "capacity_bytes": 4096,
            "rejected": 3,
        }
    )
    monkeypatch.setattr(
        runtime,
        "_open_socket_count",
        lambda: (_ for _ in ()).throw(AssertionError("snapshot must not scan /proc")),
    )

    snapshot = gauges.snapshot()

    assert snapshot["inflight_requests"] == 7
    assert snapshot["request_active"] == 7
    assert snapshot["request_waiters"] == 11
    assert snapshot["middleware_inflight_requests"] == 99
    assert snapshot["request_body_reserved_weighted_bytes"] == 1234
    assert snapshot["upstream_response_reserved_weighted_bytes"] == 4321
    assert snapshot["request_retained_reserved_weighted_bytes"] == 5555
    assert snapshot["request_large_body_threshold_weighted_bytes"] == 1024
    assert snapshot["request_large_body_limit"] == 2
    assert snapshot["request_large_body_active"] == 1
    assert snapshot["request_deferred_memory_requests"] == 2
    assert snapshot["request_deferred_memory_weighted_bytes"] == 3333
    assert snapshot["stream_parser_reserved_bytes"] == 777
    assert snapshot["stream_parser_budget_bytes"] == 4096
    assert snapshot["stream_parser_peak_bytes"] == 999
    assert snapshot["stream_parser_rejected_total"] == 3
    assert "request_body_reserved_bytes" not in snapshot
    assert snapshot["request_admission_rejected_total"] == 3
    assert snapshot["open_sockets"] == 4
    assert snapshot["tcp_close_wait"] == 1


def test_bounded_env_int_rejects_unsafe_override(monkeypatch):
    monkeypatch.setenv("TEST_BOUNDED_LIMIT", "101")
    with pytest.raises(ValueError, match="startup safety limit 100"):
        runtime._bounded_env_int("TEST_BOUNDED_LIMIT", 50, 100)


def test_bounded_env_int_rejects_invalid_override(monkeypatch):
    monkeypatch.setenv("TEST_BOUNDED_LIMIT", "not-an-int")
    with pytest.raises(ValueError, match="must be an integer"):
        runtime._bounded_env_int("TEST_BOUNDED_LIMIT", 50, 100)


def test_positive_env_int_rejects_nonpositive_override(monkeypatch):
    monkeypatch.setenv("TEST_POSITIVE_LIMIT", "0")
    with pytest.raises(ValueError, match="must be positive"):
        runtime._positive_env_int("TEST_POSITIVE_LIMIT", 50)


def test_runtime_observability_endpoint_bypasses_request_admission():
    assert runtime._bypass_request_admission(
        {"method": "GET", "path": "/v1/observability/runtime"}
    )
    assert not runtime._bypass_request_admission(
        {"method": "POST", "path": "/v1/observability/runtime"}
    )


def test_runtime_receive_failure_does_not_fabricate_disconnect():
    async def scenario():
        event = asyncio.Event()

        class Request:
            async def receive(self):
                raise RuntimeError("adapter failed")

        await runtime.monitor_disconnect(Request(), event)
        assert not event.is_set()

    asyncio.run(scenario())


def test_runtime_stream_queue_metrics_track_live_and_retired_totals():
    async def scenario():
        gauges = RuntimeGauges()
        queue = ByteBoundedQueue(max_items=1, max_bytes=16)
        gauges.register_stream_queue(queue)
        await queue.put(b"first")
        blocked = asyncio.create_task(queue.put(b"second"))
        await asyncio.sleep(0)

        live = gauges.snapshot()
        assert live["stream_queue_active"] == 1
        assert live["stream_queue_bytes"] == 5
        assert live["stream_queue_waiting_putters"] == 1

        lease = await queue.get_lease()
        assert lease.item == b"first"
        await lease.release()
        await blocked
        await queue.close(discard=True)
        gauges.unregister_stream_queue(queue)

        retired = gauges.snapshot()
        assert retired["stream_queue_active"] == 0
        assert retired["stream_queue_bytes"] == 0
        assert retired["stream_queue_blocked_puts"] == 1

    asyncio.run(scenario())


def test_runtime_network_sampler_runs_proc_scans_off_event_loop(monkeypatch):
    async def scenario():
        gauges = RuntimeGauges()
        monkeypatch.setattr(runtime, "_open_socket_count", lambda: 8)
        monkeypatch.setattr(
            runtime,
            "_tcp_state_counts",
            lambda: {"ESTABLISHED": 6},
        )

        await gauges.start_network_sampler(interval_seconds=60)
        try:
            assert gauges.snapshot()["open_sockets"] == 8
            assert gauges.snapshot()["tcp_states"] == {"ESTABLISHED": 6}
        finally:
            await gauges.stop_network_sampler()

    asyncio.run(scenario())
