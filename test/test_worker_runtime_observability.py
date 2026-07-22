import asyncio
import math

from uni_api.observability import worker_runtime
from uni_api.observability.worker_runtime import (
    CumulativeHistogram,
    WorkerRuntimeObserver,
)


def test_cumulative_histogram_records_inclusive_buckets():
    histogram = CumulativeHistogram((10, 100, 1000))

    assert histogram.observe(10)
    assert histogram.observe(50)
    assert histogram.observe(500)
    assert not histogram.observe(-1)
    assert not histogram.observe(float("nan"))

    snapshot = histogram.snapshot()
    assert snapshot["count"] == 3
    assert snapshot["sum_ms"] == 560
    assert snapshot["cumulative_buckets"] == {
        "10": 1,
        "100": 2,
        "1000": 3,
    }
    assert snapshot["infinite_bucket"] == 3


def test_worker_sample_reports_cpu_stream_rates_inflight_and_cpu_per_mib():
    emitted = []
    wall = [100.0]
    cpu = [20.0]
    observer = WorkerRuntimeObserver(
        inflight_supplier=lambda: 7,
        snapshot_emitter=emitted.append,
        cpu_profile_enabled=False,
        monotonic=lambda: wall[0],
        process_time=lambda: cpu[0],
    )
    observer.record_sse_chunk(2 * 1024 * 1024)
    observer.record_sse_event(100)
    observer.record_sse_event(100)

    wall[0] = 105.0
    cpu[0] = 24.5
    snapshot = observer.sample_now()

    assert math.isclose(snapshot["worker_cpu_cores"], 0.9)
    assert math.isclose(snapshot["worker_sse_events_per_second"], 0.4)
    assert math.isclose(
        snapshot["worker_sse_bytes_per_second"],
        (2 * 1024 * 1024) / 5,
    )
    assert math.isclose(
        snapshot["worker_cpu_seconds_per_sse_mebibyte"],
        2.25,
    )
    assert snapshot["worker_inflight_requests"] == 7
    assert emitted == [snapshot]


def test_terminal_hop_observation_updates_histogram_and_fail_open_emitter():
    emitted = []
    observer = WorkerRuntimeObserver(
        terminal_hop_emitter=emitted.append,
        cpu_profile_enabled=False,
    )

    assert observer.record_terminal_hop(
        {
            "lag_ms": 7654.25,
            "request_id": "request-safe",
            "terminal_wire_sha256": "a" * 64,
        }
    )
    assert not observer.record_terminal_hop({"lag_ms": -3})

    snapshot = observer.snapshot()
    histogram = snapshot["oaix_terminal_flush_to_ember_receive_histogram"]
    assert histogram["count"] == 1
    assert histogram["cumulative_buckets"]["5000"] == 0
    assert histogram["cumulative_buckets"]["10000"] == 1
    assert snapshot["oaix_terminal_flush_to_ember_receive_invalid_total"] == 1
    assert emitted[0]["lag_ms"] == 7654.25


def test_sustained_cpu_threshold_triggers_once_then_obeys_cooldown(monkeypatch):
    async def scenario():
        wall = [100.0]
        cpu = [20.0]
        profiles = []
        observer = WorkerRuntimeObserver(
            cpu_profile_enabled=True,
            cpu_profile_trigger_cores=0.9,
            cpu_profile_trigger_samples=2,
            cpu_profile_cooldown_seconds=900,
            monotonic=lambda: wall[0],
            process_time=lambda: cpu[0],
        )

        async def fake_profile(trigger_cpu_cores):
            profiles.append(trigger_cpu_cores)

        monkeypatch.setattr(worker_runtime.sys, "platform", "linux")
        monkeypatch.setattr(observer, "_run_profile", fake_profile)

        wall[0] = 105.0
        cpu[0] = 24.6
        observer.sample_now(emit=False)
        assert profiles == []

        wall[0] = 110.0
        cpu[0] = 29.2
        observer.sample_now(emit=False)
        assert observer._profile_task is not None
        await observer._profile_task
        assert len(profiles) == 1
        assert math.isclose(profiles[0], 0.92)

        wall[0] = 115.0
        cpu[0] = 33.8
        observer.sample_now(emit=False)
        wall[0] = 120.0
        cpu[0] = 38.4
        observer.sample_now(emit=False)
        await asyncio.sleep(0)
        assert len(profiles) == 1

    asyncio.run(scenario())
