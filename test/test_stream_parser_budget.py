import asyncio
import gc
import threading
from types import SimpleNamespace

import pytest

import uni_api.streaming.sse as sse_module
from uni_api.admission import (
    RequestAdmissionController,
    bind_request_admission_lease,
    reset_request_admission_lease,
)
from uni_api.admission.memory import AdaptiveMemoryGovernor, ProcessMemorySample
from uni_api.streaming.sse import (
    IncrementalLineParser,
    IncrementalSSEParser,
    SSEProtocolError,
    StreamParserBufferBudgetExhausted,
    parse_owned_sse_event,
    sse_event_has_data_field,
)


def test_parser_budget_snapshot_allows_frame_finalizer_reentry():
    budget = sse_module._StreamParserRetainedBudget(64)
    budget.reserve(4)
    holder = {
        "frame": sse_module._RetainedTextFrame(
            "data",
            4,
            budget,
        )
    }

    class FinalizingMemoryGovernor:
        def snapshot(self):
            frame = holder.pop("frame")
            del frame
            return SimpleNamespace(capacity_bytes=64)

        def release(self, _category, _size):
            return None

    budget.memory_governor = FinalizingMemoryGovernor()
    result = {}
    errors = []

    def take_snapshot():
        try:
            result.update(budget.snapshot())
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=take_snapshot, daemon=True)
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive(), "snapshot deadlocked during frame finalization"
    assert errors == []
    assert result["used_bytes"] == 0


def test_incomplete_sse_frames_share_a_process_wide_byte_budget(monkeypatch):
    budget = sse_module._StreamParserRetainedBudget(10)
    monkeypatch.setattr(sse_module, "_STREAM_PARSER_RETAINED_BUDGET", budget)
    first = IncrementalSSEParser()
    second = IncrementalSSEParser()

    assert first.feed(b"data:a") == []
    assert budget.snapshot()["used_bytes"] == 6
    with pytest.raises(StreamParserBufferBudgetExhausted):
        second.feed(b"data:b")
    assert budget.snapshot()["used_bytes"] == 6

    del first
    gc.collect()
    assert budget.snapshot()["used_bytes"] == 0


def test_parser_budget_competes_with_adaptive_parent_and_releases_on_gc(
    monkeypatch,
):
    governor = AdaptiveMemoryGovernor(
        source=lambda: ProcessMemorySample(100, 1000, source="fake"),
        guard_bytes=100,
        guard_ratio=0,
        sample_cache_seconds=0,
    )
    assert governor.reserve_nowait("request_body", 795)
    budget = sse_module._StreamParserRetainedBudget(
        800,
        memory_governor=governor,
    )
    monkeypatch.setattr(sse_module, "_STREAM_PARSER_RETAINED_BUDGET", budget)

    parser = IncrementalSSEParser()
    assert parser.feed(b"data") == []
    assert governor.snapshot().reservations == {
        "request_body": 795,
        "stream_parser": 4,
    }
    with pytest.raises(StreamParserBufferBudgetExhausted):
        parser.feed(b"xx")

    del parser
    gc.collect()
    assert governor.snapshot().reservations == {"request_body": 795}
    governor.release("request_body", 795)
    assert governor.snapshot().reserved_bytes == 0


def test_emitted_sse_frame_owns_bytes_until_consumer_drops_it(monkeypatch):
    budget = sse_module._StreamParserRetainedBudget(64)
    monkeypatch.setattr(sse_module, "_STREAM_PARSER_RETAINED_BUDGET", budget)
    parser = IncrementalSSEParser()

    frames = parser.feed(b"data: ok\n\n")
    assert frames == ["data: ok"]
    assert budget.snapshot()["used_bytes"] == len(b"data: ok")

    del frames
    gc.collect()
    assert budget.snapshot()["used_bytes"] == 0


def test_parser_and_emitted_frame_release_the_budget_they_reserved(monkeypatch):
    original_budget = sse_module._StreamParserRetainedBudget(128)
    monkeypatch.setattr(
        sse_module,
        "_STREAM_PARSER_RETAINED_BUDGET",
        original_budget,
    )
    pending_parser = IncrementalSSEParser()
    emitted_parser = IncrementalSSEParser()
    assert pending_parser.feed(b"data:a") == []
    frames = emitted_parser.feed(b"data: ok\n\n")
    assert original_budget.snapshot()["used_bytes"] == 6 + len(b"data: ok")

    replacement_budget = sse_module._StreamParserRetainedBudget(128)
    monkeypatch.setattr(
        sse_module,
        "_STREAM_PARSER_RETAINED_BUDGET",
        replacement_budget,
    )
    pending_parser.discard()
    del frames
    gc.collect()

    assert original_budget.snapshot()["used_bytes"] == 0
    assert replacement_budget.snapshot()["used_bytes"] == 0


def test_line_parser_transfers_pending_budget_to_returned_line(monkeypatch):
    budget = sse_module._StreamParserRetainedBudget(64)
    monkeypatch.setattr(sse_module, "_STREAM_PARSER_RETAINED_BUDGET", budget)
    parser = IncrementalLineParser()

    lines = parser.feed(b"hello\n")
    assert lines == ["hello"]
    assert budget.snapshot()["used_bytes"] == 5

    del lines
    gc.collect()
    assert budget.snapshot()["used_bytes"] == 0


def test_owned_sse_close_releases_all_budgets_under_cancellation(monkeypatch):
    async def scenario():
        budget = sse_module._StreamParserRetainedBudget(1024 * 1024)
        monkeypatch.setattr(sse_module, "_STREAM_PARSER_RETAINED_BUDGET", budget)
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=4 * 1024 * 1024,
            max_response_bytes=4 * 1024 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)
        raw_event = IncrementalSSEParser().feed(
            b'data: {"value":"owned"}\n\n'
        )[0]
        owner = await parse_owned_sse_event(raw_event)
        json_owner = owner._json_owner
        assert json_owner is not None
        await json_owner._lock.acquire()
        close_task = asyncio.create_task(owner.aclose())
        await asyncio.sleep(0)
        try:
            close_task.cancel()
            close_task.cancel()
            await asyncio.sleep(0)
        finally:
            json_owner._lock.release()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        await owner.aclose()
        assert controller.snapshot()["reserved_response_bytes"] == 0
        raw_event = None
        gc.collect()
        assert budget.snapshot()["used_bytes"] == 0
        reset_request_admission_lease(token)
        await lease.release()

    asyncio.run(scenario())


def test_owned_sse_parse_failure_releases_raw_frame_immediately(monkeypatch):
    async def scenario():
        budget = sse_module._StreamParserRetainedBudget(1024 * 1024)
        monkeypatch.setattr(sse_module, "_STREAM_PARSER_RETAINED_BUDGET", budget)
        nested = b"[" * 129 + b"0" + b"]" * 129
        raw_event = IncrementalSSEParser().feed(b"data: " + nested + b"\n\n")[0]
        assert budget.snapshot()["used_bytes"] > 0
        with pytest.raises(SSEProtocolError, match="materialization"):
            await parse_owned_sse_event(raw_event)
        assert budget.snapshot()["used_bytes"] > 0
        raw_event = None
        gc.collect()
        assert budget.snapshot()["used_bytes"] == 0

    asyncio.run(scenario())


def test_data_field_presence_scan_does_not_materialize_payload():
    class FailOnSlice(str):
        def __getitem__(self, key):
            if isinstance(key, slice):
                raise AssertionError("field presence scan copied event text")
            return super().__getitem__(key)

    raw_event = FailOnSlice('event: response.completed\ndata: {"value":"' + "x" * 65536 + '"}')

    assert sse_event_has_data_field(raw_event) is True
    assert sse_event_has_data_field(FailOnSlice("event: response.completed")) is False
