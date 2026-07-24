import asyncio
import threading

import pytest

from uni_api.admission import (
    RequestAdmissionController,
    bind_request_admission_lease,
    reset_request_admission_lease,
)
from uni_api.admission.json_parsing import (
    ReusableJSONParseWorkspace,
    parse_owned_json_value,
    parsed_json_value,
    run_json_cpu,
)


def test_cancellation_wins_after_json_worker_finishes_with_error():
    async def scenario():
        started = threading.Event()
        release = threading.Event()

        def fail_later():
            started.set()
            release.wait(timeout=2)
            raise ValueError("worker failure must not replace cancellation")

        task = asyncio.create_task(run_json_cpu(fail_later))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_memoryview_is_reserved_before_materializing_parse_copy():
    async def scenario():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024 * 1024,
            max_response_bytes=1024 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)
        source = memoryview(b'{"value":"' + b"x" * 64_000 + b'"}')
        try:
            async with parsed_json_value(source) as value:
                assert value["value"].startswith("x")
                assert controller.snapshot()["reserved_response_bytes"] > len(source)
        finally:
            reset_request_admission_lease(token)
            await lease.release()

        assert controller.snapshot()["reserved_response_bytes"] == 0

    asyncio.run(scenario())


def test_owned_json_close_is_cancellation_safe_while_lock_is_contended():
    async def scenario():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024 * 1024,
            max_response_bytes=1024 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)
        owner = await parse_owned_json_value(b'{"value":"owned"}')
        await owner._lock.acquire()
        close_task = asyncio.create_task(owner.aclose())
        await asyncio.sleep(0)
        try:
            close_task.cancel()
            close_task.cancel()
            await asyncio.sleep(0)
        finally:
            owner._lock.release()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        await owner.aclose()
        assert controller.snapshot()["reserved_response_bytes"] == 0
        reset_request_admission_lease(token)
        await lease.release()

    asyncio.run(scenario())


def test_reusable_json_workspace_holds_high_water_and_releases_once():
    async def scenario():
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
        workspace = await ReusableJSONParseWorkspace.create()
        try:
            first = await parse_owned_json_value(
                b'{"value":"first"}',
                workspace=workspace,
                workspace_extra_bytes=4096,
            )
            first_capacity = workspace.capacity
            assert first_capacity > 4096
            assert controller.snapshot()["reserved_response_bytes"] == (
                first_capacity
            )
            await first.aclose()
            assert first._close_task is None
            assert controller.snapshot()["reserved_response_bytes"] == (
                first_capacity
            )

            second = await parse_owned_json_value(
                b'{"value":1}',
                workspace=workspace,
                workspace_extra_bytes=1024,
            )
            await second.aclose()
            assert second._close_task is None
            assert workspace.capacity == first_capacity
            assert controller.snapshot()["reserved_response_bytes"] == (
                first_capacity
            )
        finally:
            await workspace.aclose()
            assert controller.snapshot()["reserved_response_bytes"] == 0
            reset_request_admission_lease(token)
            await lease.release()

    asyncio.run(scenario())
