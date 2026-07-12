import asyncio
import json

import pytest

import uni_api.streaming.sse as sse_module
from uni_api.admission import (
    RequestAdmissionController,
    bind_request_admission_lease,
    reset_request_admission_lease,
)
from uni_api.streaming.chat_completion_events import (
    build_chat_completion_chunk_sse,
    responses_usage_to_chat_completion_usage,
)
from uni_api.streaming.responses_events import (
    extract_responses_stream_sse_event,
    stream_responses_to_chat_completions,
)
from uni_api.streaming.sse import SSEBufferOverflowError, SSEProtocolError


def test_responses_event_parser_reads_named_event_and_payload():
    event_type, payload = extract_responses_stream_sse_event(
        'event: response.output_text.delta\ndata: {"delta": "hello"}'
    )

    assert event_type == "response.output_text.delta"
    assert payload == {"delta": "hello"}


def test_chat_completion_chunk_builder_outputs_openai_chunk_shape():
    raw = build_chat_completion_chunk_sse(
        response_id="chatcmpl_1",
        created_at=123,
        model_name="gpt-5.4",
        delta={"role": "assistant", "content": "hello"},
    )
    payload = json.loads(raw.removeprefix("data: "))

    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "hello"


def test_responses_usage_to_chat_completion_usage_maps_input_output_tokens():
    usage = responses_usage_to_chat_completion_usage(
        {
            "input_tokens": 3,
            "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 4},
        }
    )

    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 8
    assert usage["prompt_tokens_details"]["cached_tokens"] == 2
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 4


def test_responses_stream_to_chat_completions_converts_text_reasoning_and_done():
    async def upstream_iter():
        yield b"event: response.output_text.delt"
        yield b'a\ndata: {"type": "response.output_text.delta", "delta": "hello"}\n\n'
        yield b'event: response.reasoning_summary_text.delta\ndata: {"delta": "why"}\n\n'
        yield b"data: [DONE]\n\n"

    async def run():
        chunks = []
        async for chunk in stream_responses_to_chat_completions(upstream_iter(), request_model="gpt-5.4"):
            chunks.append(chunk)
        return "".join(chunks)

    body = asyncio.run(run())

    assert '"content": "hello"' in body
    assert '"reasoning_content": "why"' in body
    assert body.endswith("data: [DONE]\n\n")


def test_responses_output_item_collection_has_total_item_limit():
    async def upstream_iter():
        for index in range(3):
            yield (
                "event: response.output_item.done\n"
                f'data: {{"type":"response.output_item.done","output_index":{index},'
                f'"item":{{"type":"function_call","name":"f{index}"}}}}\n\n'
            ).encode()

    async def run():
        async for _chunk in stream_responses_to_chat_completions(
            upstream_iter(),
            request_model="gpt-test",
            max_collected_output_items=2,
        ):
            pass

    with pytest.raises(SSEBufferOverflowError, match="item count"):
        asyncio.run(run())


def test_responses_output_item_collection_has_total_byte_limit():
    async def upstream_iter():
        yield (
            "event: response.output_item.done\n"
            'data: {"type":"response.output_item.done","output_index":0,'
            '"item":{"type":"function_call","name":"large","arguments":"xxxxxxxx"}}\n\n'
        ).encode()

    async def run():
        async for _chunk in stream_responses_to_chat_completions(
            upstream_iter(),
            request_model="gpt-test",
            max_collected_output_bytes=16,
        ):
            pass

    with pytest.raises(SSEBufferOverflowError, match="collected output items"):
        asyncio.run(run())


def test_collected_output_items_remain_charged_until_request_lifecycle_ends():
    async def run():
        consumed = asyncio.Event()
        release_upstream = asyncio.Event()

        async def upstream_iter():
            for index in range(100):
                yield (
                    "event: response.output_item.done\n"
                    "data: "
                    + json.dumps(
                        {
                            "type": "response.output_item.done",
                            "output_index": index,
                            "item": {
                                "type": "function_call",
                                "name": f"f{index}",
                                "arguments": "x" * 10_000,
                            },
                        }
                    )
                    + "\n\n"
                ).encode()
            consumed.set()
            await release_upstream.wait()
            yield b"data: [DONE]\n\n"

        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=32 * 1024 * 1024,
            max_response_bytes=128 * 1024 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)
        stream = stream_responses_to_chat_completions(
            upstream_iter(),
            request_model="gpt-test",
        )
        first_output = asyncio.create_task(stream.__anext__())
        try:
            await asyncio.wait_for(consumed.wait(), timeout=2)
            snapshot = controller.snapshot()
            assert snapshot["reserved_response_bytes"] > 1_000_000

            release_upstream.set()
            await asyncio.wait_for(first_output, timeout=2)
            await stream.aclose()
            assert controller.snapshot()["reserved_response_bytes"] == 0
        finally:
            release_upstream.set()
            if not first_output.done():
                first_output.cancel()
                await asyncio.gather(first_output, return_exceptions=True)
            await stream.aclose()
            reset_request_admission_lease(token)
            await lease.release()

        assert controller.snapshot()["reserved_response_bytes"] == 0

    asyncio.run(run())


def test_responses_output_item_done_replaces_duplicate_index_without_stale_output():
    async def upstream_iter():
        for name in ("first", "second"):
            yield (
                "event: response.output_item.done\n"
                "data: "
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {"type": "function_call", "name": name},
                    }
                )
                + "\n\n"
            ).encode()

    async def run():
        return "".join(
            [
                chunk
                async for chunk in stream_responses_to_chat_completions(
                    upstream_iter(),
                    request_model="gpt-test",
                )
            ]
        )

    body = asyncio.run(run())
    assert '"name": "second"' in body
    assert '"name": "first"' not in body


def test_converter_clears_each_raw_batch_slot_before_releasing_parser_budget(
    monkeypatch,
):
    async def run():
        budget = sse_module._StreamParserRetainedBudget(1024)
        monkeypatch.setattr(sse_module, "_STREAM_PARSER_RETAINED_BUDGET", budget)

        async def upstream_iter():
            yield b": first\n\n: second\n\n"

        stream = stream_responses_to_chat_completions(
            upstream_iter(),
            request_model="gpt-test",
        )
        assert await stream.__anext__() == ": first\n\n"
        assert budget.snapshot()["used_bytes"] == len(b": first") + len(b": second")
        assert await stream.__anext__() == ": second\n\n"
        assert budget.snapshot()["used_bytes"] == len(b": second")
        await stream.aclose()
        assert budget.snapshot()["used_bytes"] == 0

    asyncio.run(run())


def test_closing_after_first_terminal_chunk_releases_event_and_collector_owners(
    monkeypatch,
):
    async def run():
        budget = sse_module._StreamParserRetainedBudget(4 * 1024 * 1024)
        monkeypatch.setattr(sse_module, "_STREAM_PARSER_RETAINED_BUDGET", budget)
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=16 * 1024 * 1024,
            max_response_bytes=16 * 1024 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)

        async def upstream_iter():
            yield (
                "event: response.completed\n"
                "data: "
                + json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "x" * 100_000,
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                )
                + "\n\n"
            ).encode()

        stream = stream_responses_to_chat_completions(
            upstream_iter(),
            request_model="gpt-test",
        )
        try:
            first = await stream.__anext__()
            assert "chat.completion.chunk" in first
            assert controller.snapshot()["reserved_response_bytes"] > 0
            await stream.aclose()
            assert controller.snapshot()["reserved_response_bytes"] == 0
            assert budget.snapshot()["used_bytes"] == 0
        finally:
            await stream.aclose()
            reset_request_admission_lease(token)
            await lease.release()

    asyncio.run(run())


def test_responses_stream_rejects_partial_eof_without_terminal_event():
    async def upstream_iter():
        yield (
            b"event: response.output_text.delta\n"
            b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'
        )

    async def run():
        async for _chunk in stream_responses_to_chat_completions(
            upstream_iter(),
            request_model="gpt-test",
        ):
            pass

    with pytest.raises(SSEProtocolError, match="without response.completed"):
        asyncio.run(run())


@pytest.mark.parametrize(
    ("event_type", "payload"),
    [
        (
            "response.failed",
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {"code": "server_error", "message": "failed"},
                },
            },
        ),
        (
            "error",
            {"type": "error", "error": {"message": "failed"}},
        ),
    ],
)
def test_responses_failure_terminals_never_synthesize_partial_success(
    event_type,
    payload,
):
    emitted = []

    async def upstream_iter():
        yield (
            "event: response.output_item.done\n"
            "data: "
            + json.dumps(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "partial"}
                        ],
                    },
                }
            )
            + "\n\n"
        ).encode()
        yield (
            f"event: {event_type}\n"
            f"data: {json.dumps(payload)}\n\n"
        ).encode()
        yield b"data: [DONE]\n\n"

    async def run():
        async for chunk in stream_responses_to_chat_completions(
            upstream_iter(),
            request_model="gpt-test",
        ):
            emitted.append(chunk)

    with pytest.raises(SSEProtocolError, match="upstream emitted"):
        asyncio.run(run())

    body = "".join(emitted)
    assert "finish_reason" not in body
    assert "data: [DONE]" not in body


@pytest.mark.parametrize(
    ("reason", "finish_reason"),
    [
        ("max_output_tokens", "length"),
        ("content_filter", "content_filter"),
    ],
)
def test_responses_incomplete_is_a_finite_partial_terminal(reason, finish_reason):
    async def upstream_iter():
        yield (
            b"event: response.output_text.delta\n"
            b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'
        )
        yield (
            "event: response.incomplete\n"
            "data: "
            + json.dumps(
                {
                    "type": "response.incomplete",
                    "response": {
                        "id": "resp_partial",
                        "status": "incomplete",
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": "partial"}
                                ],
                            }
                        ],
                        "incomplete_details": {"reason": reason},
                        "usage": {"input_tokens": 2, "output_tokens": 1},
                    },
                }
            )
            + "\n\n"
        ).encode()

    async def run():
        return "".join(
            [
                chunk
                async for chunk in stream_responses_to_chat_completions(
                    upstream_iter(),
                    request_model="gpt-test",
                )
            ]
        )

    body = asyncio.run(run())
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: {")
    ]
    assert any(
        payload["choices"][0].get("finish_reason") == finish_reason
        for payload in payloads
        if payload.get("choices")
    )
    assert body.endswith("data: [DONE]\n\n")


def test_completed_payload_output_obeys_item_and_byte_limits():
    async def run(payload, **limits):
        async def upstream_iter():
            yield (
                "event: response.completed\n"
                f"data: {json.dumps(payload)}\n\n"
            ).encode()

        async for _chunk in stream_responses_to_chat_completions(
            upstream_iter(),
            request_model="gpt-test",
            **limits,
        ):
            pass

    item_payload = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [
                {"type": "message", "content": []},
                {"type": "message", "content": []},
            ],
        },
    }
    with pytest.raises(SSEBufferOverflowError, match="item count"):
        asyncio.run(run(item_payload, max_collected_output_items=1))

    byte_payload = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "x" * 64}],
                }
            ],
        },
    }
    with pytest.raises(SSEBufferOverflowError, match="completed output items"):
        asyncio.run(run(byte_payload, max_collected_output_bytes=16))
