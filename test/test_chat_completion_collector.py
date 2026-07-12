import asyncio
import json

import pytest

from uni_api.admission import (
    RequestAdmissionController,
    UpstreamResponseBudgetExhausted,
    bind_request_admission_lease,
    reset_request_admission_lease,
)
from uni_api.streaming.chat_completion_collector import (
    collect_openai_chat_completion_from_streaming_sse,
)
from uni_api.streaming.sse import (
    SSEBufferOverflowError,
    SSEOutputLimitError,
    SSEProtocolError,
)


def _frame(payload):
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def test_collector_single_pass_preserves_tools_usage_and_closes_source():
    async def scenario():
        closed = False

        async def source():
            nonlocal closed
            try:
                yield _frame(
                    {
                        "created": 123,
                        "choices": [
                            {
                                "delta": {
                                    "content": "hello ",
                                    "reasoning_content": "think",
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "lookup",
                                                "arguments": '{"q":',
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                    }
                )
                yield _frame(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "world",
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": '"x"}'},
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                )
                yield _frame(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 5,
                            "total_tokens": 8,
                        },
                    }
                )
                yield "data: [DONE]\n\n"
                await asyncio.Event().wait()
            finally:
                closed = True

        raw = await collect_openai_chat_completion_from_streaming_sse(
            source(),
            model="gpt-test",
        )
        payload = json.loads(raw)

        assert closed is True
        assert payload["created"] == 123
        assert payload["choices"][0]["message"]["content"] is None
        assert payload["choices"][0]["message"]["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                },
            }
        ]
        assert payload["usage"]["total_tokens"] == 8

    asyncio.run(scenario())


def test_collector_preserves_content_and_reasoning_fragments():
    async def scenario():
        async def source():
            yield _frame(
                {
                    "created": 321,
                    "choices": [
                        {
                            "delta": {
                                "content": "hello ",
                                "reasoning_content": "think ",
                            }
                        }
                    ],
                }
            )
            yield _frame(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": "world",
                                "reasoning_content": "again",
                            }
                        }
                    ]
                }
            )
            yield "data: [DONE]\n\n"

        raw = await collect_openai_chat_completion_from_streaming_sse(
            source(),
            model="gpt-test",
        )
        payload = json.loads(raw)
        message = payload["choices"][0]["message"]
        assert message["content"] == "hello world"
        assert message["reasoning_content"] == "think again"

    asyncio.run(scenario())


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_collector_preserves_non_stop_finish_reason(finish_reason):
    async def scenario():
        async def source():
            yield _frame(
                {
                    "choices": [
                        {
                            "delta": {"content": "partial"},
                            "finish_reason": finish_reason,
                        }
                    ]
                }
            )
            yield "data: [DONE]\n\n"

        raw = await collect_openai_chat_completion_from_streaming_sse(
            source(),
            model="gpt-test",
        )
        assert json.loads(raw)["choices"][0]["finish_reason"] == finish_reason

    asyncio.run(scenario())


def test_collector_terminal_length_overrides_tool_calls_finish_reason():
    async def scenario():
        async def source():
            yield _frame(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_partial",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"x":',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "length",
                        }
                    ]
                }
            )
            yield "data: [DONE]\n\n"

        raw = await collect_openai_chat_completion_from_streaming_sse(
            source(),
            model="gpt-test",
        )
        payload = json.loads(raw)
        assert payload["choices"][0]["finish_reason"] == "length"
        assert payload["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ] == '{"x":'

    asyncio.run(scenario())


@pytest.mark.parametrize("bad_index", [True, "1", 1.0, -1])
def test_collector_rejects_noncanonical_tool_indices_and_closes_source(bad_index):
    async def scenario():
        closed = False

        async def source():
            nonlocal closed
            try:
                yield _frame(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": bad_index,
                                            "function": {"arguments": "{}"},
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
            finally:
                closed = True

        with pytest.raises(SSEProtocolError):
            await collect_openai_chat_completion_from_streaming_sse(
                source(),
                model="gpt-test",
            )
        assert closed is True

    asyncio.run(scenario())


def test_collector_enforces_fragment_tool_input_and_terminal_limits():
    async def collect(chunks, **limits):
        async def source():
            for chunk in chunks:
                yield chunk

        return await collect_openai_chat_completion_from_streaming_sse(
            source(),
            model="gpt-test",
            **limits,
        )

    content_frames = [
        _frame({"choices": [{"delta": {"content": "a"}}]}),
        _frame({"choices": [{"delta": {"content": "b"}}]}),
        "data: [DONE]\n\n",
    ]
    with pytest.raises(SSEOutputLimitError, match="fragments"):
        asyncio.run(collect(content_frames, max_fragments=1))

    tools = [
        {"index": 0, "function": {"arguments": "{}"}},
        {"index": 1, "function": {"arguments": "{}"}},
    ]
    with pytest.raises(SSEOutputLimitError, match="tool calls"):
        asyncio.run(
            collect(
                [
                    _frame({"choices": [{"delta": {"tool_calls": tools}}]}),
                    "data: [DONE]\n\n",
                ],
                max_tool_calls=1,
            )
        )

    with pytest.raises(SSEBufferOverflowError, match="SSE input"):
        asyncio.run(
            collect(
                content_frames,
                max_input_bytes=len(content_frames[0].encode("utf-8")),
            )
        )

    with pytest.raises(SSEProtocolError, match=r"without data: \[DONE\]"):
        asyncio.run(collect(content_frames[:-1]))


def test_collector_reserves_weighted_response_memory_until_request_release():
    async def scenario():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=4096,
            body_budget_bytes=16 * 1024,
            max_response_bytes=16 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)

        async def source():
            yield _frame({"choices": [{"delta": {"content": "ok"}}]})
            yield "data: [DONE]\n\n"

        try:
            await collect_openai_chat_completion_from_streaming_sse(
                source(),
                model="gpt-test",
            )
            reserved = controller.snapshot()["reserved_response_bytes"]
            assert reserved >= 4096
        finally:
            reset_request_admission_lease(token)
            await lease.release()

        assert controller.snapshot()["reserved_response_bytes"] == 0

    asyncio.run(scenario())


def test_dense_ignored_json_is_rejected_before_materialization_exceeds_budget():
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
        closed = False

        async def source():
            nonlocal closed
            try:
                yield _frame(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "unused": [{} for _ in range(20_000)]
                                }
                            }
                        ]
                    }
                )
            finally:
                closed = True

        try:
            with pytest.raises(UpstreamResponseBudgetExhausted):
                await collect_openai_chat_completion_from_streaming_sse(
                    source(),
                    model="gpt-test",
                )
            assert closed is True
            assert controller.snapshot()["reserved_response_bytes"] == 0
        finally:
            reset_request_admission_lease(token)
            await lease.release()

    asyncio.run(scenario())


def test_collector_accepts_large_image_below_eight_mib_and_rejects_above_limit():
    async def collect_content(content):
        async def source():
            yield _frame({"choices": [{"delta": {"content": content}}]})
            yield "data: [DONE]\n\n"

        return await collect_openai_chat_completion_from_streaming_sse(
            source(),
            model="gpt-image-test",
        )

    accepted = "A" * (4 * 1024 * 1024 + 1)
    result = asyncio.run(collect_content(accepted))
    assert len(result) > len(accepted)

    oversized = "B" * (8 * 1024 * 1024 + 1)
    with pytest.raises(SSEBufferOverflowError):
        asyncio.run(collect_content(oversized))
