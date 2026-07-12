import asyncio
import base64
import json

import pytest

import uni_api.providers.responses as responses_module
from uni_api.admission.json_parsing import parse_owned_json_value
from uni_api.providers.responses import (
    fetch_aws_response_stream,
    fetch_claude_response_stream,
    fetch_cloudflare_response_stream,
    fetch_dalle_response_stream,
    fetch_doubao_translation_response_stream,
    fetch_gemini_response_stream,
    fetch_gpt_response_stream,
)
from uni_api.streaming.sse import SSEOutputLimitError, SSEProtocolError


class _StreamingResponse:
    def __init__(self, chunks, *, content_type):
        self.status_code = 200
        self.headers = {"content-type": content_type}
        self.is_stream_consumed = False
        self._chunks = list(chunks)
        self.closed = 0

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed += 1


class _TerminalTrapResponse(_StreamingResponse):
    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk
        raise AssertionError("provider adapter read after semantic terminal")


class _StreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        await self.response.aclose()


class _Client:
    def __init__(self, response):
        self.response = response

    def stream(self, *_args, **_kwargs):
        return _StreamContext(self.response)


def test_dalle_non_sse_collects_transport_chunks_into_one_json_document():
    payload = {"created": 1, "data": [{"b64_json": "A" * 4096}]}
    encoded = json.dumps(payload).encode("utf-8")
    response = _StreamingResponse(
        [encoded[:17], encoded[17:101], encoded[101:]],
        content_type="application/json",
    )

    async def run():
        return [
            chunk
            async for chunk in fetch_dalle_response_stream(
                _Client(response),
                "https://example.test/v1/images/generations",
                {},
                {"prompt": "test"},
            )
        ]

    chunks = asyncio.run(run())
    assert chunks == [encoded]
    assert response.closed >= 1


def test_dalle_intermediate_done_event_is_not_a_full_stream_terminal():
    response = _StreamingResponse(
        [
            b"event: response.output_item.done\n"
            b'data: {"type":"response.output_item.done"}\n\n'
        ],
        content_type="text/event-stream",
    )
    emitted = []

    async def run():
        async for chunk in fetch_dalle_response_stream(
            _Client(response),
            "https://example.test/v1/images/generations",
            {},
            {"prompt": "test"},
        ):
            emitted.append(chunk)

    with pytest.raises(SSEProtocolError, match="without a terminal event"):
        asyncio.run(run())
    assert "response.output_item.done" in "".join(emitted)


def test_dalle_sse_accepts_only_a_full_image_terminal():
    response = _StreamingResponse(
        [
            b"event: image_generation.partial_image\n"
            b'data: {"type":"image_generation.partial_image"}\n\n',
            b"event: image_generation.completed\n"
            b'data: {"type":"image_generation.completed"}\n\n',
        ],
        content_type="text/event-stream",
    )

    async def run():
        return "".join(
            [
                chunk
                async for chunk in fetch_dalle_response_stream(
                    _Client(response),
                    "https://example.test/v1/images/generations",
                    {},
                    {"prompt": "test"},
                )
            ]
        )

    body = asyncio.run(run())
    assert "image_generation.partial_image" in body
    assert "image_generation.completed" in body


def test_doubao_translation_rejects_eof_without_protocol_terminal():
    response = _StreamingResponse(
        [
            b"event: response.output_text.delta\n"
            b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'
        ],
        content_type="text/event-stream",
    )

    async def run():
        async for _chunk in fetch_doubao_translation_response_stream(
            _Client(response),
            "https://example.test/v1/responses",
            {},
            {"input": "test"},
            "doubao-test",
            30,
        ):
            pass

    with pytest.raises(SSEProtocolError, match="without response.completed"):
        asyncio.run(run())


def test_gemini_compact_top_level_array_processes_every_object():
    frames = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "hello"}],
                    }
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": []},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 2,
                "candidatesTokenCount": 1,
                "totalTokenCount": 3,
            },
        },
    ]
    response = _StreamingResponse(
        [json.dumps(frames, separators=(",", ":")).encode("utf-8")],
        content_type="application/json",
    )

    async def run():
        return "".join(
            [
                chunk
                async for chunk in fetch_gemini_response_stream(
                    _Client(response),
                    "https://example.test/v1beta/models/gemini:streamGenerateContent",
                    {},
                    {"contents": []},
                    "gemini-test",
                    30,
                )
            ]
        )

    body = asyncio.run(run())
    assert '"content": "hello"' in body
    assert '"finish_reason": "stop"' in body
    assert body.endswith("data: [DONE]\n\n")


def test_gemini_sse_accepts_comments_and_data_without_space():
    content = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "hello-sse"}],
                }
            }
        ]
    }
    terminal = {
        "candidates": [
            {
                "content": {"role": "model", "parts": []},
                "finishReason": "STOP",
            }
        ]
    }
    response = _StreamingResponse(
        [
            b": keepalive\n\n"
            + b"data:"
            + json.dumps(content).encode()
            + b"\n\n"
            + b"data:"
            + json.dumps(terminal).encode()
            + b"\n\n"
        ],
        content_type="text/event-stream",
    )

    async def run():
        return "".join(
            [
                chunk
                async for chunk in fetch_gemini_response_stream(
                    _Client(response),
                    "https://example.test/gemini",
                    {},
                    {"contents": []},
                    "gemini-test",
                    30,
                )
            ]
        )

    body = asyncio.run(run())
    assert "hello-sse" in body
    assert body.endswith("data: [DONE]\n\n")


def test_gemini_compact_array_has_a_finite_objects_per_feed_limit():
    response = _StreamingResponse(
        [b"[" + b",".join([b"{}"] * 4097) + b"]"],
        content_type="application/json",
    )

    async def run():
        async for _chunk in fetch_gemini_response_stream(
            _Client(response),
            "https://example.test/gemini",
            {},
            {"contents": []},
            "gemini-test",
            30,
        ):
            pass

    with pytest.raises(SSEOutputLimitError, match="Gemini JSON objects"):
        asyncio.run(run())


@pytest.mark.parametrize(
    ("adapter", "terminal_payload"),
    [
        (
            fetch_cloudflare_response_stream,
            {"response": "done", "done": True},
        ),
        (
            fetch_claude_response_stream,
            {"type": "message_stop"},
        ),
    ],
)
def test_provider_semantic_terminal_does_not_wait_for_transport_eof(
    adapter,
    terminal_payload,
):
    event_name = (
        "message_stop" if adapter is fetch_claude_response_stream else "done"
    )
    response = _TerminalTrapResponse(
        [
            f"event: {event_name}\ndata: {json.dumps(terminal_payload)}\n\n".encode()
        ],
        content_type="text/event-stream",
    )

    async def run():
        return "".join(
            [
                chunk
                async for chunk in adapter(
                    _Client(response),
                    "https://example.test/provider",
                    {},
                    {},
                    "model-test",
                    30,
                )
            ]
        )

    body = asyncio.run(run())
    assert body.endswith("data: [DONE]\n\n")


def test_aws_invocation_metrics_terminal_does_not_wait_for_another_frame(
    monkeypatch,
):
    async def payloads(_response):
        encoded = base64.b64encode(
            json.dumps(
                {
                    "amazon-bedrock-invocationMetrics": {
                        "inputTokenCount": 2,
                        "outputTokenCount": 3,
                    }
                }
            ).encode()
        ).decode()
        owner = await parse_owned_json_value(
            json.dumps({"bytes": encoded})
        )
        try:
            yield owner
        finally:
            await owner.aclose()
        raise AssertionError("AWS adapter read after invocation metrics")

    monkeypatch.setattr(
        responses_module,
        "_iter_aws_eventstream_payloads",
        payloads,
    )
    response = _StreamingResponse([], content_type="application/vnd.amazon.eventstream")

    async def run():
        return "".join(
            [
                chunk
                async for chunk in fetch_aws_response_stream(
                    _Client(response),
                    "https://example.test/aws",
                    {},
                    {},
                    "model-test",
                    30,
                )
            ]
        )

    body = asyncio.run(run())
    assert body.endswith("data: [DONE]\n\n")


def test_gpt_responses_terminal_does_not_wait_for_transport_eof():
    response = _TerminalTrapResponse(
        [
            b"event: response.completed\n"
            b'data: {"type":"response.completed","response":{"usage":'
            b'{"input_tokens":2,"output_tokens":3}}}\n\n'
        ],
        content_type="text/event-stream",
    )

    async def run():
        return "".join(
            [
                chunk
                async for chunk in fetch_gpt_response_stream(
                    _Client(response),
                    "https://example.test/v1/chat/completions",
                    {},
                    {"model": "model-test", "messages": []},
                    30,
                )
            ]
        )

    body = asyncio.run(run())
    assert '"prompt_tokens": 2' in body
    assert '"completion_tokens": 3' in body
    assert body.endswith("data: [DONE]\n\n")


def test_gpt_long_reasoning_delta_only_retains_trailing_escape_marker():
    long_prefix = "x" * 4096
    events = [
        {"choices": [{"delta": {"reasoning": long_prefix + "\\"}}]},
        {"choices": [{"delta": {"reasoning": "n"}}]},
        {"choices": [{"delta": {"reasoning": "\\n"}}]},
        {"choices": [{"delta": {"reasoning": "tail"}}]},
    ]
    response = _StreamingResponse(
        [
            "".join(
                f"data: {json.dumps(event)}\n\n" for event in events
            ).encode()
            + b"data: [DONE]\n\n"
        ],
        content_type="text/event-stream",
    )

    async def run():
        return "".join(
            [
                chunk
                async for chunk in fetch_gpt_response_stream(
                    _Client(response),
                    "https://example.test/v1/chat/completions",
                    {},
                    {"model": "model-test", "messages": []},
                    30,
                )
            ]
        )

    body = asyncio.run(run())
    assert long_prefix in body
    assert "tail" in body
    assert body.endswith("data: [DONE]\n\n")
