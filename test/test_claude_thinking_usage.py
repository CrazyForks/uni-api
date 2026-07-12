import asyncio
import json

from uni_api.providers.responses import _yield_claude_chat_completion, fetch_claude_response_stream


class _BufferedResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    async def aread(self):
        return self._body


class _StreamingResponse:
    status_code = 200

    async def aiter_text(self):
        yield (
            'event: message_start\n'
            'data: {"type":"message_start","message":{"usage":{"input_tokens":25}}}\n\n'
            'event: message_delta\n'
            'data: {"type":"message_delta","usage":{"output_tokens":348,'
            '"output_tokens_details":{"thinking_tokens":312}}}\n\n'
        )


class _StreamContext:
    async def __aenter__(self):
        return _StreamingResponse()

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _StreamingClient:
    def stream(self, *args, **kwargs):
        return _StreamContext()


async def _collect_non_streaming_usage():
    response = _BufferedResponse(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "answer"}],
            "usage": {
                "input_tokens": 25,
                "output_tokens": 348,
                "output_tokens_details": {"thinking_tokens": 312},
            },
        }
    )

    return [chunk async for chunk in _yield_claude_chat_completion(response, "claude-fable-5")]


def test_claude_non_streaming_usage_maps_thinking_tokens_to_reasoning_tokens():
    payload = json.loads(asyncio.run(_collect_non_streaming_usage())[0])

    assert payload["usage"]["completion_tokens"] == 348
    assert payload["usage"]["completion_tokens_details"]["reasoning_tokens"] == 312
    assert payload["usage"]["total_tokens"] == 373


async def _collect_streaming_usage():
    return [
        chunk
        async for chunk in fetch_claude_response_stream(
            _StreamingClient(),
            "https://example.test/v1/messages",
            {},
            {"model": "claude-fable-5"},
            "claude-fable-5",
            30,
        )
    ]


def test_claude_streaming_usage_maps_final_thinking_tokens_to_reasoning_tokens():
    chunks = asyncio.run(_collect_streaming_usage())
    usage_chunks = [json.loads(chunk.removeprefix("data: ")) for chunk in chunks if chunk.startswith("data: {")]
    usage = next(chunk["usage"] for chunk in usage_chunks if chunk.get("usage"))

    assert usage["completion_tokens"] == 348
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 312
    assert usage["total_tokens"] == 373
