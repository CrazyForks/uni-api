import os
import sys
import asyncio
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import RequestModel
from core.request import get_payload
from core.response import fetch_gpt_response_stream, _responses_output_to_text
from core.utils import collect_openai_chat_completion_from_streaming_sse


class _DummyStreamingResponse:
    def __init__(self, chunks):
        self.status_code = 200
        self._chunks = list(chunks)

    async def aread(self):
        return b""

    async def aiter_text(self):
        for chunk in self._chunks:
            yield chunk

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk.encode("utf-8") if isinstance(chunk, str) else chunk


class _DummyStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummyClient:
    def __init__(self, chunks):
        self._response = _DummyStreamingResponse(chunks)

    def stream(self, method, url, headers=None, content=None, timeout=None):
        _ = (method, url, headers, content, timeout)
        return _DummyStreamContext(self._response)


async def _collect_stream_body(chunks):
    client = _DummyClient(chunks)
    body_parts = []
    async for chunk in fetch_gpt_response_stream(
        client,
        "https://example.com/v1/responses",
        {"Accept": "text/event-stream"},
        {
            "model": "gpt-image-2",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "draw a meme"}],
                }
            ],
            "stream": True,
        },
        30,
    ):
        if isinstance(chunk, bytes):
            body_parts.append(chunk.decode("utf-8"))
        else:
            body_parts.append(chunk)
    return "".join(body_parts)


async def _collect_non_stream_chat_completion(chunks):
    client = _DummyClient(chunks)

    async def converted_chunks():
        async for chunk in fetch_gpt_response_stream(
            client,
            "https://example.com/v1/responses",
            {"Accept": "text/event-stream"},
            {
                "model": "gpt-5.4",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "say test"}],
                    }
                ],
                "stream": True,
            },
            30,
        ):
            yield chunk

    return await collect_openai_chat_completion_from_streaming_sse(
        converted_chunks(),
        model="gpt-5.4",
    )


def test_fetch_gpt_response_stream_translates_image_output_item_done_and_drops_keepalive():
    chunks = [
        ": keepalive\n\n",
        (
            'event: response.created\n'
            'data: {"type":"response.created","response":{"id":"resp_img_done","status":"in_progress","model":"gpt-image-2","created_at":1710000000}}\n\n'
        ),
        (
            'event: keepalive\n'
            'data: {"type":"keepalive","sequence_number":7,"id":"resp_img_done"}\n\n'
        ),
        (
            'event: response.output_item.done\n'
            'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"image_generation_call","id":"ig_done","status":"completed","result":"done-image","output_format":"png"}}\n\n'
        ),
        "data: [DONE]\n\n",
    ]

    body = asyncio.run(_collect_stream_body(chunks))

    assert ": keepalive" in body
    assert '"type":"keepalive"' not in body.replace(" ", "")
    assert "data:image/png;base64,done-image" in body
    assert '"id":"resp_img_done"' in body.replace(" ", "")
    assert '"finish_reason":"stop"' in body.replace(" ", "")
    assert body.rstrip().endswith("data: [DONE]")


def test_fetch_gpt_response_stream_translates_responses_reasoning_summary():
    chunks = [
        (
            'event: response.created\n'
            'data: {"type":"response.created","response":{"id":"resp_reasoning","status":"in_progress","model":"gpt-image-2","created_at":1710000000}}\n\n'
        ),
        (
            'event: response.reasoning_summary_text.delta\n'
            'data: {"type":"response.reasoning_summary_text.delta","delta":"正在构思画面"}\n\n'
        ),
        (
            'event: response.reasoning_summary_text.done\n'
            'data: {"type":"response.reasoning_summary_text.done"}\n\n'
        ),
        "data: [DONE]\n\n",
    ]

    body = asyncio.run(_collect_stream_body(chunks))
    compact_body = body.replace(" ", "")

    assert '"reasoning_content":"正在构思画面"' in compact_body
    assert '"reasoning_content":"\\n\\n"' in compact_body
    assert body.rstrip().endswith("data: [DONE]")


def test_codex_non_stream_collection_keeps_responses_usage():
    chunks = [
        (
            'event: response.created\n'
            'data: {"type":"response.created","response":{"id":"resp_usage","status":"in_progress","model":"gpt-5.4","created_at":1710000000}}\n\n'
        ),
        (
            'event: response.output_text.delta\n'
            'data: {"type":"response.output_text.delta","delta":"test"}\n\n'
        ),
        (
            'event: response.completed\n'
            'data: {"type":"response.completed","response":{"id":"resp_usage","model":"gpt-5.4","created_at":1710000000,"usage":{"input_tokens":7,"output_tokens":2,"total_tokens":9,"input_tokens_details":{"cached_tokens":1},"output_tokens_details":{"reasoning_tokens":3}}}}\n\n'
        ),
        "data: [DONE]\n\n",
    ]

    response_json = asyncio.run(_collect_non_stream_chat_completion(chunks))
    response = json.loads(response_json)

    assert response["choices"][0]["message"]["content"] == "test"
    assert response["usage"]["prompt_tokens"] == 7
    assert response["usage"]["completion_tokens"] == 2
    assert response["usage"]["total_tokens"] == 9
    assert response["usage"]["prompt_tokens_details"]["cached_tokens"] == 1
    assert response["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3


def test_responses_output_to_text_includes_image_generation_call():
    content, reasoning_content = _responses_output_to_text(
        {
            "output": [
                {
                    "type": "image_generation_call",
                    "result": "img-b64",
                    "output_format": "jpeg",
                }
            ]
        }
    )

    assert content == "![image](data:image/jpeg;base64,img-b64)"
    assert reasoning_content == ""


def test_codex_gpt_image_2_payload_omits_chat_defaults_even_with_overrides():
    request = RequestModel(
        model="gpt-image-2",
        messages=[{"role": "user", "content": "draw a meme"}],
        stream=True,
    )
    provider = {
        "provider": "fugue-codex",
        "engine": "codex",
        "base_url": "https://oaix.fugue.pro/v1/responses",
        "api": "test-key",
        "model": ["gpt-image-2"],
        "preferences": {
            "post_body_parameter_overrides": {
                "parallel_tool_calls": True,
                "reasoning": {"effort": "medium", "summary": "auto"},
                "include": ["reasoning.encrypted_content"],
            }
        },
        "tools": True,
    }

    _, _, payload = asyncio.run(get_payload(request, "codex", provider, api_key="test-key"))

    assert payload["model"] == "gpt-image-2"
    assert payload["stream"] is True
    assert payload["store"] is False
    assert "parallel_tool_calls" not in payload
    assert "reasoning" not in payload
    assert "include" not in payload
