import asyncio

import httpx

from uni_api.idempotency import (
    OAIX_ROUTING_ATTEMPT_HEADER,
    apply_oaix_routing_attempt_id,
)
from uni_api.providers.responses import fetch_response, fetch_response_stream


def header_value(headers: dict[str, str], key: str) -> str:
    lower = key.lower()
    for header_key, value in headers.items():
        if header_key.lower() == lower:
            return value
    raise KeyError(key)


class DummyStreamContext:
    def __init__(self, response: httpx.Response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        await self.response.aclose()


class DummyClient:
    def __init__(self, response: httpx.Response):
        self.response = response

    async def post(self, url, headers=None, content=None, timeout=None):
        _ = url, headers, content, timeout
        return self.response

    def stream(self, method, url, headers=None, content=None, timeout=None):
        _ = method, url, headers, content, timeout
        return DummyStreamContext(self.response)


async def _fetch_response_captures_oaix_headers():
    response = httpx.Response(
        200,
        headers={
            "X-OAIX-Request-ID": "req_123",
            "X-OAIX-Token-ID": "456",
            "X-OAIX-Token-Owner-User-ID": "789",
            "X-OAIX-Connection-ID": "oaixc-nonstream",
        },
        json={"id": "chatcmpl-test", "choices": []},
    )
    captured = {}

    chunks = [
        chunk
        async for chunk in fetch_response(
            DummyClient(response),
            "https://oaix.example/v1/chat/completions",
            {},
            {"model": "gpt-test", "messages": []},
            "gpt",
            "gpt-test",
            response_headers_sink=captured.update,
        )
    ]

    assert chunks == [{"id": "chatcmpl-test", "choices": []}]
    assert header_value(captured, "X-OAIX-Request-ID") == "req_123"
    assert header_value(captured, "X-OAIX-Token-ID") == "456"
    assert header_value(captured, "X-OAIX-Token-Owner-User-ID") == "789"
    assert header_value(captured, "X-OAIX-Connection-ID") == "oaixc-nonstream"


def test_fetch_response_captures_oaix_headers():
    asyncio.run(_fetch_response_captures_oaix_headers())


async def _fetch_response_stream_captures_oaix_headers():
    response = httpx.Response(
        200,
        headers={
            "X-OAIX-Request-ID": "req_stream",
            "X-OAIX-Token-ID": "654",
            "X-OAIX-Token-Owner-User-ID": "987",
            "X-OAIX-Connection-ID": "oaixc-stream",
        },
        content=b"data: [DONE]\n\n",
    )
    captured = {}

    chunks = [
        chunk
        async for chunk in fetch_response_stream(
            DummyClient(response),
            "https://oaix.example/v1/chat/completions",
            {},
            {"model": "gpt-test", "messages": [], "stream": True},
            "gpt",
            "gpt-test",
            response_headers_sink=captured.update,
        )
    ]

    assert chunks
    assert header_value(captured, "X-OAIX-Request-ID") == "req_stream"
    assert header_value(captured, "X-OAIX-Token-ID") == "654"
    assert header_value(captured, "X-OAIX-Token-Owner-User-ID") == "987"
    assert header_value(captured, "X-OAIX-Connection-ID") == "oaixc-stream"


def test_fetch_response_stream_captures_oaix_headers():
    asyncio.run(_fetch_response_stream_captures_oaix_headers())


def test_oaix_routing_attempt_header_requires_explicit_provider_capability(
    monkeypatch,
):
    monkeypatch.delenv("OAIX_ROUTING_ATTEMPT_PROVIDERS", raising=False)
    attempt_id = "attempt-123"
    disabled_headers = {}
    enabled_headers = {}

    assert apply_oaix_routing_attempt_id(
        disabled_headers,
        provider={
            "provider": "fugue-codex",
            "base_url": "https://oaix.fugue.pro/v1/responses",
            "preferences": {},
        },
        routing_attempt_id=attempt_id,
    ) is False
    assert disabled_headers == {}

    assert apply_oaix_routing_attempt_id(
        enabled_headers,
        provider={
            "provider": "fugue-codex",
            "preferences": {"oaix_routing_attempt_id": True},
        },
        routing_attempt_id=attempt_id,
    ) is True
    assert enabled_headers == {OAIX_ROUTING_ATTEMPT_HEADER: attempt_id}

    env_headers = {}
    monkeypatch.setenv("OAIX_ROUTING_ATTEMPT_PROVIDERS", "fugue-codex")
    assert apply_oaix_routing_attempt_id(
        env_headers,
        provider={"provider": "fugue-codex", "preferences": {}},
        routing_attempt_id=attempt_id,
    ) is True
    assert env_headers == {OAIX_ROUTING_ATTEMPT_HEADER: attempt_id}
