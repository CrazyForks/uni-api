import json

import pytest

from core.models import RequestModel
from uni_api.providers.payloads import (
    get_azure_databricks_payload,
    get_azure_payload,
    get_gpt_payload,
    get_openrouter_payload,
)
from uni_api.providers.responses import _yield_responses_api_chat_completion


def _grok_tool_request() -> RequestModel:
    return RequestModel(
        model="grok-4.5",
        messages=[
            {"role": "user", "content": "Inspect the project."},
            {
                "role": "assistant",
                "content": "I will inspect it.",
                "tool_calls": [
                    {
                        "id": "call_probe",
                        "type": "function",
                        "function": {
                            "name": "get_probe",
                            "arguments": '{"path":"."}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_probe",
                "content": "README.md",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_probe",
                    "description": "Inspect a path.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        tool_choice={
            "type": "function",
            "function": {"name": "get_probe"},
        },
        max_tokens=128,
        stream=True,
    )


async def test_grok_responses_payload_preserves_and_converts_tools():
    request = _grok_tool_request()
    provider = {
        "provider": "aiwave-grok",
        "base_url": "https://api.ai-wave.org/v1/responses",
        "model": ["grok-4.5"],
    }

    url, _, payload = await get_gpt_payload(request, "gpt", provider, "upstream-key")

    assert url == "https://api.ai-wave.org/v1/responses"
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "get_probe",
            "description": "Inspect a path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]
    assert payload["tool_choice"] == {"type": "function", "name": "get_probe"}
    assert payload["max_output_tokens"] == 128
    assert "messages" not in payload
    assert "max_tokens" not in payload

    assert {"role": "assistant", "content": "I will inspect it."} in payload["input"]
    assert {
        "type": "function_call",
        "call_id": "call_probe",
        "name": "get_probe",
        "arguments": '{"path":"."}',
    } in payload["input"]
    assert {
        "type": "function_call_output",
        "call_id": "call_probe",
        "output": "README.md",
    } in payload["input"]


@pytest.mark.parametrize(
    ("builder", "engine", "provider"),
    [
        (
            get_gpt_payload,
            "gpt",
            {
                "provider": "grok-chat",
                "base_url": "https://api.example.com/v1/chat/completions",
                "model": ["grok-4.5"],
            },
        ),
        (
            get_azure_payload,
            "azure",
            {
                "provider": "grok-azure",
                "base_url": "https://example.openai.azure.com",
                "model": ["grok-4.5"],
            },
        ),
        (
            get_azure_databricks_payload,
            "azure-databricks",
            {
                "provider": "grok-databricks",
                "base_url": "https://example.databricks.com",
                "model": ["grok-4.5"],
            },
        ),
        (
            get_openrouter_payload,
            "openrouter",
            {
                "provider": "grok-openrouter",
                "base_url": "https://openrouter.ai/api/v1/chat/completions",
                "model": ["grok-4.5"],
            },
        ),
    ],
)
async def test_grok_chat_payloads_preserve_tools_and_tool_history(builder, engine, provider):
    request = _grok_tool_request()

    _, _, payload = await builder(request, engine, provider, "upstream-key")

    assert payload["tools"][0]["function"]["name"] == "get_probe"
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_probe"},
    }
    assert {
        "role": "assistant",
        "content": "I will inspect it.",
        "tool_calls": [
            {
                "id": "call_probe",
                "type": "function",
                "function": {
                    "name": "get_probe",
                    "arguments": '{"path":"."}',
                },
            }
        ],
    } in payload["messages"]
    assert {
        "role": "tool",
        "tool_call_id": "call_probe",
        "content": "README.md",
    } in payload["messages"]


async def test_explicitly_disabled_tools_remain_disabled_for_grok_responses():
    request = _grok_tool_request()
    provider = {
        "provider": "no-tools",
        "base_url": "https://api.example.com/v1/responses",
        "model": ["grok-4.5"],
        "tools": False,
    }

    _, _, payload = await get_gpt_payload(request, "gpt", provider, "upstream-key")

    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert not any(
        item.get("type") in {"function_call", "function_call_output"}
        for item in payload["input"]
    )


@pytest.mark.parametrize(
    ("builder", "engine", "base_url"),
    [
        (get_gpt_payload, "gpt", "https://api.example.com/v1/chat/completions"),
        (get_openrouter_payload, "openrouter", "https://openrouter.ai/api/v1/chat/completions"),
    ],
)
async def test_explicitly_disabled_tools_remain_disabled_for_grok_chat(builder, engine, base_url):
    request = _grok_tool_request()
    provider = {
        "provider": "no-tools",
        "base_url": base_url,
        "model": ["grok-4.5"],
        "tools": False,
    }

    _, _, payload = await builder(request, engine, provider, "upstream-key")

    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert not any(
        message.get("tool_calls") or message.get("tool_call_id")
        for message in payload["messages"]
    )


async def test_grok_responses_auto_tool_choice_is_preserved():
    request = _grok_tool_request()
    request.tool_choice = "auto"
    provider = {
        "provider": "aiwave-grok",
        "base_url": "https://api.ai-wave.org/v1/responses",
        "model": ["grok-4.5"],
    }

    _, _, payload = await get_gpt_payload(request, "gpt", provider, "upstream-key")

    assert payload["tool_choice"] == "auto"


async def test_openrouter_multiple_tool_calls_use_one_assistant_message():
    request = _grok_tool_request()
    request.messages[1].tool_calls.append(
        request.messages[1].tool_calls[0].model_copy(
            update={"id": "call_probe_2"}
        )
    )
    provider = {
        "provider": "grok-openrouter",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": ["grok-4.5"],
    }

    _, _, payload = await get_openrouter_payload(
        request,
        "openrouter",
        provider,
        "upstream-key",
    )

    assistant_messages = [
        message
        for message in payload["messages"]
        if message["role"] == "assistant"
    ]
    assert len(assistant_messages) == 1
    assert len(assistant_messages[0]["tool_calls"]) == 2


async def test_non_streaming_grok_responses_function_call_is_returned_as_chat_tool_call():
    class Response:
        async def aread(self):
            return json.dumps(
                {
                    "id": "resp_probe",
                    "created": 1_783_942_000,
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "I will call the probe.",
                                }
                            ],
                        },
                        {
                            "type": "function_call",
                            "call_id": "call_probe",
                            "name": "get_probe",
                            "arguments": '{"path":"."}',
                        },
                    ],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "total_tokens": 25,
                    },
                }
            ).encode()

    chunks = [
        chunk
        async for chunk in _yield_responses_api_chat_completion(
            Response(),
            "grok-4.5",
        )
    ]
    response = json.loads(chunks[0])

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "I will call the probe."
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_probe",
            "type": "function",
            "function": {
                "name": "get_probe",
                "arguments": '{"path":"."}',
            },
        }
    ]
