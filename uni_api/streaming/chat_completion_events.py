from __future__ import annotations

import random
import string
from typing import Any

from core.utils import _build_openai_usage, end_of_line
from uni_api.admission.json_parsing import run_json_cpu
from uni_api.serialization import json


async def generate_sse_response(
    timestamp,
    model,
    content=None,
    tools_id=None,
    function_call_name=None,
    function_call_content=None,
    role=None,
    total_tokens=0,
    prompt_tokens=0,
    completion_tokens=0,
    reasoning_content=None,
    stop=None,
    cached_tokens=0,
    prompt_audio_tokens=0,
    reasoning_tokens=0,
    completion_audio_tokens=0,
    accepted_prediction_tokens=0,
    rejected_prediction_tokens=0,
):
    """Legacy-compatible chat SSE builder with cancellation-safe encoding."""

    random.seed(timestamp)
    random_str = "".join(
        random.choices(string.ascii_letters + string.digits, k=29)
    )
    delta_content = {"role": "assistant", "content": content} if content else {}
    if reasoning_content:
        delta_content = {
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning_content,
        }
    sample_data = {
        "id": f"chatcmpl-{random_str}",
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta_content,
                "logprobs": None,
                "finish_reason": None if content or reasoning_content else "stop",
            }
        ],
        "usage": None,
        "system_fingerprint": "fp_d576307f90",
    }
    if function_call_content:
        sample_data["choices"][0]["delta"] = {
            "tool_calls": [
                {"index": 0, "function": {"arguments": function_call_content}}
            ]
        }
    if tools_id and function_call_name:
        sample_data["choices"][0]["delta"] = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": tools_id,
                    "type": "function",
                    "function": {"name": function_call_name, "arguments": ""},
                }
            ]
        }
    if role:
        sample_data["choices"][0]["delta"] = {"role": role, "content": ""}
    if total_tokens:
        sample_data["usage"] = _build_openai_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            prompt_audio_tokens=prompt_audio_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_audio_tokens=completion_audio_tokens,
            accepted_prediction_tokens=accepted_prediction_tokens,
            rejected_prediction_tokens=rejected_prediction_tokens,
        )
        sample_data["choices"] = []
    if stop:
        sample_data["choices"][0]["delta"] = {}
        sample_data["choices"][0]["finish_reason"] = stop

    json_data = await run_json_cpu(json.dumps, sample_data, ensure_ascii=False)
    return f"data: {json_data}" + end_of_line


async def generate_no_stream_response(
    timestamp,
    model,
    content=None,
    tools_id=None,
    function_call_name=None,
    function_call_content=None,
    role=None,
    total_tokens=0,
    prompt_tokens=0,
    completion_tokens=0,
    reasoning_content=None,
    image_base64=None,
    audio=None,
    cached_tokens=0,
    prompt_audio_tokens=0,
    reasoning_tokens=0,
    completion_audio_tokens=0,
    accepted_prediction_tokens=0,
    rejected_prediction_tokens=0,
    tool_calls_list=None,
    preserve_content_with_tool_calls=False,
):
    """Legacy-compatible non-stream builder with bounded CPU ownership."""

    random.seed(timestamp)
    random_str = "".join(
        random.choices(string.ascii_letters + string.digits, k=29)
    )
    message = {"role": role, "content": content, "refusal": None}
    if audio is not None:
        if message.get("content") == "":
            message["content"] = None
        message["audio"] = audio
        message["annotations"] = []
    if reasoning_content:
        message["reasoning_content"] = reasoning_content

    sample_data = {
        "id": f"chatcmpl-{random_str}",
        "object": "chat.completion",
        "created": timestamp,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": None,
        "system_fingerprint": "fp_a7d06e42a7",
    }
    if function_call_name:
        if not tools_id:
            tools_id = f"call_{random_str}"
        arguments_json = await run_json_cpu(
            json.dumps,
            function_call_content,
            ensure_ascii=False,
        )
        sample_data = {
            "id": f"chatcmpl-{random_str}",
            "object": "chat.completion",
            "created": timestamp,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tools_id,
                                "type": "function",
                                "function": {
                                    "name": function_call_name,
                                    "arguments": arguments_json,
                                },
                            }
                        ],
                        "refusal": None,
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": None,
            "service_tier": "default",
            "system_fingerprint": "fp_4691090a87",
        }
    if tool_calls_list:
        random_str_tc = "".join(
            random.choices(string.ascii_letters + string.digits, k=29)
        )
        sample_data = {
            "id": f"chatcmpl-{random_str_tc}",
            "object": "chat.completion",
            "created": timestamp,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content if preserve_content_with_tool_calls else None,
                        "tool_calls": tool_calls_list,
                        "refusal": None,
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": None,
            "service_tier": "default",
            "system_fingerprint": "fp_4691090a87",
        }
    if image_base64:
        sample_data = {
            "created": timestamp,
            "data": [{"b64_json": image_base64}],
        }
    if total_tokens:
        sample_data["usage"] = _build_openai_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            prompt_audio_tokens=prompt_audio_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_audio_tokens=completion_audio_tokens,
            accepted_prediction_tokens=accepted_prediction_tokens,
            rejected_prediction_tokens=rejected_prediction_tokens,
        )
    return await run_json_cpu(json.dumps, sample_data, ensure_ascii=False)


def build_chat_completion_chunk_sse(
    *,
    response_id: str,
    created_at: int,
    model_name: str,
    delta: dict,
    finish_reason: str | None = None,
) -> str:
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_at,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return "data: " + json.dumps(payload, ensure_ascii=False) + end_of_line


def build_chat_completion_usage_chunk_sse(
    *,
    response_id: str,
    created_at: int,
    model_name: str,
    usage: dict,
) -> str:
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_at,
        "model": model_name,
        "choices": [],
        "usage": usage,
    }
    return "data: " + json.dumps(payload, ensure_ascii=False) + end_of_line


def responses_usage_to_chat_completion_usage(usage_obj: object) -> dict | None:
    if not isinstance(usage_obj, dict):
        return None
    if all(
        usage_obj.get(key) is None
        for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens")
    ):
        return None

    prompt_tokens = usage_obj.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = usage_obj.get("input_tokens")

    completion_tokens = usage_obj.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = usage_obj.get("output_tokens")

    total_tokens = usage_obj.get("total_tokens")
    if total_tokens is None:
        try:
            total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
        except Exception:
            total_tokens = 0

    prompt_details = usage_obj.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = usage_obj.get("input_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}

    completion_details = usage_obj.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = usage_obj.get("output_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}

    return _build_openai_usage(
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens or 0,
        total_tokens=total_tokens or 0,
        cached_tokens=prompt_details.get("cached_tokens", 0) or 0,
        prompt_audio_tokens=prompt_details.get("audio_tokens", 0) or 0,
        reasoning_tokens=completion_details.get("reasoning_tokens", 0) or 0,
        completion_audio_tokens=completion_details.get("audio_tokens", 0) or 0,
        accepted_prediction_tokens=completion_details.get("accepted_prediction_tokens", 0) or 0,
        rejected_prediction_tokens=completion_details.get("rejected_prediction_tokens", 0) or 0,
    )
