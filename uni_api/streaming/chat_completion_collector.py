from __future__ import annotations

from contextlib import aclosing
from time import time
from typing import Any

from uni_api.admission import get_request_admission_lease
from uni_api.admission.json_memory import JSONMemoryComplexityError
from uni_api.admission.json_parsing import parse_owned_json_value, run_json_cpu
from uni_api.serialization import json
from uni_api.streaming.chat_completion_events import generate_no_stream_response
from uni_api.streaming.sse import (
    SSEBufferOverflowError,
    SSEOutputLimitError,
    SSEProtocolError,
)


DEFAULT_MAX_COLLECTED_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_INPUT_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_COLLECTED_FRAGMENTS = 65536
DEFAULT_MAX_TOOL_CALLS = 128
_RETAINED_TEXT_MEMORY_MULTIPLIER = 4
_RETAINED_TEXT_OVERHEAD_BYTES = 128


def _nested(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _usage_snapshot(value: Any) -> dict[str, int]:
    usage = value if isinstance(value, dict) else {}

    def integer(*paths: tuple[str, ...], default: int = 0) -> int:
        for path in paths:
            candidate = _nested(usage, *path, default=None)
            if candidate is not None:
                try:
                    return int(candidate or 0)
                except (TypeError, ValueError):
                    return default
        return default

    prompt_tokens = integer(("prompt_tokens",), ("input_tokens",))
    completion_tokens = integer(("completion_tokens",), ("output_tokens",))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": integer(
            ("total_tokens",),
            default=prompt_tokens + completion_tokens,
        ),
        "cached_tokens": integer(
            ("prompt_tokens_details", "cached_tokens"),
            ("input_tokens_details", "cached_tokens"),
        ),
        "prompt_audio_tokens": integer(
            ("prompt_tokens_details", "audio_tokens"),
            ("input_tokens_details", "audio_tokens"),
        ),
        "reasoning_tokens": integer(
            ("completion_tokens_details", "reasoning_tokens"),
            ("output_tokens_details", "reasoning_tokens"),
        ),
        "completion_audio_tokens": integer(
            ("completion_tokens_details", "audio_tokens"),
            ("output_tokens_details", "audio_tokens"),
        ),
        "accepted_prediction_tokens": integer(
            ("completion_tokens_details", "accepted_prediction_tokens"),
            ("output_tokens_details", "accepted_prediction_tokens"),
        ),
        "rejected_prediction_tokens": integer(
            ("completion_tokens_details", "rejected_prediction_tokens"),
            ("output_tokens_details", "rejected_prediction_tokens"),
        ),
    }


def _finalize_collected_parts(
    content_parts: list[str],
    reasoning_parts: list[str],
    tool_calls_by_index: dict[int, dict[str, Any]],
) -> tuple[str, str, list[dict[str, Any]] | None]:
    content_text = "".join(content_parts)
    reasoning_text = "".join(reasoning_parts)
    if not tool_calls_by_index:
        return content_text, reasoning_text, None

    tool_calls: list[dict[str, Any]] = []
    for index in sorted(tool_calls_by_index):
        entry = tool_calls_by_index[index]
        tool_calls.append(
            {
                "id": entry["id"] or f"call_{index}",
                "type": "function",
                "function": {
                    "name": entry["name"],
                    "arguments": "".join(entry["arguments_parts"]),
                },
            }
        )
    return content_text, reasoning_text, tool_calls


async def collect_openai_chat_completion_from_streaming_sse(
    sse_generator: Any,
    *,
    model: str,
    role: str = "assistant",
    max_bytes: int = DEFAULT_MAX_COLLECTED_BYTES,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    max_fragments: int = DEFAULT_MAX_COLLECTED_FRAGMENTS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> str:
    """Collect one validated chat-completion SSE stream with finite memory.

    Each frame is parsed exactly once.  Raw input and retained output have
    independent hard limits, and raw JSON bytes reserve conservative weighted
    response memory on the outer request lease until the ASGI response ends.
    """

    if min(max_bytes, max_input_bytes, max_fragments, max_tool_calls) <= 0:
        raise ValueError("chat completion collector limits must be positive")

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    retained_bytes = 0
    retained_fragments = 0
    observed_input_bytes = 0
    created_ts: int | None = None
    usage = _usage_snapshot(None)
    done_seen = False
    terminal_finish_reason: str | None = None
    request_lease = get_request_admission_lease()

    async def retain(value: Any) -> str:
        nonlocal retained_bytes, retained_fragments
        if not isinstance(value, str):
            raise SSEProtocolError(
                "chat completion retained fragments must be strings"
            )
        text = value
        remaining = max_bytes - retained_bytes
        if len(text) > remaining:
            raise SSEBufferOverflowError(
                buffer_name="collected chat completion",
                limit_bytes=max_bytes,
                observed_bytes=retained_bytes + len(text),
            )
        encoded_size = 0
        for offset in range(0, len(text), 64 * 1024):
            encoded_size += len(text[offset : offset + 64 * 1024].encode("utf-8"))
            if encoded_size > remaining:
                break
        retained_fragments += 1
        if retained_fragments > max_fragments:
            raise SSEOutputLimitError(
                output_name="chat completion fragments",
                limit=max_fragments,
                observed=retained_fragments,
            )
        retained_bytes += encoded_size
        if retained_bytes > max_bytes:
            raise SSEBufferOverflowError(
                buffer_name="collected chat completion",
                limit_bytes=max_bytes,
                observed_bytes=retained_bytes,
            )
        if request_lease is not None:
            await request_lease.reserve_response_bytes(
                encoded_size * _RETAINED_TEXT_MEMORY_MULTIPLIER
                + _RETAINED_TEXT_OVERHEAD_BYTES
            )
        # Force a distinct string before the parsed frame owner is released.
        # The reservation above covers UTF-8 bytes, copied str, final join, and
        # serialized response coexistence.
        return text.encode("utf-8").decode("utf-8")

    async def process_payload(payload: object) -> None:
        nonlocal created_ts, usage, terminal_finish_reason
        if not isinstance(payload, dict):
            raise SSEProtocolError(
                "chat completion stream payload must be a JSON object"
            )

        if created_ts is None and payload.get("created") is not None:
            try:
                created_ts = int(payload["created"])
            except (TypeError, ValueError):
                created_ts = None

        choices = payload.get("choices")
        if not choices and isinstance(payload.get("usage"), dict):
            usage = _usage_snapshot(payload["usage"])
        if choices is None:
            return
        if not isinstance(choices, list):
            raise SSEProtocolError("chat completion choices must be a list")

        for choice in choices:
            if not isinstance(choice, dict):
                raise SSEProtocolError(
                    "chat completion choice must be a JSON object"
                )
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise SSEProtocolError(
                    "chat completion delta must be a JSON object"
                )
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                if not isinstance(finish_reason, str):
                    raise SSEProtocolError(
                        "chat completion finish_reason must be a string"
                    )
                retained_finish_reason = await retain(finish_reason)
                if (
                    terminal_finish_reason is not None
                    and terminal_finish_reason != retained_finish_reason
                ):
                    raise SSEProtocolError(
                        "chat completion stream contains conflicting finish reasons"
                    )
                terminal_finish_reason = retained_finish_reason
            if delta.get("content") is not None:
                content_parts.append(await retain(delta["content"]))
            if delta.get("reasoning_content") is not None:
                reasoning_parts.append(await retain(delta["reasoning_content"]))

            tool_calls = delta.get("tool_calls")
            if tool_calls is None:
                continue
            if not isinstance(tool_calls, list):
                raise SSEProtocolError(
                    "chat completion tool_calls must be a list"
                )
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    raise SSEProtocolError(
                        "chat completion tool call must be a JSON object"
                    )
                index = tool_call.get("index", 0)
                if type(index) is not int:
                    raise SSEProtocolError(
                        "tool call index must be a JSON integer"
                    )
                if index < 0:
                    raise SSEProtocolError(
                        "tool call index cannot be negative"
                    )
                if index not in tool_calls_by_index:
                    if len(tool_calls_by_index) >= max_tool_calls:
                        raise SSEOutputLimitError(
                            output_name="tool calls",
                            limit=max_tool_calls,
                            observed=len(tool_calls_by_index) + 1,
                        )
                    tool_calls_by_index[index] = {
                        "id": None,
                        "name": None,
                        "arguments_parts": [],
                    }
                entry = tool_calls_by_index[index]
                if tool_call.get("id") is not None:
                    entry["id"] = await retain(tool_call["id"])
                function = tool_call.get("function")
                if function is None:
                    continue
                if not isinstance(function, dict):
                    raise SSEProtocolError(
                        "tool call function must be a JSON object"
                    )
                if function.get("name") is not None:
                    entry["name"] = await retain(function["name"])
                if function.get("arguments") is not None:
                    arguments = function["arguments"]
                    if not isinstance(arguments, str):
                        raise SSEProtocolError(
                            "tool call arguments must be a string"
                        )
                    entry["arguments_parts"].append(await retain(arguments))

    async with aclosing(sse_generator):
        async for item in sse_generator:
            if item is None:
                continue
            if not isinstance(item, (str, bytes, bytearray, memoryview)):
                raise SSEProtocolError(
                    "chat completion stream frames must be text or bytes"
                )

            if isinstance(item, str):
                if len(item) > max_bytes:
                    frame_bytes = len(item)
                else:
                    frame_bytes = 0
                    for offset in range(0, len(item), 64 * 1024):
                        frame_bytes += len(
                            item[offset : offset + 64 * 1024].encode(
                                "utf-8",
                                errors="strict",
                            )
                        )
                        if frame_bytes > max_bytes:
                            break
            else:
                frame_bytes = len(item)

            if frame_bytes > max_bytes:
                raise SSEBufferOverflowError(
                    buffer_name="chat completion SSE frame",
                    limit_bytes=max_bytes,
                    observed_bytes=frame_bytes,
                )
            observed_input_bytes += frame_bytes
            if observed_input_bytes > max_input_bytes:
                raise SSEBufferOverflowError(
                    buffer_name="chat completion SSE input",
                    limit_bytes=max_input_bytes,
                    observed_bytes=observed_input_bytes,
                )

            frame_workspace = (
                await request_lease.reserve_temporary_response_bytes(
                    frame_bytes * 4 + 1024
                )
                if request_lease is not None
                else None
            )
            payload_owner = None
            payload = None
            raw = None
            data = None
            frame_error: BaseException | None = None
            skip_frame = False
            stop_stream = False
            try:
                try:
                    raw = (
                        item
                        if isinstance(item, str)
                        else bytes(item).decode("utf-8", errors="strict")
                    )
                except UnicodeDecodeError:
                    frame_error = SSEProtocolError(
                        "chat completion stream contains invalid UTF-8"
                    )
                if frame_error is None:
                    data = raw.strip()
                    if not data or data.startswith(":"):
                        skip_frame = True
                    else:
                        if data.startswith("data:"):
                            data = data[len("data:") :].strip()
                        if data == "[DONE]":
                            done_seen = True
                            stop_stream = True
                        else:
                            try:
                                payload_owner = await parse_owned_json_value(data)
                                payload = payload_owner.value
                                await process_payload(payload)
                            except (
                                json.JSONDecodeError,
                                UnicodeDecodeError,
                                JSONMemoryComplexityError,
                            ):
                                frame_error = SSEProtocolError(
                                    "chat completion stream contains invalid or excessive JSON"
                                )
                            except BaseException as exc:
                                exc.__traceback__ = None
                                exc.__context__ = None
                                frame_error = exc
            finally:
                payload = None
                data = None
                raw = None
                item = None
                try:
                    if payload_owner is not None:
                        await payload_owner.aclose()
                finally:
                    if frame_workspace is not None:
                        await frame_workspace.release()

            if frame_error is not None:
                raise frame_error from None
            if stop_stream:
                break
            if skip_frame:
                continue

    if not done_seen:
        raise SSEProtocolError(
            "chat completion stream ended without data: [DONE]"
        )

    if request_lease is not None:
        # Fixed response/object/list overhead that is not proportional to
        # retained text fragments.
        await request_lease.reserve_response_bytes(4096)

    content_text, reasoning_text, tool_calls = await run_json_cpu(
        _finalize_collected_parts,
        content_parts,
        reasoning_parts,
        tool_calls_by_index,
    )
    result = await generate_no_stream_response(
        created_ts if created_ts is not None else int(time()),
        model,
        content=content_text or None,
        role=role,
        total_tokens=usage["total_tokens"]
        or (usage["prompt_tokens"] + usage["completion_tokens"]),
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        reasoning_content=reasoning_text or None,
        cached_tokens=usage["cached_tokens"],
        prompt_audio_tokens=usage["prompt_audio_tokens"],
        reasoning_tokens=usage["reasoning_tokens"],
        completion_audio_tokens=usage["completion_audio_tokens"],
        accepted_prediction_tokens=usage["accepted_prediction_tokens"],
        rejected_prediction_tokens=usage["rejected_prediction_tokens"],
        tool_calls_list=tool_calls,
    )
    if terminal_finish_reason and terminal_finish_reason != "stop":
        # The core response builder defaults tool-call responses to
        # ``tool_calls`` and ordinary responses to ``stop``.  The upstream
        # terminal is authoritative even when a tool argument was truncated;
        # otherwise a client could execute partial JSON as a completed call.
        result_payload = await run_json_cpu(json.loads, result)
        choices = result_payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(
            choices[0], dict
        ):
            raise SSEProtocolError(
                "generated chat completion response is missing choices"
            )
        choices[0]["finish_reason"] = terminal_finish_reason
        result = await run_json_cpu(
            json.dumps,
            result_payload,
            ensure_ascii=False,
        )
    return result
