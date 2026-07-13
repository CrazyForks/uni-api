from __future__ import annotations

import asyncio
import copy
import hashlib
import uuid
from contextlib import aclosing, asynccontextmanager
from datetime import datetime
from typing import Any

from core.utils import end_of_line, parse_sse_event, safe_get
from uni_api.admission.json_parsing import run_json_cpu
from uni_api.serialization import json
from uni_api.upstream.responses_errors import responses_failure_error

from .chat_completion_events import (
    build_chat_completion_chunk_sse,
    build_chat_completion_usage_chunk_sse,
    responses_usage_to_chat_completion_usage,
)
from .cleanup import close_async_iterator_safely
from .sse import (
    OwnedSSEEvent,
    SSEBufferOverflowError,
    SSEProtocolError,
    iter_sse_events,
    parse_owned_sse_event,
    validate_sse_event_type_consistency,
)


DEFAULT_MAX_COLLECTED_OUTPUT_ITEMS = 128
DEFAULT_MAX_COLLECTED_OUTPUT_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_RESPONSES_METADATA_BYTES = 4096
def _compact_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def extract_responses_stream_sse_event(raw_event: str) -> tuple[str, object]:
    return parse_sse_event(raw_event)


def normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, int, float, bool)):
        return None
    text = str(value).strip()
    if len(text) > DEFAULT_MAX_RESPONSES_METADATA_BYTES:
        raise SSEBufferOverflowError(
            buffer_name="Responses metadata",
            limit_bytes=DEFAULT_MAX_RESPONSES_METADATA_BYTES,
            observed_bytes=len(text),
        )
    encoded_bytes = len(text.encode("utf-8"))
    if encoded_bytes > DEFAULT_MAX_RESPONSES_METADATA_BYTES:
        raise SSEBufferOverflowError(
            buffer_name="Responses metadata",
            limit_bytes=DEFAULT_MAX_RESPONSES_METADATA_BYTES,
            observed_bytes=encoded_bytes,
        )
    return text if text else None


def coerce_positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed >= 0 else None


def mime_type_from_output_format(output_format: str | None) -> str:
    normalized = str(output_format or "").strip().lower().lstrip(".")
    if normalized in {"jpg", "jpeg"}:
        return "image/jpeg"
    if normalized == "webp":
        return "image/webp"
    if normalized == "gif":
        return "image/gif"
    return "image/png"


def extract_response_model_name(payload: object) -> str | None:
    for candidate in (
        safe_get(payload, "model_name", default=None),
        safe_get(payload, "model", default=None),
        safe_get(payload, "response", "model_name", default=None),
        safe_get(payload, "response", "model", default=None),
    ):
        normalized = normalize_optional_text(candidate)
        if normalized is not None:
            return normalized
    return None


def chat_completion_response_id_from_payload(payload: object, fallback: str) -> str:
    for candidate in (
        safe_get(payload, "id", default=None),
        safe_get(payload, "response", "id", default=None),
    ):
        normalized = normalize_optional_text(candidate)
        if normalized is not None:
            return normalized
    return fallback


def chat_completion_created_at_from_payload(payload: object, fallback: int) -> int:
    for candidate in (
        safe_get(payload, "created", default=None),
        safe_get(payload, "created_at", default=None),
        safe_get(payload, "response", "created_at", default=None),
        safe_get(payload, "response", "created", default=None),
    ):
        normalized = coerce_positive_int(candidate)
        if normalized is not None:
            return normalized
    return fallback


def chat_completion_tool_calls_from_responses_output(output_items: object) -> list[dict]:
    if not isinstance(output_items, list):
        return []

    tool_calls: list[dict] = []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if normalize_optional_text(item.get("type")) != "function_call":
            continue

        name = normalize_optional_text(item.get("name"))
        if name is None:
            continue

        tool_calls.append(
            {
                "id": normalize_optional_text(item.get("call_id")) or f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": str(item.get("arguments") or ""),
                },
            }
        )

    return tool_calls


def chat_completion_message_from_responses_payload(payload: dict) -> tuple[dict, str]:
    output_items = safe_get(payload, "output", default=None)
    if not isinstance(output_items, list):
        output_items = safe_get(payload, "response", "output", default=[])

    message_parts: list[str] = []
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue

            item_type = normalize_optional_text(item.get("type"))
            if item_type == "message":
                content_items = item.get("content")
                if not isinstance(content_items, list):
                    continue
                text_parts: list[str] = []
                for content_item in content_items:
                    if not isinstance(content_item, dict):
                        continue
                    content_type = normalize_optional_text(content_item.get("type"))
                    if content_type in {"output_text", "input_text", "text"} and content_item.get("text") is not None:
                        text_parts.append(str(content_item.get("text")))
                if text_parts:
                    message_parts.append("".join(text_parts))
                continue

            if item_type == "image_generation_call":
                result_b64 = item.get("result")
                if not isinstance(result_b64, str) or not result_b64:
                    continue
                mime_type = mime_type_from_output_format(
                    normalize_optional_text(item.get("output_format"))
                )
                message_parts.append(f"![image](data:{mime_type};base64,{result_b64})")

    content = "\n\n".join(part for part in message_parts if part)
    tool_calls = chat_completion_tool_calls_from_responses_output(output_items)

    message: dict[str, object] = {
        "role": "assistant",
        "content": content or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
        if not content:
            message["content"] = None
        return message, "tool_calls"
    return message, "stop"


def chat_completion_usage_from_responses_payload(payload: object) -> dict | None:
    if not isinstance(payload, dict):
        return None

    usage_obj = safe_get(payload, "response", "usage", default=None)
    if usage_obj is None:
        usage_obj = payload.get("usage")
    return responses_usage_to_chat_completion_usage(usage_obj)


def collect_responses_output_item_done(
    event_payload: object,
    *,
    output_items_by_index: dict[int, dict],
    output_items_fallback: list[dict],
) -> None:
    if not isinstance(event_payload, dict):
        return

    item = event_payload.get("item")
    if not isinstance(item, dict):
        return

    output_index = coerce_positive_int(event_payload.get("output_index"))
    item_copy = copy.deepcopy(item)
    if output_index is None and event_payload.get("output_index") not in (0, "0"):
        output_items_fallback.append(item_copy)
        return
    if output_index is None:
        output_index = 0
    output_items_by_index[output_index] = item_copy


class ResponsesOutputItemCollector:
    """Bound output-item retention across the complete Responses stream."""

    def __init__(
        self,
        *,
        output_items_by_index: dict[int, dict],
        output_items_fallback: list[dict],
        max_items: int,
        max_bytes: int,
    ) -> None:
        if max_items <= 0 or max_bytes <= 0:
            raise ValueError("Responses output collection limits must be positive")
        self.output_items_by_index = output_items_by_index
        self.output_items_fallback = output_items_fallback
        self.max_items = int(max_items)
        self.max_bytes = int(max_bytes)
        self.total_bytes = 0
        self._indexed_sizes: dict[int, int] = {}
        self._indexed_reservations: dict[int, Any] = {}
        self._fallback_reservations: list[Any] = []

    async def collect(
        self,
        event_payload: object,
        *,
        event_owner: OwnedSSEEvent,
    ) -> None:
        if not isinstance(event_payload, dict):
            return
        item = event_payload.get("item")
        if not isinstance(item, dict):
            return

        serialized = await run_json_cpu(_compact_json_bytes, item)
        serialized_size = len(serialized)
        raw_output_index = event_payload.get("output_index")
        output_index = coerce_positive_int(raw_output_index)
        has_index = output_index is not None or raw_output_index in (0, "0")
        if output_index is None and has_index:
            output_index = 0

        existing_item = (
            output_index is not None and output_index in self.output_items_by_index
        )
        existing_size = (
            self._indexed_sizes.get(output_index, 0)
            if existing_item and output_index is not None
            else 0
        )
        next_items = (
            len(self.output_items_by_index)
            + len(self.output_items_fallback)
            + (0 if existing_item else 1)
        )
        if next_items > self.max_items:
            raise SSEBufferOverflowError(
                buffer_name="collected output item count",
                limit_bytes=self.max_items,
                observed_bytes=next_items,
            )

        next_bytes = self.total_bytes - existing_size + serialized_size
        if next_bytes > self.max_bytes:
            raise SSEBufferOverflowError(
                buffer_name="collected output items",
                limit_bytes=self.max_bytes,
                observed_bytes=next_bytes,
            )

        # Transfer the already-live parsed graph reservation instead of
        # releasing it when this event closes.  The collector can then return
        # the exact charge on replacement, terminal override, cancellation, or
        # downstream close.
        reservation = event_owner.take_payload_reservation()

        if output_index is None:
            self.output_items_fallback.append(item)
            self._fallback_reservations.append(reservation)
        else:
            old_reservation = self._indexed_reservations.pop(output_index, None)
            if existing_item:
                old_item = self.output_items_by_index.pop(output_index)
                del old_item
            self.output_items_by_index[output_index] = item
            self._indexed_sizes[output_index] = serialized_size
            self._indexed_reservations[output_index] = reservation
            if old_reservation is not None:
                await old_reservation.release()
        self.total_bytes = next_bytes

    async def validate_completed_output(self, event_payload: object) -> None:
        """Apply the same bounds to output embedded in a terminal event."""

        output_items = safe_get(event_payload, "response", "output", default=None)
        if output_items is None and isinstance(event_payload, dict):
            output_items = event_payload.get("output")
        if not isinstance(output_items, list):
            return
        if len(output_items) > self.max_items:
            raise SSEBufferOverflowError(
                buffer_name="completed output item count",
                limit_bytes=self.max_items,
                observed_bytes=len(output_items),
            )
        total_bytes = 0
        for item in output_items:
            total_bytes += len(await run_json_cpu(_compact_json_bytes, item))
            if total_bytes > self.max_bytes:
                raise SSEBufferOverflowError(
                    buffer_name="completed output items",
                    limit_bytes=self.max_bytes,
                    observed_bytes=total_bytes,
                )

    async def discard_collected(self) -> None:
        """Release incremental copies superseded by terminal output."""

        reservations = list(self._indexed_reservations.values())
        reservations.extend(self._fallback_reservations)
        self.output_items_by_index.clear()
        self.output_items_fallback.clear()
        self._indexed_sizes.clear()
        self._indexed_reservations.clear()
        self._fallback_reservations.clear()
        self.total_bytes = 0
        await _release_collector_reservations(reservations)

    async def aclose(self) -> None:
        await self.discard_collected()


async def _release_collector_reservations(reservations: list[Any]) -> None:
    tasks = [
        asyncio.create_task(reservation.release())
        for reservation in reservations
        if reservation is not None
    ]
    if not tasks:
        return
    group = asyncio.gather(*tasks)
    pending_cancel: asyncio.CancelledError | None = None
    while not group.done():
        try:
            await asyncio.shield(group)
        except asyncio.CancelledError as exc:
            pending_cancel = pending_cancel or exc
    group.result()
    if pending_cancel is not None:
        raise pending_cancel


def patch_responses_completed_output(
    payload: object,
    *,
    output_items_by_index: dict[int, dict],
    output_items_fallback: list[dict],
) -> object:
    if not isinstance(payload, dict):
        return payload

    response_payload = payload.get("response")
    if not isinstance(response_payload, dict):
        return payload

    output_items = response_payload.get("output")
    should_patch_output = (
        (not isinstance(output_items, list) or not output_items)
        and (output_items_by_index or output_items_fallback)
    )
    if not should_patch_output:
        return payload

    patched_payload = dict(payload)
    patched_response = dict(response_payload)
    patched_payload["response"] = patched_response

    patched_output = [
        output_items_by_index[index]
        for index in sorted(output_items_by_index)
    ]
    patched_output.extend(output_items_fallback)
    patched_response["output"] = patched_output
    return patched_payload


def build_synthetic_responses_completed_payload(
    *,
    response_id: str,
    model_name: str,
    created_at: int,
    output_items_by_index: dict[int, dict],
    output_items_fallback: list[dict],
) -> dict | None:
    if not output_items_by_index and not output_items_fallback:
        return None

    payload = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "model": model_name,
            "created_at": created_at,
            "status": "completed",
        },
    }
    patched_payload = patch_responses_completed_output(
        payload,
        output_items_by_index=output_items_by_index,
        output_items_fallback=output_items_fallback,
    )
    output_items = safe_get(patched_payload, "response", "output", default=None)
    if not isinstance(output_items, list) or not output_items:
        return None
    return patched_payload


def build_missing_responses_completed_payload(
    *,
    completed_response_seen: bool,
    error_seen: bool,
    response_id: str,
    model_name: str,
    created_at: int,
    output_items_by_index: dict[int, dict],
    output_items_fallback: list[dict],
) -> dict | None:
    if completed_response_seen or error_seen:
        return None
    return build_synthetic_responses_completed_payload(
        response_id=response_id,
        model_name=model_name,
        created_at=created_at,
        output_items_by_index=output_items_by_index,
        output_items_fallback=output_items_fallback,
    )


async def _stream_responses_to_chat_completions_impl(
    text_iterator,
    *,
    request_model: str,
    upstream_status_code: int | None = None,
    max_collected_output_items: int = DEFAULT_MAX_COLLECTED_OUTPUT_ITEMS,
    max_collected_output_bytes: int = DEFAULT_MAX_COLLECTED_OUTPUT_BYTES,
    _collector_holder: list[ResponsesOutputItemCollector],
):
    emitted_content_chars = 0
    emitted_content_hash = hashlib.sha256()
    role_sent = False
    response_id = f"chatcmpl_{uuid.uuid4().hex}"
    created_at = int(datetime.timestamp(datetime.now()))
    model_name = normalize_optional_text(request_model) or "unknown"
    completed_response_seen = False
    error_seen = False
    output_items_by_index: dict[int, dict] = {}
    output_items_fallback: list[dict] = []
    output_item_collector = ResponsesOutputItemCollector(
        output_items_by_index=output_items_by_index,
        output_items_fallback=output_items_fallback,
        max_items=max_collected_output_items,
        max_bytes=max_collected_output_bytes,
    )
    _collector_holder.append(output_item_collector)
    def emit_content_delta(content: str) -> str | None:
        nonlocal role_sent, emitted_content_chars
        if not content:
            return None
        delta = {"content": content}
        if not role_sent:
            delta["role"] = "assistant"
            role_sent = True
        emitted_content_chars += len(content)
        emitted_content_hash.update(content.encode("utf-8"))
        return build_chat_completion_chunk_sse(
            response_id=response_id,
            created_at=created_at,
            model_name=model_name,
            delta=delta,
        )
    def emit_reasoning_delta(content: str) -> str | None:
        nonlocal role_sent
        if not content:
            return None
        delta = {"content": "", "reasoning_content": content}
        if not role_sent:
            delta["role"] = "assistant"
            role_sent = True
        return build_chat_completion_chunk_sse(
            response_id=response_id,
            created_at=created_at,
            model_name=model_name,
            delta=delta,
        )
    async def emit_completed_payload(event_payload: dict):
        nonlocal completed_response_seen, role_sent, response_id, created_at, model_name
        completed_response_seen = True

        terminal_output = safe_get(
            event_payload,
            "response",
            "output",
            default=None,
        )
        if isinstance(terminal_output, list) and terminal_output:
            # A non-empty terminal output is authoritative.  Drop historical
            # incremental copies before validating/materializing it so both
            # independently bounded representations do not remain live.
            await output_item_collector.discard_collected()
        await output_item_collector.validate_completed_output(event_payload)
        patched_payload = patch_responses_completed_output(
            event_payload,
            output_items_by_index=output_items_by_index,
            output_items_fallback=output_items_fallback,
        )
        response_id = chat_completion_response_id_from_payload(patched_payload, response_id)
        created_at = chat_completion_created_at_from_payload(patched_payload, created_at)
        model_name = extract_response_model_name(patched_payload) or model_name
        message, finish_reason = chat_completion_message_from_responses_payload(patched_payload)
        response_status = str(
            safe_get(patched_payload, "response", "status", default="") or ""
        ).strip().lower()
        event_type = str(patched_payload.get("type") or "").strip().lower()
        if response_status == "incomplete" or event_type == "response.incomplete":
            incomplete_reason = str(
                safe_get(
                    patched_payload,
                    "response",
                    "incomplete_details",
                    "reason",
                    default="",
                )
                or ""
            ).strip().lower()
            if incomplete_reason in {
                "content_filter",
                "safety",
            }:
                finish_reason = "content_filter"
            else:
                finish_reason = "length"
        final_content = str(message.get("content") or "")
        if final_content:
            suffix = final_content
            if emitted_content_chars and len(final_content) >= emitted_content_chars:
                final_prefix = final_content[:emitted_content_chars]
                if hashlib.sha256(final_prefix.encode("utf-8")).digest() == (
                    emitted_content_hash.digest()
                ):
                    suffix = final_content[emitted_content_chars:]
            content_chunk = emit_content_delta(suffix)
            if content_chunk is not None:
                yield content_chunk

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            tool_call_deltas = []
            for index, tool_call in enumerate(tool_calls):
                tool_call_delta = copy.deepcopy(tool_call)
                tool_call_delta["index"] = index
                tool_call_deltas.append(tool_call_delta)
            delta = {"tool_calls": tool_call_deltas}
            if not role_sent:
                delta["role"] = "assistant"
                role_sent = True
            yield build_chat_completion_chunk_sse(
                response_id=response_id,
                created_at=created_at,
                model_name=model_name,
                delta=delta,
            )

        if not role_sent:
            yield build_chat_completion_chunk_sse(
                response_id=response_id,
                created_at=created_at,
                model_name=model_name,
                delta={"role": "assistant"},
            )
            role_sent = True

        yield build_chat_completion_chunk_sse(
            response_id=response_id,
            created_at=created_at,
            model_name=model_name,
            delta={},
            finish_reason=finish_reason,
        )
        usage = chat_completion_usage_from_responses_payload(patched_payload)
        if usage is not None:
            yield build_chat_completion_usage_chunk_sse(
                response_id=response_id,
                created_at=created_at,
                model_name=model_name,
                usage=usage,
            )
        yield "data: [DONE]" + end_of_line

    raw_event_source = iter_sse_events(text_iterator)
    async with aclosing(raw_event_source):
        async for raw_event in raw_event_source:
            event_owner = await parse_owned_sse_event(raw_event)
            event_payload = None
            event_type = None
            delta_text = None
            synthetic_completed_payload = None
            completed_chunk = None
            content_chunk = None
            reasoning_chunk = None

            @asynccontextmanager
            async def clear_event_aliases_before_owner_release():
                nonlocal event_payload, event_type, delta_text
                nonlocal synthetic_completed_payload, raw_event
                nonlocal completed_chunk, content_chunk, reasoning_chunk
                try:
                    yield
                finally:
                    event_payload = None
                    event_type = None
                    delta_text = None
                    synthetic_completed_payload = None
                    completed_chunk = None
                    content_chunk = None
                    reasoning_chunk = None
                    raw_event = None

            async with event_owner, clear_event_aliases_before_owner_release():
                    if event_owner.is_comment:
                        yield event_owner.raw_event + end_of_line
                        continue
                    if not event_owner.has_data_field:
                        # SSE blocks without a data field do not dispatch an
                        # event.  The following data-only block, if present,
                        # is parsed independently and classified by payload.type.
                        continue
                    event_type = event_owner.event_name
                    event_payload = event_owner.payload
                    if event_type == "[DONE]":
                        synthetic_completed_payload = build_missing_responses_completed_payload(
                            completed_response_seen=completed_response_seen,
                            error_seen=error_seen,
                            response_id=response_id,
                            model_name=model_name,
                            created_at=created_at,
                            output_items_by_index=output_items_by_index,
                            output_items_fallback=output_items_fallback,
                        )
                        if synthetic_completed_payload is not None:
                            async with aclosing(
                                emit_completed_payload(synthetic_completed_payload)
                            ) as completed_chunks:
                                async for completed_chunk in completed_chunks:
                                    yield completed_chunk
                            return
                        yield "data: [DONE]" + end_of_line
                        return

                    validate_sse_event_type_consistency(
                        event_owner.declared_event_name,
                        event_payload,
                        protocol_name="Responses",
                        has_event_field=event_owner.has_event_field,
                        require_event_name=True,
                    )

                    if event_type == "error":
                        error_seen = True
                        semantic_error = responses_failure_error(
                            event_payload,
                            event_type=event_type,
                            wire_status_code=upstream_status_code,
                        )
                        if semantic_error is None:
                            raise SSEProtocolError(
                                "Responses upstream error terminal has no error semantics"
                            )
                        raise semantic_error

                    if event_type == "response.failed":
                        error_seen = True
                        semantic_error = responses_failure_error(
                            event_payload,
                            event_type=event_type,
                            wire_status_code=upstream_status_code,
                        )
                        if semantic_error is None:
                            raise SSEProtocolError(
                                "Responses upstream response.failed terminal has no error semantics"
                            )
                        raise semantic_error

                    if event_type == "response.incomplete" and isinstance(
                        event_payload,
                        dict,
                    ):
                        async with aclosing(
                            emit_completed_payload(event_payload)
                        ) as completed_chunks:
                            async for completed_chunk in completed_chunks:
                                yield completed_chunk
                        return

                    if event_type == "keepalive":
                        continue

                    if isinstance(event_payload, dict):
                        response_id = chat_completion_response_id_from_payload(
                            event_payload,
                            response_id,
                        )
                        created_at = chat_completion_created_at_from_payload(
                            event_payload,
                            created_at,
                        )
                        model_name = (
                            extract_response_model_name(event_payload) or model_name
                        )

                    if event_type == "response.output_item.done":
                        await output_item_collector.collect(
                            event_payload,
                            event_owner=event_owner,
                        )
                        continue

                    if event_type == "response.output_text.delta" and isinstance(
                        event_payload,
                        dict,
                    ):
                        delta_text = str(event_payload.get("delta") or "")
                        content_chunk = emit_content_delta(delta_text)
                        if content_chunk is not None:
                            yield content_chunk
                        continue

                    if event_type == "response.reasoning_summary_text.delta" and isinstance(
                        event_payload,
                        dict,
                    ):
                        delta_text = str(event_payload.get("delta") or "")
                        reasoning_chunk = emit_reasoning_delta(delta_text)
                        if reasoning_chunk is not None:
                            yield reasoning_chunk
                        continue

                    if event_type == "response.reasoning_summary_text.done":
                        reasoning_chunk = emit_reasoning_delta("\n\n")
                        if reasoning_chunk is not None:
                            yield reasoning_chunk
                        continue

                    if event_type != "response.completed" or not isinstance(
                        event_payload,
                        dict,
                    ):
                        continue

                    async with aclosing(
                        emit_completed_payload(event_payload)
                    ) as completed_chunks:
                        async for completed_chunk in completed_chunks:
                            yield completed_chunk
                    return

    synthetic_completed_payload = build_missing_responses_completed_payload(
        completed_response_seen=completed_response_seen,
        error_seen=error_seen,
        response_id=response_id,
        model_name=model_name,
        created_at=created_at,
        output_items_by_index=output_items_by_index,
        output_items_fallback=output_items_fallback,
    )
    if synthetic_completed_payload is not None:
        async with aclosing(
            emit_completed_payload(synthetic_completed_payload)
        ) as completed_chunks:
            async for completed_chunk in completed_chunks:
                yield completed_chunk
        return
    if error_seen:
        yield "data: [DONE]" + end_of_line
        return
    raise SSEProtocolError(
        "Responses stream ended without response.completed, error, or [DONE]"
    )


async def stream_responses_to_chat_completions(
    text_iterator,
    *,
    request_model: str,
    upstream_status_code: int | None = None,
    max_collected_output_items: int = DEFAULT_MAX_COLLECTED_OUTPUT_ITEMS,
    max_collected_output_bytes: int = DEFAULT_MAX_COLLECTED_OUTPUT_BYTES,
):
    """Convert Responses SSE while closing every transferred memory owner."""

    collector_holder: list[ResponsesOutputItemCollector] = []
    implementation = _stream_responses_to_chat_completions_impl(
        text_iterator,
        request_model=request_model,
        upstream_status_code=upstream_status_code,
        max_collected_output_items=max_collected_output_items,
        max_collected_output_bytes=max_collected_output_bytes,
        _collector_holder=collector_holder,
    )
    try:
        async for chunk in implementation:
            yield chunk
    finally:
        try:
            await implementation.aclose()
        finally:
            try:
                if collector_holder:
                    await collector_holder[0].aclose()
            finally:
                await close_async_iterator_safely(
                    text_iterator,
                    label="Responses-to-chat source iterator",
                )
