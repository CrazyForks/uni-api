import re
import random
import string
import base64
import binascii
import uuid
import asyncio
import zlib
from contextlib import aclosing
from datetime import datetime
from math import isfinite
from urllib.parse import urlparse
from typing import Any, AsyncIterator, Callable

from core.log_config import logger
from uni_api.admission import get_request_admission_lease
from uni_api.admission.json_parsing import (
    parse_owned_json_value,
    parsed_json_value,
    run_json_cpu,
)
from uni_api.http_content import is_json_media_type
from uni_api.serialization import json

from core.utils import (
    safe_get,
    end_of_line,
    parse_json_safely,
    parse_sse_event,
)
from uni_api.providers.normalization import build_openai_audio_object, normalize_gemini_parts
from uni_api.streaming.sse import (
    DEFAULT_MAX_EVENT_BYTES,
    IncrementalSSEParser,
    OwnedSSEEvent,
    SSEBufferOverflowError,
    SSEOutputLimitError,
    SSEProtocolError,
    StreamParserRetainedLease,
    is_sse_comment_frame,
    iter_lines,
    iter_sse_events,
    parse_owned_sse_event,
    parsed_sse_event,
    retain_joined_parser_text,
)
from uni_api.streaming.cleanup import (
    BACKGROUND_STREAM_CLEANUP_TASKS as _BACKGROUND_STREAM_CLEANUP_TASKS,
    await_stream_cleanup_safely as _await_stream_cleanup_safely,
    call_cleanup_safely as _call_cleanup_safely,
    close_async_iterator_safely as _close_async_iterator_safely,
    drain_current_task_cancellation as _drain_current_task_cancellation,
    force_close_response_httpcore_stream_chain_safely as _force_close_response_httpcore_stream_chain_safely,
    force_release_httpcore_pool_request_safely as _force_release_httpcore_pool_request_safely,
    track_background_stream_cleanup_task as _track_background_stream_cleanup_task,
    yield_from_stream as _yield_from_stream,
)
from uni_api.streaming.chat_completion_events import (
    generate_no_stream_response,
    generate_sse_response,
    responses_usage_to_chat_completion_usage as _responses_usage_to_chat_completion_usage,
)
from uni_api.streaming.responses_events import (
    mime_type_from_output_format as _mime_type_from_output_format,
    normalize_optional_text as _normalize_optional_text,
    stream_responses_to_chat_completions as _stream_responses_to_chat_completions,
)
from uni_api.upstream.response_limits import (
    read_limited_response_body,
    upstream_json_memory_reservation_multiplier,
    upstream_success_body_max_bytes,
)

ResponseHeadersSink = Callable[[Any], None]
_MAX_TOKEN_COUNT = (1 << 63) - 1


def _coerce_token_count(value: Any) -> int:
    """Collapse untrusted usage fields to one bounded scalar immediately."""

    parsed: int
    if isinstance(value, bool):
        parsed = int(value)
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not isfinite(value):
            return 0
        parsed = int(value)
    elif isinstance(value, str):
        if len(value) > 32:
            return 0
        normalized = value.strip()
        if not normalized or not normalized.isdecimal():
            return 0
        parsed = int(normalized)
    else:
        return 0
    return min(_MAX_TOKEN_COUNT, max(0, parsed))


def _extract_named_token_count(line: str, field: str) -> int:
    match = re.search(
        rf'"{re.escape(field)}"\s*:\s*(\d{{1,20}})',
        line,
    )
    return _coerce_token_count(match.group(1)) if match is not None else 0


def _bounded_text_concat(
    current: str,
    addition: str,
    *,
    label: str,
    limit_bytes: int = 256,
) -> str:
    observed = len(current.encode("utf-8")) + len(addition.encode("utf-8"))
    if observed > limit_bytes:
        raise SSEBufferOverflowError(
            buffer_name=label,
            limit_bytes=limit_bytes,
            observed_bytes=observed,
        )
    return current + addition


class _BoundedTextAccumulator:
    def __init__(
        self,
        *,
        label: str,
        limit_bytes: int = DEFAULT_MAX_EVENT_BYTES,
        max_parts: int = 4096,
    ) -> None:
        self.label = label
        self.limit_bytes = limit_bytes
        self.parts: list[str] = []
        self.total_bytes = 0
        self.max_parts = max_parts
        self._retained = StreamParserRetainedLease()

    def append(self, value: str) -> None:
        if len(self.parts) >= self.max_parts:
            raise SSEOutputLimitError(
                output_name=f"{self.label} fragments",
                limit=self.max_parts,
                observed=self.max_parts + 1,
            )
        worst_case_bytes = len(value) * 4
        self._retained.grow(worst_case_bytes)
        try:
            encoded_bytes = len(value.encode("utf-8"))
            observed = self.total_bytes + encoded_bytes
            if observed > self.limit_bytes:
                raise SSEBufferOverflowError(
                    buffer_name=self.label,
                    limit_bytes=self.limit_bytes,
                    observed_bytes=observed,
                )
            self.parts.append(value)
            self.total_bytes = observed
        except BaseException:
            self._retained.shrink(worst_case_bytes)
            raise
        self._retained.shrink(worst_case_bytes - encoded_bytes)

    def append_slice(
        self,
        source: str,
        start: int,
        end: int,
        *,
        suffix: str = "",
    ) -> None:
        if len(self.parts) >= self.max_parts:
            raise SSEOutputLimitError(
                output_name=f"{self.label} fragments",
                limit=self.max_parts,
                observed=self.max_parts + 1,
            )
        worst_case_bytes = (max(0, end - start) + len(suffix)) * 4
        self._retained.grow(worst_case_bytes)
        try:
            value = source[start:end] + suffix
            encoded_bytes = len(value.encode("utf-8"))
            observed = self.total_bytes + encoded_bytes
            if observed > self.limit_bytes:
                raise SSEBufferOverflowError(
                    buffer_name=self.label,
                    limit_bytes=self.limit_bytes,
                    observed_bytes=observed,
                )
            self.parts.append(value)
            self.total_bytes = observed
        except BaseException:
            self._retained.shrink(worst_case_bytes)
            raise
        self._retained.shrink(worst_case_bytes - encoded_bytes)

    def take_text(self) -> str:
        frame = retain_joined_parser_text(
            self.parts,
            retained_bytes=self.total_bytes,
        )
        self.parts.clear()
        self.total_bytes = 0
        self._retained.release()
        self._retained = StreamParserRetainedLease()
        return frame

    def close(self) -> None:
        self.parts.clear()
        self.total_bytes = 0
        self._retained.release()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class _BoundedJSONObjectAccumulator:
    """Frame pretty-printed objects from a bounded top-level JSON array."""

    def __init__(
        self,
        *,
        label: str,
        limit_bytes: int = DEFAULT_MAX_EVENT_BYTES,
        max_frames_per_feed: int = 4096,
    ) -> None:
        if max_frames_per_feed <= 0:
            raise ValueError("max_frames_per_feed must be positive")
        self._text = _BoundedTextAccumulator(label=label, limit_bytes=limit_bytes)
        self._max_frames_per_feed = int(max_frames_per_feed)
        self._depth = 0
        self._in_string = False
        self._escaped = False
        self._started = False

    def feed_line(self, line: str) -> list[str]:
        frames: list[str] = []
        segment_start: int | None = 0 if self._started else None
        for index, char in enumerate(line):
            if not self._started:
                if char.isspace() or char in "[, ]":
                    continue
                if char != "{":
                    raise SSEProtocolError(
                        "Gemini JSON stream contains a non-object array item"
                    )
                self._started = True
                self._depth = 0
                self._in_string = False
                self._escaped = False
                segment_start = index

            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_string = False
            elif char == '"':
                self._in_string = True
            elif char in "{[":
                self._depth += 1
            elif char in "}]":
                self._depth -= 1
                if self._depth < 0:
                    raise SSEProtocolError("Gemini JSON stream has unbalanced brackets")

            if self._started and self._depth == 0:
                assert segment_start is not None
                if len(frames) >= self._max_frames_per_feed:
                    raise SSEOutputLimitError(
                        output_name="Gemini JSON objects",
                        limit=self._max_frames_per_feed,
                        observed=self._max_frames_per_feed + 1,
                    )
                self._text.append_slice(line, segment_start, index + 1)
                frames.append(self._text.take_text())
                self._text = _BoundedTextAccumulator(
                    label=self._text.label,
                    limit_bytes=self._text.limit_bytes,
                    max_parts=self._text.max_parts,
                )
                self._started = False
                segment_start = None

        if self._started:
            assert segment_start is not None
            self._text.append_slice(
                line,
                segment_start,
                len(line),
                suffix="\n",
            )
        return frames

    def finish(self) -> None:
        if self._started:
            raise SSEProtocolError("Gemini JSON stream ended with an incomplete object")


def _capture_response_headers(response_headers_sink: ResponseHeadersSink | None, headers: Any) -> None:
    if response_headers_sink is not None and headers is not None:
        response_headers_sink(headers)


async def _iter_openai_stream_events(response: Any):
    """Yield explicitly owned OpenAI events from SSE or JSON-lines streams."""

    content_type = str(
        getattr(response, "headers", {}).get("content-type", "") or ""
    ).lower()
    if "text/event-stream" in content_type:
        events = _iter_owned_sse_events(response.aiter_bytes())
        async with aclosing(events):
            async for event_owner in events:
                yield event_owner
        return

    lines = iter_lines(response.aiter_bytes())
    async with aclosing(lines):
        async for line in lines:
            event_owner = None
            precopy_reservation = None
            raw_event = None
            try:
                if not line or (len(line) <= 64 and line.isspace()):
                    continue

                field_start = 0
                # Only inspect a bounded protocol prefix in Python.  JSON
                # accepts arbitrary leading whitespace; scanning an 8 MiB
                # whitespace prefix character-by-character would block the
                # event loop before admission can react.
                prefix_scan_limit = min(len(line), 64)
                while (
                    field_start < prefix_scan_limit
                    and line[field_start].isspace()
                ):
                    field_start += 1
                has_sse_field = field_start < prefix_scan_limit and line.startswith(
                    ("data:", ":", "event:"),
                    field_start,
                )
                needs_copy = field_start > 0 or not has_sse_field
                if needs_copy:
                    request_lease = get_request_admission_lease()
                    if request_lease is not None:
                        precopy_reservation = (
                            await request_lease.reserve_temporary_response_bytes(
                                len(line) * 4 + 64
                            )
                        )
                if has_sse_field:
                    raw_event = line[field_start:] if field_start else line
                else:
                    # JSON parsers accept leading/trailing whitespace, so no
                    # attacker-sized strip() copy is needed.
                    raw_event = f"data: {line}"
                event_owner = await parse_owned_sse_event(raw_event)
                yield event_owner
            except json.JSONDecodeError as exc:
                raise SSEProtocolError(
                    "upstream JSON-lines frame is not valid JSON"
                ) from exc
            finally:
                line = None
                raw_event = None
                try:
                    if event_owner is not None:
                        await event_owner.aclose()
                finally:
                    if precopy_reservation is not None:
                        await precopy_reservation.release()


async def _iter_owned_sse_events(chunks):
    """Yield events whose consumers explicitly own and close each payload."""

    raw_source = iter_sse_events(chunks)
    async with aclosing(raw_source):
        async for raw_event in raw_source:
            event_owner = await parse_owned_sse_event(raw_event)
            try:
                yield event_owner
            finally:
                raw_event = None
                await event_owner.aclose()
                event_owner = None


async def _iter_owned_raw_events(raw_events):
    for index, raw_event in enumerate(raw_events):
        event_owner = await parse_owned_sse_event(raw_event)
        try:
            yield event_owner
        finally:
            raw_events[index] = None
            raw_event = None
            await event_owner.aclose()
            event_owner = None


def _normalize_search_item_defaults(item: dict) -> dict:
    normalized = dict(item or {})
    normalized.setdefault("title", "")
    normalized.setdefault("url", "")
    normalized.setdefault("description", "")
    normalized.setdefault("content", "")
    normalized.setdefault("usage", None)
    normalized.setdefault("score", None)
    normalized.setdefault("raw_content", None)
    return normalized

def normalize_search_response(url: str, response_json: object) -> dict:
    """
    Normalizes different search providers into a Jina-like shape:
      { code, status, data: [{title,url,description,content,...}], meta: {...} }
    """
    parsed = urlparse(url or "")
    host = (parsed.netloc or "").lower()

    # Tavily shape:
    # {query, results:[{url,title,content,score,raw_content}], response_time, request_id, ...}
    if isinstance(response_json, dict) and (host.endswith("tavily.com") or "results" in response_json):
        results = response_json.get("results") or []
        data = []
        for r in results:
            if not isinstance(r, dict):
                continue
            title = r.get("title") or ""
            link = r.get("url") or ""
            content = r.get("content") or ""
            description = content
            if isinstance(description, str) and len(description) > 240:
                description = description[:237] + "..."
            item = {
                "title": title,
                "url": link,
                "description": description or "",
                "content": content or "",
            }
            # keep Tavily extra fields at the same level for convenience
            for k, v in r.items():
                if k not in item:
                    item[k] = v
            data.append(_normalize_search_item_defaults(item))

        meta = {
            "provider": "tavily",
            "query": response_json.get("query"),
            "answer": response_json.get("answer"),
            "follow_up_questions": response_json.get("follow_up_questions"),
            "images": response_json.get("images"),
            "response_time": response_json.get("response_time"),
            "request_id": response_json.get("request_id"),
        }
        # preserve any additional top-level fields
        for k, v in response_json.items():
            if k not in meta and k != "results":
                meta[k] = v

        return {
            "code": 200,
            "status": 20000,
            "data": data,
            "meta": meta,
        }

    # Jina (already close to desired format).
    if isinstance(response_json, dict) and "data" in response_json:
        out = dict(response_json)
        out.setdefault("code", 200)
        out.setdefault("status", 20000)
        meta = out.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("provider", "jina")
        out["meta"] = meta
        normalized_data = []
        for item in (out.get("data") or []):
            if isinstance(item, dict):
                normalized_data.append(_normalize_search_item_defaults(item))
        out["data"] = normalized_data
        return out

    # Fallback: wrap unknown shapes without losing data.
    return {
        "code": 200,
        "status": 20000,
        "data": [],
        "meta": {"provider": "unknown", "raw": response_json},
    }

def _responses_output_to_text(response_json: dict) -> tuple[str, str]:
    """
    Best-effort extraction of text + reasoning text from an OpenAI Responses-style response.
    Returns: (content, reasoning_content)
    """
    if not isinstance(response_json, dict):
        return "", ""

    content_parts: list[str] = []
    reasoning_parts: list[str] = []

    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text:
        content_parts.append(output_text)

    output = response_json.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            if item_type in ("output_text", "text") and item.get("text"):
                content_parts.append(str(item.get("text")))
                continue
            if item_type == "image_generation_call":
                result_b64 = item.get("result")
                if isinstance(result_b64, str) and result_b64:
                    mime_type = _mime_type_from_output_format(
                        _normalize_optional_text(item.get("output_format"))
                    )
                    content_parts.append(f"![image](data:{mime_type};base64,{result_b64})")
                continue
            if item_type in ("reasoning_summary_text", "reasoning_text") and item.get("text"):
                reasoning_parts.append(str(item.get("text")))
                continue

            if item_type != "message":
                continue

            role = (item.get("role") or "").lower()
            if role and role not in ("assistant", "model"):
                continue

            msg_content = item.get("content")
            if isinstance(msg_content, str) and msg_content:
                content_parts.append(msg_content)
                continue
            if not isinstance(msg_content, list):
                continue
            for part in msg_content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in ("output_text", "text") and part.get("text"):
                    content_parts.append(str(part.get("text")))
                elif part_type in ("reasoning_summary_text", "reasoning_text") and part.get("text"):
                    reasoning_parts.append(str(part.get("text")))

    return "".join(content_parts), "".join(reasoning_parts)

def _is_responses_api_call(url: str, payload: dict) -> bool:
    if "v1/responses" in (url or ""):
        return True
    if isinstance(payload, dict) and "input" in payload and "messages" not in payload:
        return True
    return False


async def check_response(response, error_log):
    if response and not (200 <= response.status_code < 300):
        error_body = await read_limited_response_body(response)
        error_str = error_body.text()
        # Keep error diagnostics as bounded text.  Materializing arbitrary JSON
        # here would let a dense error graph escape into the returned object
        # after its parse reservation ended.
        return {
            "error": f"{error_log} HTTP Error",
            "status_code": response.status_code,
            "details": error_str,
        }
    return None

async def gemini_json_poccess(response_json):
    promptTokenCount = 0
    candidatesTokenCount = 0
    totalTokenCount = 0
    cachedContentTokenCount = 0
    thoughtsTokenCount = 0

    json_data = safe_get(response_json, "candidates", 0, "content", default=None)
    finishReason = safe_get(response_json, "candidates", 0 , "finishReason", default=None)
    usage_metadata = response_json.get("usageMetadata") if isinstance(response_json, dict) else None
    if finishReason and isinstance(usage_metadata, dict):
        promptTokenCount = _coerce_token_count(
            usage_metadata.get("promptTokenCount", promptTokenCount)
        )
        candidatesTokenCount = _coerce_token_count(
            usage_metadata.get("candidatesTokenCount", candidatesTokenCount)
        )
        totalTokenCount = _coerce_token_count(
            usage_metadata.get("totalTokenCount", totalTokenCount)
        )
        cachedContentTokenCount = _coerce_token_count(
            usage_metadata.get("cachedContentTokenCount", cachedContentTokenCount)
        )
        thoughtsTokenCount = _coerce_token_count(
            usage_metadata.get("thoughtsTokenCount", thoughtsTokenCount)
        )
        if finishReason != "STOP":
            logger.error(f"finishReason: {finishReason}")

    parts_list = safe_get(json_data, "parts", default=[])
    normalized = await normalize_gemini_parts(
        parts_list if isinstance(parts_list, list) else []
    )

    blockReason = safe_get(json_data, 0, "promptFeedback", "blockReason", default=None)

    return (
        normalized.is_thinking,
        normalized.reasoning_content,
        normalized.content,
        normalized.image_base64,
        normalized.audio_wav_base64,
        normalized.function_call.name,
        normalized.function_call.arguments_json,
        normalized.function_call.call_id,
        finishReason,
        blockReason,
        promptTokenCount,
        candidatesTokenCount,
        totalTokenCount,
        cachedContentTokenCount,
        thoughtsTokenCount,
    )

async def fetch_gemini_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await run_json_cpu(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_gemini_response_stream")
        if error_message:
            yield error_message
            return
        promptTokenCount = 0
        candidatesTokenCount = 0
        totalTokenCount = 0
        cachedContentTokenCount = 0
        thoughtsTokenCount = 0
        json_framer = _BoundedJSONObjectAccumulator(label="gemini JSON frame")
        terminal_seen = False
        content_type = str(response.headers.get("content-type") or "").lower()

        async def response_objects():
            if "text/event-stream" in content_type:
                events = _iter_owned_sse_events(response.aiter_bytes())
                async with aclosing(events):
                    async for event_owner in events:
                        event_payload = None
                        try:
                            if event_owner.is_comment:
                                continue
                            event_payload = event_owner.payload
                            if event_payload == "[DONE]":
                                return
                            if not isinstance(event_payload, dict):
                                raise SSEProtocolError(
                                    "Gemini SSE payload must be a JSON object"
                                )
                            yield event_owner
                        finally:
                            event_payload = None
                            await event_owner.aclose()
                return

            lines = iter_lines(response.aiter_bytes())
            async with aclosing(lines):
                async for bounded_line in lines:
                    frames = None
                    try:
                        if bounded_line.startswith("data:"):
                            event_owner = await parse_owned_sse_event(bounded_line)
                            event_payload = None
                            try:
                                event_payload = event_owner.payload
                                if not isinstance(event_payload, dict):
                                    continue
                                yield event_owner
                            finally:
                                event_payload = None
                                await event_owner.aclose()
                            continue

                        frames = (
                            await run_json_cpu(
                                json_framer.feed_line,
                                bounded_line,
                            )
                            if len(bounded_line) >= 64 * 1024
                            else json_framer.feed_line(bounded_line)
                        )
                        for index, frame in enumerate(frames):
                            owner = None
                            try:
                                owner = await parse_owned_json_value(frame)
                                if not isinstance(owner.value, dict):
                                    raise SSEProtocolError(
                                        "Gemini JSON stream payload must be an object"
                                    )
                                yield owner
                            except json.JSONDecodeError:
                                continue
                            finally:
                                frames[index] = None
                                frame = None
                                if owner is not None:
                                    await owner.aclose()
                    finally:
                        frames = None
                        bounded_line = None
            json_framer.finish()

        objects = response_objects()
        async with aclosing(objects):
            async for event_owner in objects:
                response_json = None
                is_thinking = None
                reasoning_content = None
                content = None
                image_base64 = None
                audio_b64_wav = None
                function_call_name = None
                function_full_response = None
                tools_id = None
                finishReason = None
                blockReason = None
                audio_obj = None
                try:
                    if isinstance(event_owner, OwnedSSEEvent):
                        response_json = event_owner.payload
                    else:
                        response_json = event_owner.value

                    # https://ai.google.dev/api/generate-content?hl=zh-cn#FinishReason
                    (
                        is_thinking,
                        reasoning_content,
                        content,
                        image_base64,
                        audio_b64_wav,
                        function_call_name,
                        function_full_response,
                        tools_id,
                        finishReason,
                        blockReason,
                        promptTokenCount,
                        candidatesTokenCount,
                        totalTokenCount,
                        cachedContentTokenCount,
                        thoughtsTokenCount,
                    ) = await gemini_json_poccess(response_json)

                    if is_thinking and reasoning_content:
                        yield await generate_sse_response(
                            timestamp, model, reasoning_content=reasoning_content
                        )
                    if not image_base64 and content:
                        yield await generate_sse_response(timestamp, model, content=content)

                    if image_base64:
                        if "flash-image" not in model and "pro-image" not in model:
                            completion_tokens = candidatesTokenCount + thoughtsTokenCount
                            openai_total_tokens = totalTokenCount or (
                                promptTokenCount + completion_tokens
                            )
                            yield await generate_no_stream_response(
                                timestamp,
                                model,
                                content=content,
                                role=None,
                                total_tokens=openai_total_tokens,
                                prompt_tokens=promptTokenCount,
                                completion_tokens=completion_tokens,
                                cached_tokens=cachedContentTokenCount,
                                reasoning_tokens=thoughtsTokenCount,
                                image_base64=image_base64,
                            )
                        else:
                            yield await generate_sse_response(
                                timestamp,
                                model,
                                content=f"\n![image](data:image/png;base64,{image_base64})",
                            )
                    if audio_b64_wav:
                        audio_obj = build_openai_audio_object(
                            audio_b64_wav,
                            transcript=content or None,
                        )
                        yield await generate_no_stream_response(
                            timestamp,
                            model,
                            content=content or None,
                            role="assistant",
                            total_tokens=totalTokenCount
                            or (
                                promptTokenCount
                                + candidatesTokenCount
                                + thoughtsTokenCount
                            ),
                            prompt_tokens=promptTokenCount,
                            completion_tokens=candidatesTokenCount + thoughtsTokenCount,
                            cached_tokens=cachedContentTokenCount,
                            reasoning_tokens=thoughtsTokenCount,
                            audio=audio_obj,
                        )

                    if function_call_name:
                        yield await generate_sse_response(
                            timestamp,
                            model,
                            content=None,
                            tools_id=tools_id,
                            function_call_name=function_call_name,
                        )
                    if function_full_response:
                        yield await generate_sse_response(
                            timestamp,
                            model,
                            content=None,
                            tools_id=tools_id,
                            function_call_name=None,
                            function_call_content=function_full_response,
                        )

                    if blockReason == "PROHIBITED_CONTENT":
                        yield await generate_sse_response(
                            timestamp,
                            model,
                            stop="PROHIBITED_CONTENT",
                        )
                        terminal_seen = True
                        break
                    if finishReason:
                        yield await generate_sse_response(timestamp, model, stop="stop")
                        terminal_seen = True
                        break
                finally:
                    response_json = None
                    is_thinking = None
                    reasoning_content = None
                    content = None
                    image_base64 = None
                    audio_b64_wav = None
                    function_call_name = None
                    function_full_response = None
                    tools_id = None
                    finishReason = None
                    blockReason = None
                    audio_obj = None
                    await event_owner.aclose()

        if not terminal_seen:
            raise SSEProtocolError(
                "Gemini upstream ended without a finish reason"
            )

        completion_tokens = candidatesTokenCount + thoughtsTokenCount
        openai_total_tokens = totalTokenCount or (promptTokenCount + completion_tokens)
        sse_string = await generate_sse_response(
            timestamp,
            model,
            None,
            None,
            None,
            None,
            None,
            openai_total_tokens,
            promptTokenCount,
            completion_tokens,
            cached_tokens=cachedContentTokenCount,
            reasoning_tokens=thoughtsTokenCount,
        )
        yield sse_string

    yield "data: [DONE]" + end_of_line

async def fetch_vertex_claude_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await run_json_cpu(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_vertex_claude_response_stream")
        if error_message:
            yield error_message
            return

        revicing_function_call = False
        function_response_buffer = _BoundedTextAccumulator(
            label="vertex tool call"
        )
        function_response_buffer.append("{")
        need_function_call = False
        is_finish = False
        promptTokenCount = 0
        candidatesTokenCount = 0
        totalTokenCount = 0
        terminal_seen = False

        lines = iter_lines(response.aiter_bytes())
        try:
            async with aclosing(lines):
                async for line in lines:
                    json_owner = None
                    precopy_reservation = None
                    json_data = None
                    content = None
                    sse_string = None
                    snippet = None
                    try:
                        if line and '\"finishReason\": \"' in line:
                            is_finish = True
                            terminal_seen = True
                        if is_finish:
                            for field_name in (
                                "promptTokenCount",
                                "candidatesTokenCount",
                                "totalTokenCount",
                            ):
                                if f'\"{field_name}\"' not in line:
                                    continue
                                value = (
                                    await run_json_cpu(
                                        _extract_named_token_count,
                                        line,
                                        field_name,
                                    )
                                    if len(line) >= 64 * 1024
                                    else _extract_named_token_count(line, field_name)
                                )
                                if field_name == "promptTokenCount":
                                    promptTokenCount = value
                                elif field_name == "candidatesTokenCount":
                                    candidatesTokenCount = value
                                else:
                                    totalTokenCount = value

                        if line and '\"text\": \"' in line and not is_finish:
                            request_lease = get_request_admission_lease()
                            if request_lease is not None:
                                precopy_reservation = (
                                    await request_lease.reserve_temporary_response_bytes(
                                        len(line) * 4 + 64
                                    )
                                )
                            snippet = "{" + line.strip().rstrip(",") + "}"
                            try:
                                json_owner = await parse_owned_json_value(snippet)
                                json_data = json_owner.value
                                if not isinstance(json_data, dict):
                                    raise SSEProtocolError(
                                        "Vertex Claude text frame must be an object"
                                    )
                                content = json_data.get("text", "")
                                if not isinstance(content, str):
                                    raise SSEProtocolError(
                                        "Vertex Claude text field must be a string"
                                    )
                                sse_string = await generate_sse_response(
                                    timestamp,
                                    model,
                                    content=content,
                                )
                                yield sse_string
                            except json.JSONDecodeError:
                                logger.error(
                                    "Unable to parse Vertex Claude JSON line: %s",
                                    line[:512],
                                )

                        if line and (
                            '\"type\": \"tool_use\"' in line
                            or revicing_function_call
                        ):
                            revicing_function_call = True
                            need_function_call = True
                            if "]" in line:
                                revicing_function_call = False
                                continue
                            function_response_buffer.append(line)
                    finally:
                        json_data = None
                        content = None
                        sse_string = None
                        snippet = None
                        line = None
                        try:
                            if json_owner is not None:
                                await json_owner.aclose()
                        finally:
                            if precopy_reservation is not None:
                                await precopy_reservation.release()
        except BaseException:
            function_response_buffer.close()
            raise

        if need_function_call:
            function_owner = None
            function_text = function_response_buffer.take_text()
            function_call = None
            function_call_name = None
            function_call_id = None
            function_input = None
            function_full_response = None
            sse_string = None
            try:
                function_owner = await parse_owned_json_value(function_text)
                function_call = function_owner.value
                if not isinstance(function_call, dict):
                    raise SSEProtocolError(
                        "Vertex Claude tool call must be a JSON object"
                    )
                function_call_name = function_call["name"]
                function_call_id = function_call["id"]
                function_input = function_call["input"]
                sse_string = await generate_sse_response(
                    timestamp,
                    model,
                    content=None,
                    tools_id=function_call_id,
                    function_call_name=function_call_name,
                )
                yield sse_string
                function_full_response = await run_json_cpu(
                    json.dumps,
                    function_input,
                )
                sse_string = await generate_sse_response(
                    timestamp,
                    model,
                    content=None,
                    tools_id=function_call_id,
                    function_call_name=None,
                    function_call_content=function_full_response,
                )
                yield sse_string
            finally:
                function_text = None
                function_call = None
                function_call_name = None
                function_call_id = None
                function_input = None
                function_full_response = None
                sse_string = None
                if function_owner is not None:
                    await function_owner.aclose()
        else:
            function_response_buffer.close()

        if not terminal_seen:
            raise SSEProtocolError(
                "Vertex Claude upstream ended without a finish reason"
            )

        sse_string = await generate_sse_response(timestamp, model, None, None, None, None, None, totalTokenCount, promptTokenCount, candidatesTokenCount)
        yield sse_string

    yield "data: [DONE]" + end_of_line

async def fetch_gpt_response_stream(client, url, headers, payload, timeout, response_headers_sink: ResponseHeadersSink | None = None):
    timestamp = int(datetime.timestamp(datetime.now()))
    random.seed(timestamp)
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=29))
    is_thinking = False
    has_send_thinking = False
    ark_tag = False
    json_payload = await run_json_cpu(json.dumps, payload)
    response = None
    completed_normally = False
    semantic_terminal_seen = False
    input_tokens = 0
    output_tokens = 0
    try:
        async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
            _capture_response_headers(response_headers_sink, getattr(response, "headers", None))
            error_message = await check_response(response, "fetch_gpt_response_stream")
            if error_message:
                yield error_message
                return

            if _is_responses_api_call(url, payload):
                async for chunk in _stream_responses_to_chat_completions(
                    response.aiter_bytes(),
                    request_model=payload["model"],
                ):
                    yield chunk
                completed_normally = True
                return

            enter_buffer = ""

            async def transform_event(event_owner: OwnedSSEEvent):
                nonlocal ark_tag
                nonlocal completed_normally
                nonlocal enter_buffer
                nonlocal has_send_thinking
                nonlocal input_tokens
                nonlocal is_thinking
                nonlocal output_tokens
                nonlocal semantic_terminal_seen

                raw_event = None
                event_payload = None
                line = None
                event_type = None
                content = None
                end_think_reasoning_content = None
                end_think_content = None
                no_stream_content = None
                openrouter_reasoning = None
                openrouter_base64_image = None
                image_data_url = None
                azure_databricks_claude_summary_content = None
                azure_databricks_claude_signature_content = None
                reasoning_prefix = None
                sse_string = None
                json_line = None
                try:
                    if event_owner.is_comment:
                        raw_event = event_owner.raw_event
                        yield raw_event + end_of_line
                        return
                    event_payload = event_owner.payload
                    if event_payload == "[DONE]":
                        completed_normally = True
                        return
                    if not isinstance(event_payload, dict):
                        if str(event_payload or "").strip():
                            raise SSEProtocolError(
                                "upstream SSE data is not valid JSON"
                            )
                        return

                    line = event_payload
                    line["id"] = f"chatcmpl-{random_str}"
                    if safe_get(
                        line,
                        "choices",
                        0,
                        "finish_reason",
                        default=None,
                    ) is not None:
                        semantic_terminal_seen = True

                    event_type = line.get("type")
                    if (
                        event_type == "response.reasoning_summary_text.delta"
                        and line.get("delta")
                    ):
                        sse_string = await generate_sse_response(
                            timestamp,
                            payload["model"],
                            reasoning_content=line.get("delta"),
                        )
                        yield sse_string
                        return
                    if event_type == "response.reasoning_summary_text.done":
                        sse_string = await generate_sse_response(
                            timestamp,
                            payload["model"],
                            reasoning_content="\n\n",
                        )
                        yield sse_string
                        return
                    if event_type == "response.output_text.delta" and line.get("delta"):
                        sse_string = await generate_sse_response(
                            timestamp,
                            payload["model"],
                            content=line.get("delta"),
                        )
                        yield sse_string
                        return
                    if event_type == "response.output_text.done":
                        sse_string = await generate_sse_response(
                            timestamp,
                            payload["model"],
                            stop="stop",
                        )
                        yield sse_string
                        return
                    if event_type == "response.completed":
                        input_tokens = _coerce_token_count(
                            safe_get(
                                line,
                                "response",
                                "usage",
                                "input_tokens",
                                default=0,
                            )
                        )
                        output_tokens = _coerce_token_count(
                            safe_get(
                                line,
                                "response",
                                "usage",
                                "output_tokens",
                                default=0,
                            )
                        )
                        semantic_terminal_seen = True
                        completed_normally = True
                        return
                    if isinstance(event_type, str) and event_type.startswith("response."):
                        return

                    content = safe_get(
                        line,
                        "choices",
                        0,
                        "delta",
                        "content",
                        default="",
                    )
                    if "<think>" in content:
                        is_thinking = True
                        ark_tag = True
                        content = content.replace("<think>", "")
                    if "</think>" in content:
                        is_thinking = False
                        if content.rstrip("\n").endswith("</think>"):
                            end_think_reasoning_content = content.replace(
                                "</think>", ""
                            ).rstrip("\n")
                        elif content.lstrip("\n").startswith("</think>"):
                            end_think_content = content.replace(
                                "</think>", ""
                            ).lstrip("\n")
                        else:
                            end_think_reasoning_content, end_think_content = (
                                content.split("</think>", 1)
                            )
                        if end_think_reasoning_content:
                            sse_string = await generate_sse_response(
                                timestamp,
                                payload["model"],
                                reasoning_content=end_think_reasoning_content,
                            )
                            yield sse_string
                        if end_think_content:
                            sse_string = await generate_sse_response(
                                timestamp,
                                payload["model"],
                                content=end_think_content,
                            )
                            yield sse_string
                        return
                    if is_thinking and ark_tag:
                        if not has_send_thinking:
                            content = content.replace("\n\n", "")
                        if content:
                            sse_string = await generate_sse_response(
                                timestamp,
                                payload["model"],
                                reasoning_content=content,
                            )
                            yield sse_string
                            has_send_thinking = True
                        return

                    if "Thinking..." in content and "\n> " in content:
                        is_thinking = True
                        content = content.replace("Thinking...", "").replace(
                            "\n> ", ""
                        )
                    if is_thinking and "\n\n" in content and not ark_tag:
                        is_thinking = False
                    if is_thinking and not ark_tag:
                        content = content.replace("\n> ", "")
                        if not has_send_thinking:
                            content = content.replace("\n", "")
                        if content:
                            sse_string = await generate_sse_response(
                                timestamp,
                                payload["model"],
                                reasoning_content=content,
                            )
                            yield sse_string
                            has_send_thinking = True
                        return

                    no_stream_content = safe_get(
                        line,
                        "choices",
                        0,
                        "message",
                        "content",
                        default=None,
                    )
                    openrouter_reasoning = safe_get(
                        line,
                        "choices",
                        0,
                        "delta",
                        "reasoning",
                        default="",
                    )
                    openrouter_base64_image = safe_get(
                        line,
                        "choices",
                        0,
                        "delta",
                        "images",
                        0,
                        "image_url",
                        "url",
                        default="",
                    )
                    if openrouter_base64_image:
                        image_data_url = (
                            openrouter_base64_image
                            if openrouter_base64_image.startswith("data:")
                            else f"data:image/png;base64,{openrouter_base64_image}"
                        )
                        sse_string = await generate_sse_response(
                            timestamp,
                            payload["model"],
                            content=f"\n![image]({image_data_url})",
                        )
                        yield sse_string
                        return
                    azure_databricks_claude_summary_content = safe_get(
                        line,
                        "choices",
                        0,
                        "delta",
                        "content",
                        0,
                        "summary",
                        0,
                        "text",
                        default="",
                    )
                    azure_databricks_claude_signature_content = safe_get(
                        line,
                        "choices",
                        0,
                        "delta",
                        "content",
                        0,
                        "summary",
                        0,
                        "signature",
                        default="",
                    )
                    if azure_databricks_claude_signature_content:
                        return
                    if azure_databricks_claude_summary_content:
                        sse_string = await generate_sse_response(
                            timestamp,
                            payload["model"],
                            reasoning_content=azure_databricks_claude_summary_content,
                        )
                        yield sse_string
                        return
                    if openrouter_reasoning:
                        if openrouter_reasoning.endswith("\\"):
                            # Only the ambiguous trailing escape marker needs
                            # to cross an event boundary.  Emitting the safe
                            # prefix immediately prevents one normal long
                            # reasoning delta from becoming a tiny-buffer
                            # protocol failure or retained-memory spike.
                            reasoning_prefix = enter_buffer + openrouter_reasoning[:-1]
                            enter_buffer = "\\"
                            if reasoning_prefix:
                                reasoning_prefix = reasoning_prefix.replace(
                                    "\\n",
                                    "\n",
                                )
                                sse_string = await generate_sse_response(
                                    timestamp,
                                    payload["model"],
                                    reasoning_content=reasoning_prefix,
                                )
                                yield sse_string
                            return
                        if enter_buffer.endswith("\\") and openrouter_reasoning == "n":
                            enter_buffer = _bounded_text_concat(
                                enter_buffer,
                                "n",
                                label="reasoning escape buffer",
                            )
                            return
                        if (
                            enter_buffer.endswith("\\n")
                            and openrouter_reasoning == "\\n"
                        ):
                            enter_buffer = _bounded_text_concat(
                                enter_buffer,
                                "\\n",
                                label="reasoning escape buffer",
                            )
                            return
                        if enter_buffer.endswith("\\n\\n"):
                            openrouter_reasoning = "\n\n" + openrouter_reasoning
                            enter_buffer = ""
                        elif enter_buffer:
                            openrouter_reasoning = enter_buffer + openrouter_reasoning
                            enter_buffer = ""
                        openrouter_reasoning = openrouter_reasoning.replace(
                            "\\n", "\n"
                        )
                        sse_string = await generate_sse_response(
                            timestamp,
                            payload["model"],
                            reasoning_content=openrouter_reasoning,
                        )
                        yield sse_string
                        return
                    if no_stream_content and not has_send_thinking:
                        sse_string = await generate_sse_response(
                            safe_get(line, "created", default=None),
                            safe_get(line, "model", default=None),
                            content=no_stream_content,
                        )
                        yield sse_string
                        return
                    if no_stream_content:
                        del line["choices"][0]["message"]
                    json_line = await run_json_cpu(json.dumps, line)
                    yield "data: " + json_line.strip() + end_of_line
                finally:
                    raw_event = None
                    event_payload = None
                    line = None
                    event_type = None
                    content = None
                    end_think_reasoning_content = None
                    end_think_content = None
                    no_stream_content = None
                    openrouter_reasoning = None
                    openrouter_base64_image = None
                    image_data_url = None
                    azure_databricks_claude_summary_content = None
                    azure_databricks_claude_signature_content = None
                    reasoning_prefix = None
                    sse_string = None
                    json_line = None
                    await event_owner.aclose()

            events = _iter_openai_stream_events(response)
            async with aclosing(events):
                async for event_owner in events:
                    transformed = transform_event(event_owner)
                    async with aclosing(transformed):
                        async for output in transformed:
                            try:
                                yield output
                            finally:
                                output = None
                    if completed_normally:
                        break
            if not completed_normally and not semantic_terminal_seen:
                raise SSEProtocolError(
                    "OpenAI-compatible upstream ended without a terminal event"
                )
    finally:
        if response is not None and not completed_normally:
            await _force_close_response_httpcore_stream_chain_safely(
                response,
                label="gpt upstream response stream",
            )

    if input_tokens and output_tokens:
        sse_string = await generate_sse_response(timestamp, payload["model"], None, None, None, None, None, total_tokens=input_tokens + output_tokens, prompt_tokens=input_tokens, completion_tokens=output_tokens)
        yield sse_string

    yield "data: [DONE]" + end_of_line

async def fetch_azure_response_stream(client, url, headers, payload, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    is_thinking = False
    has_send_thinking = False
    ark_tag = False
    json_payload = await run_json_cpu(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_azure_response_stream")
        if error_message:
            yield error_message
            return

        sse_string = ""
        terminal_seen = False
        events = _iter_owned_sse_events(response.aiter_bytes())
        async with aclosing(events):
            async for event_owner in events:
                event_payload = None
                line = None
                no_stream_content = None
                content = None
                json_line = None
                try:
                    if event_owner.is_comment:
                        continue
                    event_payload = event_owner.payload
                    if event_payload == "[DONE]":
                        terminal_seen = True
                        break
                    if isinstance(event_payload, dict):
                        line = event_payload
                        if safe_get(
                            line,
                            "choices",
                            0,
                            "finish_reason",
                            default=None,
                        ) is not None:
                            terminal_seen = True
                        no_stream_content = safe_get(
                            line,
                            "choices",
                            0,
                            "message",
                            "content",
                            default="",
                        )
                        content = safe_get(
                            line,
                            "choices",
                            0,
                            "delta",
                            "content",
                            default="",
                        )

                        if "<think>" in content:
                            is_thinking = True
                            ark_tag = True
                            content = content.replace("<think>", "")
                        if "</think>" in content:
                            is_thinking = False
                            content = content.replace("</think>", "")
                            if not content:
                                continue
                        if is_thinking and ark_tag:
                            if not has_send_thinking:
                                content = content.replace("\n\n", "")
                            if content:
                                sse_string = await generate_sse_response(
                                    timestamp,
                                    payload["model"],
                                    reasoning_content=content,
                                )
                                yield sse_string
                                has_send_thinking = True
                            continue

                        if no_stream_content or content or sse_string:
                            input_tokens = _coerce_token_count(
                                safe_get(
                                    line,
                                    "usage",
                                    "prompt_tokens",
                                    default=0,
                                )
                            )
                            output_tokens = _coerce_token_count(
                                safe_get(
                                    line,
                                    "usage",
                                    "completion_tokens",
                                    default=0,
                                )
                            )
                            total_tokens = _coerce_token_count(
                                safe_get(
                                    line,
                                    "usage",
                                    "total_tokens",
                                    default=0,
                                )
                            )
                            sse_string = await generate_sse_response(
                                timestamp,
                                safe_get(line, "model", default=None),
                                content=no_stream_content or content,
                                total_tokens=total_tokens,
                                prompt_tokens=input_tokens,
                                completion_tokens=output_tokens,
                            )
                            yield sse_string
                        else:
                            json_line = await run_json_cpu(json.dumps, line)
                            yield "data: " + json_line.strip() + end_of_line
                    elif str(event_payload or "").strip():
                        raise SSEProtocolError(
                            "Azure upstream SSE data is not valid JSON"
                        )
                finally:
                    event_payload = None
                    line = None
                    no_stream_content = None
                    content = None
                    json_line = None
                    await event_owner.aclose()
        if not terminal_seen:
            raise SSEProtocolError("Azure upstream ended without a terminal event")
    yield "data: [DONE]" + end_of_line

async def fetch_cloudflare_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await run_json_cpu(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_cloudflare_response_stream")
        if error_message:
            yield error_message
            return

        terminal_seen = False
        events = _iter_owned_sse_events(response.aiter_bytes())
        async with aclosing(events):
            async for event_owner in events:
                event_payload = None
                resp = None
                message = None
                sse_string = None
                try:
                    if event_owner.is_comment:
                        continue
                    event_payload = event_owner.payload
                    if event_payload == "[DONE]":
                        terminal_seen = True
                        break
                    if isinstance(event_payload, dict):
                        resp = event_payload
                        message = resp.get("response")
                        if message:
                            sse_string = await generate_sse_response(
                                timestamp,
                                model,
                                content=message,
                            )
                            yield sse_string
                        if resp.get("done") is True or resp.get("event") in {
                            "done",
                            "completed",
                        }:
                            terminal_seen = True
                            break
                    elif str(event_payload or "").strip():
                        raise SSEProtocolError(
                            "Cloudflare upstream SSE data is not valid JSON"
                        )
                finally:
                    event_payload = None
                    resp = None
                    message = None
                    sse_string = None
                    await event_owner.aclose()
        if not terminal_seen:
            raise SSEProtocolError(
                "Cloudflare upstream ended without a terminal event"
            )
    yield "data: [DONE]" + end_of_line

async def fetch_cohere_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await run_json_cpu(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_cohere_response_stream")
        if error_message:
            yield error_message
            return

        terminal_seen = False
        lines = iter_lines(response.aiter_bytes())
        async with aclosing(lines):
            async for line in lines:
                if not line.strip():
                    line = None
                    continue
                owner = None
                resp = None
                message = None
                sse_string = None
                try:
                    owner = await parse_owned_json_value(line)
                    resp = owner.value
                    if not isinstance(resp, dict):
                        raise SSEProtocolError(
                            "Cohere upstream frame must be a JSON object"
                        )
                    if resp.get("is_finished"):
                        terminal_seen = True
                        break
                    if resp.get("event_type") == "text-generation":
                        message = resp.get("text")
                        sse_string = await generate_sse_response(
                            timestamp,
                            model,
                            content=message,
                        )
                        yield sse_string
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise SSEProtocolError(
                        "Cohere upstream frame is not valid JSON"
                    ) from exc
                finally:
                    resp = None
                    message = None
                    sse_string = None
                    line = None
                    if owner is not None:
                        await owner.aclose()
        if not terminal_seen:
            raise SSEProtocolError("Cohere upstream ended without is_finished")
    yield "data: [DONE]" + end_of_line

async def fetch_claude_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await run_json_cpu(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_claude_response_stream")
        if error_message:
            yield error_message
            return
        input_tokens = 0
        cache_read_input_tokens = 0
        terminal_seen = False
        events = _iter_owned_sse_events(response.aiter_bytes())
        async with aclosing(events):
            async for event_owner in events:
                event_name = None
                event_payload = None
                resp = None
                text = None
                function_call_name = None
                tools_id = None
                thinking_content = None
                function_call_content = None
                sse_string = None
                try:
                    if event_owner.is_comment:
                        continue
                    event_name = event_owner.event_name
                    event_payload = event_owner.payload
                    if event_payload == "[DONE]":
                        terminal_seen = True
                        break
                    if isinstance(event_payload, dict):
                        resp = event_payload
                        if (
                            event_name == "message_stop"
                            or resp.get("type") == "message_stop"
                        ):
                            terminal_seen = True

                        input_tokens = _coerce_token_count(
                            input_tokens
                            or safe_get(
                                resp,
                                "message",
                                "usage",
                                "input_tokens",
                                default=0,
                            )
                            or safe_get(resp, "usage", "input_tokens", default=0)
                        )
                        cache_read_input_tokens = _coerce_token_count(
                            cache_read_input_tokens
                            or safe_get(
                                resp,
                                "message",
                                "usage",
                                "cache_read_input_tokens",
                                default=0,
                            )
                            or safe_get(
                                resp,
                                "usage",
                                "cache_read_input_tokens",
                                default=0,
                            )
                        )
                        output_tokens = _coerce_token_count(
                            safe_get(
                                resp,
                                "usage",
                                "output_tokens",
                                default=0,
                            )
                        )
                        if output_tokens:
                            thinking_tokens = _coerce_token_count(
                                safe_get(
                                    resp,
                                    "usage",
                                    "output_tokens_details",
                                    "thinking_tokens",
                                    default=0,
                                )
                            )
                            total_tokens = input_tokens + output_tokens
                            sse_string = await generate_sse_response(
                                timestamp,
                                model,
                                None,
                                None,
                                None,
                                None,
                                None,
                                total_tokens,
                                input_tokens,
                                output_tokens,
                                cached_tokens=cache_read_input_tokens,
                                reasoning_tokens=thinking_tokens,
                            )
                            yield sse_string
                            terminal_seen = True
                            break

                        text = safe_get(resp, "delta", "text", default="")
                        if text:
                            sse_string = await generate_sse_response(
                                timestamp,
                                model,
                                text,
                            )
                            yield sse_string
                            continue

                        function_call_name = safe_get(
                            resp,
                            "content_block",
                            "name",
                            default=None,
                        )
                        tools_id = safe_get(
                            resp,
                            "content_block",
                            "id",
                            default=None,
                        )
                        if tools_id and function_call_name:
                            sse_string = await generate_sse_response(
                                timestamp,
                                model,
                                None,
                                tools_id,
                                function_call_name,
                                None,
                            )
                            yield sse_string

                        thinking_content = safe_get(
                            resp,
                            "delta",
                            "thinking",
                            default="",
                        )
                        if thinking_content:
                            sse_string = await generate_sse_response(
                                timestamp,
                                model,
                                reasoning_content=thinking_content,
                            )
                            yield sse_string

                        function_call_content = safe_get(
                            resp,
                            "delta",
                            "partial_json",
                            default="",
                        )
                        if function_call_content:
                            sse_string = await generate_sse_response(
                                timestamp,
                                model,
                                None,
                                None,
                                None,
                                function_call_content,
                            )
                            yield sse_string
                        if terminal_seen:
                            break
                    elif str(event_payload or "").strip():
                        raise SSEProtocolError(
                            "Claude upstream SSE data is not valid JSON"
                        )
                finally:
                    event_name = None
                    event_payload = None
                    resp = None
                    text = None
                    function_call_name = None
                    tools_id = None
                    thinking_content = None
                    function_call_content = None
                    sse_string = None
                    await event_owner.aclose()

        if not terminal_seen:
            raise SSEProtocolError("Claude upstream ended without message_stop")

    yield "data: [DONE]" + end_of_line

async def _iter_aws_eventstream_payloads(response: Any):
    iterator_factory = getattr(response, "aiter_raw", None)
    if not callable(iterator_factory):
        iterator_factory = response.aiter_bytes
    pending = bytearray()
    pending_budget = StreamParserRetainedLease()
    try:
        async for raw_chunk in iterator_factory():
            chunk_size = len(raw_chunk)
            pending_budget.grow(chunk_size)
            try:
                pending.extend(raw_chunk)
            except BaseException:
                pending_budget.shrink(chunk_size)
                raise
            raw_chunk = None
            cursor = 0
            while len(pending) - cursor >= 12:
                total_length = int.from_bytes(
                    pending[cursor : cursor + 4],
                    "big",
                )
                headers_length = int.from_bytes(
                    pending[cursor + 4 : cursor + 8],
                    "big",
                )
                if (
                    total_length < 16
                    or total_length > DEFAULT_MAX_EVENT_BYTES
                    or headers_length > total_length - 16
                ):
                    raise SSEProtocolError("invalid AWS event-stream frame length")
                if len(pending) - cursor < total_length:
                    break
                frame_view = memoryview(pending)[cursor : cursor + total_length]
                expected_prelude_crc = int.from_bytes(frame_view[8:12], "big")
                prelude_valid = (
                    zlib.crc32(frame_view[:8]) & 0xFFFFFFFF
                ) == expected_prelude_crc
                expected_message_crc = int.from_bytes(frame_view[-4:], "big")
                message_valid = (
                    zlib.crc32(frame_view[:-4]) & 0xFFFFFFFF
                ) == expected_message_crc
                del frame_view
                if not prelude_valid:
                    raise SSEProtocolError("invalid AWS event-stream prelude CRC")
                if not message_valid:
                    raise SSEProtocolError("invalid AWS event-stream message CRC")
                request_lease = get_request_admission_lease()
                frame_reservation = (
                    await request_lease.reserve_temporary_response_bytes(
                        total_length * 3
                    )
                    if request_lease is not None
                    else None
                )
                owner = None
                frame = None
                payload_bytes = None
                try:
                    frame = bytes(pending[cursor : cursor + total_length])
                    payload_start = 12 + headers_length
                    payload_bytes = frame[payload_start:-4]
                    cursor += total_length
                    if not payload_bytes:
                        continue
                    owner = await parse_owned_json_value(payload_bytes)
                    yield owner
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise SSEProtocolError(
                        "AWS event-stream payload is not valid JSON"
                    ) from exc
                finally:
                    payload_bytes = None
                    frame = None
                    try:
                        if owner is not None:
                            await owner.aclose()
                    finally:
                        if frame_reservation is not None:
                            await frame_reservation.release()
            if cursor:
                del pending[:cursor]
                pending_budget.shrink(cursor)
            if len(pending) > DEFAULT_MAX_EVENT_BYTES:
                raise SSEBufferOverflowError(
                    buffer_name="AWS event-stream frame",
                    limit_bytes=DEFAULT_MAX_EVENT_BYTES,
                    observed_bytes=len(pending),
                )
        if pending:
            raise SSEProtocolError(
                "AWS event-stream ended with an incomplete frame"
            )
    finally:
        pending.clear()
        pending_budget.release()


async def fetch_aws_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await run_json_cpu(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_aws_response_stream")
        if error_message:
            yield error_message
            return

        terminal_seen = False
        events = _iter_aws_eventstream_payloads(response)
        async with aclosing(events):
            async for event_owner in events:
                chunk_data = None
                encoded = None
                decoded_bytes = None
                payload_chunk = None
                nested_owner = None
                nested_reservation = None
                text = None
                usage = None
                sse_string = None
                try:
                    chunk_data = event_owner.value
                    if not isinstance(chunk_data, dict):
                        raise SSEProtocolError(
                            "AWS event-stream payload must be a JSON object"
                        )
                    if "bytes" not in chunk_data:
                        continue
                    encoded = chunk_data["bytes"]
                    if not isinstance(encoded, str):
                        raise SSEProtocolError(
                            "AWS event-stream bytes field must be base64 text"
                        )
                    decoded_limit = ((len(encoded) + 3) // 4) * 3
                    request_lease = get_request_admission_lease()
                    nested_reservation = (
                        await request_lease.reserve_temporary_response_bytes(
                            decoded_limit * 2
                        )
                        if request_lease is not None
                        else None
                    )
                    try:
                        decoded_bytes = await run_json_cpu(
                            base64.b64decode,
                            encoded,
                            validate=True,
                        )
                    except (ValueError, binascii.Error) as exc:
                        raise SSEProtocolError(
                            "AWS event-stream bytes field is not valid base64"
                        ) from exc
                    nested_owner = await parse_owned_json_value(decoded_bytes)
                    payload_chunk = nested_owner.value
                    if not isinstance(payload_chunk, dict):
                        raise SSEProtocolError(
                            "AWS decoded event payload must be a JSON object"
                        )

                    text = safe_get(payload_chunk, "delta", "text", default="")
                    if text:
                        sse_string = await generate_sse_response(
                            timestamp,
                            model,
                            text,
                            None,
                            None,
                        )
                        yield sse_string

                    usage = safe_get(
                        payload_chunk,
                        "amazon-bedrock-invocationMetrics",
                        default="",
                    )
                    if usage:
                        input_tokens = _coerce_token_count(
                            usage.get("inputTokenCount", 0)
                        )
                        output_tokens = _coerce_token_count(
                            usage.get("outputTokenCount", 0)
                        )
                        total_tokens = input_tokens + output_tokens
                        sse_string = await generate_sse_response(
                            timestamp,
                            model,
                            None,
                            None,
                            None,
                            None,
                            None,
                            total_tokens,
                            input_tokens,
                            output_tokens,
                        )
                        yield sse_string
                        terminal_seen = True
                        break
                finally:
                    chunk_data = None
                    encoded = None
                    decoded_bytes = None
                    payload_chunk = None
                    text = None
                    usage = None
                    sse_string = None
                    try:
                        if nested_owner is not None:
                            await nested_owner.aclose()
                    finally:
                        try:
                            if nested_reservation is not None:
                                await nested_reservation.release()
                        finally:
                            await event_owner.aclose()

        if not terminal_seen:
            raise SSEProtocolError(
                "AWS event-stream ended without invocation metrics"
            )

    yield "data: [DONE]" + end_of_line

def _pop_multipart_payload(payload):
    if not isinstance(payload, dict) or "__multipart_files__" not in payload:
        return None
    files = payload.pop("__multipart_files__", None) or []
    data = payload.pop("__multipart_data__", None) or []
    return data, files

def _quote_multipart_header_value(value) -> str:
    text = str(value or "")
    return (
        text
        .replace("\\", "\\\\")
        .replace('"', "%22")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )

def _read_multipart_file_content(content) -> bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, bytearray):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8")
    if hasattr(content, "seek"):
        try:
            content.seek(0)
        except Exception:
            pass
    if hasattr(content, "read"):
        value = content.read()
        if isinstance(value, str):
            return value.encode("utf-8")
        return bytes(value or b"")
    return bytes(content or b"")

def _build_multipart_content(headers: dict, data: list, files: list) -> tuple[dict, bytes]:
    boundary = f"----uniapi-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for key, value in data:
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{_quote_multipart_header_value(key)}"\r\n\r\n'.encode("utf-8")
        )
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")

    for key, file_value in files:
        filename = "upload"
        content_type = "application/octet-stream"
        content = file_value
        if isinstance(file_value, (tuple, list)):
            if len(file_value) >= 1 and file_value[0]:
                filename = str(file_value[0])
            if len(file_value) >= 2:
                content = file_value[1]
            if len(file_value) >= 3 and file_value[2]:
                content_type = str(file_value[2])

        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{_quote_multipart_header_value(key)}"; '
                f'filename="{_quote_multipart_header_value(filename)}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(_read_multipart_file_content(content))
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    request_headers = dict(headers or {})
    for key in list(request_headers.keys()):
        if str(key).lower() == "content-type":
            request_headers.pop(key, None)
    request_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    return request_headers, b"".join(chunks)


async def _fetch_search_response(client, url, headers, payload, timeout, response_headers_sink: ResponseHeadersSink | None = None):
    content_type = None
    for key in ("Content-Type", "content-type"):
        if key in (headers or {}):
            content_type = headers.get(key)
            break
    if content_type and is_json_media_type(content_type):
        response = await client.post(url, headers=headers, json=payload, timeout=timeout)
    else:
        response = await client.get(url, headers=headers, params=payload, timeout=timeout)
    _capture_response_headers(response_headers_sink, getattr(response, "headers", None))
    return response


async def _fetch_post_response(
    client,
    url,
    headers,
    payload,
    timeout,
    response_headers_sink: ResponseHeadersSink | None = None,
    *,
    binary_response: bool = False,
):
    post = client.post
    if binary_response:
        # The managed client still enforces the same 64 MiB wire limit and
        # shared response-byte budget, but does not charge an 8x JSON object
        # expansion for trusted TTS bytes.  Raw/dummy clients keep their
        # ordinary post method for compatibility in isolated provider tests.
        post = getattr(client, "post_buffered_binary", post)
    multipart_payload = _pop_multipart_payload(payload)
    if multipart_payload is not None:
        data, files = multipart_payload
        multipart_headers, multipart_content = _build_multipart_content(headers, data, files)
        response = await post(url, headers=multipart_headers, content=multipart_content, timeout=timeout)
        _capture_response_headers(response_headers_sink, getattr(response, "headers", None))
        return response
    if payload.get("file"):
        file = payload.pop("file")
        response = await post(url, headers=headers, data=payload, files={"file": file}, timeout=timeout)
        _capture_response_headers(response_headers_sink, getattr(response, "headers", None))
        return response
    json_payload = await run_json_cpu(json.dumps, payload)
    response = await post(url, headers=headers, content=json_payload, timeout=timeout)
    _capture_response_headers(response_headers_sink, getattr(response, "headers", None))
    return response


async def _yield_search_response(response, url):
    try:
        response_json = await run_json_cpu(json.loads, response.content)
    except Exception:
        response_json = {"text": response.text}
    normalized = normalize_search_response(url, response_json)
    yield await run_json_cpu(json.dumps, normalized, ensure_ascii=False)


async def _yield_responses_api_chat_completion(response, model):
    response_bytes = await response.aread()
    response_json = await run_json_cpu(json.loads, response_bytes)
    usage = _responses_usage_to_chat_completion_usage(safe_get(response_json, "usage", default=None))
    prompt_tokens = _coerce_token_count(
        safe_get(usage, "prompt_tokens", default=0)
    )
    completion_tokens = _coerce_token_count(
        safe_get(usage, "completion_tokens", default=0)
    )
    total_tokens = _coerce_token_count(
        safe_get(usage, "total_tokens", default=0)
    )
    content, reasoning_content = _responses_output_to_text(response_json)
    timestamp = safe_get(response_json, "created", default=int(datetime.timestamp(datetime.now())))
    yield await generate_no_stream_response(
        timestamp,
        model,
        content=content or None,
        role="assistant",
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_content=reasoning_content or None,
        cached_tokens=_coerce_token_count(safe_get(usage, "prompt_tokens_details", "cached_tokens", default=0)),
        prompt_audio_tokens=_coerce_token_count(safe_get(usage, "prompt_tokens_details", "audio_tokens", default=0)),
        reasoning_tokens=_coerce_token_count(safe_get(usage, "completion_tokens_details", "reasoning_tokens", default=0)),
        completion_audio_tokens=_coerce_token_count(safe_get(usage, "completion_tokens_details", "audio_tokens", default=0)),
        accepted_prediction_tokens=_coerce_token_count(safe_get(usage, "completion_tokens_details", "accepted_prediction_tokens", default=0)),
        rejected_prediction_tokens=_coerce_token_count(safe_get(usage, "completion_tokens_details", "rejected_prediction_tokens", default=0)),
    )


def _parse_provider_json_response(response_json):
    if isinstance(response_json, str):
        raise SSEProtocolError(
            "provider returned nested JSON text instead of an object or array"
        )
    if isinstance(response_json, list):
        return response_json
    if isinstance(response_json, dict):
        return [response_json]
    raise SSEProtocolError(
        f"provider returned unsupported JSON type {type(response_json).__name__}"
    )


async def _yield_gemini_chat_completion(response, model):
    response_bytes = await response.aread()
    response_json = await run_json_cpu(json.loads, response_bytes)
    parsed_data = _parse_provider_json_response(response_json)
    parts_list = safe_get(parsed_data, 0, "candidates", 0, "content", "parts", default=[])
    normalized_parts = await normalize_gemini_parts(
        parts_list if isinstance(parts_list, list) else []
    )
    usage_metadata = safe_get(parsed_data, -1, "usageMetadata")
    prompt_tokens = _coerce_token_count(safe_get(usage_metadata, "promptTokenCount", default=0))
    candidates_tokens = _coerce_token_count(safe_get(usage_metadata, "candidatesTokenCount", default=0))
    total_tokens = _coerce_token_count(safe_get(usage_metadata, "totalTokenCount", default=0))
    cached_tokens = _coerce_token_count(safe_get(usage_metadata, "cachedContentTokenCount", default=0))
    reasoning_tokens = _coerce_token_count(safe_get(usage_metadata, "thoughtsTokenCount", default=0))
    completion_tokens = candidates_tokens + reasoning_tokens
    role = safe_get(parsed_data, -1, "candidates", 0, "content", "role")
    if role == "model":
        role = "assistant"
    else:
        logger.error("Unknown role: %s, parsed_data: %s", role, parsed_data)
        role = "assistant"

    content = normalized_parts.content
    audio_obj = build_openai_audio_object(normalized_parts.audio_wav_base64, transcript=content or None)
    if audio_obj and not content:
        content = None
    yield await generate_no_stream_response(
        int(datetime.timestamp(datetime.now())),
        model,
        content=content,
        tools_id=normalized_parts.function_call.call_id,
        function_call_name=normalized_parts.function_call.name,
        function_call_content=normalized_parts.function_call.arguments,
        role=role,
        total_tokens=total_tokens or (prompt_tokens + completion_tokens),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_content=normalized_parts.reasoning_content,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        image_base64=normalized_parts.image_base64 or "",
        audio=audio_obj,
    )


async def _yield_claude_chat_completion(response, model):
    response_bytes = await response.aread()
    response_json = await run_json_cpu(json.loads, response_bytes)
    prompt_tokens = _coerce_token_count(
        safe_get(response_json, "usage", "input_tokens", default=0)
    )
    output_tokens = _coerce_token_count(
        safe_get(response_json, "usage", "output_tokens", default=0)
    )
    thinking_tokens = _coerce_token_count(
        safe_get(
            response_json,
            "usage",
            "output_tokens_details",
            "thinking_tokens",
            default=0,
        )
    )
    yield await generate_no_stream_response(
        int(datetime.timestamp(datetime.now())),
        model,
        content=safe_get(response_json, "content", 0, "text"),
        tools_id=safe_get(response_json, "content", 1, "id", default=None),
        function_call_name=safe_get(response_json, "content", 1, "name", default=None),
        function_call_content=safe_get(response_json, "content", 1, "input", default=None),
        role=safe_get(response_json, "role"),
        total_tokens=prompt_tokens + output_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        cached_tokens=_coerce_token_count(safe_get(response_json, "usage", "cache_read_input_tokens", default=0)),
        reasoning_tokens=thinking_tokens,
    )


async def _yield_doubao_translation_chat_completion(response, model):
    response_bytes = await response.aread()
    response_json = await run_json_cpu(json.loads, response_bytes)
    if isinstance(response_json, dict) and response_json.get("error"):
        yield {
            "error": "doubao-translation upstream error",
            "status_code": 502,
            "details": response_json.get("error"),
        }
        return

    output_text = None
    for out in safe_get(response_json, "output", default=[]) or []:
        if not isinstance(out, dict) or out.get("type") != "message" or out.get("role") != "assistant":
            continue
        for content_item in (out.get("content") or []):
            if isinstance(content_item, dict) and content_item.get("type") == "output_text" and content_item.get("text"):
                output_text = content_item.get("text")
                break
        if output_text:
            break

    if not output_text:
        yield {
            "error": "doubao-translation empty output",
            "status_code": 502,
            "details": response_json,
        }
        return

    usage_obj = safe_get(response_json, "usage", default={}) or {}
    prompt_tokens = _coerce_token_count(
        usage_obj.get("input_tokens") or usage_obj.get("prompt_tokens")
    )
    completion_tokens = _coerce_token_count(
        usage_obj.get("output_tokens") or usage_obj.get("completion_tokens")
    )
    yield await generate_no_stream_response(
        int(datetime.timestamp(datetime.now())),
        model,
        content=output_text,
        role="assistant",
        total_tokens=_coerce_token_count(usage_obj.get("total_tokens"))
        or (prompt_tokens + completion_tokens),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def _yield_azure_response(response):
    response_bytes = await response.aread()
    response_json = await run_json_cpu(json.loads, response_bytes)
    if "choices" in response_json:
        for choice in response_json["choices"]:
            if "content_filter_results" in choice:
                del choice["content_filter_results"]
    if "prompt_filter_results" in response_json:
        del response_json["prompt_filter_results"]
    yield response_json


async def _yield_dashscope_multimodal_response(response):
    response_bytes = await response.aread()
    response_json = await run_json_cpu(json.loads, response_bytes)
    yield safe_get(response_json, "output", "choices", 0, "message", "content", 0, default=None)


async def _yield_embedding_response(response, model):
    response_bytes = await response.aread()
    response_json = await run_json_cpu(json.loads, response_bytes)
    content = safe_get(response_json, "embedding", "values", default=[])
    yield {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": content,
                "index": 0,
            }
        ],
        "model": model,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }


async def fetch_response(client, url, headers, payload, engine, model, timeout=200, response_headers_sink: ResponseHeadersSink | None = None):
    if engine == "search":
        response = await _fetch_search_response(client, url, headers, payload, timeout, response_headers_sink=response_headers_sink)
        error_message = await check_response(response, "fetch_response")
        if error_message:
            yield error_message
            return
        async for item in _yield_search_response(response, url):
            yield item
        return

    response = await _fetch_post_response(
        client,
        url,
        headers,
        payload,
        timeout,
        response_headers_sink=response_headers_sink,
        binary_response=engine == "tts",
    )
    error_message = await check_response(response, "fetch_response")
    if error_message:
        yield error_message
        return

    if engine == "tts":
        yield response.read()

    elif engine in ("gpt", "codex") and _is_responses_api_call(url, payload):
        async for item in _yield_responses_api_chat_completion(response, model):
            yield item

    elif engine == "gemini" or engine == "vertex-gemini" or engine == "aws":
        async for item in _yield_gemini_chat_completion(response, model):
            yield item

    elif engine == "claude" or engine == "vertex-claude":
        async for item in _yield_claude_chat_completion(response, model):
            yield item

    elif engine == "azure":
        async for item in _yield_azure_response(response):
            yield item

    elif "dashscope.aliyuncs.com" in url and "multimodal-generation" in url:
        async for item in _yield_dashscope_multimodal_response(response):
            yield item

    elif "embedContent" in url:
        async for item in _yield_embedding_response(response, model):
            yield item
    elif engine == "doubao-translation":
        async for item in _yield_doubao_translation_chat_completion(response, model):
            yield item
    else:
        response_bytes = await response.aread()
        response_json = await run_json_cpu(json.loads, response_bytes)
        yield response_json

async def fetch_doubao_translation_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await run_json_cpu(json.dumps, payload)

    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_doubao_translation_response_stream")
        if error_message:
            yield error_message
            return

        sse_parser = IncrementalSSEParser()
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        async def raw_event_batches():
            try:
                async for chunk in response.aiter_bytes():
                    batch = sse_parser.feed(chunk)
                    chunk = None
                    try:
                        yield batch
                    finally:
                        batch = None
                batch = sse_parser.finish()
                try:
                    yield batch
                finally:
                    batch = None
            finally:
                sse_parser.discard()

        batches = raw_event_batches()
        async with aclosing(batches):
            async for raw_events in batches:
                events = _iter_owned_raw_events(raw_events)
                async with aclosing(events):
                    async for event_owner in events:
                        raw_event = None
                        event_name = None
                        event_data = None
                        delta_text = None
                        usage_obj = None
                        try:
                            raw_event = event_owner.raw_event
                            event_name = event_owner.event_name
                            event_data = event_owner.payload
                            if not raw_event.strip():
                                continue
                            if not event_name and not event_data:
                                continue
                            if event_data == "[DONE]":
                                yield "data: [DONE]" + end_of_line
                                return

                            if not isinstance(event_data, dict):
                                continue

                            if event_name == "response.output_text.delta":
                                delta_text = safe_get(
                                    event_data,
                                    "delta",
                                    default=None,
                                )
                                if not delta_text:
                                    continue
                                yield await generate_sse_response(
                                    timestamp,
                                    model,
                                    content=delta_text,
                                )
                                continue

                            if event_name == "response.completed":
                                usage_obj = (
                                    safe_get(
                                        event_data,
                                        "response",
                                        "usage",
                                        default={},
                                    )
                                    or {}
                                )
                                prompt_tokens = _coerce_token_count(
                                    usage_obj.get("input_tokens")
                                )
                                completion_tokens = _coerce_token_count(
                                    usage_obj.get("output_tokens")
                                )
                                total_tokens = _coerce_token_count(
                                    usage_obj.get("total_tokens")
                                ) or (
                                    prompt_tokens + completion_tokens
                                )

                                yield await generate_sse_response(
                                    timestamp,
                                    model,
                                    stop="stop",
                                )
                                if total_tokens:
                                    yield await generate_sse_response(
                                        timestamp,
                                        model,
                                        total_tokens=total_tokens,
                                        prompt_tokens=prompt_tokens,
                                        completion_tokens=completion_tokens,
                                    )
                                yield "data: [DONE]" + end_of_line
                                return
                        finally:
                            raw_event = None
                            event_name = None
                            event_data = None
                            delta_text = None
                            usage_obj = None
                            await event_owner.aclose()
                raw_events = None

        raise SSEProtocolError(
            "Doubao translation upstream ended without response.completed or [DONE]"
        )

async def fetch_dalle_response_stream(client, url, headers, payload, timeout=200):
    multipart_payload = _pop_multipart_payload(payload)
    if multipart_payload is not None:
        data, files = multipart_payload
        headers, multipart_content = _build_multipart_content(headers, data, files)
        stream_kwargs = {"content": multipart_content}
    else:
        json_payload = await run_json_cpu(json.dumps, payload)
        stream_kwargs = {"content": json_payload}

    async with client.stream("POST", url, headers=headers, timeout=timeout, **stream_kwargs) as response:
        error_message = await check_response(response, "fetch_dalle_response_stream")
        if error_message:
            yield error_message
            return
        content_type = str(response.headers.get("content-type") or "").lower()
        if "text/event-stream" not in content_type:
            # A normal Images API response is one finite JSON document whose
            # EOF is its protocol terminal.  Transport chunks are not JSON
            # frames, so collect under the shared weighted response budget and
            # yield exactly one validated document to the legacy first-item
            # checker.
            request_lease = get_request_admission_lease()
            limited = await read_limited_response_body(
                response,
                max_bytes=upstream_success_body_max_bytes(),
                reserve_bytes=(
                    request_lease.reserve_response_bytes
                    if request_lease is not None
                    else None
                ),
                reservation_multiplier=upstream_json_memory_reservation_multiplier(),
            )
            if limited.truncated:
                raise SSEBufferOverflowError(
                    buffer_name="DALL-E JSON response",
                    limit_bytes=upstream_success_body_max_bytes(),
                    observed_bytes=limited.observed_bytes_at_least,
                )
            owner = None
            parsed = None
            try:
                owner = await parse_owned_json_value(limited.body)
                parsed = owner.value
                if not isinstance(parsed, dict):
                    raise SSEProtocolError(
                        "DALL-E upstream response must be a JSON object"
                    )
            except json.JSONDecodeError as exc:
                raise SSEProtocolError(
                    "DALL-E upstream returned an incomplete JSON document"
                ) from exc
            finally:
                parsed = None
                if owner is not None:
                    await owner.aclose()
            yield limited.body
            return

        terminal_seen = False
        events = _iter_owned_sse_events(response.aiter_bytes())
        async with aclosing(events):
            async for event_owner in events:
                raw_event = None
                event_payload = None
                payload_type = None
                normalized_event = None
                try:
                    raw_event = event_owner.raw_event
                    if event_owner.is_comment:
                        yield raw_event + end_of_line
                        continue
                    event_payload = event_owner.payload
                    payload_type = (
                        str(event_payload.get("type") or "").strip().lower()
                        if isinstance(event_payload, dict)
                        else ""
                    )
                    normalized_event = event_owner.event_name.strip().lower()
                    if event_payload == "[DONE]":
                        terminal_seen = True
                    elif normalized_event in {
                        "done",
                        "completed",
                        "response.completed",
                        "image_generation.completed",
                    } or payload_type in {
                        "done",
                        "completed",
                        "response.completed",
                        "image_generation.completed",
                    }:
                        terminal_seen = True
                    elif normalized_event in {
                        "error",
                        "response.failed",
                        "response.incomplete",
                        "image_generation.failed",
                    } or payload_type in {
                        "error",
                        "response.failed",
                        "response.incomplete",
                        "image_generation.failed",
                    }:
                        raise SSEProtocolError(
                            "DALL-E SSE upstream emitted failure terminal"
                        )
                    yield raw_event + end_of_line
                    if terminal_seen:
                        return
                finally:
                    raw_event = None
                    event_payload = None
                    payload_type = None
                    normalized_event = None
                    await event_owner.aclose()
        if not terminal_seen:
            raise SSEProtocolError(
                "DALL-E SSE upstream ended without a terminal event"
            )

async def fetch_response_stream(client, url, headers, payload, engine, model, timeout=200, response_headers_sink: ResponseHeadersSink | None = None):
    if engine == "gemini" or engine == "vertex-gemini":
        stream = fetch_gemini_response_stream(client, url, headers, payload, model, timeout)
    elif engine == "claude" or engine == "vertex-claude":
        stream = fetch_claude_response_stream(client, url, headers, payload, model, timeout)
    elif engine == "aws":
        stream = fetch_aws_response_stream(client, url, headers, payload, model, timeout)
    elif engine in ("gpt", "codex", "openrouter", "azure-databricks"):
        stream = fetch_gpt_response_stream(client, url, headers, payload, timeout, response_headers_sink=response_headers_sink)
    elif engine == "azure":
        stream = fetch_azure_response_stream(client, url, headers, payload, timeout)
    elif engine == "cloudflare":
        stream = fetch_cloudflare_response_stream(client, url, headers, payload, model, timeout)
    elif engine == "cohere":
        stream = fetch_cohere_response_stream(client, url, headers, payload, model, timeout)
    elif engine == "doubao-translation":
        stream = fetch_doubao_translation_response_stream(client, url, headers, payload, model, timeout)
    elif engine == "dalle":
        stream = fetch_dalle_response_stream(client, url, headers, payload, timeout)
    else:
        raise ValueError("Unknown response")

    async for chunk in _yield_from_stream(stream, label=f"{engine} upstream response stream"):
        yield chunk
