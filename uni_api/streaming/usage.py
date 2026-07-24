from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


_MAX_USAGE_COUNTER = (1 << 63) - 1
_USAGE_COUNTER_FIELDS = (
    "prompt_tokens",
    "input_tokens",
    "completion_tokens",
    "output_tokens",
    "total_tokens",
)


@dataclass(frozen=True, slots=True)
class StreamUsageSnapshot:
    """Bounded usage facts detached from a parsed streaming payload.

    The upstream Responses parser already owns and validates the complete JSON
    graph.  Carry only scalar facts through the stream queue so the downstream
    response does not need to frame and materialize the same SSE event again.
    """

    counters_seen: bool
    input_known: bool = False
    output_known: bool = False
    total_known: bool = False
    values_valid: bool | None = None
    alias_consistent: bool | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    parse_error: str | None = None
    complete: bool = False


def parse_usage_count(value: Any) -> tuple[int, bool]:
    if isinstance(value, bool):
        return 0, False
    if isinstance(value, int):
        return value, 0 <= value <= _MAX_USAGE_COUNTER
    if isinstance(value, float):
        if (
            math.isfinite(value)
            and 0 <= value <= _MAX_USAGE_COUNTER
            and value.is_integer()
        ):
            return int(value), True
        return 0, False
    if isinstance(value, str):
        rendered = value.strip()
        if rendered and len(rendered) <= 20 and rendered.isdigit():
            try:
                parsed = int(rendered)
            except (ValueError, OverflowError):
                return 0, False
            if parsed <= _MAX_USAGE_COUNTER:
                return parsed, True
    return 0, False


def stream_usage_snapshot(usage_obj: Any) -> StreamUsageSnapshot | None:
    if not isinstance(usage_obj, dict):
        return None
    if not any(key in usage_obj for key in _USAGE_COUNTER_FIELDS):
        return StreamUsageSnapshot(counters_seen=False)

    input_known = "prompt_tokens" in usage_obj or "input_tokens" in usage_obj
    output_known = (
        "completion_tokens" in usage_obj or "output_tokens" in usage_obj
    )
    explicit_total_known = "total_tokens" in usage_obj
    total_known = explicit_total_known or (input_known and output_known)

    prompt_raw = (
        usage_obj.get("prompt_tokens")
        if "prompt_tokens" in usage_obj
        else usage_obj.get("input_tokens")
    )
    completion_raw = (
        usage_obj.get("completion_tokens")
        if "completion_tokens" in usage_obj
        else usage_obj.get("output_tokens")
    )
    prompt_tokens, prompt_valid = parse_usage_count(prompt_raw)
    completion_tokens, completion_valid = parse_usage_count(completion_raw)
    if explicit_total_known:
        total_tokens, total_valid = parse_usage_count(
            usage_obj.get("total_tokens")
        )
    elif input_known and output_known and prompt_valid and completion_valid:
        total_tokens = prompt_tokens + completion_tokens
        total_valid = True
    else:
        total_tokens = 0
        total_valid = False

    observed_values_valid = all(
        parse_usage_count(usage_obj[field])[1]
        for field in _USAGE_COUNTER_FIELDS
        if field in usage_obj
    )
    values_valid = (
        observed_values_valid
        and (not input_known or prompt_valid)
        and (not output_known or completion_valid)
        and (not total_known or total_valid)
    )
    alias_consistent = True
    for first, second in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
    ):
        if first in usage_obj and second in usage_obj:
            first_value, first_valid = parse_usage_count(usage_obj[first])
            second_value, second_valid = parse_usage_count(usage_obj[second])
            alias_consistent = alias_consistent and (
                first_valid
                and second_valid
                and first_value == second_value
            )

    parse_error = None
    if not values_valid:
        parse_error = "invalid_usage_counter"
    elif not alias_consistent:
        parse_error = "conflicting_usage_aliases"
    complete = bool(
        values_valid
        and alias_consistent
        and input_known
        and output_known
        and total_known
    )
    return StreamUsageSnapshot(
        counters_seen=True,
        input_known=input_known,
        output_known=output_known,
        total_known=total_known,
        values_valid=values_valid,
        alias_consistent=alias_consistent,
        prompt_tokens=prompt_tokens if input_known and prompt_valid else None,
        completion_tokens=(
            completion_tokens if output_known and completion_valid else None
        ),
        total_tokens=total_tokens if total_known and total_valid else None,
        parse_error=parse_error,
        complete=complete,
    )


def stream_usage_snapshot_from_payload(
    payload: Any,
) -> StreamUsageSnapshot | None:
    if not isinstance(payload, dict):
        return None
    usage_obj = payload.get("usage")
    response = payload.get("response")
    if not isinstance(usage_obj, dict) and isinstance(response, dict):
        usage_obj = response.get("usage")
    message = payload.get("message")
    if not isinstance(usage_obj, dict) and isinstance(message, dict):
        usage_obj = message.get("usage")
    return stream_usage_snapshot(usage_obj)
