from __future__ import annotations

from typing import Any


MAX_STREAM_ERROR_TEXT_BYTES = 4 * 1024
_TRUNCATED_MARKER = " [truncated]"


def _truncate_text(value: str, limit_bytes: int) -> str:
    limit_bytes = max(1, int(limit_bytes))
    marker_bytes = _TRUNCATED_MARKER.encode("utf-8")
    # At most four UTF-8 bytes are emitted per Python character.  Slice before
    # encoding so a multi-megabyte exception string never creates an equally
    # large temporary bytes object.
    candidate = value[:limit_bytes]
    encoded = candidate.encode("utf-8", errors="replace")
    truncated = len(value) > len(candidate) or len(encoded) > limit_bytes
    if not truncated:
        return encoded.decode("utf-8", errors="replace")
    if limit_bytes <= len(marker_bytes):
        return "." * limit_bytes
    content_limit = max(0, limit_bytes - len(marker_bytes))
    content = encoded[:content_limit].decode("utf-8", errors="ignore")
    return content + _TRUNCATED_MARKER


def bounded_stream_error_text(
    value: Any,
    *,
    limit_bytes: int = MAX_STREAM_ERROR_TEXT_BYTES,
) -> str:
    """Create a finite diagnostic without invoking attacker-sized reprs."""

    if isinstance(value, BaseException):
        error_type = _truncate_text(type(value).__name__, 256)
        detail: Any = value.args[0] if value.args else ""
        if isinstance(detail, bytes):
            detail = detail[:limit_bytes].decode("utf-8", errors="replace")
        elif not isinstance(detail, str):
            if isinstance(detail, (int, float, bool, type(None))):
                detail = str(detail)
            else:
                detail = type(detail).__name__
        if not detail:
            return _truncate_text(error_type, limit_bytes)
        return _truncate_text(detail, limit_bytes)

    if isinstance(value, bytes):
        sliced = value[:limit_bytes]
        text = sliced.decode("utf-8", errors="replace")
        if len(value) > len(sliced):
            return _truncate_text(text + _TRUNCATED_MARKER, limit_bytes)
        return _truncate_text(text, limit_bytes)
    if isinstance(value, str):
        return _truncate_text(value, limit_bytes)
    if isinstance(value, (int, float, bool, type(None))):
        return _truncate_text(str(value), limit_bytes)
    return _truncate_text(type(value).__name__, limit_bytes)
