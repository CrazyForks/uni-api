from __future__ import annotations

import errno as errno_module
import hashlib
import json
import re
from typing import Any


_MAX_CAUSE_DEPTH = 8
_MAX_TEXT_BYTES = 768
_MAX_CHAIN_JSON_BYTES = 4096

_AUTH_RE = re.compile(r"(?i)\b(?:bearer|basic)\s+[^\s,;]+")
_API_KEY_RE = re.compile(
    r"(?i)\b(?:sk-[a-z0-9_-]{8,}|AIza[a-z0-9_-]{12,})"
)
_NAMED_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|token|key)\s*[=:]\s*)([^\s,;&]+)"
)
_URL_USERINFO_RE = re.compile(r"(?i)(https?://)[^/@\s]+@")
_URL_QUERY_RE = re.compile(r"(?i)(https?://[^?\s]+)\?[^\s]*")


def _bounded_text(value: Any, *, max_bytes: int = _MAX_TEXT_BYTES) -> str | None:
    if value is None:
        return None
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        text = f"<{type(value).__module__}.{type(value).__name__}>"
    text = text.strip()
    if not text:
        return None
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def redact_exception_text(value: Any, *, max_bytes: int = _MAX_TEXT_BYTES) -> str | None:
    """Return bounded diagnostic text without credentials or URL queries."""

    text = _bounded_text(value, max_bytes=max_bytes * 4)
    if not text:
        return None
    text = _AUTH_RE.sub("[redacted-auth]", text)
    text = _API_KEY_RE.sub("[redacted-api-key]", text)
    text = _NAMED_SECRET_RE.sub(r"\1[redacted]", text)
    text = _URL_USERINFO_RE.sub(r"\1[redacted]@", text)
    text = _URL_QUERY_RE.sub(r"\1?[redacted]", text)
    return _bounded_text(text, max_bytes=max_bytes)


def _safe_repr(exc: BaseException) -> str | None:
    try:
        rendered = repr(exc)
    except Exception:
        rendered = f"<{type(exc).__module__}.{type(exc).__name__}>"
    return redact_exception_text(rendered)


def _errno(exc: BaseException) -> tuple[int | None, str | None]:
    value = getattr(exc, "errno", None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None, None
    return value, errno_module.errorcode.get(value)


def _next_exception(exc: BaseException) -> tuple[str | None, BaseException | None]:
    if exc.__cause__ is not None:
        return "cause", exc.__cause__
    if exc.__context__ is not None and not exc.__suppress_context__:
        return "context", exc.__context__
    return None, None


def _bounded_chain_json(rows: list[dict[str, Any]]) -> str:
    accepted: list[dict[str, Any]] = []
    for row in rows:
        candidate = json.dumps(
            [*accepted, row],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(candidate.encode("utf-8")) > _MAX_CHAIN_JSON_BYTES:
            break
        accepted.append(row)
    return json.dumps(
        accepted,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def exception_chain(exc: BaseException) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    relation = "raised"
    while (
        current is not None
        and id(current) not in seen
        and len(rows) < _MAX_CAUSE_DEPTH
    ):
        seen.add(id(current))
        message = redact_exception_text(current)
        rendered_repr = _safe_repr(current)
        error_number, error_name = _errno(current)
        row: dict[str, Any] = {
            "relation": relation,
            "type": type(current).__name__,
            "module": type(current).__module__,
        }
        if message:
            row["message"] = message
            row["message_sha256"] = hashlib.sha256(
                message.encode("utf-8")
            ).hexdigest()
        if rendered_repr:
            row["repr"] = rendered_repr
        if error_number is not None:
            row["errno"] = error_number
        if error_name:
            row["errno_name"] = error_name
        rows.append(row)
        relation, current = _next_exception(current)
    return rows, current is not None


def classify_protocol_error(chain: list[dict[str, Any]]) -> str:
    """Classify known protocol failures using bounded, redacted cause facts."""

    rendered = "\n".join(
        " ".join(
            str(row.get(key) or "")
            for key in ("type", "module", "message", "repr")
        )
        for row in chain
    ).lower()
    if "too little data for declared content-length" in rendered:
        return "CONTENT_LENGTH_MISMATCH"
    if "too much data for declared content-length" in rendered:
        return "CONTENT_LENGTH_MISMATCH"
    if "toomanystreamserror" in rendered or "max outbound streams" in rendered:
        return "TOO_MANY_STREAMS"
    if (
        "connectioninputs.send_headers" in rendered
        and "connectionstate.closed" in rendered
    ):
        return "SEND_HEADERS_ON_CLOSED"
    if (
        "connectioninputs.send_body" in rendered
        and "connectionstate.closed" in rendered
    ):
        return "SEND_BODY_ON_CLOSED"
    if (
        "connectionterminated" in rendered
        or "goaway" in rendered
        or "go_away" in rendered
    ):
        return "REMOTE_GOAWAY"
    if "remoteprotocolerror" in rendered:
        return "REMOTE_PROTOCOL_ERROR"
    if "localprotocolerror" in rendered or "protocolerror" in rendered:
        return "LOCAL_PROTOCOL_UNCLASSIFIED"
    return "NOT_PROTOCOL_ERROR"


def exception_diagnostics(exc: BaseException) -> dict[str, Any]:
    chain, truncated = exception_chain(exc)
    first = chain[0] if chain else {}
    return {
        "exception_type": type(exc).__name__,
        "exception_module": type(exc).__module__,
        "exception_message": first.get("message"),
        "exception_repr": first.get("repr"),
        "exception_chain": chain,
        "exception_chain_json": _bounded_chain_json(chain),
        "exception_chain_depth": len(chain),
        "exception_chain_truncated": truncated,
        "protocol_error_reason": classify_protocol_error(chain),
    }
