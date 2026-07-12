from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import struct
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable

from core.utils import (
    safe_get,
)
from uni_api.providers.base64_chunks import (
    ChunkedBase64Error,
    encode_byte_stream_base64_chunks,
    inspect_base64_chunks,
    iter_strict_base64_decoded_chunks,
    predicted_decoded_base64_bytes,
)


GEMINI_THOUGHT_SIGNATURE_MAX_BYTES = 64 * 1024
_GEMINI_SIGNATURE_CACHE_MAX_ITEMS = 100
_GEMINI_SIGNATURE_CACHE_MAX_BYTES = 4 * 1024 * 1024
_GEMINI_IMAGE_KEY_MAX_CHARS = 16 * 1024 * 1024
_GEMINI_IMAGE_KEY_MAX_DECODED_BYTES = (
    (_GEMINI_IMAGE_KEY_MAX_CHARS + 3) // 4
) * 3
_HASH_CHUNK_CHARS = 64 * 1024
_GEMINI_AUDIO_BASE64_MAX_CHARS = 8 * 1024 * 1024
_GEMINI_AUDIO_PCM_MAX_BYTES = 6 * 1024 * 1024


class GeminiThoughtSignatureTooLarge(ValueError):
    status_code = 502
    reason = "gemini_thought_signature_too_large"


def _bounded_utf8_size(value: str, limit_bytes: int) -> int:
    if len(value) > limit_bytes:
        return limit_bytes + 1
    observed = 0
    for offset in range(0, len(value), _HASH_CHUNK_CHARS):
        observed += len(value[offset : offset + _HASH_CHUNK_CHARS].encode("utf-8"))
        if observed > limit_bytes:
            return observed
    return observed


def _image_cache_key(data_base64: str) -> str | None:
    """Hash decoded image bytes so equivalent base64 spellings share a key."""

    if (
        not isinstance(data_base64, str)
        or not data_base64
        or len(data_base64) > _GEMINI_IMAGE_KEY_MAX_CHARS
    ):
        return None
    try:
        padded = data_base64 + "=" * ((-len(data_base64)) % 4)
        decoded = base64.b64decode(padded, validate=True)
    except (ValueError, binascii.Error, UnicodeEncodeError):
        return None
    return hashlib.sha256(decoded).hexdigest()


async def _image_cache_key_async(data_base64: str) -> str | None:
    try:
        inspection = await inspect_base64_chunks(
            data_base64,
            max_encoded_chars=_GEMINI_IMAGE_KEY_MAX_CHARS,
            max_decoded_bytes=_GEMINI_IMAGE_KEY_MAX_DECODED_BYTES,
        )
    except ChunkedBase64Error:
        return None
    return inspection.digest_hex


class _ByteBoundedSignatureCache:
    def __init__(self) -> None:
        self._data: OrderedDict[str, tuple[str, int]] = OrderedDict()
        self._bytes = 0
        self._rejected = 0
        self._lock = threading.Lock()

    def put(self, key: str, signature: str) -> bool:
        size = _bounded_utf8_size(
            signature,
            GEMINI_THOUGHT_SIGNATURE_MAX_BYTES,
        )
        if size > GEMINI_THOUGHT_SIGNATURE_MAX_BYTES:
            with self._lock:
                self._rejected += 1
            return False
        with self._lock:
            previous = self._data.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            self._data[key] = (signature, size)
            self._bytes += size
            while (
                len(self._data) > _GEMINI_SIGNATURE_CACHE_MAX_ITEMS
                or self._bytes > _GEMINI_SIGNATURE_CACHE_MAX_BYTES
            ):
                _old_key, (_old_value, old_size) = self._data.popitem(last=False)
                self._bytes -= old_size
        return True

    def get(self, key: str) -> str | None:
        with self._lock:
            value = self._data.get(key)
            return value[0] if value is not None else None

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "items": len(self._data),
                "bytes": self._bytes,
                "max_items": _GEMINI_SIGNATURE_CACHE_MAX_ITEMS,
                "max_bytes": _GEMINI_SIGNATURE_CACHE_MAX_BYTES,
                "rejected": self._rejected,
            }

    def reject(self) -> None:
        with self._lock:
            self._rejected += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0


_GEMINI_IMAGE_THOUGHT_SIGNATURE_CACHE = _ByteBoundedSignatureCache()


def cache_put_gemini_image_thought_signature(
    inline_data_base64: str,
    thought_signature: str,
) -> bool:
    if not isinstance(thought_signature, str) or not thought_signature:
        return False
    key = _image_cache_key(inline_data_base64)
    if key is None:
        return False
    return _GEMINI_IMAGE_THOUGHT_SIGNATURE_CACHE.put(key, thought_signature)


def cache_get_gemini_image_thought_signature(
    inline_data_base64: str,
) -> str | None:
    key = _image_cache_key(inline_data_base64)
    if key is None:
        return None
    return _GEMINI_IMAGE_THOUGHT_SIGNATURE_CACHE.get(key)


async def cache_put_gemini_image_thought_signature_async(
    inline_data_base64: str,
    thought_signature: str,
) -> bool:
    if not isinstance(thought_signature, str) or not thought_signature:
        return False
    if (
        _bounded_utf8_size(
            thought_signature,
            GEMINI_THOUGHT_SIGNATURE_MAX_BYTES,
        )
        > GEMINI_THOUGHT_SIGNATURE_MAX_BYTES
    ):
        # Reject before hashing an attacker-sized image key.
        _GEMINI_IMAGE_THOUGHT_SIGNATURE_CACHE.reject()
        return False
    key = await _image_cache_key_async(inline_data_base64)
    if key is None:
        return False
    return _GEMINI_IMAGE_THOUGHT_SIGNATURE_CACHE.put(key, thought_signature)


async def cache_get_gemini_image_thought_signature_async(
    inline_data_base64: str,
) -> str | None:
    key = await _image_cache_key_async(inline_data_base64)
    if key is None:
        return None
    return _GEMINI_IMAGE_THOUGHT_SIGNATURE_CACHE.get(key)


def cache_get_gemini_image_thought_signature_by_key(key: str) -> str | None:
    """Look up a key already derived while strictly validating image bytes."""

    if not isinstance(key, str) or len(key) != hashlib.sha256().digest_size * 2:
        return None
    return _GEMINI_IMAGE_THOUGHT_SIGNATURE_CACHE.get(key)


def gemini_thought_signature_cache_snapshot() -> dict[str, int]:
    return _GEMINI_IMAGE_THOUGHT_SIGNATURE_CACHE.snapshot()


def clear_gemini_thought_signature_cache() -> None:
    _GEMINI_IMAGE_THOUGHT_SIGNATURE_CACHE.clear()


@dataclass(frozen=True)
class GeminiInlinePart:
    mime_type: str
    data_base64: str
    thought_signature: str | None = None


@dataclass(frozen=True)
class GeminiFunctionCall:
    name: str | None
    arguments: Any = None
    call_id: str | None = None

    @property
    def arguments_json(self) -> str | None:
        if self.arguments in (None, ""):
            return None
        return json.dumps(self.arguments, ensure_ascii=False)


@dataclass(frozen=True)
class GeminiPartsNormalization:
    content: str
    reasoning_content: str
    image_base64: str | None
    audio_wav_base64: str | None
    is_thinking: bool
    function_call: GeminiFunctionCall


def _part_inline_data(part: dict[str, Any]) -> GeminiInlinePart | None:
    mime_type = safe_get(part, "inlineData", "mimeType", default=None)
    if not mime_type:
        mime_type = safe_get(part, "inline_data", "mime_type", default=None)
    data_base64 = safe_get(part, "inlineData", "data", default=None)
    if not data_base64:
        data_base64 = safe_get(part, "inline_data", "data", default=None)
    if (
        not isinstance(mime_type, str)
        or not isinstance(data_base64, str)
        or not mime_type
        or len(mime_type) > 256
        or not data_base64
    ):
        return None

    thought_signature = safe_get(part, "thoughtSignature", default=None)
    if not thought_signature:
        thought_signature = safe_get(part, "thought_signature", default=None)
    return GeminiInlinePart(
        mime_type=mime_type,
        data_base64=data_base64,
        thought_signature=(
            thought_signature
            if isinstance(thought_signature, str) and thought_signature
            else None
        ),
    )


async def _cache_image_thought_signature(inline: GeminiInlinePart) -> None:
    if inline.thought_signature:
        await cache_put_gemini_image_thought_signature_async(
            inline.data_base64,
            inline.thought_signature,
        )


def _function_call_id_from_thought_signature(thought_signature: Any) -> str | None:
    if not thought_signature:
        return None
    if not isinstance(thought_signature, str):
        return None
    if (
        _bounded_utf8_size(
            thought_signature,
            GEMINI_THOUGHT_SIGNATURE_MAX_BYTES,
        )
        > GEMINI_THOUGHT_SIGNATURE_MAX_BYTES
    ):
        raise GeminiThoughtSignatureTooLarge(
            "Gemini thought signature exceeded the local response limit"
        )
    encoded = base64.urlsafe_b64encode(
        thought_signature.encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"call_{encoded}.{uuid.uuid4().hex}"


def _pcm_wav_header(*, pcm_bytes: int, sample_rate: int) -> bytes:
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + pcm_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        pcm_bytes,
    )


async def _gemini_audio_inline_data_to_wav_base64(
    mime_type: str,
    data_base64: str,
) -> str | None:
    """Convert Gemini PCM to WAV with bounded, interleaved base64 work."""

    if not mime_type or len(mime_type) > 256 or not data_base64:
        return None
    normalized_mime = mime_type.lower()
    if (
        not normalized_mime.startswith("audio/")
        or "l16" not in normalized_mime
        or "pcm" not in normalized_mime
    ):
        return None
    try:
        rate_match = re.search(r"rate=(\d{1,10})(?!\d)", normalized_mime)
        sample_rate = int(rate_match.group(1)) if rate_match else 24000
        if sample_rate <= 0 or sample_rate > 0x7FFFFFFF // 2:
            return None
        pcm_bytes = predicted_decoded_base64_bytes(data_base64)
        if pcm_bytes > _GEMINI_AUDIO_PCM_MAX_BYTES:
            return None
        header = _pcm_wav_header(
            pcm_bytes=pcm_bytes,
            sample_rate=sample_rate,
        )

        async def pcm_chunks():
            async for decoded_chunk in iter_strict_base64_decoded_chunks(
                data_base64,
                max_encoded_chars=_GEMINI_AUDIO_BASE64_MAX_CHARS,
                max_decoded_bytes=_GEMINI_AUDIO_PCM_MAX_BYTES,
            ):
                yield decoded_chunk

        return await encode_byte_stream_base64_chunks(
            pcm_chunks(),
            prefix=header,
        )
    except (ChunkedBase64Error, OverflowError, struct.error, ValueError):
        return None


async def normalize_gemini_parts(parts: Iterable[Any]) -> GeminiPartsNormalization:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    image_base64: str | None = None
    audio_wav_base64: str | None = None
    is_thinking = False
    function_call = GeminiFunctionCall(name=None)

    for part in parts or []:
        if not isinstance(part, dict):
            continue

        text = safe_get(part, "text", default=None)
        part_is_thinking = bool(safe_get(part, "thought", default=False))
        if part_is_thinking:
            is_thinking = True
        if text:
            if part_is_thinking:
                reasoning_parts.append(str(text))
            else:
                content_parts.append(str(text))

        inline = _part_inline_data(part)
        if inline is not None:
            mime_type = inline.mime_type.lower()
            if mime_type.startswith("image/"):
                image_base64 = inline.data_base64
                await _cache_image_thought_signature(inline)
            elif mime_type.startswith("audio/"):
                converted_audio = await _gemini_audio_inline_data_to_wav_base64(
                    inline.mime_type,
                    inline.data_base64,
                )
                audio_wav_base64 = converted_audio or audio_wav_base64

        function_name = safe_get(part, "functionCall", "name", default=None)
        if function_name and not function_call.name:
            thought_signature = safe_get(part, "thoughtSignature", default=None)
            if not thought_signature:
                thought_signature = safe_get(part, "thought_signature", default=None)
            function_call = GeminiFunctionCall(
                name=str(function_name),
                arguments=safe_get(part, "functionCall", "args", default=None),
                call_id=_function_call_id_from_thought_signature(thought_signature),
            )

    return GeminiPartsNormalization(
        content="".join(content_parts),
        reasoning_content="".join(reasoning_parts),
        image_base64=image_base64,
        audio_wav_base64=audio_wav_base64,
        is_thinking=is_thinking,
        function_call=function_call,
    )


def build_openai_audio_object(audio_wav_base64: str | None, *, transcript: str | None = None) -> dict | None:
    if not audio_wav_base64:
        return None
    return {
        "id": f"audio_{uuid.uuid4().hex[:24]}",
        "data": audio_wav_base64,
        "expires_at": None,
        "transcript": transcript or None,
        "format": "wav",
    }
