from __future__ import annotations


def normalized_media_type(value: object) -> str:
    return str(value or "").split(";", 1)[0].strip().lower()


def is_json_media_type(value: object) -> bool:
    media_type = normalized_media_type(value)
    return media_type == "application/json" or media_type.endswith("+json")
