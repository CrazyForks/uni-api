import asyncio
import json

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from uni_api.api.media_parsers import parse_image_edit_request


def _request(content_type: str, payload: bytes) -> Request:
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/images/edits",
            "headers": [(b"content-type", content_type.encode("latin-1"))],
        },
        receive,
    )


def test_image_edit_rejects_application_jsonp_instead_of_parsing_it_as_json():
    request = _request("application/jsonp", b'{"prompt":"unsafe downgrade"}')

    with pytest.raises(HTTPException) as rejected:
        asyncio.run(parse_image_edit_request(request))

    assert rejected.value.status_code == 400
    assert "Unsupported Content-Type" in str(rejected.value.detail)


def test_image_edit_accepts_structured_json_suffix_media_type():
    request = _request(
        "application/problem+json; charset=utf-8",
        json.dumps({"prompt": "draw a bounded queue"}).encode(),
    )

    parsed = asyncio.run(parse_image_edit_request(request))

    assert parsed.prompt == "draw a bounded queue"
    assert parsed.request_type == "image"
