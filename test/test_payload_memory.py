import asyncio
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from uni_api.admission import (
    RequestAdmissionController,
    bind_request_admission_lease,
    reset_request_admission_lease,
)
import uni_api.providers.payloads as payloads_module
from uni_api.providers.payloads import (
    _parse_tool_call_arguments,
    get_upload_certificate,
    get_whisper_payload,
    upload_file_to_oss,
)


def _tool_call(arguments: str):
    return SimpleNamespace(
        function=SimpleNamespace(name="bounded_tool", arguments=arguments)
    )


def test_nested_tool_argument_graph_is_charged_for_request_lifetime():
    async def scenario():
        arguments = json.dumps([{} for _ in range(1000)], separators=(",", ":"))
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024 * 1024,
            body_budget_bytes=16 * 1024 * 1024,
            max_response_bytes=16 * 1024 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)
        try:
            parsed = await _parse_tool_call_arguments(_tool_call(arguments))
            assert len(parsed) == 1000
            assert (
                controller.snapshot()["reserved_response_bytes"]
                > len(arguments) * 20
            )
        finally:
            reset_request_admission_lease(token)
            await lease.release()

        assert controller.snapshot()["reserved_response_bytes"] == 0

    asyncio.run(scenario())


def test_invalid_nested_tool_arguments_are_a_client_error():
    with pytest.raises(HTTPException) as rejected:
        asyncio.run(_parse_tool_call_arguments(_tool_call("{invalid")))

    assert rejected.value.status_code == 400
    assert "bounded_tool" in str(rejected.value.detail)


def test_legal_but_excessively_nested_tool_arguments_return_413():
    arguments = "[" * 129 + "0" + "]" * 129

    with pytest.raises(HTTPException) as rejected:
        asyncio.run(_parse_tool_call_arguments(_tool_call(arguments)))

    assert rejected.value.status_code == 413
    assert "complexity limit" in str(rejected.value.detail)


def test_dashscope_certificate_request_uses_explicit_30_second_httpx_timeout():
    observed_timeout = None

    async def handler(request):
        nonlocal observed_timeout
        observed_timeout = request.extensions.get("timeout")
        return httpx.Response(
            200,
            json={
                "data": {
                    "upload_host": "https://oss.example",
                    "upload_dir": "dir",
                }
            },
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            return await get_upload_certificate(client, "secret", "whisper-test")

    certificate = asyncio.run(scenario())

    assert certificate["upload_host"] == "https://oss.example"
    assert observed_timeout == {
        "connect": 30.0,
        "read": 30.0,
        "write": 30.0,
        "pool": 30.0,
    }


def test_dashscope_oss_upload_preserves_one_hour_httpx_timeout():
    observed_timeout = None

    async def handler(request):
        nonlocal observed_timeout
        observed_timeout = request.extensions.get("timeout")
        await request.aread()
        return httpx.Response(200)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            return await upload_file_to_oss(
                client,
                {
                    "upload_host": "https://oss.example",
                    "upload_dir": "dir",
                },
                ("audio.wav", io.BytesIO(b"audio"), "audio/wav"),
            )

    oss_url = asyncio.run(scenario())

    assert oss_url == "oss://dir/audio.wav"
    assert observed_timeout == {
        "connect": 3600.0,
        "read": 3600.0,
        "write": 3600.0,
        "pool": 3600.0,
    }


@pytest.mark.parametrize("certificate_ok", [True, False])
def test_dashscope_whisper_auxiliary_client_is_closed_on_every_path(
    monkeypatch,
    certificate_ok,
):
    closed = 0

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc_info):
            nonlocal closed
            closed += 1

    async def certificate(_client, _api_key, _model):
        if not certificate_ok:
            return None
        return {
            "upload_host": "https://oss.example",
            "upload_dir": "dir",
        }

    async def upload(_client, _certificate, _file):
        return "oss://dir/audio.wav"

    monkeypatch.setattr(
        payloads_module.httpx,
        "AsyncClient",
        lambda **_kwargs: FakeClient(),
    )
    monkeypatch.setattr(payloads_module, "get_upload_certificate", certificate)
    monkeypatch.setattr(payloads_module, "upload_file_to_oss", upload)
    request = SimpleNamespace(
        model="whisper-test",
        file=("audio.wav", object(), "audio/wav"),
        prompt=None,
        response_format=None,
        temperature=None,
        language=None,
        timestamp_granularities=None,
    )
    provider = {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": ["whisper-test"],
    }

    result = asyncio.run(
        get_whisper_payload(
            request,
            "whisper",
            provider,
            api_key="secret",
        )
    )

    assert closed == 1
    assert (result is not None) is certificate_ok
