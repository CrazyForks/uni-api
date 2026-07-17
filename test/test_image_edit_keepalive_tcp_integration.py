import asyncio
import contextlib
import socket

import httpx
import uvicorn
from starlette.responses import JSONResponse

import main


class _ProviderKeys:
    def get_items_count(self):
        return 1

    async def is_all_rate_limited(self, _model):
        return False

    async def next(self, _model):
        return "provider-key"

    async def set_cooling(self, _item, _cooling_time):
        raise AssertionError("completed image stream must not cool the key")


class _ChannelManager:
    cooldown_period = 300

    async def exclude_model(self, _provider, _model):
        raise AssertionError("completed image stream must not exclude the channel")


class _ImageStreamResponse:
    status_code = 200
    headers = {"content-type": "text/event-stream"}
    is_stream_consumed = False

    def __init__(self):
        self.closed = 0

    async def aiter_bytes(self):
        yield (
            b"event: image_edit.completed\n"
            b'data: {"type":"image_edit.completed","b64_json":"QUJD",'
            b'"usage":{"images":1}}\n\n'
        )

    async def aclose(self):
        self.closed += 1


class _ImageStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, _exc_type, _exc, _tb):
        await self.response.aclose()


class _ImageClient:
    def __init__(self, response):
        self.response = response

    def stream(self, *_args, **_kwargs):
        return _ImageStreamContext(self.response)


class _ImageClientManager:
    def __init__(self, response):
        self.response = response

    @contextlib.asynccontextmanager
    async def get_client(self, *_args, **_kwargs):
        yield _ImageClient(self.response)


def test_image_edit_completed_then_responses_reuses_real_tcp_connection(
    monkeypatch,
):
    image_response = _ImageStreamResponse()

    async def fake_resolver(
        request_model_name,
        _config,
        _api_index,
        _algorithm,
    ):
        return [
            {
                "provider": "oaix-test",
                "engine": "dalle",
                "model": [request_model_name],
                "_model_dict_cache": {
                    request_model_name: request_model_name,
                },
                "base_url": "https://oaix.test/v1/images/edits",
                "api": ["provider-key"],
                "preferences": {"api_key_cooldown_period": 300},
            }
        ]

    async def fake_responses_api_response(**_kwargs):
        return JSONResponse(
            status_code=200,
            content={"id": "resp_keepalive", "status": "completed"},
        )

    monkeypatch.setattr(main, "DISABLE_DATABASE", True)
    monkeypatch.setattr(main, "get_right_order_providers", fake_resolver)
    monkeypatch.setattr(
        main,
        "responses_api_response",
        fake_responses_api_response,
    )
    monkeypatch.setattr(main, "model_handler", main.ModelRequestHandler())
    config = {
        "api_keys": [
            {
                "api": "sk-local-test",
                "model": ["gpt-image-2", "gpt-5.6-sol"],
                "preferences": {"AUTO_RETRY": True},
            }
        ],
        "preferences": {"rate_limit": "999999/min"},
    }
    monkeypatch.setattr(main.app.state, "config", config, raising=False)
    monkeypatch.setattr(main.app.state, "runtime_config", None, raising=False)
    monkeypatch.setattr(
        main.app.state,
        "runtime_config_source_id",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        main.app.state,
        "api_list",
        ["sk-local-test"],
        raising=False,
    )
    monkeypatch.setattr(
        main.app.state,
        "models_list",
        {"sk-local-test": ["gpt-image-2", "gpt-5.6-sol"]},
        raising=False,
    )
    monkeypatch.setattr(
        main.app.state,
        "api_keys_db",
        [{"api": "sk-local-test"}],
        raising=False,
    )
    monkeypatch.setattr(
        main.app.state,
        "user_api_keys_rate_limit",
        main._build_user_api_keys_rate_limit(config, ["sk-local-test"]),
        raising=False,
    )
    monkeypatch.setattr(
        main.app.state,
        "provider_timeouts",
        {"global": {"default": 30}},
        raising=False,
    )
    monkeypatch.setattr(
        main.app.state,
        "keepalive_interval",
        {"global": {"default": 99999}},
        raising=False,
    )
    monkeypatch.setattr(
        main.app.state,
        "channel_manager",
        _ChannelManager(),
        raising=False,
    )
    monkeypatch.setattr(
        main.app.state,
        "client_manager",
        _ImageClientManager(image_response),
        raising=False,
    )
    monkeypatch.setattr(
        main.app.state,
        "error_triggers",
        [],
        raising=False,
    )
    monkeypatch.setitem(
        main.provider_api_circular_list,
        "oaix-test",
        _ProviderKeys(),
    )

    async def scenario():
        protocol, protocol_stats = main.build_bounded_h11_protocol(
            connection_limit=16,
            header_timeout_seconds=2,
        )
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                main.app,
                http=protocol,
                lifespan="off",
                limit_concurrency=None,
                log_level="critical",
                timeout_keep_alive=10,
            )
        )
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            while not server.started:
                await asyncio.sleep(0.001)
            headers = {
                "Authorization": "Bearer sk-local-test",
                "Content-Type": "application/json",
            }
            limits = httpx.Limits(
                max_connections=1,
                max_keepalive_connections=1,
                keepalive_expiry=30,
            )
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                limits=limits,
                timeout=5,
            ) as client:
                image_edit = await client.post(
                    "/v1/images/edits",
                    headers=headers,
                    json={
                        "model": "gpt-image-2",
                        "prompt": "edit",
                        "images": [
                            {
                                "image_url": (
                                    "data:image/png;base64,"
                                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
                                )
                            }
                        ],
                        "stream": True,
                        "response_format": "b64_json",
                    },
                )
                assert image_edit.status_code == 200
                assert "event: image_edit.completed" in image_edit.text
                assert '"type":"image_edit.completed"' in image_edit.text
                assert "event: error" not in image_edit.text

                responses = await client.post(
                    "/v1/responses",
                    headers=headers,
                    json={
                        "model": "gpt-5.6-sol",
                        "input": "same connection",
                        "stream": False,
                    },
                )
                assert responses.status_code == 200
                assert responses.json() == {
                    "id": "resp_keepalive",
                    "status": "completed",
                }

            assert protocol_stats.accepted_connections == 1
            assert protocol_stats.rejected_connections == 0
        finally:
            server.should_exit = True
            await asyncio.wait_for(server_task, timeout=5)
            listener.close()

    asyncio.run(scenario())
    assert image_response.closed >= 1
