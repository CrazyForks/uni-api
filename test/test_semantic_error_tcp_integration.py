import asyncio
import contextlib
import json
import socket

import httpx
import uvicorn

import main
from uni_api.upstream.responses_errors import responses_failure_error


class _ProviderKeys:
    def __init__(self):
        self.cooling_calls = []

    def get_items_count(self):
        return 1

    async def is_all_rate_limited(self, _model):
        return False

    async def next(self, _model):
        return "provider-key"

    async def set_cooling(self, item, cooling_time):
        self.cooling_calls.append((item, cooling_time))


class _ChannelManager:
    cooldown_period = 300

    def __init__(self):
        self.excluded = []

    async def exclude_model(self, provider, model):
        self.excluded.append((provider, model))


class _DummyClientManager:
    @contextlib.asynccontextmanager
    async def get_client(self, *_args, **_kwargs):
        yield object()


def test_semantic_context_error_over_real_local_http_connection(monkeypatch):
    provider_keys = _ProviderKeys()
    channel_manager = _ChannelManager()
    channel_results = []
    fetch_calls = []

    async def fake_resolver(
        request_model_name,
        _config,
        _api_index,
        _algorithm,
    ):
        return [
            {
                "provider": "provider-a",
                "engine": "gpt",
                "model": [request_model_name],
                "_model_dict_cache": {
                    request_model_name: request_model_name,
                },
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["provider-key"],
                "preferences": {"api_key_cooldown_period": 300},
            }
        ]

    def semantic_error():
        error = responses_failure_error(
            {
                "type": "error",
                "error": {
                    "message": (
                        "Your input exceeds the context window of this model."
                    ),
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                    "param": "input",
                },
            },
            event_type="error",
            wire_status_code=200,
        )
        assert error is not None
        return error

    async def fake_fetch_response_stream(
        _client,
        _url,
        _headers,
        payload,
        _engine,
        _model,
        _timeout,
        response_headers_sink=None,
    ):
        if response_headers_sink is not None:
            response_headers_sink({"content-type": "text/event-stream"})
        serialized_payload = json.dumps(payload, sort_keys=True)
        content = "precommit" if "precommit" in serialized_payload else "postcommit"
        fetch_calls.append(content)
        if content == "precommit":
            raise semantic_error()
        yield (
            'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        )
        raise semantic_error()

    monkeypatch.setattr(main, "DISABLE_DATABASE", True)
    monkeypatch.setattr(main, "get_right_order_providers", fake_resolver)
    monkeypatch.setattr(
        main,
        "fetch_response_stream",
        fake_fetch_response_stream,
    )
    monkeypatch.setattr(main, "model_handler", main.ModelRequestHandler())
    monkeypatch.setattr(
        main,
        "_schedule_channel_stats_bounded",
        lambda *_args, success, **_kwargs: channel_results.append(success),
    )
    config = {
        "api_keys": [
            {
                "api": "sk-local-test",
                "model": ["gpt-5.5"],
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
        {"sk-local-test": ["gpt-5.5"]},
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
        channel_manager,
        raising=False,
    )
    monkeypatch.setattr(
        main.app.state,
        "client_manager",
        _DummyClientManager(),
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
        "provider-a",
        provider_keys,
    )

    async def scenario():
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                main.app,
                lifespan="off",
                log_level="critical",
            )
        )
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            while not server.started:
                await asyncio.sleep(0.001)
            headers = {
                "Authorization": "Bearer sk-local-test",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
            ) as client:
                precommit = await client.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "gpt-5.5",
                        "messages": [
                            {"role": "user", "content": "precommit"}
                        ],
                        "stream": True,
                    },
                )
                assert precommit.status_code == 400
                assert precommit.json()["error"]["code"] == (
                    "context_length_exceeded"
                )

                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "gpt-5.5",
                        "messages": [
                            {"role": "user", "content": "postcommit"}
                        ],
                        "stream": True,
                    },
                ) as postcommit:
                    body = (await postcommit.aread()).decode("utf-8")
                    assert postcommit.status_code == 200
                    assert "partial" in body
                    assert body.count("event: error") == 1
                    assert '"status_code": 400' in body
                    assert '"code": "context_length_exceeded"' in body
                    assert "Streaming error" not in body
                    assert "data: [DONE]" not in body
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=5)
            listener.close()

    asyncio.run(scenario())

    assert fetch_calls == ["precommit", "postcommit"]
    assert channel_results == []
    assert channel_manager.excluded == []
    assert provider_keys.cooling_calls == []
