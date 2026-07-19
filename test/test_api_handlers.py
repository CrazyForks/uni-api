import hashlib
from pathlib import Path
from types import SimpleNamespace

from uni_api.api.admin import api_config_response, api_config_update_response
from uni_api.api.health import healthz_response, observability_runtime_response
from uni_api.api.models import codex_models_payload, list_models_payload
from uni_api.api.stats import (
    add_credits_response,
    api_keys_states_response,
    channel_key_rankings_response,
    token_usage_response,
)


class _RuntimeGauges:
    async def record_event_loop_lag(self):
        self.recorded = True

    def snapshot(self):
        return {"inflight_requests": 0}


class _ClientManager:
    def snapshot(self):
        return {"client_count": 1}


async def test_health_and_observability_handlers_are_pure_api_boundaries():
    assert await healthz_response("1.2.3") == {"status": "ok", "version": "1.2.3"}

    runtime = await observability_runtime_response(
        _RuntimeGauges(),
        _ClientManager(),
        idempotency_snapshot=lambda: {"mode": "memory-single-process"},
    )

    assert runtime["inflight_requests"] == 0
    assert runtime["upstream_http_clients"] == {"client_count": 1}
    assert runtime["idempotency"] == {"mode": "memory-single-process"}


def test_list_models_payload_uses_cache_before_fallback():
    calls = []

    def build_models(api_index, config, api_list, models_list):
        calls.append((api_index, config, api_list, models_list))
        return [{"id": "fallback"}]

    cached = list_models_payload(
        api_index=0,
        api_list=["sk-test"],
        model_response_cache={"sk-test": [{"id": "cached"}]},
        config={},
        models_list={},
        build_models=build_models,
    )

    fallback = list_models_payload(
        api_index=1,
        api_list=["sk-test", "sk-miss"],
        model_response_cache={},
        config={"api_keys": []},
        models_list={},
        build_models=build_models,
    )

    assert cached["data"] == [{"id": "cached"}]
    assert fallback["data"] == [{"id": "fallback"}]
    assert len(calls) == 1


def test_codex_models_payload_uses_pro_metadata_and_filters_unroutable_models():
    payload = codex_models_payload(
        api_index=0,
        api_list=["sk-test"],
        model_response_cache={
            "sk-test": [
                {"id": "gpt-5.4"},
                {"id": "gpt-5.5"},
                {"id": "gpt-5.6-sol"},
                {"id": "gpt-5.6-terra"},
                {"id": "gpt-5.6-luna"},
                {"id": "gpt-5.6-sol-max"},
                {"id": "gpt-5.5-fast"},
                {"id": "gpt-default"},
                {"id": "gpt-image-2"},
            ]
        },
        config={},
        models_list={},
        build_models=lambda *_: [],
    )

    models = {model["slug"]: model for model in payload["models"]}
    assert list(models) == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.6-sol-max",
        "gpt-5.5-fast",
        "gpt-default",
    ]
    assert "gpt-5.3-codex-spark" not in models
    assert "codex-auto-review" not in models
    assert "gpt-image-2" not in models

    sol = models["gpt-5.6-sol"]
    assert sol["default_reasoning_level"] == "low"
    assert [level["effort"] for level in sol["supported_reasoning_levels"]] == [
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultra",
    ]
    assert sol["context_window"] == 372000
    assert sol["max_context_window"] == 372000
    assert sol["use_responses_lite"] is True
    assert sol["tool_mode"] == "code_mode_only"
    assert sol["multi_agent_version"] == "v2"
    assert sol["base_instructions"]
    assert sol["model_messages"]["instructions_template"]

    luna = models["gpt-5.6-luna"]
    assert [level["effort"] for level in luna["supported_reasoning_levels"]][-1] == "max"
    assert luna["multi_agent_version"] == "v1"
    assert models["gpt-5.4"]["max_context_window"] == 1000000
    assert models["gpt-5.5"]["max_context_window"] == 272000
    assert models["gpt-5.6-sol-max"]["context_window"] == 372000
    assert [
        level["effort"]
        for level in models["gpt-5.6-sol-max"]["supported_reasoning_levels"]
    ][-2:] == ["max", "ultra"]
    assert models["gpt-5.5-fast"]["comp_hash"] == "2911"
    assert models["gpt-default"]["description"] == "Available through uni-api."
    assert models["gpt-default"]["priority"] >= 100


def test_codex_pro_models_snapshot_matches_verified_official_response():
    snapshot = (
        Path(__file__).parents[1]
        / "uni_api"
        / "api"
        / "codex_models_pro_0_144_0.json"
    )
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == (
        "c21a449d1a9785661087e9a6d2aaa217c4b77813a69f1190fa02728b5bd68345"
    )


async def test_admin_config_handlers_read_and_update_runtime_state():
    response = await api_config_response({"providers": [{"provider": "a"}]})
    assert response.status_code == 200

    app = SimpleNamespace(state=SimpleNamespace(config={"providers": [], "api_keys": []}))
    calls = []

    async def update_config(config, use_config_url=False):
        calls.append(("update", list(config["providers"]), use_config_url))
        return config, [{"api": "sk"}], ["sk"]

    async def refresh_runtime_state(app_obj):
        calls.append(("refresh", app_obj.state.api_list))

    update_response = await api_config_update_response(
        app=app,
        config_patch={"providers": [{"provider": "b"}]},
        update_config=update_config,
        refresh_runtime_state=refresh_runtime_state,
    )

    assert update_response.status_code == 200
    assert app.state.api_list == ["sk"]
    assert calls == [
        ("update", [{"provider": "b"}], False),
        ("refresh", ["sk"]),
    ]


class _StatsRepository:
    async def query_token_usage(self, **kwargs):
        self.token_usage_kwargs = kwargs
        return [
            {
                "api_key_prefix": "sk-test",
                "model": "gpt-4.1",
                "total_prompt_tokens": 2,
                "total_completion_tokens": 3,
                "total_tokens": 5,
                "request_count": 1,
            }
        ]

    async def query_channel_key_stats(self, **kwargs):
        self.rankings_kwargs = kwargs
        return [
            {
                "api_key": "upstream-key",
                "success_count": 9,
                "total_requests": 10,
                "success_rate": 0.9,
            }
        ]


async def test_token_usage_handler_resolves_user_filter_and_paid_state():
    repository = _StatsRepository()

    async def update_paid_key_state(api_key):
        assert api_key == "sk-user"
        return 10.0, 1.5

    response = await token_usage_response(
        repository=repository,
        database_disabled=False,
        config={"api_keys": [{"api": "sk-user"}]},
        admin_api_keys=[],
        api_index=0,
        model="gpt-4.1",
        last_n_days=1,
        update_paid_key_state=update_paid_key_state,
    )

    assert response.query_details.api_key_filter == "self"
    assert response.query_details.balance == "$8.5"
    assert response.usage[0].total_tokens == 5
    assert repository.token_usage_kwargs["filter_api_key"] == "sk-user"


async def test_channel_rankings_handler_delegates_time_range_to_repository():
    repository = _StatsRepository()

    response = await channel_key_rankings_response(
        repository=repository,
        database_disabled=False,
        provider_name="provider-a",
        last_n_days=1,
    )

    assert response.rankings[0].api_key == "upstream-key"
    assert response.query_details.api_key_filter == "provider-a"
    assert repository.rankings_kwargs["provider_name"] == "provider-a"


def test_api_keys_state_and_add_credits_handlers_are_pure_state_updates():
    states = {
        "sk-paid": {
            "credits": 1.0,
            "created_at": __import__("datetime").datetime(2026, 1, 1),
            "all_tokens_info": [],
            "total_cost": 2.5,
            "enabled": False,
        }
    }

    before = api_keys_states_response(states)
    response = add_credits_response(paid_api_keys_states=states, paid_key="sk-paid", amount=2.0)

    assert before.api_keys_states["sk-paid"].enabled is False
    assert states["sk-paid"]["credits"] == 3.0
    assert states["sk-paid"]["enabled"] is True
    assert response.status_code == 200
