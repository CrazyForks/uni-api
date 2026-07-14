import httpx
from fastapi import HTTPException

from uni_api.upstream.policies import CooldownPolicy, ProviderErrorClassifier, RetryPolicy
from uni_api.upstream.responses_errors import responses_failure_error


def _safe_get(data, *keys, default=None):
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _get_engine(provider, endpoint=None, original_model=None):
    _ = endpoint, original_model
    return provider.get("engine", "gpt"), None


def test_provider_error_classifier_normalizes_http_and_network_errors():
    classifier = ProviderErrorClassifier(_safe_get)

    assert classifier.normalize_exception(HTTPException(status_code=418, detail="teapot")) == (418, "teapot")
    assert classifier.normalize_exception(httpx.ConnectError("no route")) == (503, "Unable to connect to service")
    assert classifier.remap_status_code(500, "string_above_max_length") == 413


def test_provider_error_classifier_preserves_local_upstream_admission_503():
    classifier = ProviderErrorClassifier(_safe_get)

    class LocalAdmissionError(Exception):
        status_code = 503
        reason = "upstream_wait_timeout"
        local_admission_rejection = True

    assert classifier.normalize_exception(LocalAdmissionError()) == (
        503,
        "upstream_wait_timeout",
    )


def test_provider_error_classifier_preserves_responses_semantic_400():
    classifier = ProviderErrorClassifier(_safe_get)
    retry_policy = RetryPolicy(classifier, _get_engine)
    error = responses_failure_error(
        {
            "error": {
                "code": "oaix_gateway_error",
                "message": "Your input exceeds the context window of this model.",
                "status": 400,
                "type": "gateway_error",
            }
        },
        event_type="error",
    )

    assert error is not None
    status_code, detail = classifier.normalize_exception(error)
    assert status_code == 400
    assert '"code":"oaix_gateway_error"' in detail
    assert retry_policy.should_retry(
        True,
        status_code,
        {"base_url": "https://example.com/v1/responses"},
        error_message=detail,
        endpoint="/v1/chat/completions",
        original_model="gpt-5.5",
    ) is False


def test_responses_semantic_error_bounds_attacker_sized_message():
    error = responses_failure_error(
        {
            "type": "error",
            "error": {
                "code": "server_error",
                "message": "x" * (1024 * 1024),
            },
        },
        event_type="error",
    )

    assert error is not None
    assert len(error.message.encode("utf-8")) <= 4096
    assert len(error.detail_json.encode("utf-8")) < 8192
    assert error.message.endswith(" [truncated]")
    assert error.passthrough_error_body is None


def test_response_failed_has_detached_bounded_responses_terminal():
    error = responses_failure_error(
        {
            "type": "response.failed",
            "sequence_number": 7,
            "response": {
                "id": "resp_ctx",
                "object": "response",
                "model": "gpt-test",
                "status": "failed",
                "error": {
                    "code": " Context_Length_Exceeded ",
                    "type": " Invalid_Request_Error ",
                    "message": "x" * (1024 * 1024),
                    "param": "input",
                    "ignored": {"large": "y" * (1024 * 1024)},
                },
                "ignored": ["z" * (1024 * 1024)],
            },
        },
        event_type="response.failed",
        wire_status_code=200,
    )

    assert error is not None
    assert error.sse_payload["type"] == "error"
    assert error.responses_sse_event_type == "response.failed"
    assert error.responses_sse_payload == {
        "type": "response.failed",
        "sequence_number": 7,
        "response": {
            "id": "resp_ctx",
            "object": "response",
            "model": "gpt-test",
            "status": "failed",
            "error": {
                "code": "context_length_exceeded",
                "type": "invalid_request_error",
                "message": error.message,
                "param": "input",
            },
        },
    }
    assert len(str(error.responses_sse_payload).encode("utf-8")) < 8192


def test_preserved_response_failed_http_body_does_not_retain_large_graph():
    ignored = "y" * (7 * 1024 * 1024)
    error = responses_failure_error(
        {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {
                    "code": "context_length_exceeded",
                    "message": "input is too long",
                    "ignored": {"attacker_owned": ignored},
                },
            },
        },
        event_type="response.failed",
        preserve_error_body=True,
    )

    assert error is not None
    assert error.passthrough_error_body == {
        "error": {
            "code": "context_length_exceeded",
            "message": "input is too long",
        }
    }
    assert ignored not in str(error.passthrough_error_body)
    assert len(str(error.passthrough_error_body).encode("utf-8")) < 8192


def test_generic_error_event_is_not_promoted_to_response_failed():
    error = responses_failure_error(
        {
            "type": "error",
            "error": {
                "code": "context_length_exceeded",
                "message": "input is too long",
            },
        },
        event_type="error",
    )

    assert error is not None
    assert error.responses_sse_event_type == "error"
    assert error.responses_sse_payload is error.sse_payload


def test_response_failed_rejects_non_string_status_without_stringifying():
    class ExplosiveStatus(list):
        def __str__(self):
            raise AssertionError("protocol status must not be stringified")

    error = responses_failure_error(
        {
            "type": "response.failed",
            "response": {
                "status": ExplosiveStatus(["x" * (1024 * 1024)]),
                "error": {
                    "code": "context_length_exceeded",
                    "message": "input is too long",
                },
            },
        },
        event_type="response.failed",
    )

    assert error is None


def test_retry_policy_does_not_retry_missing_persisted_response_item():
    classifier = ProviderErrorClassifier(_safe_get)
    retry_policy = RetryPolicy(classifier, _get_engine)
    error = {
        "error": {
            "message": "Item with id 'rs_1' not found. Items are not persisted when `store` is set to false.",
            "type": "invalid_request_error",
        }
    }

    assert retry_policy.should_retry(
        True,
        404,
        {"base_url": "https://example.com/v1/responses"},
        error_message=str(error),
        endpoint="/v1/responses",
        original_model="gpt-5.4",
    ) is False


def test_retry_policy_retries_codex_chatgpt_model_unsupported():
    classifier = ProviderErrorClassifier(_safe_get)
    retry_policy = RetryPolicy(classifier, _get_engine)

    assert retry_policy.should_retry(
        True,
        400,
        {"base_url": "https://chatgpt.com/backend-api/codex", "engine": "codex"},
        error_message='{"error":{"message":"model is not supported when using codex with a ChatGPT account"}}',
        endpoint="/v1/responses",
        original_model="gpt-5.5",
    ) is True


def test_cooldown_policy_uses_retry_after_and_configured_minimum():
    classifier = ProviderErrorClassifier(_safe_get)
    cooldown_policy = CooldownPolicy(classifier, _get_engine)
    details = (
        '{"error":{"code":"rate_limit_exceeded",'
        '"message":"Rate limit reached. Please try again in 2500ms."}}'
    )

    assert cooldown_policy.rate_limit_cooling_time(
        {"preferences": {"api_key_rate_limit_cooldown_period": 1}},
        429,
        details,
    ) == 3


def test_cooldown_policy_identifies_quota_and_codex_auth_cooldowns():
    classifier = ProviderErrorClassifier(_safe_get)
    retry_policy = RetryPolicy(classifier, _get_engine)
    cooldown_policy = CooldownPolicy(classifier, _get_engine)

    assert cooldown_policy.should_use_quota_cooldown(
        {"engine": "gpt"},
        429,
        "insufficient_quota",
        endpoint="/v1/responses",
        original_model="gpt-5.4",
        retry_policy=retry_policy,
    ) is True

    assert cooldown_policy.should_use_quota_cooldown(
        {"engine": "codex"},
        403,
        '{"error":{"code":"account_deactivated","message":"account has been deactivated"}}',
        endpoint="/v1/responses",
        original_model="gpt-5.4",
        retry_policy=retry_policy,
    ) is True
