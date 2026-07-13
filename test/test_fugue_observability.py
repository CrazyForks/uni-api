import json
import re
import asyncio

import main
from fugue_observability import (
    FugueObservabilityClient,
    FugueObservabilityConfig,
    build_uni_api_ember_request_telemetry,
    fugue_observability_config_from_env,
)


def test_fugue_observability_disabled_without_endpoint(monkeypatch):
    monkeypatch.delenv("FUGUE_OBSERVABILITY_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    config = fugue_observability_config_from_env(service_version="test")

    assert config.enabled is False


def test_uni_api_ember_telemetry_redacts_secrets_and_body():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"tenant_id": "tenant_123", "app_id": "app_123"},
        current_info={
            "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
            "request_id": "request_123",
            "parent_span_id": "00f067aa0ba902b7",
            "endpoint": "POST /v1/responses",
            "model": "gpt-5.4",
            "provider": "oaix",
            "role": "sk-test",
            "stream": True,
            "status_code": 200,
            "wire_status_code": 200,
            "response_committed": True,
            "process_time": 1.25,
            "api_key": "sk-secret-api-key",
            "text": "this is request body content",
            "authorization": "Bearer ember-secret-token",
            "headers": {
                "Authorization": "Bearer ember-secret-token",
                "Cookie": "session=ember-cookie-secret",
            },
            "cookie": "session=ember-cookie-secret",
            "database_url": "postgresql://user:pass@db/ember",
            "body": {"input": "ember request body secret"},
            "email": "ember@example.com",
            "source_ip": "203.0.113.88",
            "token": "ember-upstream-token-secret",
            "message_roles": "system/user",
            "role_counts": "system:1,user:1",
            "retry_count": 1,
            "cooldown_count": 1,
            "timing_spans": {
                "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
                "parent_span_id": "00f067aa0ba902b7",
                "request_received": 0,
                "body_parsed": 3,
                "provider_selected": 8,
                "provider_key_selected": 10,
                "retry_started": 20,
                "retry_count": 1,
                "retry_status_code": 503,
                "retry_provider": "oaix",
                "upstream_pool_wait_ms": 17,
                "Authorization": "Bearer ember-secret-token",
                "Cookie": "session=ember-cookie-secret",
                "database_url": "postgresql://user:pass@db/ember",
                "body": "ember request body secret",
                "email": "ember@example.com",
                "source_ip": "203.0.113.88",
                "upstream_send_start": 30,
                "upstream_headers_received": 90,
                "upstream_first_chunk": 140,
                "downstream_response_start": 145,
                "stream_end": 1250,
            },
            "upstream_attempts": [
                {
                    "index": 1,
                    "endpoint": "/v1/responses/compact",
                    "provider": "fugue-codex",
                    "model": "gpt-5.4",
                    "actual_model": "gpt-5.4",
                    "engine": "codex",
                    "upstream_host": "oaix.internal",
                    "payload_bytes": 29658219,
                    "timeout_seconds": 120,
                    "timeout_adjusted_from_seconds": 20,
                    "wants_compact": True,
                    "stream": False,
                    "started_ms": 31,
                    "duration_ms": 20061,
                    "status_code": 504,
                    "success": False,
                    "error_type": "ReadTimeout",
                    "error_message": "Bearer ember-secret-token",
                    "stream_diagnostics": {
                        "semantic_status": "error",
                        "diagnosis": "responses_partial_event_abort",
                        "failure_stage": "postcommit",
                        "oaix_connection_id": "oaixc-observe-1",
                        "http_version": "HTTP/2",
                        "httpcore_stream_id": 17,
                        "explicit_proxy_configured": False,
                        "transport_local_endpoint_hmac": "a" * 64,
                        "transport_peer_endpoint_hmac": "b" * 64,
                        "transport_four_tuple_hmac": "c" * 64,
                        "transport_socket_hmac": "d" * 64,
                        "upstream_body_bytes": 4096,
                        "upstream_chunk_count": 8,
                        "complete_event_count": 5,
                        "last_event_type": "response.output_text.delta",
                        "last_event_ordinal": 5,
                        "last_event_bytes": 128,
                        "last_event_sha256": "e" * 64,
                        "partial_event_bytes": 77,
                        "partial_event_sha256": "f" * 64,
                        "hash_scope": "ember_normalized_sse_event_lf_v1",
                        "partial_hash_scope": "normalized_prefix_plus_utf8_tail_v1",
                        "upstream_eof_seen": False,
                        "upstream_terminal_seen": False,
                        "upstream_terminal_validated": False,
                        "downstream_terminal_seen": False,
                        "downstream_terminal_asgi_write_completed": False,
                        "error_event_seen": True,
                        "usage_seen": False,
                        "exception_type": "ReadError",
                        "exception_origin": "postcommit_stream",
                        "exception_errno": 104,
                        "exception_errno_name": "ECONNRESET",
                        "exception_chain_depth": 3,
                        "exception_chain_truncated": False,
                        "exception_chain": [
                            {"type": "ReadError", "relation": "raised"},
                            {"type": "ReadError", "relation": "cause"},
                            {
                                "type": "ConnectionResetError",
                                "relation": "cause",
                                "errno": 104,
                            },
                        ],
                        "cleanup_owner": "responses_proxy_finally",
                        "cleanup_trigger": "after_upstream_read_or_stream_failure",
                        "cleanup_method": "cooperative_response_aclose",
                        "cleanup_result": "succeeded",
                        "cleanup_transport_evicted": False,
                        "cleanup_transport_safe": True,
                    },
                }
            ],
            "responses_stream_diagnostics": {
                "semantic_status": "error",
                "diagnosis": "responses_partial_event_abort",
                "failure_stage": "postcommit",
                "oaix_connection_id": "oaixc-observe-1",
                "upstream_body_bytes": 4096,
                "upstream_chunk_count": 8,
                "last_event_type": "response.output_text.delta",
                "last_event_ordinal": 5,
                "last_event_bytes": 128,
                "last_event_sha256": "e" * 64,
                "partial_event_bytes": 77,
                "partial_event_sha256": "f" * 64,
                "hash_scope": "ember_normalized_sse_event_lf_v1",
                "upstream_terminal_seen": False,
                "upstream_terminal_validated": False,
                "downstream_terminal_seen": False,
                "downstream_terminal_asgi_write_completed": False,
                "error_event_seen": True,
                "usage_seen": False,
                "exception_type": "ReadError",
                "exception_origin": "postcommit_stream",
                "cleanup_result": "succeeded",
            },
        },
        runtime_metrics={
            "inflight_requests": 12,
            "request_body_reserved_weighted_bytes": 8192,
            "upstream_response_reserved_weighted_bytes": 16384,
            "request_retained_reserved_weighted_bytes": 24576,
            "request_deferred_memory_requests": 2,
            "request_deferred_memory_weighted_bytes": 4096,
            "waiting_first_byte": 4,
            "event_loop_lag_ms": 2,
            "upstream_pool_in_use": 3,
            "stream_parser_reserved_bytes": 2048,
            "stream_parser_rejected_total": 1,
        },
    )

    serialized = json.dumps(telemetry, sort_keys=True)
    assert "sk-secret-api-key" not in serialized
    assert "this is request body content" not in serialized
    for secret in {
        "Bearer ember-secret-token",
        "ember-cookie-secret",
        "postgresql://user:pass@db/ember",
        "ember request body secret",
        "ember@example.com",
        "203.0.113.88",
        "ember-upstream-token-secret",
    }:
        assert secret not in serialized
    assert "api_key_hash" in serialized
    assert "system/user" in serialized

    log_event = telemetry["logs"][0]
    assert log_event["level"] == "info"
    assert log_event["service"] == "uni-api-ember"
    assert log_event["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert log_event["request_id"] == "request_123"
    assert log_event["event"] == "request_summary"
    assert log_event["event_type"] == "request_summary"
    assert log_event["message"] == "uni-api-ember request finished"
    assert log_event["app_id"] == "app_123"
    assert log_event["path"] == "/v1/responses"
    assert log_event["status_code"] == 200
    summary = log_event["summary"]
    assert summary["wire_status_code"] == "200"
    assert summary["semantic_status"] == "error"
    assert summary["upstream_terminal_seen"] == "false"
    assert summary["upstream_terminal_validated"] == "false"
    assert summary["downstream_terminal_seen"] == "false"
    assert summary["usage_seen"] == "false"
    assert summary["diagnosis"] == "responses_partial_event_abort"
    assert summary["failure_stage"] == "postcommit"
    assert summary["oaix_connection_id"] == "oaixc-observe-1"
    attempt_event = next(event for event in telemetry["logs"] if event["event"] == "upstream_attempt")
    attempt_attrs = attempt_event["attributes"]
    assert attempt_attrs["provider"] == "fugue-codex"
    assert attempt_attrs["attempt_status_code"] == "504"
    assert attempt_attrs["attempt_error_type"] == "ReadTimeout"
    assert attempt_attrs["payload_bytes"] == "29658219"
    assert attempt_attrs["timeout_seconds"] == "120"
    assert attempt_attrs["timeout_adjusted_from_seconds"] == "20"
    assert attempt_attrs["diagnosis"] == "responses_partial_event_abort"
    assert attempt_attrs["failure_stage"] == "postcommit"
    assert attempt_attrs["upstream_terminal_validated"] == "false"
    assert attempt_attrs["oaix_connection_id"] == "oaixc-observe-1"
    assert attempt_attrs["upstream_http_version"] == "HTTP/2"
    assert attempt_attrs["exception_errno_name"] == "ECONNRESET"
    assert attempt_attrs["exception_chain_depth"] == "3"
    assert "ConnectionResetError" in attempt_attrs["exception_chain_json"]
    assert attempt_attrs["cleanup_owner"] == "responses_proxy_finally"

    stages = {
        event["attributes"]["stage"]
        for event in telemetry["traces"]
    }
    stage_ms = {
        event["attributes"]["stage"]: event["attributes"]["stage_ms"]
        for event in telemetry["traces"]
    }
    assert {
        "request_received",
        "body_parsed",
        "provider_selected",
        "provider_key_selected",
        "retry_started",
        "client_pool_acquired",
        "upstream_send_start",
        "upstream_headers_received",
        "upstream_first_chunk",
        "downstream_response_start",
        "stream_end",
    }.issubset(stages)
    assert stage_ms["upstream_first_chunk"] == "140"
    assert stage_ms["downstream_response_start"] == "5"

    for metric in telemetry["metrics"]:
        attrs = metric["attributes"]
        assert "trace_id" not in attrs
        assert "request_id" not in attrs
        assert "api_key_hash" not in attrs

    metrics = {event["metric"]: event for event in telemetry["metrics"]}
    assert metrics["uniapi_ember_upstream_errors_total"]["value"] == 1
    assert metrics["uniapi_ember_exposed_5xx_total"]["value"] == 0
    assert (
        metrics["uniapi_ember_request_body_reserved_weighted_bytes"]["value"]
        == 8192
    )
    assert (
        metrics[
            "uniapi_ember_upstream_response_reserved_weighted_bytes"
        ]["value"]
        == 16384
    )
    assert (
        metrics[
            "uniapi_ember_request_retained_reserved_weighted_bytes"
        ]["value"]
        == 24576
    )
    assert metrics["uniapi_ember_request_deferred_memory_requests"]["value"] == 2
    assert (
        metrics["uniapi_ember_request_deferred_memory_weighted_bytes"]["value"]
        == 4096
    )
    assert metrics["uniapi_ember_stream_parser_reserved_bytes"]["value"] == 2048
    assert metrics["uniapi_ember_stream_parser_rejected_total"]["value"] == 1
    assert "route_id" not in metrics["uniapi_ember_inflight_requests"]["attributes"]
    assert metrics["uniapi_ember_request_duration_ms"]["attributes"]["route_id"]


def test_local_admission_503_is_not_counted_as_upstream_failure():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 503,
            "admission_rejected": True,
            "error_type": "queue_full",
            "process_time": 0.01,
            "upstream_attempts": [],
        },
        runtime_metrics={"inflight_requests": 100, "request_waiters": 900},
    )

    metrics = {event["metric"]: event["value"] for event in telemetry["metrics"]}
    assert metrics["uniapi_ember_upstream_errors_total"] == 0
    assert metrics["uniapi_ember_exposed_5xx_total"] == 1
    assert metrics["uniapi_ember_request_admission_rejected_total"] == 1


def test_post_commit_stream_failure_keeps_wire_200_and_has_failure_metric():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "wire_status_code": 200,
            "stream": True,
            "stream_outcome": "local_backpressure_abort",
            "stream_error_status_code": 503,
            "stream_error_after_response_start": True,
            "error_type": "StreamQueuePutTimeout",
            "process_time": 1.0,
        },
    )

    log_event = telemetry["logs"][0]
    assert log_event["status_code"] == 200
    assert log_event["level"] == "error"
    assert log_event["attributes"]["stream_outcome"] == "local_backpressure_abort"
    assert log_event["attributes"]["stream_error_status_code"] == "503"
    metrics = {event["metric"]: event["value"] for event in telemetry["metrics"]}
    assert metrics["uniapi_ember_exposed_5xx_total"] == 0
    assert metrics["uniapi_ember_stream_failures_total"] == 1


def test_responses_diagnostics_are_exported_with_valid_bounded_json():
    diagnostic = {
        "schema_version": 1,
        "semantic_status": "error",
        "diagnosis": "responses_read_error",
        "failure_stage": "upstream_headers",
        "terminal_consistency_status": "inconsistent",
        "terminal_semantics_consistent": False,
        "terminal_semantics_inconsistency": ["declared_outcome_mismatch"],
        "usage_object_seen": True,
        "usage_counters_seen": True,
        "usage_input_known": True,
        "usage_output_known": False,
        "usage_total_known": True,
        "usage_values_valid": True,
        "usage_seen": False,
        "downstream_usage_object_seen": True,
        "downstream_usage_counters_seen": True,
        "downstream_usage_input_known": True,
        "downstream_usage_output_known": False,
        "downstream_usage_total_known": True,
        "downstream_usage_values_valid": True,
        "downstream_usage_seen": False,
        "downstream_usage_observer_status": "completed",
        "transport_error_code": "peer_closed_incomplete_chunked_body",
        "transport_error_code_source": "known_message_pattern",
        "transport_end_trigger": "httpcore_body_read_failure",
        "response_start_asgi_write_attempted": True,
        "response_start_asgi_write_completed": True,
        "downstream_final_body_attempted": False,
        "downstream_final_body_completed": False,
        "cleanup_result": "incomplete",
        "cleanup_failure": True,
        "cleanup_failure_stage": "cleanup",
        "cleanup_actions": [
            {
                "actor": "responses_proxy_finally",
                "method": "cooperative_response_aclose",
                "transport_safe": False,
                "padding": "x" * 300,
            }
            for _ in range(32)
        ],
        "cleanup_actions_truncated": True,
        "exception_chain": [
            {
                "relation": "cause",
                "type": "RemoteProtocolError",
                "message_sha256": "a" * 64,
                "padding": "y" * 300,
            }
            for _ in range(32)
        ],
        "exception_chain_truncated": True,
    }
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "wire_status_code": 200,
            "response_committed": True,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "usage_parse_error": "missing_usage_components",
            "responses_stream_diagnostics": diagnostic,
            "upstream_attempts": [
                {
                    "index": 1,
                    "status_code": 200,
                    "stream_diagnostics": diagnostic,
                }
            ],
        },
    )

    summary = telemetry["logs"][0]["summary"]
    attempt = telemetry["logs"][1]["attributes"]
    for attrs in (summary, attempt):
        assert attrs["failure_stage"] == "precommit"
        assert attrs["transport_error_code"] == (
            "peer_closed_incomplete_chunked_body"
        )
        assert attrs["terminal_semantics_consistent"] == "false"
        assert attrs["usage_input_known"] == "true"
        assert attrs["usage_output_known"] == "false"
        assert attrs["downstream_usage_output_known"] == "false"
        assert attrs["cleanup_failure_stage"] == "cleanup"
        for field in (
            "terminal_semantics_inconsistency_json",
            "exception_chain_json",
            "cleanup_actions_json",
        ):
            json.loads(attrs[field])
            assert len(attrs[field].encode("utf-8")) <= 4096
        assert attrs["exception_chain_json_truncated"] == "true"
        assert attrs["cleanup_actions_json_truncated"] == "true"
        assert len(attrs["exception_chain_json_sha256"]) == 64

    assert summary["response_committed"] == "true"
    assert summary["prompt_tokens"] == "0"
    assert "completion_tokens" not in summary
    assert summary["total_tokens"] == "0"
    assert summary["usage_parse_error"] == "missing_usage_components"


def test_responses_summary_exports_real_known_zero_usage():
    no_diagnostics = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )
    no_diagnostics_summary = no_diagnostics["logs"][0]["summary"]
    assert "prompt_tokens" not in no_diagnostics_summary
    assert "completion_tokens" not in no_diagnostics_summary
    assert "total_tokens" not in no_diagnostics_summary

    unknown = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "responses_stream_diagnostics": {
                "downstream_usage_input_known": False,
                "downstream_usage_output_known": False,
                "downstream_usage_total_known": False,
                "downstream_usage_seen": False,
            },
        },
    )
    unknown_summary = unknown["logs"][0]["summary"]
    assert "prompt_tokens" not in unknown_summary
    assert "completion_tokens" not in unknown_summary
    assert "total_tokens" not in unknown_summary

    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "responses_stream_diagnostics": {
                "downstream_usage_input_known": True,
                "downstream_usage_output_known": True,
                "downstream_usage_total_known": True,
                "downstream_usage_values_valid": True,
                "downstream_usage_alias_consistent": True,
                "downstream_usage_seen": True,
            },
        },
    )

    summary = telemetry["logs"][0]["summary"]
    assert summary["prompt_tokens"] == "0"
    assert summary["completion_tokens"] == "0"
    assert summary["total_tokens"] == "0"


def test_responses_request_summary_is_never_sampled_away():
    client = FugueObservabilityClient(
        FugueObservabilityConfig(
            endpoint="https://observability.invalid",
            sample_rate=0.0,
        )
    )
    client._queue = asyncio.Queue(maxsize=10)

    client.emit_request(
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "wire_status_code": 200,
            "stream_outcome": "completed",
            "responses_stream_diagnostics": {
                "diagnosis": "responses_completed_with_usage",
                "semantic_status": "completed",
                "usage_seen": True,
            },
        }
    )
    assert client._queue.qsize() == 1

    client._queue.get_nowait()
    client._queue.task_done()
    client.emit_request(
        current_info={
            "endpoint": "GET /healthz",
            "status_code": 200,
            "stream_outcome": "completed",
        }
    )
    assert client._queue.qsize() == 0


def test_traceparent_is_inherited_and_forwarded():
    incoming = main._incoming_trace_context(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "tracestate": "vendor=value",
            "x-request-id": "legacy-request",
        }
    )

    trace = main.RequestTrace(
        trace_id=incoming["trace_id"],
        parent_span_id=incoming["parent_span_id"],
        trace_flags=incoming["trace_flags"],
        tracestate=incoming["tracestate"],
    )
    headers = main._trace_headers_for_upstream(
        {
            "trace_id": trace.trace_id,
            "request_id": "request_123",
            "trace": trace,
            "tracestate": trace.tracestate,
        }
    )

    assert incoming["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert incoming["parent_span_id"] == "00f067aa0ba902b7"
    assert incoming["x_request_id"] == "legacy-request"
    assert headers["x-request-id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert headers["tracestate"] == "vendor=value"
    assert headers["traceparent"].startswith("00-4bf92f3577b34da6a3ce929d0e0e4736-")


def test_missing_trace_headers_generate_w3c_trace_id():
    incoming = main._incoming_trace_context({})

    assert re.match(r"^[0-9a-f]{32}$", incoming["trace_id"])


def test_request_trace_uses_nonzero_ms_for_observed_stages(monkeypatch):
    trace = main.RequestTrace(trace_id="4bf92f3577b34da6a3ce929d0e0e4736")
    monkeypatch.setattr(main, "time", lambda: trace.started_at)

    trace.mark("request_received")
    trace.mark("upstream_first_chunk")

    spans = trace.snapshot()
    assert spans["request_received"] == 0
    assert spans["upstream_first_chunk"] == 1
