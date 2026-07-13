from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import time
from typing import Any

import httpx


logger = logging.getLogger("uni-api")

_TRACE_ENDPOINT = "/v1/traces"
_LOG_ENDPOINT = "/v1/logs"
_METRIC_ENDPOINT = "/v1/metrics"
_DEFAULT_SERVICE_NAME = "uni-api-ember"
_DEFAULT_QUEUE_MAX_SIZE = 10000
_DEFAULT_EXPORT_WORKER_COUNT = 4
_DEFAULT_EXPORT_TIMEOUT_SECONDS = 2.0
_DEFAULT_SAMPLE_RATE = 1.0

_STAGE_ORDER = [
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
]


@dataclass(frozen=True)
class FugueObservabilityConfig:
    endpoint: str | None
    service_name: str = _DEFAULT_SERVICE_NAME
    service_version: str | None = None
    queue_max_size: int = _DEFAULT_QUEUE_MAX_SIZE
    export_worker_count: int = _DEFAULT_EXPORT_WORKER_COUNT
    export_timeout_seconds: float = _DEFAULT_EXPORT_TIMEOUT_SECONDS
    sample_rate: float = _DEFAULT_SAMPLE_RATE
    identity_attrs: dict[str, str] = field(default_factory=dict)
    emit_request_summaries: bool = True
    emit_stage_spans: bool = True
    emit_metrics: bool = True

    @property
    def enabled(self) -> bool:
        return bool((self.endpoint or "").strip())


class FugueObservabilityClient:
    def __init__(self, config: FugueObservabilityConfig) -> None:
        self.config = config
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._client: httpx.AsyncClient | None = None
        self._dropped = 0
        self._export_errors = 0

    async def start(self) -> None:
        if not self.config.enabled or self._tasks:
            return
        self._queue = asyncio.Queue(maxsize=max(1, int(self.config.queue_max_size)))
        self._client = httpx.AsyncClient(timeout=self.config.export_timeout_seconds)
        worker_count = max(1, int(self.config.export_worker_count))
        self._tasks = [
            asyncio.create_task(
                self._worker(),
                name=f"uni-api-ember-fugue-observability-exporter-{index}",
            )
            for index in range(worker_count)
        ]
        logger.info(
            "Fugue observability exporter enabled for service=%s workers=%s queue_max_size=%s",
            self.config.service_name,
            worker_count,
            self.config.queue_max_size,
        )

    async def stop(self) -> None:
        tasks = self._tasks
        self._tasks = []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()
        self._queue = None

    def emit_request(self, *, current_info: dict[str, Any], runtime_metrics: dict[str, Any] | None = None) -> None:
        if not self.config.enabled:
            return
        status_code = _safe_int(current_info.get("status_code"), 0)
        stream_failure = _is_stream_failure(current_info)
        downstream_disconnected = _safe_bool(
            current_info.get("downstream_disconnected")
        ) or status_code == 499
        retain_for_responses_correlation = _is_responses_request(current_info)
        sampled_out = (
            status_code < 400
            and not stream_failure
            and not downstream_disconnected
            and self.config.sample_rate < 1.0
            and (
                self.config.sample_rate <= 0.0
                or random.random() > self.config.sample_rate
            )
        )
        if sampled_out and not retain_for_responses_correlation:
            return
        telemetry = build_uni_api_ember_request_telemetry(
            service_name=self.config.service_name,
            service_version=self.config.service_version,
            identity_attrs=self.config.identity_attrs,
            current_info=current_info,
            runtime_metrics=runtime_metrics,
        )
        if self.config.emit_request_summaries:
            self._emit_events(_LOG_ENDPOINT, telemetry["logs"])
        # When ordinary success sampling excludes a Responses request, retain
        # only its compact correlation logs.  Stage spans and per-request
        # metric batches must not crowd those logs out of the bounded queue.
        if self.config.emit_stage_spans and not sampled_out:
            self._emit_events(_TRACE_ENDPOINT, telemetry["traces"])
        if self.config.emit_metrics and not sampled_out:
            self._emit_events(_METRIC_ENDPOINT, telemetry["metrics"])

    def _emit_events(self, path: str, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        queue = self._queue
        if queue is None:
            return
        try:
            queue.put_nowait((path, {"events": events}))
        except asyncio.QueueFull:
            self._dropped += len(events)
            if self._dropped == len(events) or self._dropped % 100 == 0:
                logger.warning("Fugue observability queue full; dropped %s event(s)", self._dropped)

    async def _worker(self) -> None:
        assert self._queue is not None
        while True:
            path, payload = await self._queue.get()
            try:
                await self._post_json(path, payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._export_errors += 1
                if self._export_errors == 1 or self._export_errors % 100 == 0:
                    logger.warning("Fugue observability export failed: %s", type(exc).__name__)
            finally:
                self._queue.task_done()

    async def _post_json(self, path: str, payload: dict[str, Any]) -> None:
        client = self._client
        if client is None:
            return
        response = await client.post(_endpoint_url(self.config.endpoint or "", path), json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"observability endpoint returned HTTP {response.status_code}")


_client: FugueObservabilityClient | None = None


async def start_fugue_observability_from_env(*, service_version: str | None = None) -> None:
    global _client
    config = fugue_observability_config_from_env(service_version=service_version)
    if not config.enabled:
        _client = None
        return
    client = FugueObservabilityClient(config)
    await client.start()
    _client = client


async def stop_fugue_observability() -> None:
    global _client
    client = _client
    _client = None
    if client is not None:
        await client.stop()


def fugue_observability_config_from_env(*, service_version: str | None = None) -> FugueObservabilityConfig:
    endpoint = _env_text("FUGUE_OBSERVABILITY_ENDPOINT") or _env_text("OTEL_EXPORTER_OTLP_ENDPOINT")
    return FugueObservabilityConfig(
        endpoint=endpoint,
        service_name=_env_text("FUGUE_OBSERVABILITY_SERVICE_NAME") or _DEFAULT_SERVICE_NAME,
        service_version=_env_text("FUGUE_OBSERVABILITY_SERVICE_VERSION") or service_version,
        queue_max_size=_env_int("FUGUE_OBSERVABILITY_QUEUE_MAX_SIZE", _DEFAULT_QUEUE_MAX_SIZE),
        export_worker_count=_env_int("FUGUE_OBSERVABILITY_EXPORT_WORKERS", _DEFAULT_EXPORT_WORKER_COUNT),
        export_timeout_seconds=_env_float(
            "FUGUE_OBSERVABILITY_EXPORT_TIMEOUT_SECONDS",
            _DEFAULT_EXPORT_TIMEOUT_SECONDS,
        ),
        sample_rate=max(0.0, min(1.0, _env_float("FUGUE_OBSERVABILITY_SAMPLE_RATE", _DEFAULT_SAMPLE_RATE))),
        identity_attrs=_identity_attrs_from_env(),
        emit_request_summaries=_env_bool("FUGUE_OBSERVABILITY_REQUEST_SUMMARY_ENABLED", True),
        emit_stage_spans=_env_bool("FUGUE_OBSERVABILITY_STAGE_SPANS_ENABLED", True),
        emit_metrics=_env_bool("FUGUE_OBSERVABILITY_METRICS_ENABLED", True),
    )


def emit_uni_api_ember_request_observability(**kwargs: Any) -> None:
    client = _client
    if client is None:
        return
    try:
        client.emit_request(**kwargs)
    except Exception:
        logger.exception("Failed to enqueue Fugue request observability event")


def build_uni_api_ember_request_telemetry(
    *,
    service_name: str,
    service_version: str | None,
    identity_attrs: dict[str, str] | None,
    current_info: dict[str, Any],
    runtime_metrics: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    now = datetime.now(timezone.utc)
    spans = dict(current_info.get("timing_spans") or {})
    trace_id = _safe_text(current_info.get("trace_id") or spans.get("trace_id"))
    request_id = _safe_text(current_info.get("request_id"))
    endpoint = _safe_text(current_info.get("endpoint"))
    method, path_template = _split_endpoint(endpoint)
    status_code = _safe_int(current_info.get("status_code"), 0)
    route_id = _route_id(endpoint)
    duration_ms = _duration_ms_from_info(current_info)
    ttft_ms = _ttft_ms(spans)
    error_type = _safe_text(current_info.get("error_type")) or _classify_error(status_code)
    stream_outcome = _safe_text(current_info.get("stream_outcome"))
    stream_error_status_code = _safe_int(
        current_info.get("stream_error_status_code"), 0
    )
    retry_count = _safe_int(current_info.get("retry_count"), 0)
    cooldown_count = _safe_int(current_info.get("cooldown_count"), 0)
    is_stream = _safe_bool(current_info.get("stream"))
    api_key_hash = _secret_hash(current_info.get("api_key"))
    responses_diagnostics = _responses_stream_diagnostics(current_info)

    base = _base_attrs(
        service_name=service_name,
        service_version=service_version,
        identity_attrs=identity_attrs,
        trace_id=trace_id,
        request_id=request_id,
        parent_span_id=_safe_text(current_info.get("parent_span_id") or spans.get("parent_span_id")),
        endpoint=endpoint,
        method=method,
        path_template=path_template,
        route_id=route_id,
        model=_safe_text(current_info.get("model")),
        provider=_safe_text(current_info.get("provider")),
        role=_safe_text(current_info.get("role")),
        is_stream=is_stream,
        status_code=status_code,
        error_type=error_type,
        retry_count=retry_count,
        cooldown_count=cooldown_count,
        api_key_hash=api_key_hash,
    )

    logs = [
        {
            "timestamp": _iso_timestamp(now),
            "level": _event_level(stream_error_status_code or status_code),
            "service": service_name,
            "trace_id": trace_id,
            "request_id": request_id,
            "event": "request_summary",
            "event_type": "request_summary",
            "source": service_name,
            "message": "uni-api-ember request finished",
            # Fugue request_facts indexes these fields at the event top level.
            # Keep the nested attributes too for existing log consumers.
            "app_id": _safe_text((identity_attrs or {}).get("app_id")),
            "path": path_template or endpoint,
            "status_code": status_code,
            "attributes": _drop_empty(
                {
                    **base,
                    "duration_ms": _int_text(duration_ms),
                    "total_ms": _int_text(duration_ms),
                    "ttfb_ms": _int_text(ttft_ms),
                    "ttft_ms": _int_text(ttft_ms),
                    "upstream_ms": _int_text(_stage_delta_ms(spans, "upstream_headers_received", "upstream_send_start")),
                    "status_class": _status_class(status_code),
                    "request_kind": _safe_text(current_info.get("request_kind")),
                    "stream_outcome": stream_outcome,
                    "stream_error_status_code": _optional_int_text(
                        stream_error_status_code
                    ),
                    "stream_error_after_response_start": _bool_text(
                        _safe_bool(
                            current_info.get("stream_error_after_response_start")
                        )
                    ),
                    "downstream_disconnected": _bool_text(
                        _safe_bool(current_info.get("downstream_disconnected"))
                    ),
                }
            ),
            "summary": _drop_empty(
                {
                    "message_roles": _safe_text(current_info.get("message_roles")),
                    "role_counts": _safe_text(current_info.get("role_counts")),
                    "client_pool_wait_ms": _int_text(_span_ms(spans, "upstream_pool_wait_ms")),
                    "request_admission_wait_ms": _int_text(
                        _span_ms(spans, "request_admission_wait_ms")
                    ),
                    "event_loop_lag_ms": _int_text(_runtime_int(runtime_metrics, "event_loop_lag_ms")),
                    "inflight_requests": _int_text(_runtime_int(runtime_metrics, "inflight_requests")),
                    "request_waiters": _int_text(_runtime_int(runtime_metrics, "request_waiters")),
                    "request_body_reserved_weighted_bytes": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "request_body_reserved_weighted_bytes",
                        )
                    ),
                    "upstream_response_reserved_weighted_bytes": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "upstream_response_reserved_weighted_bytes",
                        )
                    ),
                    "request_retained_reserved_weighted_bytes": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "request_retained_reserved_weighted_bytes",
                        )
                    ),
                    "waiting_first_byte": _int_text(_runtime_int(runtime_metrics, "waiting_first_byte")),
                    "upstream_pool_in_use": _int_text(
                        _runtime_int(runtime_metrics, "upstream_pool_in_use")
                    ),
                    "upstream_pool_waiters": _int_text(
                        _runtime_int(runtime_metrics, "upstream_pool_waiters")
                    ),
                    "stream_queue_bytes": _int_text(
                        _runtime_int(runtime_metrics, "stream_queue_bytes")
                    ),
                    "stream_queue_peak_bytes": _int_text(
                        _safe_int(current_info.get("stream_queue_peak_bytes"), 0)
                    ),
                    "wire_status_code": _optional_int_text(
                        current_info.get("wire_status_code")
                    ),
                    "response_committed": _bool_text(
                        _safe_bool(current_info.get("response_committed"))
                    ),
                    "prompt_tokens": _responses_token_text(
                        current_info,
                        responses_diagnostics,
                        value_key="prompt_tokens",
                        known_key="downstream_usage_input_known",
                    ),
                    "completion_tokens": _responses_token_text(
                        current_info,
                        responses_diagnostics,
                        value_key="completion_tokens",
                        known_key="downstream_usage_output_known",
                    ),
                    "total_tokens": _responses_token_text(
                        current_info,
                        responses_diagnostics,
                        value_key="total_tokens",
                        known_key="downstream_usage_total_known",
                    ),
                    "usage_parse_error": _safe_text(
                        current_info.get("usage_parse_error"), max_len=80
                    ),
                    "semantic_status": _safe_text(
                        responses_diagnostics.get("semantic_status")
                    ),
                    "upstream_terminal_seen": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get("upstream_terminal_seen")
                        )
                    ),
                    "upstream_terminal_validated": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get(
                                "upstream_terminal_validated"
                            )
                        )
                    ),
                    "terminal_frame_seen": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get("terminal_frame_seen")
                        )
                    ),
                    "declared_terminal_type": _safe_text(
                        responses_diagnostics.get("declared_terminal_type")
                    ),
                    "declared_terminal_ordinal": _optional_int_text(
                        responses_diagnostics.get("declared_terminal_ordinal")
                    ),
                    "declared_terminal_bytes": _optional_int_text(
                        responses_diagnostics.get("declared_terminal_bytes")
                    ),
                    "declared_terminal_sha256": _safe_text(
                        responses_diagnostics.get("declared_terminal_sha256")
                    ),
                    "semantic_terminal_type": _safe_text(
                        responses_diagnostics.get("semantic_terminal_type")
                    ),
                    "semantic_terminal_outcome": _safe_text(
                        responses_diagnostics.get("semantic_terminal_outcome")
                    ),
                    "semantic_terminal_bytes": _optional_int_text(
                        responses_diagnostics.get("semantic_terminal_bytes")
                    ),
                    "semantic_terminal_sha256": _safe_text(
                        responses_diagnostics.get("semantic_terminal_sha256")
                    ),
                    "semantic_terminal_sequence_number": _optional_int_text(
                        responses_diagnostics.get(
                            "semantic_terminal_sequence_number"
                        )
                    ),
                    "downstream_terminal_seen": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get("downstream_terminal_seen")
                        )
                    ),
                    "ember_queue_terminal_handoff_completed": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get(
                                "ember_queue_terminal_handoff_completed"
                            )
                        )
                    ),
                    "downstream_terminal_asgi_write_completed": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get(
                                "downstream_terminal_asgi_write_completed"
                            )
                        )
                    ),
                    "error_event_seen": _bool_text(
                        _safe_bool(responses_diagnostics.get("error_event_seen"))
                    ),
                    "usage_seen": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get(
                                "usage_seen",
                                current_info.get("usage_seen"),
                            )
                        )
                    ),
                    "diagnosis": _safe_text(
                        responses_diagnostics.get("diagnosis")
                    ),
                    "failure_stage": _responses_failure_stage(
                        current_info,
                        responses_diagnostics,
                    ),
                    "oaix_connection_id": _safe_text(
                        responses_diagnostics.get("oaix_connection_id")
                    ),
                    "upstream_body_bytes": _optional_int_text(
                        responses_diagnostics.get("upstream_body_bytes")
                    ),
                    "upstream_chunk_count": _optional_int_text(
                        responses_diagnostics.get("upstream_chunk_count")
                    ),
                    "last_event_type": _safe_text(
                        responses_diagnostics.get("last_event_type")
                    ),
                    "last_event_ordinal": _optional_int_text(
                        responses_diagnostics.get("last_event_ordinal")
                    ),
                    "last_event_bytes": _optional_int_text(
                        responses_diagnostics.get("last_event_bytes")
                    ),
                    "last_event_sha256": _safe_text(
                        responses_diagnostics.get("last_event_sha256")
                    ),
                    "partial_event_bytes": _optional_int_text(
                        responses_diagnostics.get("partial_event_bytes")
                    ),
                    "partial_event_sha256": _safe_text(
                        responses_diagnostics.get("partial_event_sha256")
                    ),
                    "event_hash_scope": _safe_text(
                        responses_diagnostics.get("hash_scope")
                    ),
                    **_responses_diagnostic_attrs(
                        responses_diagnostics,
                        current_info=current_info,
                    ),
                }
            ),
        }
    ]
    logs.extend(_upstream_attempt_log_events(now, service_name, base, current_info))

    traces = []
    for stage, stage_ms, stage_attrs in _stage_rows(spans, duration_ms):
        traces.append(
            {
                "timestamp": _iso_timestamp(now),
                "kind": "span",
                "event_type": "request_span",
                "source": service_name,
                "message": stage,
                "attributes": _drop_empty(
                    {
                        **base,
                        **stage_attrs,
                        "span_id": _span_id(trace_id, request_id, stage),
                        "parent_span_id": _safe_text(current_info.get("parent_span_id") or spans.get("parent_span_id")),
                        "stage": stage,
                        "stage_ms": _int_text(stage_ms),
                    }
                ),
            }
        )

    metrics = _request_metric_events(
        service_name=service_name,
        identity_attrs=identity_attrs,
        timestamp=now,
        method=method,
        status_code=status_code,
        route_id=route_id,
        values={
            "uniapi_ember_request_duration_ms": duration_ms,
            "uniapi_ember_request_admission_wait_ms": _span_ms(
                spans, "request_admission_wait_ms"
            ),
            "uniapi_ember_request_ttfb_ms": ttft_ms,
            "uniapi_ember_inflight_requests": _runtime_int(runtime_metrics, "inflight_requests"),
            "uniapi_ember_request_waiters": _runtime_int(runtime_metrics, "request_waiters"),
            "uniapi_ember_request_body_reserved_weighted_bytes": _runtime_int(
                runtime_metrics, "request_body_reserved_weighted_bytes"
            ),
            "uniapi_ember_upstream_response_reserved_weighted_bytes": _runtime_int(
                runtime_metrics, "upstream_response_reserved_weighted_bytes"
            ),
            "uniapi_ember_request_retained_reserved_weighted_bytes": _runtime_int(
                runtime_metrics, "request_retained_reserved_weighted_bytes"
            ),
            "uniapi_ember_request_deferred_memory_requests": _runtime_int(
                runtime_metrics, "request_deferred_memory_requests"
            ),
            "uniapi_ember_request_deferred_memory_weighted_bytes": _runtime_int(
                runtime_metrics, "request_deferred_memory_weighted_bytes"
            ),
            "uniapi_ember_waiting_first_byte": _runtime_int(runtime_metrics, "waiting_first_byte"),
            "uniapi_ember_event_loop_lag_ms": _runtime_int(runtime_metrics, "event_loop_lag_ms"),
            "uniapi_ember_client_pool_in_use": _runtime_int(runtime_metrics, "upstream_pool_in_use"),
            "uniapi_ember_client_pool_waiters": _runtime_int(
                runtime_metrics, "upstream_pool_waiters"
            ),
            "uniapi_ember_client_pool_wait_ms": _span_ms(spans, "upstream_pool_wait_ms"),
            "uniapi_ember_stream_queue_bytes": _runtime_int(
                runtime_metrics, "stream_queue_bytes"
            ),
            "uniapi_ember_stream_queue_waiting_putters": _runtime_int(
                runtime_metrics, "stream_queue_waiting_putters"
            ),
            "uniapi_ember_stream_buffer_reserved_bytes": _runtime_int(
                runtime_metrics, "stream_buffer_reserved_bytes"
            ),
            "uniapi_ember_stream_buffer_budget_waiters": _runtime_int(
                runtime_metrics, "stream_buffer_budget_waiters"
            ),
            "uniapi_ember_stream_parser_reserved_bytes": _runtime_int(
                runtime_metrics, "stream_parser_reserved_bytes"
            ),
            "uniapi_ember_stream_parser_rejected_total": _runtime_int(
                runtime_metrics, "stream_parser_rejected_total"
            ),
            "uniapi_ember_stream_queue_peak_bytes": _safe_int(
                current_info.get("stream_queue_peak_bytes"), 0
            ),
            "uniapi_ember_retry_total": retry_count,
            "uniapi_ember_provider_cooldown_total": cooldown_count,
            "uniapi_ember_upstream_errors_total": _actual_upstream_error_count(
                current_info
            ),
            "uniapi_ember_exposed_5xx_total": 1 if status_code >= 500 else 0,
            "uniapi_ember_request_admission_rejected_total": 1
            if _safe_bool(current_info.get("admission_rejected"))
            else 0,
            "uniapi_ember_stream_failures_total": 1
            if _is_stream_failure(current_info)
            else 0,
            "uniapi_ember_downstream_disconnects_total": 1
            if _safe_bool(current_info.get("downstream_disconnected"))
            else 0,
        },
    )
    return {"logs": logs, "traces": traces, "metrics": metrics}


def _upstream_attempt_log_events(
    timestamp: datetime,
    service_name: str,
    base: dict[str, str],
    current_info: dict[str, Any],
) -> list[dict[str, Any]]:
    attempts = current_info.get("upstream_attempts")
    if not isinstance(attempts, list):
        return []

    events: list[dict[str, Any]] = []
    for raw_attempt in attempts[:16]:
        if not isinstance(raw_attempt, dict):
            continue
        attempt_status = _safe_int(raw_attempt.get("status_code"), 0)
        attempt_provider = _safe_text(raw_attempt.get("provider"))
        attempt_error_type = _safe_text(raw_attempt.get("error_type"), max_len=80)
        stream_diagnostics = raw_attempt.get("stream_diagnostics")
        if not isinstance(stream_diagnostics, dict):
            stream_diagnostics = {}
        timeout_adjusted_from = _optional_int_text(raw_attempt.get("timeout_adjusted_from_seconds"))
        started_ms = _optional_int_text(raw_attempt.get("started_ms"))
        duration_ms = _optional_int_text(raw_attempt.get("duration_ms"))
        events.append(
            {
                "timestamp": _iso_timestamp(timestamp),
                "level": _event_level(attempt_status),
                "service": service_name,
                "trace_id": base.get("trace_id"),
                "request_id": base.get("request_id"),
                "event": "upstream_attempt",
                "event_type": "upstream_attempt",
                "source": service_name,
                "message": "uni-api-ember upstream attempt",
                "attributes": _drop_empty(
                    {
                        **base,
                        "provider": attempt_provider,
                        "channel": attempt_provider,
                        "model": _safe_text(raw_attempt.get("model")) or base.get("model"),
                        "actual_model": _safe_text(raw_attempt.get("actual_model")),
                        "engine": _safe_text(raw_attempt.get("engine")),
                        "upstream_host": _safe_text(raw_attempt.get("upstream_host")),
                        "attempt_index": _int_text(_safe_int(raw_attempt.get("index"), 0)),
                        "attempt_status_code": _int_text(attempt_status),
                        "attempt_status_class": _status_class(attempt_status),
                        "attempt_success": _bool_text(_safe_bool(raw_attempt.get("success"))),
                        "attempt_error_type": attempt_error_type,
                        "payload_bytes": _int_text(_safe_int(raw_attempt.get("payload_bytes"), 0)),
                        "timeout_seconds": _int_text(_safe_int(raw_attempt.get("timeout_seconds"), 0)),
                        "timeout_adjusted_from_seconds": timeout_adjusted_from,
                        "wants_compact": _bool_text(_safe_bool(raw_attempt.get("wants_compact"))),
                        "stream": _bool_text(_safe_bool(raw_attempt.get("stream"))),
                        "started_ms": started_ms,
                        "duration_ms": duration_ms,
                        "semantic_status": _safe_text(
                            stream_diagnostics.get("semantic_status")
                        ),
                        "diagnosis": _safe_text(
                            stream_diagnostics.get("diagnosis")
                        ),
                        "failure_stage": _safe_text(
                            stream_diagnostics.get("failure_stage")
                        ),
                        "oaix_connection_id": _safe_text(
                            stream_diagnostics.get("oaix_connection_id")
                        ),
                        "upstream_http_version": _safe_text(
                            stream_diagnostics.get("http_version")
                        ),
                        "httpcore_stream_id": _optional_int_text(
                            stream_diagnostics.get("httpcore_stream_id")
                        ),
                        "explicit_proxy_configured": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "explicit_proxy_configured"
                                )
                            )
                        ),
                        "transport_local_endpoint_hmac": _safe_text(
                            stream_diagnostics.get(
                                "transport_local_endpoint_hmac"
                            )
                        ),
                        "transport_peer_endpoint_hmac": _safe_text(
                            stream_diagnostics.get(
                                "transport_peer_endpoint_hmac"
                            )
                        ),
                        "transport_four_tuple_hmac": _safe_text(
                            stream_diagnostics.get("transport_four_tuple_hmac")
                        ),
                        "transport_socket_hmac": _safe_text(
                            stream_diagnostics.get("transport_socket_hmac")
                        ),
                        "transport_local_family": _safe_text(
                            stream_diagnostics.get("transport_local_family")
                        ),
                        "transport_peer_family": _safe_text(
                            stream_diagnostics.get("transport_peer_family")
                        ),
                        "upstream_body_bytes": _optional_int_text(
                            stream_diagnostics.get("upstream_body_bytes")
                        ),
                        "upstream_chunk_count": _optional_int_text(
                            stream_diagnostics.get("upstream_chunk_count")
                        ),
                        "complete_event_count": _optional_int_text(
                            stream_diagnostics.get("complete_event_count")
                        ),
                        "last_event_type": _safe_text(
                            stream_diagnostics.get("last_event_type")
                        ),
                        "last_event_ordinal": _optional_int_text(
                            stream_diagnostics.get("last_event_ordinal")
                        ),
                        "last_event_bytes": _optional_int_text(
                            stream_diagnostics.get("last_event_bytes")
                        ),
                        "last_event_sha256": _safe_text(
                            stream_diagnostics.get("last_event_sha256")
                        ),
                        "partial_event_bytes": _optional_int_text(
                            stream_diagnostics.get("partial_event_bytes")
                        ),
                        "partial_event_sha256": _safe_text(
                            stream_diagnostics.get("partial_event_sha256")
                        ),
                        "event_hash_scope": _safe_text(
                            stream_diagnostics.get("hash_scope")
                        ),
                        "partial_hash_scope": _safe_text(
                            stream_diagnostics.get("partial_hash_scope")
                        ),
                        "upstream_eof_seen": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get("upstream_eof_seen")
                            )
                        ),
                        "upstream_terminal_seen": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "upstream_terminal_seen"
                                )
                            )
                        ),
                        "upstream_terminal_validated": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "upstream_terminal_validated"
                                )
                            )
                        ),
                        "terminal_frame_seen": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get("terminal_frame_seen")
                            )
                        ),
                        "declared_terminal_type": _safe_text(
                            stream_diagnostics.get("declared_terminal_type")
                        ),
                        "declared_terminal_ordinal": _optional_int_text(
                            stream_diagnostics.get("declared_terminal_ordinal")
                        ),
                        "declared_terminal_bytes": _optional_int_text(
                            stream_diagnostics.get("declared_terminal_bytes")
                        ),
                        "declared_terminal_sha256": _safe_text(
                            stream_diagnostics.get("declared_terminal_sha256")
                        ),
                        "semantic_terminal_type": _safe_text(
                            stream_diagnostics.get("semantic_terminal_type")
                        ),
                        "semantic_terminal_outcome": _safe_text(
                            stream_diagnostics.get("semantic_terminal_outcome")
                        ),
                        "semantic_terminal_bytes": _optional_int_text(
                            stream_diagnostics.get("semantic_terminal_bytes")
                        ),
                        "semantic_terminal_sha256": _safe_text(
                            stream_diagnostics.get("semantic_terminal_sha256")
                        ),
                        "semantic_terminal_sequence_number": _optional_int_text(
                            stream_diagnostics.get(
                                "semantic_terminal_sequence_number"
                            )
                        ),
                        "downstream_terminal_seen": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "downstream_terminal_seen"
                                )
                            )
                        ),
                        "ember_queue_terminal_handoff_completed": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "ember_queue_terminal_handoff_completed"
                                )
                            )
                        ),
                        "downstream_terminal_asgi_write_completed": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "downstream_terminal_asgi_write_completed"
                                )
                            )
                        ),
                        "error_event_seen": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get("error_event_seen")
                            )
                        ),
                        "usage_seen": _bool_text(
                            _safe_bool(stream_diagnostics.get("usage_seen"))
                        ),
                        "exception_type": _safe_text(
                            stream_diagnostics.get("exception_type"),
                            max_len=80,
                        ),
                        "exception_origin": _safe_text(
                            stream_diagnostics.get("exception_origin"),
                            max_len=80,
                        ),
                        "exception_errno": _optional_int_text(
                            stream_diagnostics.get("exception_errno")
                        ),
                        "exception_errno_name": _safe_text(
                            stream_diagnostics.get("exception_errno_name"),
                            max_len=80,
                        ),
                        "exception_chain_depth": _optional_int_text(
                            stream_diagnostics.get("exception_chain_depth")
                        ),
                        "exception_chain_truncated": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "exception_chain_truncated"
                                )
                            )
                        ),
                        "exception_chain_json": _diagnostic_json(
                            stream_diagnostics.get("exception_chain")
                        ),
                        "httpcore_events_json": _diagnostic_json(
                            stream_diagnostics.get("httpcore_events")
                        ),
                        "httpcore_response_close_trigger": _safe_text(
                            stream_diagnostics.get(
                                "httpcore_response_close_trigger"
                            )
                        ),
                        "cleanup_owner": _safe_text(
                            stream_diagnostics.get("cleanup_owner")
                        ),
                        "cleanup_trigger": _safe_text(
                            stream_diagnostics.get("cleanup_trigger")
                        ),
                        "cleanup_method": _safe_text(
                            stream_diagnostics.get("cleanup_method")
                        ),
                        "cleanup_result": _safe_text(
                            stream_diagnostics.get("cleanup_result")
                        ),
                        "cleanup_transport_evicted": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "cleanup_transport_evicted"
                                )
                            )
                        ),
                        "cleanup_transport_safe": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get("cleanup_transport_safe")
                            )
                        ),
                        "cleanup_detached": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "cleanup_detached_cleanup"
                                )
                            )
                        ),
                        "pool_sweeper_close_observed": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "pool_sweeper_close_observed"
                                )
                            )
                        ),
                        "pool_sweeper_trigger": _safe_text(
                            stream_diagnostics.get("pool_sweeper_trigger")
                        ),
                        **_responses_diagnostic_attrs(
                            stream_diagnostics,
                            current_info=current_info,
                        ),
                    }
                ),
            }
        )
    return events


def _stage_rows(spans: dict[str, Any], duration_ms: int | None) -> list[tuple[str, int, dict[str, str]]]:
    rows: list[tuple[str, int, dict[str, str]]] = []
    previous_stage = ""
    for stage in _STAGE_ORDER:
        if stage == "client_pool_acquired":
            stage_ms = _span_ms(spans, "upstream_pool_wait_ms")
            attrs = {
                "client_pool_acquire_start_ms": _int_text(_span_ms(spans, "client_pool_acquire_start")),
                "client_pool_acquire_end_ms": _int_text(_span_ms(spans, "client_pool_acquire_end")),
            }
        elif stage == "retry_started":
            stage_ms = _stage_delta_ms(spans, stage, previous_stage)
            attrs = {
                "retry_count": _int_text(_span_ms(spans, "retry_count")),
                "retry_status_code": _int_text(_span_ms(spans, "retry_status_code")),
                "retry_provider": _safe_text(spans.get("retry_provider")),
            }
        elif stage == "stream_end" and _span_ms(spans, stage) <= 0 and duration_ms is not None:
            stage_ms = max(0, int(duration_ms))
            attrs = {}
        elif stage == "upstream_first_chunk":
            stage_ms = _span_ms(spans, stage)
            attrs = {}
        else:
            stage_ms = _stage_delta_ms(spans, stage, previous_stage)
            attrs = {}
        rows.append((stage, max(0, int(stage_ms or 0)), attrs))
        if _span_ms(spans, stage) > 0 or stage == "request_received":
            previous_stage = stage
    return rows


def _request_metric_events(
    *,
    service_name: str,
    identity_attrs: dict[str, str] | None,
    timestamp: datetime,
    method: str | None,
    status_code: int,
    route_id: str | None,
    values: dict[str, int | None],
) -> list[dict[str, Any]]:
    request_attrs = _drop_empty(
        {
            **(identity_attrs or {}),
            "component": service_name,
            "route_id": route_id,
            "method": method,
            "status_class": _status_class(status_code),
        }
    )
    global_attrs = _drop_empty(
        {
            **(identity_attrs or {}),
            "component": service_name,
        }
    )
    global_metrics = {
        "uniapi_ember_inflight_requests",
        "uniapi_ember_request_waiters",
        "uniapi_ember_request_body_reserved_weighted_bytes",
        "uniapi_ember_upstream_response_reserved_weighted_bytes",
        "uniapi_ember_request_retained_reserved_weighted_bytes",
        "uniapi_ember_request_deferred_memory_requests",
        "uniapi_ember_request_deferred_memory_weighted_bytes",
        "uniapi_ember_waiting_first_byte",
        "uniapi_ember_event_loop_lag_ms",
        "uniapi_ember_client_pool_in_use",
        "uniapi_ember_client_pool_waiters",
        "uniapi_ember_stream_queue_bytes",
        "uniapi_ember_stream_queue_waiting_putters",
        "uniapi_ember_stream_buffer_reserved_bytes",
        "uniapi_ember_stream_buffer_budget_waiters",
        "uniapi_ember_stream_parser_reserved_bytes",
    }
    events = []
    for metric, value in values.items():
        if value is None:
            continue
        events.append(
            {
                "timestamp": _iso_timestamp(timestamp),
                "kind": "metric",
                "source": service_name,
                "message": metric,
                "metric": metric,
                "value": max(0, int(value)),
                "attributes": global_attrs if metric in global_metrics else request_attrs,
            }
        )
    return events


def _actual_upstream_error_count(current_info: dict[str, Any]) -> int | None:
    attempts = current_info.get("upstream_attempts")
    if not isinstance(attempts, list):
        # Legacy routes do not yet expose unified attempt facts.  Omitting the
        # metric is truthful; emitting zero would silently claim knowledge we
        # do not have.
        return None
    count = 0
    for attempt in attempts[:16]:
        if not isinstance(attempt, dict) or _safe_bool(attempt.get("success")):
            continue
        if _safe_bool(attempt.get("local_admission_rejected")):
            continue
        status_code = _safe_int(attempt.get("status_code"), 0)
        error_type = _safe_text(attempt.get("error_type")) or ""
        if error_type in {"UpstreamAdmissionRejected", "StreamBufferBudgetTimeout"}:
            continue
        if status_code >= 500:
            count += 1
    return count


def _responses_stream_diagnostics(current_info: dict[str, Any]) -> dict[str, Any]:
    diagnostics = current_info.get("responses_stream_diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else {}


def _is_responses_request(current_info: dict[str, Any]) -> bool:
    """Responses correlation logs are never sampled away.

    A request may look completely healthy inside Ember while a downstream
    consumer fails to parse its terminal usage.  Retaining every compact
    Responses summary is what makes a later 0-0 ``200 / unknown usage`` row
    joinable by request/connection/event hashes.
    """

    _method, path = _split_endpoint(_safe_text(current_info.get("endpoint")))
    if not path:
        return False
    normalized = path.split("?", 1)[0].rstrip("/")
    return normalized in {"/v1/responses", "/v1/responses/compact"}


def _responses_token_text(
    current_info: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    value_key: str,
    known_key: str,
) -> str | None:
    if _is_responses_request(current_info) and not diagnostics:
        if _safe_bool(current_info.get("usage_seen")) is not True:
            return None
    if diagnostics:
        if _safe_bool(diagnostics.get(known_key)) is not True:
            return None
        if _safe_bool(diagnostics.get("downstream_usage_values_valid")) is not True:
            return None
        if _safe_bool(diagnostics.get("downstream_usage_alias_consistent")) is False:
            return None
    return _optional_int_text(current_info.get(value_key))


def _responses_failure_stage(
    current_info: dict[str, Any],
    diagnostics: dict[str, Any],
) -> str | None:
    if not diagnostics:
        return None
    explicit_stage = (_safe_text(diagnostics.get("failure_stage")) or "").lower()
    if explicit_stage in {"precommit", "postcommit", "downstream", "cleanup"}:
        return explicit_stage
    if explicit_stage in {
        "upstream",
        "upstream_headers",
        "upstream_response_headers",
        "headers",
    } or explicit_stage.startswith("precommit"):
        return "precommit"
    if explicit_stage.startswith("postcommit"):
        return "postcommit"
    if explicit_stage.startswith("downstream"):
        return "downstream"
    if explicit_stage.startswith("cleanup"):
        return "cleanup"
    if diagnostics.get("cleanup_result") == "incomplete":
        return "cleanup"
    outcome = _safe_text(current_info.get("stream_outcome")) or ""
    if outcome in {
        "downstream_disconnected",
        "downstream_write_timeout",
        "downstream_send_error",
        "local_backpressure_abort",
    }:
        return "downstream"
    origin = _safe_text(diagnostics.get("exception_origin")) or ""
    if origin.startswith("precommit") or origin == "upstream_response_headers":
        return "precommit"
    if diagnostics.get("exception_type") or diagnostics.get(
        "failure_terminal_seen"
    ):
        return "postcommit" if current_info.get("response_committed") else "precommit"
    return None


def _diagnostic_json(value: Any, *, max_len: int = 4096) -> str | None:
    encoded, _truncated, _count, _digest = _diagnostic_json_details(
        value,
        max_len=max_len,
    )
    return encoded


def _diagnostic_json_details(
    value: Any,
    *,
    max_len: int = 4096,
) -> tuple[str | None, bool, int | None, str | None]:
    if value in (None, [], {}):
        return None, False, None, None
    try:
        full = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return None, False, None, None

    full_bytes = full.encode("utf-8")
    digest = hashlib.sha256(full_bytes).hexdigest()
    count = len(value) if isinstance(value, (list, dict)) else 1
    if len(full_bytes) <= max_len:
        return full, False, count, digest

    if isinstance(value, list):
        bounded: Any = []
        for item in value:
            candidate = [*bounded, item]
            try:
                rendered = json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                break
            if len(rendered.encode("utf-8")) > max_len:
                break
            bounded = candidate
    elif isinstance(value, dict):
        bounded = {}
        for key in sorted(value, key=lambda candidate: str(candidate)):
            candidate = {**bounded, key: value[key]}
            try:
                rendered = json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                break
            if len(rendered.encode("utf-8")) > max_len:
                break
            bounded = candidate
    else:
        bounded = {"truncated": True, "value_type": type(value).__name__}

    encoded = json.dumps(
        bounded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > max_len:
        encoded = "[]" if isinstance(value, list) else "{}"
    return encoded, True, count, digest


def _diagnostic_json_attrs(
    prefix: str,
    value: Any,
    *,
    source_truncated: bool = False,
) -> dict[str, str | None]:
    encoded, bounded_truncated, count, digest = _diagnostic_json_details(value)
    if encoded is None:
        return {}
    return {
        f"{prefix}_json": encoded,
        f"{prefix}_json_truncated": _bool_text(
            bool(source_truncated or bounded_truncated)
        ),
        f"{prefix}_original_count": _optional_int_text(count),
        f"{prefix}_json_sha256": digest,
    }


def _responses_diagnostic_attrs(
    diagnostics: dict[str, Any],
    *,
    current_info: dict[str, Any],
) -> dict[str, str]:
    if not diagnostics:
        return {}

    attrs: dict[str, Any] = {
        "responses_diagnostic_schema_version": _optional_int_text(
            diagnostics.get("schema_version")
        ),
        "semantic_status": _safe_text(diagnostics.get("semantic_status")),
        "diagnosis": _safe_text(diagnostics.get("diagnosis")),
        "failure_stage": _responses_failure_stage(current_info, diagnostics),
        "terminal_consistency_status": _safe_text(
            diagnostics.get("terminal_consistency_status")
        ),
        "declared_terminal_type": _safe_text(
            diagnostics.get("declared_terminal_type")
        ),
        "semantic_terminal_type": _safe_text(
            diagnostics.get("semantic_terminal_type")
        ),
        "semantic_terminal_outcome": _safe_text(
            diagnostics.get("semantic_terminal_outcome")
        ),
        "downstream_declared_terminal_type": _safe_text(
            diagnostics.get("downstream_declared_terminal_type")
        ),
        "downstream_semantic_status": _safe_text(
            diagnostics.get("downstream_semantic_status")
        ),
        "oaix_connection_id": _safe_text(
            diagnostics.get("oaix_connection_id")
        ),
        "upstream_http_version": _safe_text(diagnostics.get("http_version")),
        "transport_error_code": _safe_text(
            diagnostics.get("transport_error_code")
        ),
        "transport_error_code_source": _safe_text(
            diagnostics.get("transport_error_code_source")
        ),
        "transport_end_trigger": _safe_text(
            diagnostics.get("transport_end_trigger")
        ),
        "local_end_origin": _safe_text(diagnostics.get("local_end_origin")),
        "exception_type": _safe_text(
            diagnostics.get("exception_type"), max_len=80
        ),
        "exception_origin": _safe_text(
            diagnostics.get("exception_origin"), max_len=80
        ),
        "exception_errno_name": _safe_text(
            diagnostics.get("exception_errno_name"), max_len=80
        ),
        "httpcore_body_read_failure_type": _safe_text(
            diagnostics.get("httpcore_body_read_failure_type"), max_len=80
        ),
        "httpcore_response_close_trigger": _safe_text(
            diagnostics.get("httpcore_response_close_trigger")
        ),
        "downstream_usage_observer_status": _safe_text(
            diagnostics.get("downstream_usage_observer_status")
        ),
        "downstream_usage_observer_error_type": _safe_text(
            diagnostics.get("downstream_usage_observer_error_type"), max_len=80
        ),
        "downstream_usage_observer_abort_reason": _safe_text(
            diagnostics.get("downstream_usage_observer_abort_reason"), max_len=80
        ),
        "downstream_usage_completeness": _safe_text(
            diagnostics.get("downstream_usage_completeness")
        ),
        "downstream_failure_outcome": _safe_text(
            diagnostics.get("downstream_failure_outcome")
        ),
        "response_start_asgi_write_outcome": _safe_text(
            diagnostics.get("response_start_asgi_write_outcome")
        ),
        "response_start_asgi_write_error_type": _safe_text(
            diagnostics.get("response_start_asgi_write_error_type"), max_len=80
        ),
        "downstream_final_body_outcome": _safe_text(
            diagnostics.get("downstream_final_body_outcome")
        ),
        "downstream_final_body_error_type": _safe_text(
            diagnostics.get("downstream_final_body_error_type"), max_len=80
        ),
        "downstream_final_body_skip_reason": _safe_text(
            diagnostics.get("downstream_final_body_skip_reason"), max_len=80
        ),
        "cleanup_owner": _safe_text(diagnostics.get("cleanup_owner")),
        "cleanup_trigger": _safe_text(diagnostics.get("cleanup_trigger")),
        "cleanup_method": _safe_text(diagnostics.get("cleanup_method")),
        "cleanup_result": _safe_text(diagnostics.get("cleanup_result")),
        "cleanup_transport_action_actor": _safe_text(
            diagnostics.get("cleanup_transport_action_actor")
        ),
        "cleanup_transport_result_actor": _safe_text(
            diagnostics.get("cleanup_transport_result_actor")
        ),
        "cleanup_context_exit_actor": _safe_text(
            diagnostics.get("cleanup_context_exit_actor")
        ),
        "cleanup_failure_stage": _safe_text(
            diagnostics.get("cleanup_failure_stage")
        ),
        "pool_sweeper_trigger": _safe_text(
            diagnostics.get("pool_sweeper_trigger")
        ),
    }

    bool_fields = (
        "transport_metadata_available",
        "response_start_asgi_write_attempted",
        "response_start_asgi_write_completed",
        "upstream_eof_seen",
        "terminal_frame_seen",
        "terminal_frame_structured",
        "terminal_semantics_consistent",
        "upstream_terminal_seen",
        "upstream_terminal_validated",
        "response_completed_validated",
        "response_incomplete_validated",
        "failure_terminal_validated",
        "usage_object_seen",
        "usage_counters_seen",
        "usage_input_known",
        "usage_output_known",
        "usage_total_known",
        "usage_values_valid",
        "usage_alias_consistent",
        "usage_seen",
        "ember_queue_terminal_handoff_completed",
        "downstream_terminal_seen",
        "downstream_terminal_asgi_write_completed",
        "error_event_seen",
        "downstream_usage_object_seen",
        "downstream_usage_counters_seen",
        "downstream_usage_input_known",
        "downstream_usage_output_known",
        "downstream_usage_total_known",
        "downstream_usage_values_valid",
        "downstream_usage_alias_consistent",
        "downstream_usage_seen",
        "downstream_final_body_attempted",
        "downstream_final_body_completed",
        "local_cleanup_claimed_before_body_read_failure",
        "httpcore_events_truncated",
        "cleanup_transport_evicted",
        "cleanup_transport_isolated",
        "cleanup_transport_safe",
        "cleanup_context_exit_succeeded",
        "cleanup_detached_cleanup",
        "cleanup_actions_truncated",
        "cleanup_failure",
        "pool_sweeper_close_observed",
        "pool_sweeper_close_succeeded",
    )
    for key in bool_fields:
        attrs[key] = _bool_text(_safe_bool(diagnostics.get(key)))

    int_fields = (
        "httpcore_stream_id",
        "upstream_body_bytes",
        "upstream_chunk_count",
        "complete_event_count",
        "last_event_ordinal",
        "last_event_bytes",
        "partial_event_bytes",
        "declared_terminal_ordinal",
        "declared_terminal_bytes",
        "semantic_terminal_bytes",
        "semantic_terminal_sequence_number",
        "exception_errno",
        "exception_chain_depth",
        "cleanup_attempt_count",
    )
    for key in int_fields:
        attrs[key] = _optional_int_text(diagnostics.get(key))

    text_fields = (
        "last_event_type",
        "last_event_sha256",
        "partial_event_sha256",
        "hash_scope",
        "partial_hash_scope",
        "declared_terminal_sha256",
        "semantic_terminal_sha256",
        "first_upstream_body_at",
        "response_start_asgi_write_attempted_at",
        "response_start_asgi_write_completed_at",
        "response_start_asgi_write_error_at",
        "last_event_received_at",
        "declared_terminal_received_at",
        "terminal_frame_structured_at",
        "semantic_terminal_classified_at",
        "upstream_eof_at",
        "exception_at",
        "local_end_at",
        "ember_queue_terminal_handoff_completed_at",
        "downstream_terminal_asgi_write_completed_at",
        "downstream_failure_at",
        "downstream_usage_observer_aborted_at",
        "downstream_final_body_attempted_at",
        "downstream_final_body_completed_at",
        "downstream_final_body_error_at",
        "httpcore_body_read_failed_at",
        "httpcore_body_read_cancelled_at",
        "httpcore_response_close_started_at",
        "httpcore_response_close_completed_at",
        "httpcore_response_close_failed_at",
        "cleanup_started_at",
        "cleanup_completed_at",
        "pool_sweeper_close_started_at",
        "pool_sweeper_close_completed_at",
    )
    for key in text_fields:
        attrs[key] = _safe_text(diagnostics.get(key))

    attrs.update(
        _diagnostic_json_attrs(
            "terminal_semantics_inconsistency",
            diagnostics.get("terminal_semantics_inconsistency"),
        )
    )
    attrs.update(
        _diagnostic_json_attrs(
            "exception_chain",
            diagnostics.get("exception_chain"),
            source_truncated=bool(diagnostics.get("exception_chain_truncated")),
        )
    )
    attrs.update(
        _diagnostic_json_attrs(
            "httpcore_events",
            diagnostics.get("httpcore_events"),
            source_truncated=bool(diagnostics.get("httpcore_events_truncated")),
        )
    )
    attrs.update(
        _diagnostic_json_attrs(
            "cleanup_actions",
            diagnostics.get("cleanup_actions"),
            source_truncated=bool(diagnostics.get("cleanup_actions_truncated")),
        )
    )
    return _drop_empty(attrs)


def _is_stream_failure(current_info: dict[str, Any]) -> bool:
    outcome = _safe_text(current_info.get("stream_outcome")) or ""
    return bool(
        outcome
        and outcome not in {"completed", "downstream_disconnected"}
    )


def _base_attrs(
    *,
    service_name: str,
    service_version: str | None,
    identity_attrs: dict[str, str] | None,
    trace_id: str | None,
    request_id: str | None,
    parent_span_id: str | None,
    endpoint: str | None,
    method: str | None,
    path_template: str | None,
    route_id: str | None,
    model: str | None,
    provider: str | None,
    role: str | None,
    is_stream: bool | None,
    status_code: int,
    error_type: str | None,
    retry_count: int,
    cooldown_count: int,
    api_key_hash: str | None,
) -> dict[str, str]:
    return _drop_empty(
        {
            **(identity_attrs or {}),
            "service": service_name,
            "component": service_name,
            "service_version": _safe_text(service_version),
            "trace_id": _safe_text(trace_id),
            "request_id": _safe_text(request_id),
            "parent_span_id": _safe_text(parent_span_id),
            "route": _safe_text(endpoint),
            "route_id": route_id,
            "path_template": _safe_text(path_template or endpoint),
            "method": _safe_text(method),
            "request_kind": _safe_text(path_template or endpoint),
            "model": _safe_text(model),
            "provider": _safe_text(provider),
            "channel": _safe_text(provider),
            "role": _safe_text(role),
            "stream": _bool_text(is_stream),
            "streaming": _bool_text(is_stream),
            "status_code": _int_text(status_code),
            "status_class": _status_class(status_code),
            "error_type": error_type,
            "retry_count": _int_text(retry_count),
            "cooldown_count": _int_text(cooldown_count),
            "api_key_hash": api_key_hash,
        }
    )


def _identity_attrs_from_env() -> dict[str, str]:
    env_map = {
        "tenant_id": "FUGUE_OBSERVABILITY_TENANT_ID",
        "project_id": "FUGUE_OBSERVABILITY_PROJECT_ID",
        "app_id": "FUGUE_OBSERVABILITY_APP_ID",
        "runtime_id": "FUGUE_OBSERVABILITY_RUNTIME_ID",
        "pod": "HOSTNAME",
    }
    return _drop_empty({key: _env_text(env_name) for key, env_name in env_map.items()})


def _duration_ms_from_info(current_info: dict[str, Any]) -> int | None:
    process_time = current_info.get("process_time")
    try:
        if process_time is not None:
            return max(0, int(round(float(process_time) * 1000)))
    except (TypeError, ValueError):
        pass
    started_at = current_info.get("start_time")
    try:
        if started_at is not None:
            return max(0, int(round((time() - float(started_at)) * 1000)))
    except (TypeError, ValueError):
        pass
    return None


def _ttft_ms(spans: dict[str, Any]) -> int | None:
    value = _span_ms(spans, "upstream_first_chunk")
    if value > 0:
        return value
    value = _span_ms(spans, "upstream_headers_received")
    return value if value > 0 else None


def _stage_delta_ms(spans: dict[str, Any], stage: str, previous_stage: str) -> int:
    current = _span_ms(spans, stage)
    if current <= 0:
        return 0
    previous = _span_ms(spans, previous_stage)
    return current if previous <= 0 else max(0, current - previous)


def _runtime_int(runtime_metrics: dict[str, Any] | None, key: str) -> int | None:
    if not runtime_metrics:
        return None
    value = runtime_metrics.get(key)
    if value is None:
        return None
    return _safe_int(value, 0)


def _span_ms(spans: dict[str, Any], name: str) -> int:
    value = spans.get(name)
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return 0


def _split_endpoint(endpoint: str | None) -> tuple[str | None, str | None]:
    text = _safe_text(endpoint)
    if not text:
        return None, None
    parts = text.split(" ", 1)
    if len(parts) == 2 and parts[0].isalpha():
        return parts[0].upper(), parts[1].strip() or None
    return None, text


def _route_id(endpoint: str | None) -> str | None:
    _, path = _split_endpoint(endpoint)
    if not path:
        return None
    route = path.split("?", 1)[0].strip().rstrip("/") or "/"
    return route[:160]


def _endpoint_url(endpoint: str, path: str) -> str:
    base = endpoint.strip().rstrip("/")
    if base.endswith(("/v1/logs", "/v1/metrics", "/v1/traces")):
        base = base.rsplit("/v1/", 1)[0]
    return base + path


def _env_text(name: str) -> str | None:
    value = str(os.getenv(name, "")).strip()
    return value or None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "")).strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, "")).strip() or default)
    except ValueError:
        return default


def _safe_text(value: Any, *, max_len: int = 256) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:max_len]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _bool_text(value: bool | None) -> str | None:
    if value is None:
        return None
    return "true" if value else "false"


def _int_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(max(0, int(value)))
    except (TypeError, ValueError):
        return None


def _optional_int_text(value: Any) -> str | None:
    if value is None:
        return None
    return _int_text(value)


def _iso_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _status_class(status_code: int) -> str:
    if status_code <= 0:
        return "unknown"
    return f"{status_code // 100}xx"


def _event_level(status_code: int) -> str:
    if status_code >= 500:
        return "error"
    if status_code >= 400:
        return "warning"
    return "info"


def _classify_error(status_code: int) -> str | None:
    if status_code <= 0 or status_code < 400:
        return None
    if status_code == 499:
        return "client_closed"
    if status_code == 429:
        return "rate_limited"
    if 400 <= status_code < 500:
        return "client_error"
    if status_code >= 500:
        return "upstream_or_server_error"
    return "error"


def _secret_hash(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _span_id(trace_id: str | None, request_id: str | None, stage: str) -> str:
    seed = "|".join([_safe_text(trace_id) or "", _safe_text(request_id) or "", stage])
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _drop_empty(values: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        result[str(key)] = text
    return result
