from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from dataclasses import dataclass
from time import time
from typing import Any, Awaitable, Callable, Optional

from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from uni_api.admission.json_parsing import run_json_cpu
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse as StarletteStreamingResponse

from core.log_config import logger
from core.models import ModerationRequest
from core.utils import safe_get
from uni_api.admission import AdmissionRejected
from uni_api.http_content import is_json_media_type
from uni_api.middleware.request_decompression import (
    RequestBodyTooComplex,
    RequestBodyTooLarge,
)
from uni_api.middleware.request_decompression import (
    DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY,
    RequestBodyDisconnected,
    RequestBodyReadTimeout,
)
from uni_api.serialization import json
from uni_api.observability.request_context import (
    RequestContext,
    get_request_info,
    reset_request_info,
    set_request_info,
)
from uni_api.observability.request_inspection import inspect_request_body
from uni_api.observability.spans import merge_timing_spans
from uni_api.streaming.logging_response import (
    LoggingStreamingResponse,
    await_with_hard_deadline,
)
from uni_api.streaming.sse import DEFAULT_MAX_PENDING_BYTES
from uni_api.upstream.client_pool import UpstreamAdmissionRejected


@dataclass(frozen=True)
class StatsMiddlewareDependencies:
    app_state: Any
    database_disabled: bool
    runtime_gauges: Any
    trace_factory: Callable[..., Any]
    incoming_trace_context: Callable[[Any], dict[str, str]]
    get_api_key: Callable[[Request], Awaitable[Optional[str]]]
    get_client_ip: Callable[[Request], str]
    parse_request_body: Callable[[Request], Awaitable[Any]]
    message_role_summary: Callable[[Any], tuple[Optional[str], Optional[str]]]
    messages_request_last_text: Callable[[Any], Optional[str]]
    is_public_health_request: Callable[[Request], bool]
    is_video_or_asset_request_path: Callable[[str], bool]
    lingjing_request_model_for_openapi: Callable[[Optional[dict[str, Any]], Any], str]
    video_prompt_from_body: Callable[[dict[str, Any]], str]
    monitor_disconnect: Callable[[Request, asyncio.Event], Awaitable[None]]
    log_debug_request_headers: Callable[..., None]
    log_debug_request_body: Callable[..., None]
    mask_secret_for_log: Callable[[Any], str]
    update_stats: Callable[[dict[str, Any]], Awaitable[None]]
    emit_request_observability: Callable[[dict[str, Any]], None]
    mark_first_byte_observed: Callable[[dict[str, Any]], None]
    moderation_handler: Callable[[ModerationRequest, BackgroundTasks, int], Awaitable[Any]]
    responses_usage_buffer_limit_bytes: int = DEFAULT_MAX_PENDING_BYTES
    logging_response_class: type[LoggingStreamingResponse] = LoggingStreamingResponse
    debug: bool = False


class _DisconnectFanoutReceive:
    """Serialize the post-body receive channel and broadcast disconnect.

    ``BaseHTTPMiddleware`` gives the downstream app a cached-body receive
    wrapper, while the request disconnect monitor historically read the raw
    receive channel directly.  After the body was consumed, both could race
    for the single ``http.disconnect`` message.  This adapter is installed
    only after body parsing has completed; non-disconnect messages keep their
    one-consumer semantics, while a disconnect becomes sticky for every
    subsequent receiver.
    """

    def __init__(self, receive: Callable[[], Awaitable[dict[str, Any]]], event: asyncio.Event) -> None:
        self._receive = receive
        self._event = event
        self._lock = asyncio.Lock()

    async def __call__(self) -> dict[str, Any]:
        if self._event.is_set():
            return {"type": "http.disconnect"}
        async with self._lock:
            if self._event.is_set():
                return {"type": "http.disconnect"}
            message = await self._receive()
            if message.get("type") == "http.disconnect":
                self._event.set()
            return message


class StatsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, dependencies: StatsMiddlewareDependencies):
        super().__init__(app)
        self.dependencies = dependencies

    def _forbidden_response(self, trace) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={"error": "Invalid or missing API Key"},
            headers={"x-request-id": trace.trace_id},
        )

    def _api_index_for_token(self, token: str) -> int | None:
        api_key_index = getattr(self.dependencies.app_state, "api_key_index", None)
        if api_key_index is None:
            api_list = getattr(self.dependencies.app_state, "api_list", []) or []
            api_key_index = {api_key: index for index, api_key in enumerate(api_list)}
            self.dependencies.app_state.api_key_index = api_key_index
        return api_key_index.get(token)

    def _request_context(self, request: Request, *, trace, incoming_trace: dict[str, str], start_time: float, token: str) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        return RequestContext(
            request_id=request_id,
            trace_id=trace.trace_id,
            span_id=trace.span_id,
            parent_span_id=trace.parent_span_id,
            trace_flags=trace.trace_flags,
            tracestate=trace.tracestate,
            x_request_id=incoming_trace.get("x_request_id"),
            start_time=start_time,
            endpoint=f"{request.method} {request.url.path}",
            client_ip=self.dependencies.get_client_ip(request),
            api_key=token,
            timing_spans=trace.snapshot(),
            extras={"trace": trace},
        ).to_dict()

    def _paid_key_enabled_response(self, token: str, api_index: int, request: Request, trace) -> JSONResponse | None:
        deps = self.dependencies
        if deps.database_disabled or request.url.path.startswith("/v1/token_usage"):
            return None
        check_api_key = safe_get(deps.app_state.config, "api_keys", api_index, "api")
        if safe_get(getattr(deps.app_state, "paid_api_keys_states", {}), check_api_key, "enabled", default=None) is not False:
            return None
        _ = token
        return JSONResponse(
            status_code=429,
            content={"error": "Balance is insufficient, please check your account."},
            headers={"x-request-id": trace.trace_id},
        )

    async def _parse_and_log_body(self, request: Request, request_id: str, trace) -> Any:
        deps = self.dependencies
        deps.log_debug_request_headers(
            "DEBUG client request headers",
            request.headers,
            method=request.method,
            endpoint=request.url.path,
            request_id=request_id,
        )
        parsed_body = await deps.parse_request_body(request)
        trace.mark("body_parsed")
        if parsed_body is not None:
            deps.log_debug_request_body(
                "DEBUG client request body",
                parsed_body,
                method=request.method,
                endpoint=request.url.path,
                request_id=request_id,
            )
        return parsed_body

    async def _start_disconnect_monitor(self, request: Request, current_info: dict[str, Any]) -> tuple[Optional[asyncio.Event], Optional[asyncio.Task]]:
        shared_event = getattr(
            request.state,
            DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY,
            None,
        )
        if isinstance(shared_event, asyncio.Event):
            current_info["disconnect_event"] = shared_event
            return shared_event, None
        if request.method != "POST" or not is_json_media_type(
            request.headers.get("content-type", "")
        ):
            return None, None
        disconnect_event = asyncio.Event()
        current_info["disconnect_event"] = disconnect_event
        # ``Request.body()`` has completed before this method runs.  Replacing
        # the remaining raw receive channel avoids two independent consumers
        # stealing the one-shot disconnect from one another.
        request._receive = _DisconnectFanoutReceive(  # type: ignore[attr-defined]
            request.receive,
            disconnect_event,
        )
        return disconnect_event, asyncio.create_task(self.dependencies.monitor_disconnect(request, disconnect_event))

    async def _apply_body_policy(
        self,
        request: Request,
        *,
        parsed_body: Any,
        api_index: int,
        enable_moderation: bool,
        current_info: dict[str, Any],
        start_time: float,
    ) -> JSONResponse | None:
        if not parsed_body or request.url.path.startswith("/v1/api_config"):
            return None

        deps = self.dependencies
        final_api_key = deps.app_state.api_list[api_index]
        moderated_content = await self._rate_limit_and_extract_moderation_text(
            request,
            parsed_body=parsed_body,
            current_info=current_info,
            final_api_key=final_api_key,
        )
        if isinstance(moderated_content, JSONResponse):
            return moderated_content
        if enable_moderation and moderated_content:
            return await self._moderation_response_if_flagged(
                moderated_content,
                api_index=api_index,
                current_info=current_info,
                start_time=start_time,
            )
        return None

    async def _rate_limit_and_extract_moderation_text(
        self,
        request: Request,
        *,
        parsed_body: Any,
        current_info: dict[str, Any],
        final_api_key: str,
    ) -> str | JSONResponse | None:
        deps = self.dependencies
        if request.url.path.rstrip("/") == "/v1/messages":
            if isinstance(parsed_body, dict):
                model = str(parsed_body.get("model") or "").strip()
                if model:
                    current_info["model"] = model
                    limited_response = await self._rate_limit_response(final_api_key, model, current_info)
                    if limited_response is not None:
                        return limited_response
            return deps.messages_request_last_text(parsed_body)

        if deps.is_video_or_asset_request_path(request.url.path):
            model = deps.lingjing_request_model_for_openapi(
                parsed_body if isinstance(parsed_body, dict) else None,
                request.query_params,
            )
            current_info["model"] = model
            limited_response = await self._rate_limit_response(final_api_key, model, current_info)
            if limited_response is not None:
                return limited_response
            if isinstance(parsed_body, dict):
                moderated_content = str(safe_get(parsed_body, "taskParams", "input", "prompt", default="") or "").strip()
                return moderated_content or deps.video_prompt_from_body(parsed_body)
            return None

        inspection = inspect_request_body(parsed_body)
        model = inspection.model
        current_info["model"] = model
        if model:
            limited_response = await self._rate_limit_response(final_api_key, model, current_info)
            if limited_response is not None:
                return limited_response
        if inspection.request_type is None:
            logger.error("Unknown request type for middleware inspection: %s", request.url.path)
        return inspection.moderated_content

    async def _rate_limit_response(self, final_api_key: str, model: str, current_info: dict[str, Any]) -> JSONResponse | None:
        try:
            await self.dependencies.app_state.user_api_keys_rate_limit[final_api_key].next(model)
        except HTTPException as exc:
            if exc.status_code != 429:
                raise
            current_info["status_code"] = 429
            current_info["error_type"] = "rate_limited"
            return JSONResponse(status_code=429, content={"error": "Too many requests"})
        return None

    async def _moderation_response_if_flagged(
        self,
        moderated_content: str,
        *,
        api_index: int,
        current_info: dict[str, Any],
        start_time: float,
    ) -> JSONResponse | None:
        moderation_response = await self.moderate_content(moderated_content, api_index, BackgroundTasks())
        is_flagged = moderation_response.get("results", [{}])[0].get("flagged", False)
        if not is_flagged:
            return None
        logger.error("Content did not pass the moral check: %s", moderated_content)
        current_info["process_time"] = time() - start_time
        current_info["is_flagged"] = is_flagged
        current_info["text"] = moderated_content
        current_info["status_code"] = 400
        current_info["error_type"] = "moderation_flagged"
        try:
            persisted = await await_with_hard_deadline(
                self.dependencies.update_stats(current_info),
                timeout=5.0,
                label="moderation request stats write",
            )
            if persisted is False:
                current_info["stats_write_failed"] = True
        except asyncio.TimeoutError:
            current_info["stats_write_timeout"] = True
        return JSONResponse(
            status_code=400,
            content={"error": "Content did not pass the moral check, please modify and try again."},
        )

    async def _wrap_response_for_observability(
        self,
        request: Request,
        response,
        current_info: dict[str, Any],
        trace,
        *,
        lifecycle_close: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
    ):
        deps = self.dependencies
        response_info = getattr(response, "current_info", None)
        if isinstance(response_info, dict) and response_info is not current_info:
            current_info.update(response_info)
        response_trace = current_info.get("trace") if isinstance(current_info, dict) else None
        if response_trace is not None and hasattr(response_trace, "mark") and hasattr(response_trace, "trace_id"):
            trace = response_trace
        trace.mark("downstream_response_start")
        merge_timing_spans(current_info, trace.snapshot())
        response.headers["x-request-id"] = trace.trace_id
        current_info["status_code"] = getattr(response, "status_code", 0) or 0
        if isinstance(response, StarletteStreamingResponse) or type(response).__name__ == "_StreamingResponse":
            current_info["_defer_observability_until_stream_end"] = True
            return deps.logging_response_class(
                content=response.body_iterator,
                status_code=response.status_code,
                media_type=response.media_type,
                headers=response.headers,
                current_info=current_info,
                mark_first_byte_observed=deps.mark_first_byte_observed,
                emit_request_observability=deps.emit_request_observability,
                update_stats=None if deps.database_disabled else deps.update_stats,
                trace_type=type(trace),
                debug=deps.debug,
                disconnect_event=current_info.get("disconnect_event"),
                lifecycle_close=lifecycle_close,
                usage_buffer_limit_bytes=(
                    getattr(
                        deps,
                        "responses_usage_buffer_limit_bytes",
                        DEFAULT_MAX_PENDING_BYTES,
                    )
                    if request.url.path in {
                        "/v1/responses",
                        "/v1/responses/compact",
                    }
                    else DEFAULT_MAX_PENDING_BYTES
                ),
            )
        if not request.url.path.startswith("/v1") or deps.database_disabled:
            return response
        if hasattr(response, "json"):
            logger.info("Response: %s", await response.json())
        else:
            logger.info(
                "Response: type=%s, status_code=%s, headers=%s",
                type(response).__name__,
                response.status_code,
                response.headers,
            )
        return response

    async def dispatch(self, request: Request, call_next):
        deps = self.dependencies
        runtime_gauges = deps.runtime_gauges

        if request.method == "OPTIONS":
            return await call_next(request)
        if deps.is_public_health_request(request):
            return await call_next(request)

        start_time = time()
        runtime_gauges.begin_inflight()

        trace = None
        token: Optional[str] = None
        current_request_info = None
        current_info: Optional[dict[str, Any]] = None
        disconnect_task: Optional[asyncio.Task] = None
        lifecycle_transferred = False
        lifecycle_finished = False
        disconnect_finished = False
        fields_finalized = False
        observability_finished = False
        inflight_finished = False
        context_finished = False
        lifecycle_lock = asyncio.Lock()

        async def finish_request_lifecycle(_info: Optional[dict[str, Any]] = None) -> None:
            """Release request-owned state exactly once, after the body is done."""
            nonlocal lifecycle_finished
            nonlocal disconnect_finished, fields_finalized
            nonlocal observability_finished, inflight_finished, context_finished
            async with lifecycle_lock:
                if lifecycle_finished:
                    return

                if not disconnect_finished and disconnect_task is not None:
                    disconnect_task.cancel()
                    try:
                        await disconnect_task
                    except asyncio.CancelledError:
                        if not disconnect_task.done():
                            raise
                    except Exception:
                        logger.exception("Disconnect monitor failed during request cleanup")
                    disconnect_finished = True
                elif disconnect_task is None:
                    disconnect_finished = True

                try:
                    if current_info is not None and not fields_finalized:
                        final_trace = current_info.get("trace") or trace
                        if final_trace is not None:
                            final_trace.mark("stream_end")
                            if current_info.get("_defer_observability_until_stream_end"):
                                final_trace.mark("usage_recorded")
                            merge_timing_spans(current_info, final_trace.snapshot())
                        current_info["process_time"] = time() - start_time
                        logger.info(
                            "trace_span trace_id=%s request_id=%s endpoint=%s spans=%s",
                            current_info.get("trace_id"),
                            current_info.get("request_id"),
                            current_info.get("endpoint"),
                            current_info.get("timing_spans"),
                        )
                        fields_finalized = True
                    if current_info is not None and not observability_finished:
                        try:
                            deps.emit_request_observability(current_info)
                        except Exception:
                            logger.exception("Failed to emit request observability")
                        observability_finished = True
                finally:
                    try:
                        if (
                            current_info is not None
                            and current_info.get("_waiting_first_byte_active")
                        ):
                            end_waiting = getattr(
                                runtime_gauges,
                                "end_waiting_first_byte",
                                None,
                            )
                            if callable(end_waiting):
                                end_waiting(current_info)
                        if not inflight_finished:
                            runtime_gauges.end_inflight()
                            inflight_finished = True
                    finally:
                        if current_request_info is not None and not context_finished:
                            try:
                                reset_request_info(current_request_info)
                            except (LookupError, ValueError):
                                logger.exception("Failed to reset request context")
                            context_finished = True
                        elif current_request_info is None:
                            context_finished = True
                lifecycle_finished = all(
                    (
                        disconnect_finished,
                        fields_finalized or current_info is None,
                        observability_finished or current_info is None,
                        inflight_finished,
                        context_finished,
                    )
                )

        try:
            await runtime_gauges.record_event_loop_lag()
            incoming_trace = deps.incoming_trace_context(request.headers)
            trace = deps.trace_factory(
                trace_id=incoming_trace["trace_id"],
                parent_span_id=incoming_trace.get("parent_span_id"),
                trace_flags=incoming_trace.get("trace_flags"),
                tracestate=incoming_trace.get("tracestate"),
            )
            if incoming_trace.get("x_request_id"):
                trace.set_tag("x_request_id", incoming_trace.get("x_request_id"))
            trace.mark("request_received")
            admission_wait_ms = getattr(
                request.state,
                "uni_api_admission_wait_ms",
                None,
            )
            if admission_wait_ms is not None and hasattr(trace, "add_ms"):
                trace.add_ms("request_admission_wait_ms", admission_wait_ms)

            request_info_data = self._request_context(
                request,
                trace=trace,
                incoming_trace=incoming_trace,
                start_time=start_time,
                token="",
            )
            current_request_info = set_request_info(request_info_data)
            current_info = get_request_info()
            request.state.uni_api_request_info = current_info
            request.state.uni_api_trace = trace

            token = await deps.get_api_key(request)
            if not token:
                current_info["status_code"] = 403
                current_info["error_type"] = "invalid_api_key"
                current_info["success"] = False
                return self._forbidden_response(trace)
            current_info["api_key"] = token

            api_index = self._api_index_for_token(token)
            if api_index is not None:
                enable_moderation = safe_get(
                    deps.app_state.config,
                    "api_keys",
                    api_index,
                    "preferences",
                    "ENABLE_MODERATION",
                    default=False,
                )
                paid_key_response = self._paid_key_enabled_response(token, api_index, request, trace)
                if paid_key_response is not None:
                    current_info["status_code"] = 429
                    current_info["error_type"] = "insufficient_balance"
                    current_info["success"] = False
                    return paid_key_response
            else:
                current_info["status_code"] = 403
                current_info["error_type"] = "invalid_api_key"
                current_info["success"] = False
                return self._forbidden_response(trace)
            trace.mark("auth_done")

            parsed_body = await self._parse_and_log_body(
                request,
                current_info["request_id"],
                trace,
            )
            if isinstance(parsed_body, dict):
                current_info["stream"] = parsed_body.get("stream")
                current_info["request_kind"] = request.url.path
                message_roles, role_counts = deps.message_role_summary(parsed_body)
                current_info["message_roles"] = message_roles
                current_info["role_counts"] = role_counts
            _, disconnect_task = await self._start_disconnect_monitor(request, current_info)
            policy_response = await self._apply_body_policy(
                request,
                parsed_body=parsed_body,
                api_index=api_index,
                enable_moderation=enable_moderation,
                current_info=current_info,
                start_time=start_time,
            )
            if policy_response is not None:
                return policy_response

            # Body policy and low-cardinality summaries are complete.  Do not
            # retain this first full JSON object while FastAPI/Pydantic builds
            # the endpoint model from the cached raw bytes.
            parsed_body = None

            response = await call_next(request)
            response = await self._wrap_response_for_observability(
                request,
                response,
                current_info,
                trace,
                lifecycle_close=finish_request_lifecycle,
            )
            lifecycle_transferred = isinstance(response, deps.logging_response_class)
            return response

        except RequestBodyTooLarge:
            # The outer body middleware owns the protocol-safe 413 response.
            # Swallowing this here would incorrectly turn chunked overflow into
            # a 500 after the request body crossed its hard limit.
            if current_info is not None:
                current_info["status_code"] = 413
                current_info["error_type"] = "body_too_large"
                current_info["admission_rejected"] = True
                current_info["admission_reason"] = "body_too_large"
                current_info["success"] = False
            raise
        except RequestBodyTooComplex:
            # The outer body middleware owns the protocol-safe 413 response.
            # This dedicated type cannot be confused with complexity failures
            # while parsing upstream responses or SSE events.
            if current_info is not None:
                current_info["status_code"] = 413
                current_info["error_type"] = "body_too_complex"
                current_info["admission_rejected"] = True
                current_info["admission_reason"] = "body_too_complex"
                current_info["success"] = False
            raise
        except RequestBodyDisconnected:
            if current_info is not None:
                current_info["status_code"] = 499
                current_info["error_type"] = "request_body_disconnected"
                current_info["stream_outcome"] = "downstream_disconnected"
                current_info["downstream_disconnected"] = True
                current_info["success"] = False
            raise
        except RequestBodyReadTimeout:
            if current_info is not None:
                current_info["status_code"] = 408
                current_info["error_type"] = "request_body_timeout"
                current_info["success"] = False
            raise
        except AdmissionRejected as exc:
            # Request/body admission is translated by the outer pure-ASGI
            # middleware.  Do not silently turn its intentional 413/503 into
            # an unrelated 500 response here.
            if current_info is not None:
                current_info["status_code"] = exc.status_code
                current_info["error_type"] = exc.reason
                current_info["admission_rejected"] = True
                current_info["admission_reason"] = exc.reason
                current_info["success"] = False
            raise
        except UpstreamAdmissionRejected as exc:
            if current_info is not None:
                current_info["status_code"] = exc.status_code
                current_info["error_type"] = exc.reason
                current_info["admission_rejected"] = True
                current_info["admission_reason"] = exc.reason
                current_info["success"] = False
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "message": "Upstream concurrency is at local capacity",
                        "type": "local_overload",
                        "code": exc.reason,
                    }
                },
                headers={
                    "retry-after": str(exc.retry_after_seconds),
                    "x-uni-api-admission-reason": exc.reason,
                },
            )
        except HTTPException as e:
            if current_info is not None:
                current_info["status_code"] = getattr(e, "status_code", 500)
                current_info["error_type"] = "http_exception"
            raise
        except ValidationError as e:
            logger.error("API key: %s, invalid request body: %s", deps.mask_secret_for_log(token), e.errors())
            content = await run_json_cpu(jsonable_encoder, {"detail": e.errors()})
            if current_info is not None:
                current_info["status_code"] = 422
                current_info["error_type"] = "validation_error"
            return JSONResponse(status_code=422, content=content)
        except Exception as e:
            if deps.debug:
                import traceback

                traceback.print_exc()
            logger.error("Error processing request: %s", e)
            if current_info is not None:
                current_info["status_code"] = 500
                current_info["error_type"] = type(e).__name__
            return JSONResponse(status_code=500, content={"error": f"Internal server error: {e}"})

        finally:
            if not lifecycle_transferred:
                await finish_request_lifecycle()

    async def moderate_content(self, content: str, api_index: int, background_tasks: BackgroundTasks) -> dict[str, Any]:
        response = await self.dependencies.moderation_handler(
            ModerationRequest(input=content),
            background_tasks,
            api_index,
        )

        moderation_result = bytearray()
        moderation_result_limit = 1024 * 1024
        async for chunk in response.body_iterator:
            if isinstance(chunk, str):
                chunk_bytes = chunk.encode("utf-8")
            else:
                chunk_bytes = bytes(chunk)
            if len(moderation_result) + len(chunk_bytes) > moderation_result_limit:
                raise RuntimeError("moderation response exceeded 1 MiB local limit")
            moderation_result.extend(chunk_bytes)

        return json.loads(bytes(moderation_result).decode("utf-8"))
