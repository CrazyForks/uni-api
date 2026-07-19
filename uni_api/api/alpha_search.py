from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Callable

from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from core.utils import get_engine, provider_api_circular_list, safe_get
from uni_api.admission.json_parsing import run_json_cpu
from uni_api.idempotency import apply_oaix_routing_attempt_id
from uni_api.providers.header_passthrough import apply_provider_preference_headers
from uni_api.providers.payloads import force_codex_client_headers
from uni_api.routing.planner import (
    RoutingPlan,
    get_right_order_providers,
    select_provider_api_key_raw,
)
from uni_api.routing.search_affinity import (
    SearchAffinityBinding,
    SearchAffinityStore,
)
from uni_api.upstream.urls import normalize_alpha_search_upstream_url
from upstream import UpstreamRunner


ALPHA_SEARCH_ENDPOINT = "/v1/alpha/search"
ALPHA_SEARCH_MAX_ATTEMPTS = 3
ALPHA_SEARCH_MAX_ID_CHARS = 512
ALPHA_SEARCH_MAX_MODEL_CHARS = 256

_SENSITIVE_CLIENT_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
    "chatgpt-account-id",
}
_UNSAFE_RESPONSE_HEADERS = {
    "api-key",
    "authorization",
    "chatgpt-account-id",
    "connection",
    "content-encoding",
    "content-length",
    "cookie",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "set-cookie",
    "set-cookie2",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "x-api-key",
}
_SAFE_OAIX_RESPONSE_HEADERS = {
    "x-oaix-connection-id",
    "x-oaix-request-id",
}


class _RetryableAlphaSearchResponse(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = int(status_code)
        self.reason = f"alpha_search_upstream_status_{self.status_code}"
        super().__init__(self.reason)


class _AlphaSearchProtocolError(RuntimeError):
    status_code = 502
    reason = "alpha_search_upstream_protocol_error"


def validate_alpha_search_request(value: Any) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise HTTPException(
            status_code=400,
            detail="alpha/search request body must be a JSON object",
        )
    search_id = value.get("id")
    model = value.get("model")
    if not isinstance(search_id, str) or not search_id.strip():
        raise HTTPException(
            status_code=400,
            detail="alpha/search request id must be a non-empty string",
        )
    if len(search_id) > ALPHA_SEARCH_MAX_ID_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"alpha/search request id exceeds {ALPHA_SEARCH_MAX_ID_CHARS} characters",
        )
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(
            status_code=400,
            detail="alpha/search request model must be a non-empty string",
        )
    if len(model) > ALPHA_SEARCH_MAX_MODEL_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"alpha/search request model exceeds {ALPHA_SEARCH_MAX_MODEL_CHARS} characters",
        )
    return search_id, model


async def validate_alpha_search_response(raw: bytes) -> None:
    try:
        payload = await run_json_cpu(json.loads, raw)
    except Exception as exc:
        raise _AlphaSearchProtocolError(
            "alpha/search upstream returned invalid JSON"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("output"), str):
        raise _AlphaSearchProtocolError(
            "alpha/search upstream response must contain a string output"
        )
    encrypted_output = payload.get("encrypted_output")
    if "encrypted_output" in payload and encrypted_output is not None and not isinstance(
        encrypted_output,
        str,
    ):
        raise _AlphaSearchProtocolError(
            "alpha/search upstream encrypted_output must be a string or null"
        )


def _safe_provider_label(provider_name: str) -> str:
    if provider_name.startswith("sk-"):
        digest = hashlib.sha256(provider_name.encode("utf-8")).hexdigest()[:16]
        return f"local-api-key:{digest}"
    return provider_name[:256]


def _safe_client_header_source(http_request: Request) -> SimpleNamespace:
    safe_headers: dict[str, str] = {}
    for name, value in (getattr(http_request, "headers", None) or {}).items():
        normalized = str(name).lower()
        if normalized in _SENSITIVE_CLIENT_HEADERS or normalized.startswith("x-oaix-"):
            continue
        safe_headers[str(name)] = str(value)
    return SimpleNamespace(headers=safe_headers)


def _set_header(headers: dict[str, Any], name: str, value: Any) -> None:
    normalized = name.lower()
    for existing_name in list(headers):
        if str(existing_name).lower() == normalized:
            headers.pop(existing_name, None)
    if value is not None and str(value) != "":
        headers[name] = str(value)


def _get_header(headers: Any, name: str) -> str | None:
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value is not None:
            return str(value)
    normalized = name.lower()
    for existing_name, value in (headers or {}).items():
        if str(existing_name).lower() == normalized:
            return str(value)
    return None


def _copy_alpha_search_response_headers(headers: Any) -> dict[str, str]:
    grouped: dict[str, tuple[str, list[str]]] = {}
    raw_headers = getattr(headers, "raw", None)
    pairs = raw_headers if raw_headers else (headers or {}).items()
    for raw_name, raw_value in pairs:
        name = (
            raw_name.decode("latin-1", errors="replace")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )
        value = (
            raw_value.decode("latin-1", errors="replace")
            if isinstance(raw_value, bytes)
            else str(raw_value)
        ).strip(" \t")
        normalized = name.lower()
        if normalized in _UNSAFE_RESPONSE_HEADERS or not value:
            continue
        if (
            normalized.startswith("x-oaix-")
            and normalized not in _SAFE_OAIX_RESPONSE_HEADERS
        ):
            continue
        if "\r" in name or "\n" in name or "\r" in value or "\n" in value:
            continue
        if normalized not in grouped:
            grouped[normalized] = (name, [value])
        else:
            grouped[normalized][1].append(value)
    copied = {name: ", ".join(values) for name, values in grouped.values()}
    _set_header(copied, "Cache-Control", "no-store")
    return copied


def _json_error_response(status_code: int) -> JSONResponse:
    status = int(status_code)
    if status < 400 or status > 599:
        status = 502
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": "alpha/search upstream request failed",
                "type": "upstream_error",
                "code": "alpha_search_upstream_error",
            }
        },
        headers={"Cache-Control": "no-store"},
    )


class AlphaSearchRequestHandler:
    def __init__(
        self,
        *,
        app: Any,
        get_runtime_api_list: Callable[[], list[str]],
        api_key_has_model_rules: Callable[[Any, int], bool],
        resolve_codex_upstream_auth: Callable[..., Any],
        resolve_timeout: Callable[..., Any],
        add_trace_headers: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None,
        record_plan_observability: Callable[[dict[str, Any], RoutingPlan], Any] | None = None,
        record_retry_observability: Callable[..., Any] | None = None,
        provider_resolver: Callable[..., Any] = get_right_order_providers,
        debug: Callable[[], bool] | None = None,
        affinity_store: SearchAffinityStore | None = None,
    ) -> None:
        self.app = app
        self.get_runtime_api_list = get_runtime_api_list
        self.api_key_has_model_rules = api_key_has_model_rules
        self.resolve_codex_upstream_auth = resolve_codex_upstream_auth
        self.resolve_timeout = resolve_timeout
        self.add_trace_headers = add_trace_headers
        self.record_plan_observability = record_plan_observability
        self.record_retry_observability = record_retry_observability
        self.provider_resolver = provider_resolver
        self.debug = debug or (lambda: False)
        self.affinity_store = affinity_store or SearchAffinityStore()
        self.last_provider_indices = defaultdict(lambda: -1)
        self.locks = defaultdict(asyncio.Lock)

    async def request_search(
        self,
        *,
        http_request: Request,
        request_body: Any,
        api_index: int,
        background_tasks: BackgroundTasks | None = None,
    ) -> Response:
        _ = background_tasks
        search_id, request_model = validate_alpha_search_request(request_body)
        api_list = list(self.get_runtime_api_list())
        if api_index < 0 or api_index >= len(api_list):
            raise HTTPException(status_code=401, detail="Invalid API key")
        if not self.api_key_has_model_rules(self.app, api_index):
            raise HTTPException(
                status_code=404,
                detail=f"No matching model found: {request_model}",
            )

        session_key = self.affinity_store.session_key(
            api_list[api_index],
            search_id,
        )
        binding = await self.affinity_store.get(session_key)
        self._validate_binding_model(binding, request_model)
        if binding is not None:
            return await self._run_with_binding(
                http_request=http_request,
                request_body=dict(request_body),
                request_model=request_model,
                api_index=api_index,
                session_key=session_key,
                binding=binding,
            )

        async with self.affinity_store.session(session_key):
            binding = await self.affinity_store.get(session_key)
            self._validate_binding_model(binding, request_model)
            if binding is None:
                return await self._run_with_binding(
                    http_request=http_request,
                    request_body=dict(request_body),
                    request_model=request_model,
                    api_index=api_index,
                    session_key=session_key,
                    binding=None,
                )

        return await self._run_with_binding(
            http_request=http_request,
            request_body=dict(request_body),
            request_model=request_model,
            api_index=api_index,
            session_key=session_key,
            binding=binding,
        )

    @staticmethod
    def _validate_binding_model(
        binding: SearchAffinityBinding | None,
        request_model: str,
    ) -> None:
        if binding is not None and binding.request_model != request_model:
            raise HTTPException(
                status_code=409,
                detail="alpha/search id is already bound to a different model",
            )

    async def _run_with_binding(
        self,
        *,
        http_request: Request,
        request_body: dict[str, Any],
        request_model: str,
        api_index: int,
        session_key: str,
        binding: SearchAffinityBinding | None,
    ) -> Response:
        encoded_request = await run_json_cpu(
            json.dumps,
            request_body,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            plan = await RoutingPlan.create(
                self.app,
                request_model,
                api_index,
                self.last_provider_indices,
                self.locks,
                endpoint=ALPHA_SEARCH_ENDPOINT,
                request_body_bytes=len(encoded_request.encode("utf-8")),
                debug=bool(self.debug()),
                provider_resolver=self.provider_resolver,
            )
        except HTTPException as exc:
            if exc.status_code == 404:
                raise HTTPException(
                    status_code=503,
                    detail="No providers are available for /v1/alpha/search",
                ) from exc
            raise

        if binding is not None:
            bound_provider_name = next(
                (
                    str(provider.get("provider") or "")
                    for provider in plan.matching_providers
                    if hmac.compare_digest(
                        self.affinity_store.provider_fingerprint(
                            str(provider.get("provider") or "")
                        ),
                        binding.provider_fingerprint,
                    )
                ),
                None,
            )
            if not bound_provider_name or not plan.restrict_to_provider(
                bound_provider_name
            ):
                raise HTTPException(
                    status_code=503,
                    detail="Bound alpha/search provider is unavailable",
                )
            current_original_model = plan.matching_providers[0][
                "_model_dict_cache"
            ][request_model]
            if current_original_model != binding.original_model:
                raise HTTPException(
                    status_code=503,
                    detail="Bound alpha/search model mapping changed",
                )

        current_info = getattr(
            getattr(http_request, "state", None),
            "uni_api_request_info",
            None,
        )
        if not isinstance(current_info, dict):
            current_info = {}
        current_info["stream"] = False
        current_info["model"] = request_model
        current_info["alpha_search_affinity_hit"] = binding is not None
        if self.record_plan_observability is not None:
            result = self.record_plan_observability(current_info, plan)
            if inspect.isawaitable(result):
                await result

        execution = _AlphaSearchExecution(
            handler=self,
            http_request=http_request,
            request_body=request_body,
            request_model=request_model,
            session_key=session_key,
            binding=binding,
            plan=plan,
            current_info=current_info,
        )
        return await execution.run()


class _AlphaSearchExecution:
    def __init__(
        self,
        *,
        handler: AlphaSearchRequestHandler,
        http_request: Request,
        request_body: dict[str, Any],
        request_model: str,
        session_key: str,
        binding: SearchAffinityBinding | None,
        plan: RoutingPlan,
        current_info: dict[str, Any],
    ) -> None:
        self.handler = handler
        self.http_request = http_request
        self.request_body = request_body
        self.request_model = request_model
        self.session_key = session_key
        self.binding = binding
        self.plan = plan
        self.current_info = current_info
        self.last_retry_response: Response | None = None
        selector = (
            self._select_bound_credential
            if binding is not None
            else select_provider_api_key_raw
        )
        self.runner = UpstreamRunner(
            plan,
            endpoint=ALPHA_SEARCH_ENDPOINT,
            debug=bool(handler.debug()),
            provider_api_key_selector=selector,
            observability_context=current_info,
        )

    async def run(self) -> Response:
        response = await self.runner.run(
            self._execute_attempt,
            prepare_attempt=self._prepare_attempt,
            after_failure=self._after_failure,
            build_error_response=self._build_error_response,
            build_final_response=self._build_final_response,
            allow_channel_exclusion=False,
            should_cool_down=lambda *_args: False,
            retry_decider=self._retry_decider,
            max_attempts=(
                1 if self.binding is not None else ALPHA_SEARCH_MAX_ATTEMPTS
            ),
            on_retry=self.handler.record_retry_observability,
        )
        if isinstance(response, Response):
            response.headers["Cache-Control"] = "no-store"
        return response

    async def _select_bound_credential(
        self,
        provider: dict[str, Any],
        original_model: str,
        api_list: list[str],
    ) -> str | None:
        assert self.binding is not None
        provider_name = str(provider.get("provider") or "")
        if not hmac.compare_digest(
            self.handler.affinity_store.provider_fingerprint(provider_name),
            self.binding.provider_fingerprint,
        ):
            raise HTTPException(
                status_code=503,
                detail="Bound alpha/search provider changed",
            )

        target = self.binding.credential_fingerprint
        if provider_name.startswith("sk-") and provider_name in api_list:
            candidate = provider_name
            if target is not None and hmac.compare_digest(
                self.handler.affinity_store.credential_fingerprint(candidate)
                or "",
                target,
            ):
                return candidate
            raise HTTPException(
                status_code=503,
                detail="Bound alpha/search credential changed",
            )

        if target is None:
            if provider.get("api"):
                raise HTTPException(
                    status_code=503,
                    detail="Bound alpha/search credential changed",
                )
            return None

        pool = provider_api_circular_list.get(provider_name)
        if pool is None or not hasattr(pool, "claim_by_fingerprint"):
            raise HTTPException(
                status_code=503,
                detail="Bound alpha/search credential is unavailable",
            )
        claimed = await pool.claim_by_fingerprint(
            target,
            self.handler.affinity_store.credential_fingerprint,
            original_model,
        )
        if claimed is None:
            raise HTTPException(
                status_code=503,
                detail="Bound alpha/search credential is unavailable",
            )
        return claimed

    async def _prepare_attempt(self, attempt: Any) -> None:
        provider = attempt.provider
        provider_name = attempt.provider_name
        original_model = attempt.original_model
        if self.binding is not None and original_model != self.binding.original_model:
            raise HTTPException(
                status_code=503,
                detail="Bound alpha/search model mapping changed",
            )

        try:
            upstream_url = normalize_alpha_search_upstream_url(
                provider.get("base_url", "")
            )
        except ValueError as exc:
            raise _AlphaSearchProtocolError(str(exc)) from exc
        engine, _stream_mode = get_engine(
            provider,
            endpoint=ALPHA_SEARCH_ENDPOINT,
            original_model=original_model,
        )
        proxy = safe_get(self.handler.app.state.config, "preferences", "proxy")
        proxy = safe_get(provider, "preferences", "proxy", default=proxy)
        attempt.provider_api_key_raw = await self.runner.select_provider_api_key(
            attempt
        )
        api_key = attempt.provider_api_key_raw
        codex_account_id = None
        if engine == "codex" and api_key:
            api_key, codex_account_id = await self.handler.resolve_codex_upstream_auth(
                provider_name,
                api_key,
                proxy,
            )
        timeout = self.handler.resolve_timeout(
            provider_name=provider_name,
            original_model=original_model,
            request_model=self.request_model,
            role=self.plan.role,
            engine=engine,
        )
        if inspect.isawaitable(timeout):
            timeout = await timeout
        attempt.state.update(
            {
                "upstream_url": upstream_url,
                "engine": engine,
                "proxy": proxy,
                "api_key": api_key,
                "codex_account_id": codex_account_id,
                "timeout": timeout,
            }
        )

    async def _execute_attempt(self, attempt: Any) -> Response:
        payload = dict(self.request_body)
        payload["model"] = attempt.original_model
        json_payload = await run_json_cpu(
            json.dumps,
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        headers = self._build_headers(attempt)
        if self.handler.add_trace_headers is not None:
            result = self.handler.add_trace_headers(headers, self.current_info)
            if inspect.isawaitable(result):
                await result

        async with self.handler.app.state.client_manager.get_client(
            attempt.state["upstream_url"],
            attempt.state["proxy"],
            http2=False if attempt.state["engine"] == "codex" else None,
        ) as client:
            upstream_response = await client.post(
                attempt.state["upstream_url"],
                headers=headers,
                content=json_payload,
                timeout=attempt.state["timeout"],
            )

        raw = bytes(upstream_response.content)
        attempt.state["routing_wire_status_code"] = int(
            upstream_response.status_code
        )
        downstream = Response(
            content=raw,
            status_code=upstream_response.status_code,
            headers=_copy_alpha_search_response_headers(
                upstream_response.headers
            ),
        )
        if not 200 <= upstream_response.status_code < 300:
            self.last_retry_response = downstream
            attempt.state["alpha_retry_response"] = downstream
            self.current_info["success"] = False
            if upstream_response.status_code == 429 or upstream_response.status_code >= 500:
                raise _RetryableAlphaSearchResponse(
                    upstream_response.status_code
                )
            return downstream

        try:
            await validate_alpha_search_response(raw)
        except _AlphaSearchProtocolError:
            self.last_retry_response = _json_error_response(502)
            attempt.state["alpha_retry_response"] = self.last_retry_response
            raise

        candidate_binding = SearchAffinityBinding(
            provider_fingerprint=self.handler.affinity_store.provider_fingerprint(
                attempt.provider_name
            ),
            request_model=self.request_model,
            original_model=attempt.original_model,
            credential_fingerprint=self.handler.affinity_store.credential_fingerprint(
                attempt.provider_api_key_raw
            ),
        )
        winner = await self.handler.affinity_store.bind_if_absent(
            self.session_key,
            candidate_binding,
        )
        if winner != candidate_binding:
            self.last_retry_response = _json_error_response(503)
            attempt.state["alpha_retry_response"] = self.last_retry_response
            raise _AlphaSearchProtocolError(
                "alpha/search affinity changed concurrently"
            )

        self.current_info["success"] = True
        self.current_info["provider"] = _safe_provider_label(
            attempt.provider_name
        )
        self.current_info["actual_model"] = attempt.original_model
        return downstream

    def _build_headers(self, attempt: Any) -> dict[str, Any]:
        headers: dict[str, Any] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        apply_provider_preference_headers(
            headers,
            attempt.provider,
            http_request=_safe_client_header_source(self.http_request),
        )
        if attempt.state.get("api_key"):
            _set_header(
                headers,
                "Authorization",
                f"Bearer {attempt.state['api_key']}",
            )
        if attempt.state["engine"] == "codex":
            request_headers = getattr(self.http_request, "headers", None) or {}
            if _get_header(headers, "Openai-Beta") is None:
                _set_header(
                    headers,
                    "Openai-Beta",
                    _get_header(request_headers, "Openai-Beta")
                    or "responses=experimental",
                )
            if _get_header(headers, "Originator") is None:
                _set_header(
                    headers,
                    "Originator",
                    _get_header(request_headers, "Originator")
                    or "codex_cli_rs",
                )
            _set_header(headers, "Session_id", self.request_body["id"])
            if attempt.state.get("codex_account_id"):
                _set_header(
                    headers,
                    "Chatgpt-Account-Id",
                    attempt.state["codex_account_id"],
                )
            force_codex_client_headers(headers)
        _set_header(headers, "Content-Type", "application/json")
        _set_header(headers, "Accept", "application/json")
        apply_oaix_routing_attempt_id(
            headers,
            provider=attempt.provider,
            routing_attempt_id=attempt.routing_attempt_id,
        )
        return headers

    async def _retry_decider(
        self,
        exc: Exception,
        status_code: int,
        _error_message: Any,
        _attempt: Any,
        _prepare_failure: bool,
    ) -> bool:
        if self.binding is not None or not self.plan.auto_retry:
            return False
        if bool(getattr(exc, "local_admission_rejection", False)):
            return False
        return int(status_code) == 429 or 500 <= int(status_code) <= 599

    async def _after_failure(
        self,
        attempt: Any,
        _exc: Exception,
        _status_code: int,
        _error_message: Any,
    ) -> None:
        self.last_retry_response = attempt.state.get("alpha_retry_response")
        self.current_info["success"] = False

    async def _build_error_response(
        self,
        status_code: int,
        _error_message: Any,
    ) -> Response:
        return self.last_retry_response or _json_error_response(status_code)

    async def _build_final_response(self, plan: RoutingPlan) -> Response:
        return self.last_retry_response or _json_error_response(plan.status_code)
