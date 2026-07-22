from __future__ import annotations

from typing import Any

from fugue_observability import (
    emit_uni_api_ember_admission_503_response_write_outcome,
    emit_uni_api_ember_large_body_admission_decision,
    emit_uni_api_ember_request_observability,
    emit_uni_api_ember_response_buffer_event,
    fugue_observability_delivery_snapshot,
)
from uni_api.admission.observability import (
    Admission503ResponseWriteOutcome,
    LargeBodyAdmissionDecision,
    ResponseBufferEvent,
)


def emit_request_observability(current_info: dict[str, Any], runtime_metrics: dict[str, Any]) -> None:
    emit_uni_api_ember_request_observability(
        current_info=current_info,
        runtime_metrics=runtime_metrics,
    )


def emit_large_body_admission_decision(
    decision: LargeBodyAdmissionDecision,
) -> bool | None:
    return emit_uni_api_ember_large_body_admission_decision(decision)


def emit_response_buffer_event(event: ResponseBufferEvent) -> bool | None:
    return emit_uni_api_ember_response_buffer_event(event)


def observability_exporter_snapshot() -> dict[str, int]:
    return fugue_observability_delivery_snapshot()


def emit_admission_503_response_write_outcome(
    outcome: Admission503ResponseWriteOutcome,
) -> bool | None:
    return emit_uni_api_ember_admission_503_response_write_outcome(outcome)
