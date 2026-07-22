from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestBodyObservation:
    """Body-free facts carried into one atomic admission decision."""

    request_id: str | None = None
    trace_id: str | None = None
    method: str | None = None
    path: str | None = None
    declared_content_length_bytes: int | None = None
    wire_bytes: int = 0
    decoded_bytes: int = 0
    decoder_workspace_bytes: int = 0
    json_raw_bytes: int | None = None
    json_structural_item_count: int | None = None
    json_depth: int | None = None
    json_peak_depth: int | None = None
    json_scalar_bytes: int | None = None
    json_estimated_bytes: int | None = None
    json_raw_memory_multiplier: int | None = None
    json_structural_item_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class LargeBodyHolderSnapshot:
    claim_id: str
    lease_id: str
    request_id: str | None
    trace_id: str | None
    claimed_at_unix_ms: int
    held_ms: int
    request_self_body_reserved_weighted_bytes: int


@dataclass(frozen=True, slots=True)
class LargeBodyAdmissionDecision:
    """Immutable claim/reject/release record created under the body lock."""

    schema_version: int
    sequence: int
    decision: str
    reason: str
    occurred_at_unix_ms: int
    release_reason: str | None
    release_finalizer: str | None
    request_self_lease_id: str
    request_self_request_id: str | None
    request_self_trace_id: str | None
    request_self_method: str | None
    request_self_path: str | None
    request_self_declared_content_length_bytes: int | None
    request_self_wire_bytes: int
    request_self_decoded_bytes: int
    request_self_decoder_workspace_bytes: int
    request_self_json_raw_bytes: int | None
    request_self_json_structural_item_count: int | None
    request_self_json_depth: int | None
    request_self_json_peak_depth: int | None
    request_self_json_scalar_bytes: int | None
    request_self_json_estimated_bytes: int | None
    request_self_json_raw_memory_multiplier: int | None
    request_self_json_structural_item_memory_bytes: int | None
    request_self_body_reserved_weighted_before_bytes: int
    request_self_body_reserved_weighted_attempted_after_bytes: int
    request_self_body_reserved_weighted_committed_after_bytes: int
    runtime_global_large_body_threshold_weighted_bytes: int
    runtime_global_large_body_active_before: int
    runtime_global_large_body_active_after: int
    runtime_global_large_body_limit: int
    runtime_global_request_body_reserved_weighted_before_bytes: int
    runtime_global_request_body_reserved_weighted_after_bytes: int
    runtime_global_upstream_response_reserved_weighted_before_bytes: int
    runtime_global_upstream_response_reserved_weighted_after_bytes: int
    runtime_global_retained_reserved_weighted_before_bytes: int
    runtime_global_retained_reserved_weighted_after_bytes: int
    runtime_global_request_body_budget_weighted_bytes: int
    runtime_global_request_body_budget_hard_weighted_bytes: int
    runtime_global_cgroup_memory_source: str | None
    runtime_global_cgroup_memory_current_bytes_sampled: int | None
    runtime_global_cgroup_memory_limit_bytes_sampled: int | None
    runtime_global_cgroup_memory_high_bytes_sampled: int | None
    runtime_global_cgroup_memory_soft_limit_bytes_sampled: int | None
    runtime_global_cgroup_memory_guard_bytes_sampled: int | None
    runtime_global_cgroup_memory_capacity_bytes_sampled: int | None
    runtime_global_cgroup_memory_available_bytes_sampled: int | None
    runtime_global_cgroup_memory_reserved_bytes_sampled: int | None
    runtime_global_cgroup_memory_sample_sequence: int | None
    runtime_global_cgroup_memory_sample_age_ms_at_decision: int | None
    runtime_global_cgroup_memory_sample_error: str | None
    holder: LargeBodyHolderSnapshot | None
    blocking_holders: tuple[LargeBodyHolderSnapshot, ...]


@dataclass(frozen=True, slots=True)
class ResponseBufferEvent:
    """One response-buffer admission or ownership lifecycle fact."""

    schema_version: int
    sequence: int
    event: str
    outcome: str
    admission_branch: str | None
    occurred_at_unix_ms: int
    request_self_lease_id: str
    request_self_request_id: str | None
    request_self_trace_id: str | None
    routing_attempt_id: str | None
    routing_attempt_index: int | None
    provider: str | None
    request_model: str | None
    actual_model: str | None
    allocation_id: str
    allocation_kind: str
    requested_bytes: int
    allocation_reserved_before_bytes: int
    allocation_reserved_after_bytes: int
    allocation_reserve_call_count: int
    request_response_reserved_before_bytes: int
    request_response_reserved_projected_bytes: int
    request_response_reserved_after_bytes: int
    request_retained_reserved_before_bytes: int
    request_retained_reserved_projected_bytes: int
    request_retained_reserved_after_bytes: int
    runtime_global_response_reserved_before_bytes: int
    runtime_global_response_reserved_projected_bytes: int
    runtime_global_response_reserved_after_bytes: int
    runtime_global_retained_reserved_before_bytes: int
    runtime_global_retained_reserved_projected_bytes: int
    runtime_global_retained_reserved_after_bytes: int
    retained_from_prior_attempts_bytes: int
    crosses_retry_boundary: bool
    request_response_limit_bytes: int
    request_retained_limit_bytes: int
    runtime_global_hard_budget_bytes: int
    parent_governor_allowed: bool | None
    parent_governor_reserved_before_bytes: int | None
    parent_governor_projected_reserved_bytes: int | None
    parent_governor_reserved_after_bytes: int | None
    parent_governor_available_before_bytes: int | None
    parent_governor_available_after_bytes: int | None
    cgroup_memory_source: str | None
    cgroup_memory_current_bytes_sampled: int | None
    cgroup_memory_limit_bytes_sampled: int | None
    cgroup_memory_high_bytes_sampled: int | None
    cgroup_memory_soft_limit_bytes_sampled: int | None
    cgroup_memory_guard_bytes_sampled: int | None
    cgroup_memory_capacity_bytes_sampled: int | None
    cgroup_memory_sample_sequence: int | None
    cgroup_memory_sample_age_ms_at_decision: int | None
    cgroup_memory_sample_error: str | None
    reserve_started_count: int = 0
    commit_count: int = 0
    committed_bytes: int = 0
    rollback_count: int = 0
    rolled_back_bytes: int = 0
    release_count: int = 0
    released_bytes: int = 0
    rejection_count: int = 0


@dataclass(frozen=True, slots=True)
class Admission503ResponseWriteOutcome:
    """Per-request ASGI write result, distinct from admission's decision."""

    schema_version: int
    occurred_at_unix_ms: int
    reason: str
    intended_status_code: int
    asgi_response_write_completed: bool
    request_self_lease_id: str | None
    request_self_request_id: str
    request_self_trace_id: str
    request_self_method: str | None
    request_self_path: str | None
    runtime_global_admission_503_response_write_completed_total_after: int
    runtime_global_admission_503_response_write_failed_total_after: int
