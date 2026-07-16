"""Bounded, cancellation-safe admission primitives."""

from uni_api.admission.core import (
    AdmissionLease,
    AdmissionRejected,
    BoundedAdmissionGate,
    LargeBodyCapacityExhausted,
    PendingBodyReservation,
    RequestAdmissionController,
    RequestAdmissionLease,
    RequestBodyBudgetExhausted,
    RequestBodyTooLarge,
    TemporaryResponseBytesReservation,
    UpstreamResponseBudgetExhausted,
    bind_request_admission_lease,
    get_request_admission_lease,
    reset_request_admission_lease,
)
from uni_api.admission.observability import (
    Admission503ResponseWriteOutcome,
    LargeBodyAdmissionDecision,
    LargeBodyHolderSnapshot,
    RequestBodyObservation,
)

__all__ = [
    "AdmissionLease",
    "Admission503ResponseWriteOutcome",
    "AdmissionRejected",
    "BoundedAdmissionGate",
    "LargeBodyCapacityExhausted",
    "LargeBodyAdmissionDecision",
    "LargeBodyHolderSnapshot",
    "PendingBodyReservation",
    "RequestAdmissionController",
    "RequestAdmissionLease",
    "RequestBodyBudgetExhausted",
    "RequestBodyObservation",
    "RequestBodyTooLarge",
    "TemporaryResponseBytesReservation",
    "UpstreamResponseBudgetExhausted",
    "bind_request_admission_lease",
    "get_request_admission_lease",
    "reset_request_admission_lease",
]
