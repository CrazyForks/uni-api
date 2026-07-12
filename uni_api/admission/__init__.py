"""Bounded, cancellation-safe admission primitives."""

from uni_api.admission.core import (
    AdmissionLease,
    AdmissionRejected,
    BoundedAdmissionGate,
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

__all__ = [
    "AdmissionLease",
    "AdmissionRejected",
    "BoundedAdmissionGate",
    "PendingBodyReservation",
    "RequestAdmissionController",
    "RequestAdmissionLease",
    "RequestBodyBudgetExhausted",
    "RequestBodyTooLarge",
    "TemporaryResponseBytesReservation",
    "UpstreamResponseBudgetExhausted",
    "bind_request_admission_lease",
    "get_request_admission_lease",
    "reset_request_admission_lease",
]
