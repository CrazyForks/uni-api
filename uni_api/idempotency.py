from __future__ import annotations

import os
from typing import Any


OAIX_ROUTING_ATTEMPT_HEADER = "X-OAIX-Routing-Attempt-ID"
OAIX_ROUTING_ATTEMPT_PREFERENCE = "oaix_routing_attempt_id"
OAIX_ROUTING_ATTEMPT_PROVIDERS_ENV = "OAIX_ROUTING_ATTEMPT_PROVIDERS"


def provider_supports_oaix_routing_attempt_id(provider: Any) -> bool:
    """Return the provider's explicit OAIX attempt-id capability flag.

    Hostname or provider-name inference is intentionally avoided: sending an
    internal idempotency header to an unrelated external provider would create
    an undocumented protocol dependency.
    """

    if not isinstance(provider, dict):
        return False
    preferences = provider.get("preferences")
    if (
        isinstance(preferences, dict)
        and preferences.get(OAIX_ROUTING_ATTEMPT_PREFERENCE) is True
    ):
        return True
    configured = {
        item.strip()
        for item in os.getenv(OAIX_ROUTING_ATTEMPT_PROVIDERS_ENV, "").split(",")
        if item.strip()
    }
    return str(provider.get("provider") or "") in configured


def apply_oaix_routing_attempt_id(
    headers: dict[str, Any],
    *,
    provider: Any,
    routing_attempt_id: Any,
) -> bool:
    """Attach the attempt identifier only to explicitly capable providers."""

    attempt_id = str(routing_attempt_id or "").strip()
    if not attempt_id or not provider_supports_oaix_routing_attempt_id(provider):
        return False
    headers[OAIX_ROUTING_ATTEMPT_HEADER] = attempt_id
    return True
