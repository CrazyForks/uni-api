import asyncio
from types import SimpleNamespace

from fastapi import HTTPException

from upstream import maybe_exclude_failed_channel
from uni_api.routing.health import ProviderModelCircuitBreaker


def test_provider_model_circuit_opens_only_after_sustained_403_404():
    clock = [10.0]
    circuit = ProviderModelCircuitBreaker(
        failure_threshold=3,
        failure_window_seconds=120,
        open_seconds=300,
        clock=lambda: clock[0],
    )

    assert circuit.record_failure("provider-a", "model-x", 404) is False
    clock[0] += 1
    assert circuit.record_failure("provider-a", "model-x", 403) is False
    clock[0] += 1
    assert circuit.record_failure("provider-a", "model-x", 404) is True
    assert circuit.is_open("provider-a", "model-x") is True
    assert circuit.is_open("provider-a", "model-y") is False

    snapshot = circuit.snapshot()
    assert snapshot["open_route_count"] == 1
    assert snapshot["failure_total_by_status"] == {403: 1, 404: 2}
    assert "provider-a" not in str(snapshot["open_routes"])
    assert "model-x" not in str(snapshot["open_routes"])


def test_provider_model_circuit_expires_and_success_resets_evidence():
    clock = [0.0]
    circuit = ProviderModelCircuitBreaker(
        failure_threshold=2,
        failure_window_seconds=10,
        open_seconds=20,
        clock=lambda: clock[0],
    )

    assert circuit.record_failure("provider-a", "model-x", 404) is False
    assert circuit.record_success("provider-a", "model-x") is True
    assert circuit.record_failure("provider-a", "model-x", 404) is False
    assert circuit.record_failure("provider-a", "model-x", 404) is True
    clock[0] = 21.0
    assert circuit.is_open("provider-a", "model-x") is False
    assert circuit.snapshot()["open_route_count"] == 0


def test_deterministic_failure_circuit_ignores_legacy_cooldown_exclusions():
    async def run():
        circuit = ProviderModelCircuitBreaker(
            failure_threshold=3,
            failure_window_seconds=120,
            open_seconds=300,
        )

        class Manager:
            cooldown_period = 0

            async def record_model_failure(self, provider, model, status_code):
                return circuit.record_failure(provider, model, status_code)

        class Plan:
            num_matching_providers = 1
            app = SimpleNamespace(
                state=SimpleNamespace(channel_manager=Manager())
            )

            async def refresh_matching_providers(self, *, debug=False):
                _ = debug
                raise HTTPException(
                    status_code=503,
                    detail="No available providers at the moment",
                )

        plan = Plan()
        for _ in range(2):
            assert (
                await maybe_exclude_failed_channel(
                    plan,
                    "provider-a",
                    "model-x",
                    403,
                    "User location is not supported for the API use",
                    exclude_error_substrings=[
                        "User location is not supported for the API use"
                    ],
                )
                is None
            )
        assert (
            await maybe_exclude_failed_channel(
                plan,
                "provider-a",
                "model-x",
                403,
                "User location is not supported for the API use",
                exclude_error_substrings=[
                    "User location is not supported for the API use"
                ],
            )
            == "opened_no_alternative"
        )
        assert circuit.is_open("provider-a", "model-x") is True

    asyncio.run(run())
