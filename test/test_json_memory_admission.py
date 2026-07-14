import pytest

from uni_api.admission.json_memory import (
    IncrementalJSONMemoryEstimator,
    JSONMemoryComplexityError,
    JSONMemoryComplexityReason,
    JSONMemoryComplexityTriggerPhase,
    estimate_json_memory_bytes,
)


def test_dense_objects_are_charged_by_structure_not_only_raw_bytes():
    payload = b"[" + b",".join([b"{}"] * 10_000) + b"]"
    snapshot = estimate_json_memory_bytes(payload)

    assert snapshot.tokens == 10_001
    assert snapshot.estimated_bytes > len(payload) * 300


def test_large_single_string_remains_bounded_by_raw_memory_weight():
    payload = b'{"image":"' + (b"A" * (1024 * 1024)) + b'"}'
    snapshot = estimate_json_memory_bytes(payload)

    assert snapshot.tokens == 3
    assert len(payload) * 5 <= snapshot.estimated_bytes < len(payload) * 6


def test_incremental_scanner_is_split_invariant_across_escapes_and_utf8():
    payload = '{"text":"a\\\"😀","items":[1,true,null,{}]}'.encode()
    expected = estimate_json_memory_bytes(payload)

    for split_at in range(len(payload) + 1):
        estimator = IncrementalJSONMemoryEstimator()
        estimator.feed(payload[:split_at])
        estimator.feed(payload[split_at:])
        assert estimator.snapshot() == expected


def test_excessive_json_depth_is_rejected_before_materialization():
    estimator = IncrementalJSONMemoryEstimator(max_depth=4)
    with pytest.raises(JSONMemoryComplexityError, match="nesting"):
        estimator.feed(b"[[[[[]]]]]")


def test_pathological_scalar_is_finite():
    estimator = IncrementalJSONMemoryEstimator(max_scalar_bytes=4)
    with pytest.raises(JSONMemoryComplexityError, match="scalar"):
        estimator.feed(b"[12345]")


@pytest.mark.parametrize("split_at", range(4))
def test_depth_rejection_carries_split_invariant_body_free_diagnostics(
    split_at,
):
    payload = b"[[["
    estimator = IncrementalJSONMemoryEstimator(max_depth=2)

    with pytest.raises(JSONMemoryComplexityError, match="nesting") as caught:
        estimator.feed(payload[:split_at])
        estimator.feed(payload[split_at:])

    observation = caught.value.observation
    assert observation is not None
    assert observation.reason is JSONMemoryComplexityReason.MAX_DEPTH
    assert observation.trigger_phase is JSONMemoryComplexityTriggerPhase.DEPTH_SCAN
    assert observation.raw_bytes == 3
    assert observation.structural_item_count == 3
    assert observation.depth == 3
    assert observation.peak_depth == 3
    assert observation.estimated_bytes == 3 * 5 + 3 * 1024
    assert observation.configured_limit == 2
    assert observation.max_depth == 2


@pytest.mark.parametrize("split_at", range(7))
def test_scalar_rejection_carries_split_invariant_body_free_diagnostics(
    split_at,
):
    payload = b"[12345"
    estimator = IncrementalJSONMemoryEstimator(max_scalar_bytes=4)

    with pytest.raises(JSONMemoryComplexityError, match="scalar") as caught:
        estimator.feed(payload[:split_at])
        estimator.feed(payload[split_at:])

    observation = caught.value.observation
    assert observation is not None
    assert observation.reason is JSONMemoryComplexityReason.MAX_SCALAR_BYTES
    assert observation.trigger_phase is JSONMemoryComplexityTriggerPhase.SCALAR_SCAN
    assert observation.raw_bytes == 6
    assert observation.structural_item_count == 2
    assert observation.scalar_bytes == 5
    assert observation.configured_limit == 4
    assert observation.max_scalar_bytes == 4


@pytest.mark.parametrize("split_at", range(3))
def test_estimate_rejection_carries_split_invariant_body_free_diagnostics(
    split_at,
):
    payload = b"[0"
    estimator = IncrementalJSONMemoryEstimator(max_estimated_bytes=1500)

    with pytest.raises(JSONMemoryComplexityError, match="materialization") as caught:
        estimator.feed(payload[:split_at])
        estimator.feed(payload[split_at:])

    observation = caught.value.observation
    assert observation is not None
    assert observation.reason is JSONMemoryComplexityReason.MAX_ESTIMATED_BYTES
    assert (
        observation.trigger_phase
        is JSONMemoryComplexityTriggerPhase.STRUCTURAL_ITEM_SCAN
    )
    assert observation.raw_bytes == 2
    assert observation.structural_item_count == 2
    assert observation.estimated_bytes == 2 * 5 + 2 * 1024
    assert observation.configured_limit == 1500
    assert observation.max_estimated_bytes == 1500


def test_raw_chunk_charge_phase_is_distinguished_from_structure_growth():
    estimator = IncrementalJSONMemoryEstimator(max_estimated_bytes=10)

    with pytest.raises(JSONMemoryComplexityError, match="materialization") as caught:
        estimator.feed(b"abc")

    observation = caught.value.observation
    assert observation is not None
    assert observation.reason is JSONMemoryComplexityReason.MAX_ESTIMATED_BYTES
    assert (
        observation.trigger_phase
        is JSONMemoryComplexityTriggerPhase.CHUNK_RAW_CHARGE
    )
    assert observation.raw_bytes == 3
    assert observation.structural_item_count == 0
    assert observation.estimated_bytes == 15
