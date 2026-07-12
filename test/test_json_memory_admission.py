import pytest

from uni_api.admission.json_memory import (
    IncrementalJSONMemoryEstimator,
    JSONMemoryComplexityError,
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
