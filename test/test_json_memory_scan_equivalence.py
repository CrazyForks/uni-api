import random

from uni_api.admission.json_memory import (
    IncrementalJSONMemoryEstimator,
    JSONMemoryComplexityError,
    JSONMemoryComplexityReason,
    JSONMemoryComplexityTriggerPhase,
)


class _BytewiseReferenceEstimator(IncrementalJSONMemoryEstimator):
    """Pre-optimization scanner retained only for differential testing."""

    def feed(self, chunk: bytes | bytearray | memoryview) -> int:
        view = memoryview(chunk).cast("B")
        self.raw_bytes += len(view)
        if self.estimated_bytes > self.max_estimated_bytes:
            self._raise_complexity(
                reason=JSONMemoryComplexityReason.MAX_ESTIMATED_BYTES,
                trigger_phase=(
                    JSONMemoryComplexityTriggerPhase.CHUNK_RAW_CHARGE
                ),
                message=(
                    "JSON materialization estimate exceeds "
                    f"{self.max_estimated_bytes} bytes"
                ),
            )

        for value in view:
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif value == 0x5C:
                    self._escaped = True
                elif value == 0x22:
                    self._in_string = False
                continue

            if value == 0x22:
                self._finish_scalar()
                self._count_token()
                self._in_string = True
                continue

            if value in (0x7B, 0x5B):
                self._finish_scalar()
                self._count_token()
                self.depth += 1
                self.peak_depth = max(self.peak_depth, self.depth)
                if self.depth > self.max_depth:
                    self._raise_complexity(
                        reason=JSONMemoryComplexityReason.MAX_DEPTH,
                        trigger_phase=(
                            JSONMemoryComplexityTriggerPhase.DEPTH_SCAN
                        ),
                        message=f"JSON nesting exceeds {self.max_depth} levels",
                    )
                continue

            if value in (0x7D, 0x5D):
                self._finish_scalar()
                self.depth = max(0, self.depth - 1)
                continue

            if value in (0x20, 0x09, 0x0A, 0x0D, 0x2C, 0x3A):
                self._finish_scalar()
                continue

            if not self._scalar_active:
                self._scalar_active = True
                self._scalar_bytes = 0
                self._count_token()
            self._scalar_bytes += 1
            if self._scalar_bytes > self.max_scalar_bytes:
                self._raise_complexity(
                    reason=JSONMemoryComplexityReason.MAX_SCALAR_BYTES,
                    trigger_phase=(
                        JSONMemoryComplexityTriggerPhase.SCALAR_SCAN
                    ),
                    message=(
                        f"JSON scalar exceeds {self.max_scalar_bytes} bytes"
                    ),
                )

        return self.estimated_bytes


def _outcome(estimator_type, chunks, limits):
    estimator = estimator_type(**limits)
    values = []
    try:
        for chunk in chunks:
            values.append(estimator.feed(chunk))
    except JSONMemoryComplexityError as exc:
        return ("error", str(exc), exc.observation, estimator.snapshot())
    return ("ok", values, estimator.snapshot())


def test_optimized_scanner_matches_bytewise_reference_for_chunked_bytes():
    random_source = random.Random(20260723)

    for _ in range(2_000):
        payload = bytes(
            random_source.randrange(256)
            for _ in range(random_source.randrange(384))
        )
        offsets = sorted(
            {
                0,
                len(payload),
                *(
                    random_source.randrange(len(payload) + 1)
                    for _ in range(8)
                ),
            }
        )
        raw_chunks = [
            payload[start:end]
            for start, end in zip(offsets, offsets[1:])
        ]
        chunk_type = random_source.randrange(3)
        if chunk_type == 1:
            chunks = [bytearray(chunk) for chunk in raw_chunks]
        elif chunk_type == 2:
            chunks = [memoryview(chunk) for chunk in raw_chunks]
        else:
            chunks = raw_chunks
        limits = {
            "raw_memory_multiplier": random_source.randrange(1, 8),
            "token_memory_bytes": random_source.randrange(1, 2048),
            "max_depth": random_source.randrange(1, 32),
            "max_scalar_bytes": random_source.randrange(1, 128),
            "max_estimated_bytes": random_source.randrange(1, 256 * 1024),
        }

        assert _outcome(
            IncrementalJSONMemoryEstimator,
            chunks,
            limits,
        ) == _outcome(_BytewiseReferenceEstimator, chunks, limits)
