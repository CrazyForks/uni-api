from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


DEFAULT_JSON_RAW_MEMORY_MULTIPLIER = 5
DEFAULT_JSON_TOKEN_MEMORY_BYTES = 1024
DEFAULT_JSON_MAX_DEPTH = 128
DEFAULT_JSON_MAX_SCALAR_BYTES = 4096
DEFAULT_JSON_MAX_ESTIMATED_BYTES = 256 * 1024 * 1024
_TEXT_SCAN_CHUNK_CHARACTERS = 256 * 1024


class JSONMemoryComplexityReason(StrEnum):
    MAX_DEPTH = "max_depth"
    MAX_SCALAR_BYTES = "max_scalar_bytes"
    MAX_ESTIMATED_BYTES = "max_estimated_bytes"


class JSONMemoryComplexityTriggerPhase(StrEnum):
    CHUNK_RAW_CHARGE = "chunk_raw_charge"
    STRUCTURAL_ITEM_SCAN = "structural_item_scan"
    DEPTH_SCAN = "depth_scan"
    SCALAR_SCAN = "scalar_scan"


@dataclass(frozen=True, slots=True)
class JSONMemoryComplexityObservation:
    """Body-free primitives captured at the exact rejection decision.

    The scanner preserves its existing whole-chunk raw-memory charge. The
    cumulative ``raw_bytes`` and low-cardinality trigger phase identify the
    rejected input frame without adding per-byte work to accepted requests.
    """

    schema_version: int
    reason: JSONMemoryComplexityReason
    trigger_phase: JSONMemoryComplexityTriggerPhase
    raw_bytes: int
    structural_item_count: int
    depth: int
    peak_depth: int
    scalar_bytes: int
    estimated_bytes: int
    configured_limit: int
    max_depth: int
    max_scalar_bytes: int
    max_estimated_bytes: int
    raw_memory_multiplier: int
    structural_item_memory_bytes: int


class JSONMemoryComplexityError(ValueError):
    """A JSON document exceeds a finite structural/memory envelope."""

    def __init__(
        self,
        message: str,
        *,
        observation: JSONMemoryComplexityObservation | None = None,
    ) -> None:
        super().__init__(message)
        self.observation = observation


@dataclass(frozen=True, slots=True)
class JSONMemorySnapshot:
    raw_bytes: int
    tokens: int
    depth: int
    peak_depth: int
    scalar_bytes: int
    estimated_bytes: int
    raw_memory_multiplier: int
    structural_item_memory_bytes: int


class IncrementalJSONMemoryEstimator:
    """Estimate JSON materialization memory with O(1) retained state.

    Raw-byte multipliers alone are unsafe for object-dense JSON: a three-byte
    ``{}`` can become a dict, list slot, and (for typed routes) validation
    objects.  This scanner therefore charges both source bytes and every
    string/scalar/container token.  The defaults deliberately overestimate
    measured CPython 3.11 + Pydantic peaks while still allowing large single
    strings such as base64 image inputs.

    The scanner is not a JSON validator.  Malformed documents keep their
    existing FastAPI error behavior; we only reject unreasonably deep/large
    structures before a parser can materialize them.
    """

    def __init__(
        self,
        *,
        raw_memory_multiplier: int = DEFAULT_JSON_RAW_MEMORY_MULTIPLIER,
        token_memory_bytes: int = DEFAULT_JSON_TOKEN_MEMORY_BYTES,
        max_depth: int = DEFAULT_JSON_MAX_DEPTH,
        max_scalar_bytes: int = DEFAULT_JSON_MAX_SCALAR_BYTES,
        max_estimated_bytes: int = DEFAULT_JSON_MAX_ESTIMATED_BYTES,
    ) -> None:
        if raw_memory_multiplier <= 0 or token_memory_bytes <= 0:
            raise ValueError("JSON memory weights must be positive")
        if max_depth <= 0 or max_scalar_bytes <= 0 or max_estimated_bytes <= 0:
            raise ValueError("JSON complexity limits must be positive")
        self.raw_memory_multiplier = int(raw_memory_multiplier)
        self.token_memory_bytes = int(token_memory_bytes)
        self.max_depth = int(max_depth)
        self.max_scalar_bytes = int(max_scalar_bytes)
        self.max_estimated_bytes = int(max_estimated_bytes)

        self.raw_bytes = 0
        self.tokens = 0
        self.depth = 0
        self.peak_depth = 0
        self._in_string = False
        self._escaped = False
        self._scalar_active = False
        self._scalar_bytes = 0

    @property
    def estimated_bytes(self) -> int:
        return (
            self.raw_bytes * self.raw_memory_multiplier
            + self.tokens * self.token_memory_bytes
        )

    def snapshot(self) -> JSONMemorySnapshot:
        return JSONMemorySnapshot(
            raw_bytes=self.raw_bytes,
            tokens=self.tokens,
            depth=self.depth,
            peak_depth=self.peak_depth,
            scalar_bytes=self._scalar_bytes,
            estimated_bytes=self.estimated_bytes,
            raw_memory_multiplier=self.raw_memory_multiplier,
            structural_item_memory_bytes=self.token_memory_bytes,
        )

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
                elif value == 0x5C:  # backslash
                    self._escaped = True
                elif value == 0x22:  # quote
                    self._in_string = False
                continue

            if value == 0x22:  # quote starts a key or value string
                self._finish_scalar()
                self._count_token()
                self._in_string = True
                continue

            if value in (0x7B, 0x5B):  # { [
                self._finish_scalar()
                self._count_token()
                self.depth += 1
                self.peak_depth = max(self.peak_depth, self.depth)
                if self.depth > self.max_depth:
                    self._raise_complexity(
                        reason=JSONMemoryComplexityReason.MAX_DEPTH,
                        trigger_phase=JSONMemoryComplexityTriggerPhase.DEPTH_SCAN,
                        message=f"JSON nesting exceeds {self.max_depth} levels",
                    )
                continue

            if value in (0x7D, 0x5D):  # } ]
                self._finish_scalar()
                self.depth = max(0, self.depth - 1)
                continue

            if value in (0x20, 0x09, 0x0A, 0x0D, 0x2C, 0x3A):
                # JSON whitespace, comma, or colon terminates a scalar token.
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
                    trigger_phase=JSONMemoryComplexityTriggerPhase.SCALAR_SCAN,
                    message=(
                        f"JSON scalar exceeds {self.max_scalar_bytes} bytes"
                    ),
                )

        return self.estimated_bytes

    def _count_token(self) -> None:
        self.tokens += 1
        if self.estimated_bytes > self.max_estimated_bytes:
            self._raise_complexity(
                reason=JSONMemoryComplexityReason.MAX_ESTIMATED_BYTES,
                trigger_phase=(
                    JSONMemoryComplexityTriggerPhase.STRUCTURAL_ITEM_SCAN
                ),
                message=(
                    "JSON materialization estimate exceeds "
                    f"{self.max_estimated_bytes} bytes"
                ),
            )

    def _raise_complexity(
        self,
        *,
        reason: JSONMemoryComplexityReason,
        trigger_phase: JSONMemoryComplexityTriggerPhase,
        message: str,
    ) -> None:
        configured_limit = {
            JSONMemoryComplexityReason.MAX_DEPTH: self.max_depth,
            JSONMemoryComplexityReason.MAX_SCALAR_BYTES: self.max_scalar_bytes,
            JSONMemoryComplexityReason.MAX_ESTIMATED_BYTES: (
                self.max_estimated_bytes
            ),
        }[reason]
        raise JSONMemoryComplexityError(
            message,
            observation=JSONMemoryComplexityObservation(
                schema_version=1,
                reason=reason,
                trigger_phase=trigger_phase,
                raw_bytes=self.raw_bytes,
                structural_item_count=self.tokens,
                depth=self.depth,
                peak_depth=self.peak_depth,
                scalar_bytes=self._scalar_bytes,
                estimated_bytes=self.estimated_bytes,
                configured_limit=configured_limit,
                max_depth=self.max_depth,
                max_scalar_bytes=self.max_scalar_bytes,
                max_estimated_bytes=self.max_estimated_bytes,
                raw_memory_multiplier=self.raw_memory_multiplier,
                structural_item_memory_bytes=self.token_memory_bytes,
            ),
        )

    def _finish_scalar(self) -> None:
        self._scalar_active = False
        self._scalar_bytes = 0


def estimate_json_memory_bytes(
    payload: bytes | bytearray | memoryview,
    **limits: int,
) -> JSONMemorySnapshot:
    estimator = IncrementalJSONMemoryEstimator(**limits)
    estimator.feed(payload)
    return estimator.snapshot()


def estimate_json_text_memory_bytes(
    payload: str,
    **limits: int,
) -> JSONMemorySnapshot:
    """Scan text without allocating a second attacker-sized UTF-8 copy."""

    estimator = IncrementalJSONMemoryEstimator(**limits)
    for offset in range(0, len(payload), _TEXT_SCAN_CHUNK_CHARACTERS):
        estimator.feed(
            payload[offset : offset + _TEXT_SCAN_CHUNK_CHARACTERS].encode(
                "utf-8",
                errors="strict",
            )
        )
    return estimator.snapshot()
