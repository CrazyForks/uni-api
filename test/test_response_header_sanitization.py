import httpx
import h11

from uni_api.runtime import _copy_upstream_response_headers


def _assert_h11_accepts(headers: dict[str, str]) -> None:
    h11.Response(
        status_code=200,
        headers=[(key.encode("latin-1"), value.encode("latin-1")) for key, value in headers.items()],
    )


def test_copy_upstream_response_headers_drops_duplicate_empty_values_that_h11_rejects():
    upstream_headers = httpx.Headers([(b"x-empty", b""), (b"x-empty", b"")])

    copied = _copy_upstream_response_headers(upstream_headers)

    assert "x-empty" not in copied
    _assert_h11_accepts(copied)


def test_copy_upstream_response_headers_preserves_valid_duplicate_values():
    upstream_headers = httpx.Headers([(b"x-upstream", b"alpha"), (b"x-upstream", b"beta")])

    copied = _copy_upstream_response_headers(upstream_headers)

    assert copied["x-upstream"] == "alpha, beta"
    _assert_h11_accepts(copied)


def test_copy_upstream_response_headers_drops_invalid_names_and_values():
    upstream_headers = httpx.Headers(
        [
            (b"x-valid", b"ok"),
            (b"x-invalid-value", b"bad\r\nvalue"),
            (b"bad name", b"ok"),
        ]
    )

    copied = _copy_upstream_response_headers(upstream_headers)

    assert copied == {"x-valid": "ok"}
    _assert_h11_accepts(copied)


def test_copy_upstream_response_headers_drops_hop_by_hop_and_comma_only_values():
    upstream_headers = httpx.Headers(
        [
            (b"content-length", b"123"),
            (b"transfer-encoding", b"chunked"),
            (b"x-empty-joined", b", "),
            (b"x-trimmed", b" ok\t"),
        ]
    )

    copied = _copy_upstream_response_headers(upstream_headers)

    assert copied == {"x-trimmed": "ok"}
    _assert_h11_accepts(copied)
