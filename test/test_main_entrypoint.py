import pytest
import uvicorn

import main


def test_main_cli_preserves_common_uvicorn_overrides_with_safe_protocol(monkeypatch):
    captured = {}
    for name in (
        "UVICORN_CONNECTION_LIMIT",
        "UVICORN_HTTP_PROTOCOL",
        "BOUNDED_HTTP_PROTOCOL_STATS",
    ):
        monkeypatch.setattr(main._runtime, name, getattr(main._runtime, name))
    monkeypatch.setattr(
        uvicorn.main,
        "callback",
        lambda **parameters: captured.update(parameters),
    )

    main._run_uvicorn_cli(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "9123",
            "--log-level",
            "warning",
            "--limit-concurrency",
            "10",
        ]
    )

    assert captured["app"] == "uni_api.runtime:app"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9123
    assert captured["log_level"] == "warning"
    assert captured["limit_concurrency"] is None
    assert captured["workers"] == 1
    assert captured["http"].__name__ == "BoundedH11Protocol"


def test_main_cli_rejects_multiple_workers_and_unsafe_connection_override():
    with pytest.raises(SystemExit, match="--workers must remain 1"):
        main._run_uvicorn_cli(["--workers", "2"])

    with pytest.raises(SystemExit, match="must not exceed"):
        main._run_uvicorn_cli(
            [
                "--limit-concurrency",
                str(main._runtime.UVICORN_CONNECTION_LIMIT + 1),
            ]
        )


@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_main_cli_eager_options_exit_cleanly(argument, capsys):
    with pytest.raises(SystemExit) as exited:
        main._run_uvicorn_cli([argument])
    assert exited.value.code == 0
    output = capsys.readouterr().out
    assert "Traceback" not in output
