# SPDX-License-Identifier: MIT
"""A range body cut short by a dropped connection is requested again (issue 73), and the API key file is read."""

from __future__ import annotations

import http.server
import pathlib
import sys
import threading

import pytest

try:
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from singlet.s1 import transport as T  # noqa: E402

DATA = bytes(range(256)) * 8


def _server(short_first: int):
    state = {"n": 0}

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # noqa: D401
            pass

        def do_GET(self):  # noqa: N802
            state["n"] += 1
            a, b = [int(x) for x in self.headers["Range"].split("=")[1].split("-")]
            body = DATA[a : b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(DATA)}")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body[: len(body) // 2] if state["n"] <= short_first else body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def test_short_body_is_retried(monkeypatch):
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    srv, state = _server(short_first=2)
    try:
        t = T.HttpTransport(f"http://127.0.0.1:{srv.server_port}")
        r = t.get("x", rng=(10, 109))
        assert bytes(r.body) == DATA[10:110] and state["n"] == 3
    finally:
        srv.shutdown()


def test_short_body_keeps_failing_as_integrity_error(monkeypatch):
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    srv, _ = _server(short_first=99)
    try:
        t = T.HttpTransport(f"http://127.0.0.1:{srv.server_port}", retries=2)
        with pytest.raises(T.IntegrityError, match="body is"):
            t.get("x", rng=(0, 99))
    finally:
        srv.shutdown()


def test_api_key_file_is_read_when_the_variable_is_unset(tmp_path, monkeypatch):
    import importlib

    find = importlib.import_module("singlet.find")

    (tmp_path / ".singlet").mkdir()
    (tmp_path / ".singlet" / "api_key").write_text("sk-test\n", encoding="utf-8")
    monkeypatch.delenv("SINGLET_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(find, "_API_KEY", None)
    assert find._api_key() == "sk-test"
    monkeypatch.setenv("SINGLET_API_KEY", "from-env")
    assert find._api_key() == "from-env"
