# SPDX-License-Identifier: MIT
"""A small threaded HTTP server for tests: serves a directory with Range, ETag / If-None-Match and Content-Range like the data
Worker does, and records every request (path, range, status, bytes sent) so tests can assert on request counts and byte totals."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_RANGE = re.compile(r"^bytes=(\d+)-(\d*)$")


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.log = []  # (method, path, range, status, nbytes)
        self.fail_next = []  # list of (status) to inject, consumed in order
        self.user_agents = set()

    def add(self, *row):
        with self.lock:
            self.log.append(row)

    @property
    def requests(self):
        return len(self.log)

    @property
    def bytes(self):
        return sum(r[4] for r in self.log)

    def paths(self, prefix=""):
        return [r for r in self.log if r[1].startswith(prefix)]


def _handler(root: str, rec: Recorder, ignore_range: bool, total_override):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence
            pass

        def do_GET(self):  # noqa: N802
            rec.user_agents.add(self.headers.get("User-Agent", ""))
            path = self.path.split("?", 1)[0]
            with rec.lock:
                inject = rec.fail_next.pop(0) if rec.fail_next else None
            if inject:
                self.send_response(inject)
                self.send_header("Content-Length", "0")
                self.end_headers()
                rec.add("GET", path, None, inject, 0)
                return
            fs = os.path.normpath(os.path.join(root, *path.lstrip("/").split("/")))
            if not fs.startswith(os.path.normpath(root)) or not os.path.isfile(fs):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                rec.add("GET", path, None, 404, 0)
                return
            size = os.path.getsize(fs)
            st = os.stat(fs)
            etag = (
                '"%s"'
                % hashlib.md5(("%d-%d" % (size, int(st.st_mtime * 1000))).encode()).hexdigest()
            )
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Content-Length", "0")
                self.end_headers()
                rec.add("GET", path, None, 304, 0)
                return
            rng = self.headers.get("Range")
            m = _RANGE.match(rng) if rng and not ignore_range else None
            with open(fs, "rb") as fh:
                if m:
                    a = int(m.group(1))
                    b = int(m.group(2)) if m.group(2) else size - 1
                    b = min(b, size - 1)
                    if a >= size or a > b:
                        self.send_response(416)
                        self.send_header("Content-Range", "bytes */%d" % size)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        rec.add("GET", path, (a, b), 416, 0)
                        return
                    fh.seek(a)
                    body = fh.read(b - a + 1)
                    self.send_response(206)
                    tot = total_override if total_override is not None else size
                    self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, tot))
                    self.send_header("ETag", etag)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header(
                        "Cache-Control",
                        "public, max-age=31536000, immutable" if "/s/" in path else "max-age=300",
                    )
                    self.end_headers()
                    self.wfile.write(body)
                    rec.add("GET", path, (a, b), 206, len(body))
                else:
                    body = fh.read()
                    self.send_response(200)
                    self.send_header("ETag", etag)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    rec.add("GET", path, None, 200, len(body))

    return H


@contextmanager
def serve(root: str, *, ignore_range: bool = False, total_override=None):
    """Yield (base_url, Recorder). ``total_override`` lies about the object size in Content-Range (to test the size check)."""
    rec = Recorder()
    srv = ThreadingHTTPServer(
        ("127.0.0.1", 0), _handler(os.path.abspath(root), rec, ignore_range, total_override)
    )
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1], rec
    finally:
        srv.shutdown()
        srv.server_close()


if (
    __name__ == "__main__"
):  # python -P tests/python/_s1_range_server.py <root>  : serve until killed, print the base URL (used by the R tests)
    import sys

    root = sys.argv[1]
    rec = Recorder()
    srv = ThreadingHTTPServer(
        ("127.0.0.1", int(sys.argv[2]) if len(sys.argv) > 2 else 0),
        _handler(os.path.abspath(root), rec, False, None),
    )
    print("http://127.0.0.1:%d" % srv.server_address[1], flush=True)
    srv.serve_forever()
