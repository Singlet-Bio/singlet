# SPDX-License-Identifier: MIT
"""A small threaded HTTP server for tests: serves a directory with Range, ETag / If-None-Match and Content-Range like the data
Worker does, and records every request (path, range, status, bytes sent) so tests can assert on request counts and byte totals."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_RANGE = re.compile(r"^bytes=(\d+)-(\d*)$")


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.log = []  # (method, path, range, status, nbytes)
        self.fail_next = []  # list of (status) to inject, consumed in order
        self.user_agents = set()
        self.inflight = 0  # GETs being answered now, and the most at once
        self.max_inflight = 0

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


def _handler(root: str, rec: Recorder, ignore_range: bool, total_override, range_status: int = 206, delay: float = 0.0):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence
            pass

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/_stats":  # the recorder, for clients in another process (the R tests)
                with rec.lock:
                    body = json.dumps(
                        {"requests": len(rec.log), "bytes": sum(r[4] for r in rec.log), "max_inflight": rec.max_inflight}
                    ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            with rec.lock:
                rec.inflight += 1
                rec.max_inflight = max(rec.max_inflight, rec.inflight)
            try:
                if delay:
                    time.sleep(delay)  # a slow link: concurrent requests overlap here
                self._get(path)
            finally:
                with rec.lock:
                    rec.inflight -= 1

        def _get(self, path):
            rec.user_agents.add(self.headers.get("User-Agent", ""))
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
                    self.send_response(range_status)
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
                    rec.add("GET", path, (a, b), range_status, len(body))
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
def serve(root: str, *, ignore_range: bool = False, total_override=None, range_status: int = 206, delay: float = 0.0):
    """Yield (base_url, Recorder). ``total_override`` lies about the object size in Content-Range (to test the size check);
    ``range_status=200`` answers a range with the slice and its Content-Range but status 200, as ``rclone serve s3`` does;
    ``delay`` holds every GET that many seconds (a high-latency link; ``Recorder.max_inflight`` then shows the concurrency)."""
    rec = Recorder()
    srv = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _handler(os.path.abspath(root), rec, ignore_range, total_override, range_status, delay),
    )
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1], rec
    finally:
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    # python -P tests/python/_s1_range_server.py <root> [port] [--range-status 200] [--ignore-range] [--delay S] [--ttl S] [--url-file F]
    # serves until killed, GET /_stop or --ttl seconds, and prints the base URL (used by the R tests); GET /_stats is the recorder.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("port", nargs="?", type=int, default=0)
    ap.add_argument("--range-status", type=int, default=206)
    ap.add_argument("--ignore-range", action="store_true")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--ttl", type=float, default=0.0, help="exit after this many seconds (0: never)")
    ap.add_argument("--url-file", help="also write the base URL to this file (written whole, then renamed)")
    a = ap.parse_args()
    rec = Recorder()
    Base = _handler(os.path.abspath(a.root), rec, a.ignore_range, None, a.range_status, a.delay)

    class Stoppable(Base):
        def do_GET(self):  # noqa: N802 - GET /_stop ends the process
            if self.path == "/_stop":
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.wfile.flush()
                threading.Timer(0.05, lambda: os._exit(0)).start()
                return
            super().do_GET()

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Stoppable)
    srv.daemon_threads = True
    if a.ttl > 0:
        threading.Timer(a.ttl, lambda: os._exit(0)).start()
    url = "http://127.0.0.1:%d" % srv.server_address[1]
    if a.url_file:
        with open(a.url_file + ".part", "w") as fh:
            fh.write(url + "\n")
        os.replace(a.url_file + ".part", a.url_file)
    print(url, flush=True)
    srv.serve_forever()
