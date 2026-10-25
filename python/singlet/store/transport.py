# SPDX-License-Identifier: MIT
"""Byte transports for the store client: ranged HTTP GET with persistent connections, retries and response checks, and a local
directory that mimics the server layout (for tests, mirrors and offline use).

A transport is addressed by *keys* (``store/m/GSE145370.json``, ``store/s/GSM.../<d>.s1``) relative to a base:

* ``https://data.singlet.bio`` (default) or any ``http(s)://`` URL;
* a directory path or ``file://`` URL whose layout is the server's.

Every ranged read is checked: HTTP 206, ``Content-Range: bytes a-b/TOTAL`` equal to the request and (when the caller knows it, and
for immutable objects it always does) to the expected object size, body length equal to the range. ``data.singlet.bio`` answers 403
to the default Python-urllib User-Agent, so every request carries ``User-Agent: singlet-client/2``.
"""

from __future__ import annotations

import http.client
import os
import random
import re
import socket
import ssl
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Mapping, Optional

USER_AGENT = "singlet-client/2 (+https://singlet.bio)"
DEFAULT_BASE = "https://data.singlet.bio"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")
_RETRY_STATUS = {429, 500, 502, 503, 504}


class TransportError(RuntimeError):
    """A request failed (network, HTTP status, or a response that does not match the request)."""


class NotFound(TransportError):
    """HTTP 404 / missing local file."""


class IntegrityError(TransportError):
    """The server returned bytes that cannot be the requested object (wrong total size, wrong range, short body)."""


@dataclass
class Response:
    status: int
    headers: dict
    body: object = b""  # bytes / bytearray / memoryview
    total: Optional[int] = None  # object size from Content-Range / Content-Length
    etag: Optional[str] = None


@dataclass
class Stats:
    requests: int = 0
    bytes: int = 0
    retries: int = 0
    by_kind: dict = field(default_factory=dict)

    def add(self, kind: str, nbytes: int) -> None:
        self.requests += 1
        self.bytes += nbytes
        k = self.by_kind.setdefault(kind, [0, 0])
        k[0] += 1
        k[1] += nbytes


def data_base() -> str:
    """Base URL (or directory) of the data host: ``$SINGLET_DATA_BASE`` (default ``https://data.singlet.bio``).

    A value ending in ``/data`` (the form the R client documents) is accepted and stripped."""
    base = os.environ.get("SINGLET_DATA_BASE", "").strip() or DEFAULT_BASE
    base = base.rstrip("/")
    if base.endswith("/data"):
        base = base[: -len("/data")]
    return base


def make_transport(base: Optional[str] = None, **kw) -> "Transport":
    base = (base or data_base()).strip()
    if re.match(r"^https?://", base, re.I):
        return HttpTransport(base, **kw)
    if base.lower().startswith("file://"):
        p = urllib.parse.unquote(urllib.parse.urlsplit(base).path)
        if re.match(r"^/[A-Za-z]:", p):  # file:///C:/x
            p = p[1:]
        return DirTransport(p)
    return DirTransport(base)


class Transport:
    """Interface: ``get(key, rng=None, ...)`` -> :class:`Response`."""

    stats: Stats
    base: str

    def get(
        self,
        key: str,
        rng: Optional[tuple] = None,
        *,
        expect_total: Optional[int] = None,
        if_none_match: Optional[str] = None,
        kind: str = "range",
    ) -> Response:  # pragma: no cover - interface
        raise NotImplementedError

    def url(self, key: str) -> str:
        return f"{self.base}/{key.lstrip('/')}"

    def is_local(self) -> bool:
        return False


# ============================================================================================================== local directory
class DirTransport(Transport):
    """Serve keys from a directory laid out like the server (``<root>/store/m/GSE.json``, ``<root>/store/s/GSM/<d>.s1``)."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.base = self.root
        self.stats = Stats()

    def is_local(self) -> bool:
        return True

    def path(self, key: str) -> str:
        p = os.path.normpath(os.path.join(self.root, *key.lstrip("/").split("/")))
        if os.path.commonpath([p, self.root]) != self.root:
            raise TransportError(f"key escapes the data directory: {key!r}")
        return p

    def get(self, key, rng=None, *, expect_total=None, if_none_match=None, kind="range"):
        p = self.path(key)
        try:
            size = os.path.getsize(p)
        except OSError as e:
            raise NotFound(f"{p}: not found") from e
        if expect_total is not None and size != expect_total:
            raise IntegrityError(f"{key}: object is {size} bytes, expected {expect_total}")
        etag = f'"{size}-{int(os.path.getmtime(p) * 1000)}"'
        if if_none_match is not None and rng is None and if_none_match == etag:
            self.stats.add(kind, 0)
            return Response(304, {}, b"", size, etag)
        if rng is None:
            a, b = 0, size - 1
        else:
            a, b = rng
            if a < 0 or b < a or b >= size:
                raise IntegrityError(f"{key}: range {a}-{b} outside the {size}-byte object")
        n = b - a + 1
        buf = bytearray(n)
        with open(p, "rb") as fh:
            fh.seek(a)
            got = fh.readinto(buf) if n else 0
        if got != n:
            raise IntegrityError(f"{key}: short read ({got} of {n} bytes)")
        self.stats.add(kind, n)
        return Response(206 if rng is not None else 200, {}, buf, size, etag)


# ============================================================================================================== HTTP
class HttpTransport(Transport):
    """Persistent-connection HTTP client (one connection per thread), with retries and response validation."""

    def __init__(
        self,
        base: str,
        *,
        timeout: float = 60.0,
        retries: int = 4,
        user_agent: str = USER_AGENT,
        max_redirects: int = 3,
    ):
        u = urllib.parse.urlsplit(base)
        self.scheme = u.scheme.lower()
        self.netloc = u.netloc
        self.prefix = u.path.rstrip("/")
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.user_agent = user_agent
        self.max_redirects = max_redirects
        self.stats = Stats()
        self._tls = threading.local()
        self._lock = threading.Lock()
        self._ctx = ssl.create_default_context() if self.scheme == "https" else None

    # ------------------------------------------------------------------ connections
    def _conn(self, netloc: str, scheme: str):
        cache = getattr(self._tls, "conns", None)
        if cache is None:
            cache = self._tls.conns = {}
        c = cache.get((scheme, netloc))
        if c is None:
            if scheme == "https":
                c = http.client.HTTPSConnection(netloc, timeout=self.timeout, context=self._ctx)
            else:
                c = http.client.HTTPConnection(netloc, timeout=self.timeout)
            cache[(scheme, netloc)] = c
        return c

    def _drop(self, netloc: str, scheme: str) -> None:
        cache = getattr(self._tls, "conns", None)
        if cache:
            c = cache.pop((scheme, netloc), None)
            if c is not None:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass

    def close(self) -> None:
        cache = getattr(self._tls, "conns", None) or {}
        for c in cache.values():
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        self._tls.conns = {}

    # ------------------------------------------------------------------ requests
    def get(self, key, rng=None, *, expect_total=None, if_none_match=None, kind="range"):
        path = f"{self.prefix}/{urllib.parse.quote(key.lstrip('/'), safe='/')}"
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "identity",
            "Connection": "keep-alive",
        }
        if rng is not None:
            headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
        if if_none_match:
            headers["If-None-Match"] = if_none_match
        scheme, netloc = self.scheme, self.netloc
        last: Optional[BaseException] = None
        redirects = 0
        attempt = 0
        stale_ok = True  # one free retry when a kept-alive connection was closed by the server
        while attempt <= self.retries:
            conn = self._conn(netloc, scheme)
            reused = conn.sock is not None
            try:
                conn.request("GET", path, headers=headers)
                resp = conn.getresponse()
                status = resp.status
                if status in (301, 302, 307, 308) and redirects < self.max_redirects:
                    loc = resp.getheader("Location") or ""
                    resp.read()
                    nu = urllib.parse.urlsplit(
                        urllib.parse.urljoin(f"{scheme}://{netloc}{path}", loc)
                    )
                    if nu.scheme not in ("http", "https") or (
                        self.scheme == "https" and nu.scheme != "https"
                    ):
                        raise TransportError(f"refusing redirect to {loc!r}")
                    scheme, netloc, path = (
                        nu.scheme,
                        nu.netloc,
                        nu.path + (("?" + nu.query) if nu.query else ""),
                    )
                    redirects += 1
                    continue
                hdr = {k.lower(): v for k, v in resp.getheaders()}
                if status in _RETRY_STATUS:
                    resp.read()
                    last = TransportError(f"HTTP {status} for {key}")
                    delay = self._retry_after(hdr, attempt)
                    self._drop(netloc, scheme)
                else:
                    r = self._finish(resp, status, hdr, key, rng, expect_total, kind)
                    if hdr.get("connection", "").lower() == "close":
                        self._drop(netloc, scheme)
                    return r
            except (OSError, http.client.HTTPException, socket.timeout) as e:
                last = e
                self._drop(netloc, scheme)
                if (
                    reused
                    and stale_ok
                    and isinstance(
                        e,
                        (
                            http.client.RemoteDisconnected,
                            ConnectionResetError,
                            BrokenPipeError,
                            ConnectionAbortedError,
                        ),
                    )
                ):
                    stale_ok = False
                    continue
                delay = min(8.0, 0.5 * (2**attempt)) * (0.75 + 0.5 * random.random())
            attempt += 1
            if attempt <= self.retries:
                with self._lock:
                    self.stats.retries += 1
                time.sleep(delay)
        raise TransportError(
            f"GET {self.url(key)} failed after {self.retries + 1} attempts: {last}"
        )

    @staticmethod
    def _retry_after(hdr: Mapping[str, str], attempt: int) -> float:
        ra = hdr.get("retry-after")
        if ra and ra.isdigit():
            return min(30.0, float(ra))
        return min(8.0, 0.5 * (2**attempt)) * (0.75 + 0.5 * random.random())

    def _finish(self, resp, status, hdr, key, rng, expect_total, kind) -> Response:
        if status == 304:
            resp.read()
            self.stats.add(kind, 0)
            return Response(304, hdr, b"", None, hdr.get("etag"))
        if status == 404:
            resp.read()
            raise NotFound(f"{self.url(key)}: HTTP 404")
        if status == 403:
            resp.read()
            raise TransportError(
                f"{self.url(key)}: HTTP 403 (the data host rejects this client; check the User-Agent and $SINGLET_DATA_BASE)"
            )
        if status == 416:
            resp.read()
            raise IntegrityError(f"{self.url(key)}: HTTP 416, range {rng} is outside the object")
        if status not in (200, 206):
            resp.read()
            raise TransportError(f"{self.url(key)}: HTTP {status}")
        total: Optional[int] = None
        if status == 206:
            m = _CONTENT_RANGE.match(hdr.get("content-range", ""))
            if not m or rng is None:
                resp.read()
                raise IntegrityError(
                    f"{self.url(key)}: missing or unexpected Content-Range {hdr.get('content-range')!r}"
                )
            a, b = int(m.group(1)), int(m.group(2))
            total = None if m.group(3) == "*" else int(m.group(3))
            if (a, b) != tuple(rng):
                resp.read()
                raise IntegrityError(
                    f"{self.url(key)}: asked bytes {rng[0]}-{rng[1]}, server sent {a}-{b}"
                )
            if expect_total is not None and total != expect_total:
                resp.read()
                raise IntegrityError(
                    f"{self.url(key)}: object is {total} bytes, expected {expect_total} (stale or replaced object)"
                )
            want = b - a + 1
        else:  # 200: the whole object
            cl = hdr.get("content-length")
            total = int(cl) if cl and cl.isdigit() else None
            if rng is not None and not (rng[0] == 0 and total is not None and rng[1] >= total - 1):
                resp.read()
                raise IntegrityError(f"{self.url(key)}: server ignored the Range header")
            if expect_total is not None and total is not None and total != expect_total:
                resp.read()
                raise IntegrityError(
                    f"{self.url(key)}: object is {total} bytes, expected {expect_total}"
                )
            want = total
        body = self._read_body(resp, want)
        if want is not None and len(body) != want:
            raise IntegrityError(f"{self.url(key)}: body is {len(body)} bytes, expected {want}")
        self.stats.add(kind, len(body))
        return Response(status, hdr, body, total, hdr.get("etag"))

    @staticmethod
    def _read_body(resp, want: Optional[int]):
        if want is None:
            return resp.read()
        buf = bytearray(want)
        mv = memoryview(buf)
        pos = 0
        while pos < want:
            n = resp.readinto(mv[pos:])
            if not n:
                break
            pos += n
        return buf if pos == want else buf[:pos]
