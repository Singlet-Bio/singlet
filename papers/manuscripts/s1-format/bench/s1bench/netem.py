"""Network tiers for the minibatch stage: an S3-compatible object store behind Toxiproxy (bandwidth and latency toxics).

The store is `rclone serve s3` over a local directory (netem.store: rclone, the default) or a MinIO server
(netem.store: minio; MinIO stopped publishing community binaries in 2025, so rclone is what the Clipper runs use).
No root is needed: each is a single static binary. Objects are served anonymously over plain HTTP so that every reader
can use HTTP range requests or a whole-object GET through the same throttled port. Before the timed runs, `calibrate`
measures what the proxy sustains with no toxic; a tier above that ceiling is reported as capped, not as measured.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .util import Config, log


def _api(cfg: Config, method: str, path: str, body: Optional[dict] = None) -> dict:
    port = cfg["netem"]["toxiproxy_api_port"]
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        txt = r.read()
    return json.loads(txt) if txt else {}


def _wait(url: str, timeout: float = 30.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception as e:  # MinIO answers 403 on / when it is up
            if getattr(e, "code", None) in (400, 403):
                return
            time.sleep(0.3)
    raise RuntimeError(f"{url} did not come up")


@contextmanager
def object_store(cfg: Config, files: dict[str, Path], workdir: Path) -> Iterator[str]:
    """Start MinIO and Toxiproxy, upload `files` (key -> local path) and yield the proxied base URL."""
    n = cfg["netem"]
    workdir.mkdir(parents=True, exist_ok=True)
    if n.get("store", "rclone") == "rclone":
        with _rclone_store(cfg, files, workdir) as base:
            yield base
        return
    env = {"MINIO_ROOT_USER": n["access_key"], "MINIO_ROOT_PASSWORD": n["secret_key"], "PATH": "/usr/bin:/bin"}
    minio = subprocess.Popen([n["minio"], "server", str(workdir / "minio"), "--address", f"127.0.0.1:{n['minio_port']}", "--quiet"],
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tox = subprocess.Popen([n["toxiproxy_server"], "-port", str(n["toxiproxy_api_port"])], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    try:
        _wait(f"http://127.0.0.1:{n['minio_port']}/minio/health/live")
        _wait(f"http://127.0.0.1:{n['toxiproxy_api_port']}/version")
        mc = [n["mc"], "--config-dir", str(workdir / "mc")]
        subprocess.run([*mc, "alias", "set", "b", f"http://127.0.0.1:{n['minio_port']}", n["access_key"], n["secret_key"]], check=True,
                       capture_output=True)
        subprocess.run([*mc, "mb", "--ignore-existing", f"b/{n['bucket']}"], check=True, capture_output=True)
        subprocess.run([*mc, "anonymous", "set", "download", f"b/{n['bucket']}"], check=True, capture_output=True)
        for key, p in files.items():
            subprocess.run([*mc, "cp", "--recursive" if p.is_dir() else "--quiet", str(p), f"b/{n['bucket']}/{key}"], check=True,
                           capture_output=True)
        _api(cfg, "POST", "/proxies", {"name": "s3", "listen": f"127.0.0.1:{n['proxy_port']}",
                                       "upstream": f"127.0.0.1:{n['minio_port']}", "enabled": True})
        yield f"http://127.0.0.1:{n['proxy_port']}/{n['bucket']}"
    finally:
        for p in (tox, minio):
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(workdir / "minio", ignore_errors=True)


def _place(src: Path, dst: Path) -> None:
    """Put a file (or a directory tree) at dst by hard link when it is on the same filesystem, else by copy."""
    import os

    dst.parent.mkdir(parents=True, exist_ok=True)

    def link(a, b):
        try:
            os.link(a, b)
        except OSError:
            shutil.copy2(a, b)

    if src.is_dir():
        shutil.copytree(src, dst, copy_function=link, dirs_exist_ok=True)
    else:
        link(src, dst)


@contextmanager
def _rclone_store(cfg: Config, files: dict[str, Path], workdir: Path) -> Iterator[str]:
    """`rclone serve s3` over workdir/s3 (bucket = a directory, anonymous access, path-style), behind Toxiproxy."""
    n = cfg["netem"]
    root = workdir / "s3"
    for key, p in files.items():
        _place(p, root / n["bucket"] / key)
    port = n["minio_port"]
    srv = subprocess.Popen([n.get("rclone", "rclone"), "serve", "s3", str(root), "--addr", f"127.0.0.1:{port}",
                            "--force-path-style", "--vfs-cache-mode", "off", "--log-level", "ERROR"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tox = subprocess.Popen([n["toxiproxy_server"], "-port", str(n["toxiproxy_api_port"])], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    try:
        _wait(f"http://127.0.0.1:{port}/")
        _wait(f"http://127.0.0.1:{n['toxiproxy_api_port']}/version")
        _api(cfg, "POST", "/proxies", {"name": "s3", "listen": f"127.0.0.1:{n['proxy_port']}",
                                       "upstream": f"127.0.0.1:{port}", "enabled": True})
        yield f"http://127.0.0.1:{n['proxy_port']}/{n['bucket']}"
    finally:
        for p in (tox, srv):
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(root, ignore_errors=True)


def set_tier(cfg: Config, tier: dict) -> None:
    """Replace the toxics on the proxy with the tier's bandwidth (both directions) and latency."""
    for t in _api(cfg, "GET", "/proxies/s3/toxics") or []:
        _api(cfg, "DELETE", f"/proxies/s3/toxics/{t['name']}")
    if tier.get("gbit"):
        rate_kb = int(tier["gbit"] * 1e9 / 8 / 1000)   # toxiproxy bandwidth rate is in KB/s
        for stream in ("downstream", "upstream"):
            _api(cfg, "POST", "/proxies/s3/toxics", {"name": f"bw_{stream}", "type": "bandwidth", "stream": stream,
                                                     "attributes": {"rate": rate_kb}})
    if tier.get("latency_ms"):
        _api(cfg, "POST", "/proxies/s3/toxics", {"name": "lat", "type": "latency", "stream": "downstream",
                                                 "attributes": {"latency": int(tier["latency_ms"]), "jitter": 0}})


def calib_blob(cfg: Config, tmp: Path) -> Path:
    """A random object of netem.calib_mb (default 1024) MB for `calibrate`: a small file measures TCP ramp-up, not the
    ceiling. Made once per data_dir."""
    import os

    mb = int(cfg["netem"].get("calib_mb", 1024))
    p = tmp / f"calib_{mb}mb.bin"
    if not p.exists() or p.stat().st_size != mb << 20:
        tmp.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            for _ in range(mb // 16):
                f.write(os.urandom(16 << 20))
    return p


def _get(url: str, rng: Optional[tuple[int, int]] = None) -> int:
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1] - 1}"} if rng else {})
    n = 0
    with urllib.request.urlopen(req) as r:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            n += len(b)
    return n


def calibrate(cfg: Config, url: str, nbytes_hint: int, streams: int = 1) -> float:
    """Gbit/s the proxy sustains with no toxics: one GET of the whole object, or `streams` concurrent range GETs that
    split it. The tier toxics act on each connection separately, so the one-stream figure is what caps a tier."""
    from concurrent.futures import ThreadPoolExecutor

    set_tier(cfg, {})
    t0 = time.perf_counter()
    if streams <= 1:
        n = _get(url)
    else:
        size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
        cuts = [size * k // streams for k in range(streams + 1)]
        with ThreadPoolExecutor(streams) as ex:
            n = sum(ex.map(lambda k: _get(url, (cuts[k], cuts[k + 1])), range(streams)))
    g = n * 8 / (time.perf_counter() - t0) / 1e9
    log(f"proxy ceiling: {g:.2f} Gbit/s over {n} bytes, {streams} stream(s)")
    return g
