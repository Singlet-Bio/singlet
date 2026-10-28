"""Network tiers for the minibatch stage: a MinIO object store behind Toxiproxy (bandwidth and latency toxics).

No root is needed: both are single static binaries. Objects are served anonymously over plain HTTP so that every reader
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


def calibrate(cfg: Config, url: str, nbytes_hint: int) -> float:
    """Gbit/s the proxy sustains for one large GET with no toxics."""
    set_tier(cfg, {})
    t0 = time.perf_counter()
    with urllib.request.urlopen(url) as r:
        n = 0
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            n += len(b)
    g = n * 8 / (time.perf_counter() - t0) / 1e9
    log(f"proxy ceiling: {g:.2f} Gbit/s over {n} bytes")
    return g
