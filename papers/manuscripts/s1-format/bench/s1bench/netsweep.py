"""Stage `netsweep` (review round 3): what round-trip time, requests in flight and DataLoader workers do to the readers that
fetch by range, and the same readers against a real object store.

Emulated grids, on the store of netem.py at `netsweep.gbit` with bwproxy counting the bytes and requests it serves:
  latency   every loader in `netsweep.loaders` at each added round-trip time in `netsweep.rtt_ms`
  inflight  s1_stream at each `netsweep.inflight_rtt_ms`, for each (io_threads, ahead) pair in `netsweep.inflight`:
            io_threads requests at once, the next `ahead` minibatches' requests sent while one decodes
  workers   s1_cells and scdataset under a torch DataLoader with each count in `netsweep.workers`, from the node's
            disk and from the store at `netsweep.workers_rtt_ms`
Real store (`--r2`): the loaders in `netsweep.r2_loaders` against Cloudflare R2, read over the public internet from the
timing node through a Worker that serves the bucket by range (bench/r2worker). S1_R2_URL names the bucket's base URL
(it contains the read capability); S1_R2_WRITE is the upload token. Objects are uploaded from the cluster once.

Each run is one seed in a fresh process (minibatch.measure_one): the first minibatch, then `netsweep.n_batches` sustained
ones with the time of each kept for the latency distributions.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .util import Config, Unit, log, read_tsv, result_path, write_tsv
from .write import format_dir


def _files(cfg: Config, u: Unit, loaders) -> dict[str, Path]:
    from . import formats
    from .minibatch import LOADERS

    return {ld: formats.get([LOADERS[ld].fmt])[0].path(format_dir(cfg, u)) for ld in loaders}


def _sets(ns: dict, **kw) -> list[str]:
    s = [f"minibatch.n_batches={ns['n_batches']}", f"minibatch.warmup_batches={ns['warmup_batches']}"]
    return s + [f"minibatch._{k}={v}" for k, v in kw.items() if v is not None]


def run(cfg: Config, units: list[Unit], force: bool = False, overrides=(), r2: bool = False) -> None:
    from . import netem
    from .minibatch import _child

    ns = cfg["netsweep"]
    H = int(cfg["threads"]["headline"])
    for u in units:
        out = result_path(cfg, "netsweep", u, ".r2" if r2 else "")
        if out.exists() and not force:
            log(f"{u.key}: netsweep results exist")
            continue
        n_cells = int(read_tsv(result_path(cfg, "stats", u))[0]["cells"])
        rows: list[dict] = []

        def go(grid, ld, src, seed=0, extra_sets=(), **meta):
            res = _child(cfg, [*overrides, *extra_sets], u.key, ld, src, seed, H, n_cells)
            row = {"dataset": u.id, "matrix": u.name, "grid": grid, "loader": ld, "seed": seed, "threads": H, **meta, **res}
            rows.append(row)
            log(f"{u.key} {grid} {ld} {meta}: first {res.get('first_batch_s', res.get('note'))} s, "
                f"{res.get('sustained_cells_per_s', '')} cells/s")

        if r2:
            _r2(cfg, u, ns, go)
            write_tsv(rows, out)
            continue
        loaders = sorted(set(ns["loaders"]) | {"s1_stream", "s1_cells", "scdataset"})
        files = _files(cfg, u, loaders)
        work = cfg.data_dir / "tmp" / f"netsweep_{u.key}"
        stats = work / "proxy.stats"
        keys = {f"{u.key}/{p.name}": p for p in set(files.values())}
        with netem.object_store(cfg, keys, work, gbit=float(ns["gbit"]), stats=stats) as base:
            for k, f in keys.items():   # rclone's first request into a directory lists it (about a second): not timed
                if Path(f).is_file():
                    netem._get(f"{base}/{k}", (0, 1))
            url = {ld: f"{base}/{u.key}/{files[ld].name}" for ld in loaders}
            for rtt in ns["rtt_ms"]:
                netem.set_tier(cfg, {"latency_ms": rtt})
                for ld in ns["loaders"]:
                    go("latency", ld, url[ld], extra_sets=_sets(ns, proxy_stats=stats), gbit=ns["gbit"], rtt_ms=rtt)
            for rtt in ns["inflight_rtt_ms"]:
                netem.set_tier(cfg, {"latency_ms": rtt})
                for io, ahead in ns["inflight"]:
                    go("inflight", "s1_stream", url["s1_stream"], extra_sets=_sets(ns, proxy_stats=stats, io_threads=io, ahead=ahead),
                       gbit=ns["gbit"], rtt_ms=rtt, io_threads=io, ahead=ahead)
            netem.set_tier(cfg, {"latency_ms": ns["workers_rtt_ms"]})
            for w in ns["workers"]:
                for ld in ("s1_cells", "scdataset"):
                    go("workers", ld, url[ld], extra_sets=_sets(ns, proxy_stats=stats, workers=w), gbit=ns["gbit"],
                       rtt_ms=ns["workers_rtt_ms"], workers=w, where="store")
        for w in ns["workers"]:
            for ld in ("s1_cells", "scdataset"):
                go("workers", ld, str(files[ld]), extra_sets=_sets(ns, workers=w), workers=w, where="local")
        write_tsv(rows, out)


# ----------------------------------------------------------------------------------------------- real object store (R2)
def _r2_base() -> str:
    base = os.environ.get("S1_R2_URL", "").rstrip("/")
    if not base:
        raise SystemExit("S1_R2_URL is not set (see bench/r2worker/README.md)")
    return base


def _pool():
    import urllib3

    return urllib3.PoolManager(maxsize=16, retries=urllib3.Retry(total=4, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504]),
                               timeout=urllib3.Timeout(connect=30, read=300))


def upload(base: str, key: str, path: Path, token: str, part: int = 64 << 20) -> int:
    """Put one file at base/key through the Worker (multipart above `part` bytes); skips an object of the same size."""
    import json

    pool = _pool()
    size = path.stat().st_size
    h = pool.request("HEAD", f"{base}/{key}")
    if h.status == 200 and int(h.headers.get("Content-Length", -1)) == size:
        return 0
    auth = {"x-s1-write": token}
    if size <= part:
        r = pool.request("PUT", f"{base}/{key}", body=path.read_bytes(), headers=auth)
        if r.status != 200:
            raise IOError(f"PUT {key}: HTTP {r.status} {r.data[:200]!r}")
        return size
    r = pool.request("POST", f"{base}/{key}?uploads", headers=auth)
    uid = json.loads(r.data)["uploadId"]
    parts = []
    with path.open("rb") as f:
        n = 1
        while True:
            b = f.read(part)
            if not b:
                break
            r = pool.request("PUT", f"{base}/{key}?uploadId={uid}&partNumber={n}", body=b, headers=auth)
            if r.status != 200:
                raise IOError(f"part {n} of {key}: HTTP {r.status}")
            parts.append(json.loads(r.data))
            n += 1
    r = pool.request("POST", f"{base}/{key}?uploadId={uid}", body=json.dumps(parts).encode(), headers=auth)
    if r.status != 200:
        raise IOError(f"complete {key}: HTTP {r.status} {r.data[:200]!r}")
    return size


def upload_tree(base: str, prefix: str, path: Path, token: str, threads: int = 8) -> int:
    from concurrent.futures import ThreadPoolExecutor

    items = [(f"{prefix}/{path.name}", path)] if path.is_file() else \
        [(f"{prefix}/{path.name}/{q.relative_to(path).as_posix()}", q) for q in sorted(path.rglob("*")) if q.is_file()]
    with ThreadPoolExecutor(threads) as ex:
        return sum(ex.map(lambda kv: upload(base, kv[0], kv[1], token), items))


def link(base: str, key: str, size: int, n_rtt: int = 20) -> dict:
    """The path to the store: median and 90th percentile time of a one-byte range GET on a warm connection (the request
    round trip, including the store's own time to first byte), and the rate of reading `key` whole on one connection and
    in 8 ranges at once."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    pool = _pool()
    url = f"{base}/{key}"
    pool.request("GET", url, headers={"Range": "bytes=0-0"})   # connection set-up and TLS, not counted
    ts = []
    for _ in range(n_rtt):
        t0 = time.perf_counter()
        pool.request("GET", url, headers={"Range": "bytes=0-0"})
        ts.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    n1 = len(pool.request("GET", url).data)
    one = n1 * 8 / (time.perf_counter() - t0) / 1e9
    cuts = [size * k // 8 for k in range(9)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as ex:
        n8 = sum(ex.map(lambda k: len(pool.request("GET", url, headers={"Range": f"bytes={cuts[k]}-{cuts[k + 1] - 1}"}).data), range(8)))
    eight = n8 * 8 / (time.perf_counter() - t0) / 1e9
    return {"rtt_median_ms": 1e3 * float(np.median(ts)), "rtt_p90_ms": 1e3 * float(np.quantile(ts, 0.9)),
            "gbit_1conn": one, "gbit_8conn": eight, "calib_bytes": size}


def _r2(cfg: Config, u: Unit, ns: dict, go) -> None:
    base = _r2_base()
    token = os.environ.get("S1_R2_WRITE", "")
    loaders = list(ns["r2_loaders"])
    files = _files(cfg, u, loaders)
    prefix = f"bench/{u.key}"
    if token:
        t0 = time.perf_counter()
        n = sum(upload_tree(base, prefix, p, token) for p in set(files.values()))
        log(f"{u.key}: uploaded {n / 2**20:.0f} MiB to R2 in {time.perf_counter() - t0:.0f} s")
    s1 = files["s1_stream"]
    lk = link(base, f"{prefix}/{s1.name}", s1.stat().st_size)
    log(f"{u.key}: R2 link {lk}")
    for ld in loaders:
        for seed in cfg["minibatch"]["seeds"]:
            go("r2", ld, f"{base}/{prefix}/{files[ld].name}", seed=seed, extra_sets=_sets(ns), store="r2", **lk)
