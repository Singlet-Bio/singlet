"""Stage `minibatch` (Q3, Q4): cold start to the first minibatch, and sustained cells per second, per loader and tier.

Workload: a seeded sequence of minibatches, each made of `batch_size / block_size` blocks of `block_size` contiguous
cells drawn without replacement. Every loader serves the same sequence (scDataset uses its own block sampler with the
same block size and is marked as such). Every loader returns a cells x genes torch.sparse_csr_tensor.

Each (dataset, loader, tier, seed) runs in a fresh process pinned to `threads.headline` cores with a cold page cache.
Remote tiers read through the throttled object store (netem.py): loaders that can do range reads use them, the others
download the object first (recorded in the `access` column).

Loader adapters below follow the pinned versions in environment.yml. If an API moved, fix the adapter, keep the
workload, and note the change in the PR.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from .util import BENCH_DIR, Config, Unit, evict, log, pin_cores, result_path, write_tsv
from .write import format_dir


# ----------------------------------------------------------------------------------------------- workload
def workload(n_cells: int, batch_size: int, block_size: int, n_batches: int, seed: int) -> list[list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n_cells - block_size + 1, block_size)
    rng.shuffle(starts)
    per = batch_size // block_size
    need = per * n_batches
    starts = np.resize(starts, need) if len(starts) < need else starts[:need]
    return [sorted((int(s), int(s) + block_size) for s in starts[i * per:(i + 1) * per]) for i in range(n_batches)]


def _idx(blocks) -> np.ndarray:
    return np.concatenate([np.arange(a, b) for a, b in blocks])


def _csr_to_tensor(x):
    import torch

    x = x.tocsr()
    return torch.sparse_csr_tensor(torch.from_numpy(x.indptr.astype(np.int64)), torch.from_numpy(x.indices.astype(np.int64)),
                                   torch.from_numpy(x.data.astype(np.float32)), size=x.shape)


# ----------------------------------------------------------------------------------------------- loaders
class Loader:
    """open() once, then read(blocks) -> scipy CSR or tensor; to_tensor() converts. `access` is 'range' or 'download'.
    `modules` are imported before the clock starts: a training script has them loaded already."""

    fmt = ""
    access = "download"
    returns_tensor = False
    modules: tuple = ("anndata", "h5py", "scipy.sparse")

    def open(self, src: str, threads: int):
        raise NotImplementedError

    def read(self, blocks):
        raise NotImplementedError

    def to_tensor(self, x):
        return _csr_to_tensor(x)

    def split(self) -> dict:
        return {}


def _next_epoch(loader, restart):
    """Next batch of a loader with its own sampler; on a small matrix the sustained loop outlasts one epoch, and a training
    loop would then start the next one."""
    try:
        return next(loader.it)
    except StopIteration:
        loader.it = restart()
        return next(loader.it)


class AnndataInmem(Loader):
    fmt = "h5ad_gzip"

    def open(self, src, threads):
        import anndata as ad

        self.x = ad.read_h5ad(src).X.tocsr()

    def read(self, blocks):
        return self.x[_idx(blocks)]


class AnndataBacked(Loader):
    fmt = "h5ad_gzip"

    def open(self, src, threads):
        import h5py

        try:
            from anndata.io import sparse_dataset
        except ImportError:
            from anndata.experimental import sparse_dataset
        self.f = h5py.File(src, "r")
        self.x = sparse_dataset(self.f["X"])

    def read(self, blocks):
        return self.x[_idx(blocks)]


class AnndataZarrLazy(AnndataBacked):
    fmt = "zarr"
    access = "range"
    modules = ("anndata", "zarr", "fsspec", "scipy.sparse")

    def open(self, src, threads):
        import zarr

        try:
            from anndata.io import sparse_dataset
        except ImportError:
            from anndata.experimental import sparse_dataset
        try:
            from numcodecs import blosc

            blosc.set_nthreads(threads)
        except Exception:
            pass
        if src.startswith("http"):
            import fsspec

            g = zarr.open_consolidated(fsspec.get_mapper(src), mode="r")
        else:
            g = zarr.open(src, mode="r")
        self.x = sparse_dataset(g["X"])


class ScDataset(Loader):
    """scDataset block shuffling over backed h5ad (D'Ascenzo and Cultrera di Montesano 2025)."""

    fmt = "h5ad_gzip"
    own_sampler = True
    modules = ("anndata", "h5py", "scdataset")
    default_fetch = False   # True: the package's own fetch_factor (16 batches per fetch) instead of one batch per fetch

    def open(self, src, threads):
        import anndata as ad
        from scdataset import BlockShuffling, scDataset

        self.a = ad.read_h5ad(src, backed="r")
        # a fetch is a view of the backed file; batches are sliced from it, and backed AnnData cannot view a view, so each
        # fetch is read into memory (the fetch_transform of the scDataset AnnData examples)
        kw = {} if self.default_fetch else {"fetch_factor": self.fetch_factor}
        self.ds = scDataset(self.a, BlockShuffling(block_size=self.block_size), batch_size=self.batch_size,
                            fetch_transform=lambda v: v.to_memory(), **kw)
        self.src = self.ds if not self.workers else _dataloader(self.ds, self.workers)
        self.it = iter(self.src)

    def read(self, blocks):
        b = _next_epoch(self, lambda: iter(self.src))
        return getattr(b, "X", b)


class SomaMl(Loader):
    fmt = "soma"
    access = "range"
    own_sampler = True
    modules = ("tiledbsoma", "tiledbsoma_ml")

    def open(self, src, threads):
        import tiledbsoma
        import tiledbsoma_ml as sml

        cfg = {"sm.compute_concurrency_level": threads, "sm.io_concurrency_level": threads}
        if src.startswith("http"):
            # MinIO through the proxy: TileDB talks S3 to the proxied endpoint
            host, bucket_key = src.split("://", 1)[1].split("/", 1)
            cfg.update({"vfs.s3.endpoint_override": host, "vfs.s3.scheme": "http", "vfs.s3.use_virtual_addressing": "false",
                        "vfs.s3.no_sign_request": "true"})
            src = "s3://" + bucket_key
        ctx = tiledbsoma.SOMATileDBContext(tiledb_config=cfg)
        self.exp = tiledbsoma.Experiment.open(src, context=ctx)
        self.q = self.exp.axis_query("RNA")
        # the shared workload: blocks of block_size cells, one batch per read, no larger shuffle buffer to fill first
        ds = sml.ExperimentDataset(self.q, layer_name="data", batch_size=self.batch_size, shuffle=True, seed=self.seed,
                                   return_sparse_X=True, shuffle_chunk_size=self.block_size, io_batch_size=self.batch_size)
        self.dl = sml.experiment_dataloader(ds)
        self.it = iter(self.dl)

    def read(self, blocks):
        x, _obs = _next_epoch(self, lambda: iter(self.dl))
        return x

    def to_tensor(self, x):
        import scipy.sparse as sp

        return x if hasattr(x, "is_sparse_csr") else _csr_to_tensor(sp.csr_matrix(x))


class S1StreamLoader(Loader):
    fmt = "s1"
    access = "range"
    returns_tensor = True
    order = "native"
    modules = ("singlet.torch", "requests")

    # a batch's range requests sent together: 8 over HTTP (the singlet 1.3.2 default, which the main tables measured; 1.3.3
    # raised "auto" to 32), 1 for a local file
    io_threads = "auto"

    def open(self, src, threads):
        from singlet.torch import S1Stream   # the package's streaming reader (local path or HTTP ranges)

        io = getattr(self, "io_override", None) or self.io_threads
        if io == "auto":
            io = 8 if src.startswith("http") else 1
        self.s = S1Stream(src, member="counts", threads=threads, order=self.order, io_threads=io)

    def read(self, blocks):
        return self.s.batch(blocks)

    def iterate(self, batches):
        # as S1Cells reads (prefetch=1): the next batch's requests are in flight while one decodes
        return self.s.iter_batches(batches, ahead=getattr(self, "ahead", 1))

    def to_tensor(self, x):
        return x

    def split(self):
        return {"fetch_s": self.s.fetch_s, "decode_s": self.s.decode_s, "bytes": self.s.bytes_fetched, "requests": self.s.requests}


class S1StreamCanonical(S1StreamLoader):
    order = "canonical"


class S1StreamSeq(S1StreamLoader):
    """The same reader sending a batch's range requests one at a time, to show what sending them together buys."""
    io_threads = 1


class ScDatasetDefault(ScDataset):
    """scDataset as its documentation configures it: fetch_factor left at the package default."""
    default_fetch = True


def _dataloader(ds, workers: int):
    import torch

    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, persistent_workers=True, prefetch_factor=2)


class S1CellsLoader(Loader):
    """The package's torch dataset (S1Cells, its own block sampler) under a torch DataLoader with `workers` processes, each
    with threads // workers decode threads; workers = 0 reads in the training process."""

    fmt = "s1"
    access = "range"
    returns_tensor = True
    own_sampler = True
    modules = ("singlet.torch", "torch")

    def open(self, src, threads):
        from singlet.torch import S1Cells

        per = max(1, threads // max(1, self.workers))
        self.ds = S1Cells(src, batch_size=self.batch_size, block_size=self.block_size, seed=self.seed, threads=per)
        self.src = self.ds if not self.workers else _dataloader(self.ds, self.workers)
        self.it = iter(self.src)

    def read(self, blocks):
        return _next_epoch(self, lambda: iter(self.src))["X"]

    def to_tensor(self, x):
        return x


class Zarr3ShardLoader(Loader):
    """CSR in sharded Zarr v3 (formats.Zarr3Shard): indptr read at open, then per block the data and indices slices it
    needs, all of a batch's slices at once on a thread pool (zarr fetches the inner chunks of each slice itself)."""

    fmt = "zarr3"
    access = "range"
    modules = ("zarr", "fsspec", "scipy.sparse")

    def open(self, src, threads):
        from concurrent.futures import ThreadPoolExecutor

        import zarr

        zarr.config.set({"async.concurrency": 64, "threading.max_workers": threads})
        if src.startswith("http"):
            store = zarr.storage.FsspecStore.from_url(src, read_only=True)
        else:
            store = zarr.storage.LocalStore(src, read_only=True)
        g = zarr.open_group(store, mode="r")
        self.shape = tuple(g.attrs["shape"])
        self.indptr = g["indptr"][:]
        self.d, self.ix = g["data"], g["indices"]
        self.pool = ThreadPoolExecutor(max(8, threads))

    def read(self, blocks):
        import scipy.sparse as sp

        runs = list(_merge(blocks))
        spans = [(int(self.indptr[a]), int(self.indptr[b])) for a, b in runs]
        fd = [self.pool.submit(lambda s=s: self.d[s[0]:s[1]]) for s in spans]
        fi = [self.pool.submit(lambda s=s: self.ix[s[0]:s[1]]) for s in spans]
        data = np.concatenate([f.result() for f in fd]) if fd else np.zeros(0, np.uint32)
        ind = np.concatenate([f.result() for f in fi]) if fi else np.zeros(0, np.uint32)
        ip = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(self.indptr[a:b + 1]) for a, b in runs]))])
        return sp.csr_matrix((data, ind, ip), shape=(len(ip) - 1, self.shape[1]))


def _merge(blocks):
    cur = None
    for a, b in sorted(blocks):
        if cur and a <= cur[1]:
            cur = (cur[0], max(cur[1], b))
        else:
            if cur:
                yield cur
            cur = (a, b)
    if cur:
        yield cur


LOADERS = {
    "anndata_inmem": AnndataInmem,
    "anndata_backed": AnndataBacked,
    "scdataset": ScDataset,
    "anndata_zarr_lazy": AnndataZarrLazy,
    "soma_ml": SomaMl,
    "s1_stream": S1StreamLoader,
    "s1_stream_canonical": S1StreamCanonical,
    "s1_stream_seq": S1StreamSeq,
    "scdataset_default": ScDatasetDefault,
    "s1_cells": S1CellsLoader,
    "zarr3_shard": Zarr3ShardLoader,
}


# ----------------------------------------------------------------------------------------------- one measurement (child process)
def _download(url: str, dst: Path) -> None:
    # a named client: Cloudflare refuses urllib's default User-Agent (HTTP 403 from the R2 Worker)
    req = urllib.request.Request(url, headers={"User-Agent": "s1bench/1"})
    with urllib.request.urlopen(req) as r, dst.open("wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)


def measure_one(cfg: Config, unit_key: str, loader: str, src: str, seed: int, threads: int) -> dict:
    """Runs inside the child process: cold open, first minibatch, then sustained throughput."""
    pin_cores(threads)
    mb = cfg["minibatch"]
    import torch  # noqa: F401  (import cost outside the timed region, as in a training script)

    cuda = torch.cuda.is_available() and mb.get("device", "auto") in ("auto", "cuda")
    if cuda:  # CUDA context creation is a one-off cost of the training script, not of the loader
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()

    L = LOADERS[loader]()
    import importlib

    for m in L.modules:
        importlib.import_module(m)
    L.batch_size, L.block_size, L.seed = int(mb["batch_size"]), int(mb["block_size"]), seed
    L.fetch_factor = int(mb.get("fetch_factor", 1))   # one batch per fetch: the same 8 blocks of 64 cells as every loader
    L.workers = int(mb.get("_workers") or 0)          # DataLoader worker processes (s1_cells, scdataset)
    if mb.get("_io_threads"):                         # the in-flight sweep (netsweep): requests at once, batches ahead
        L.io_override = int(mb["_io_threads"])
    if mb.get("_ahead") not in (None, ""):
        L.ahead = int(mb["_ahead"])
    rec: dict = {"access": L.access if src.startswith("http") else "local"}
    stats = _proxy_reader(mb.get("_proxy_stats"))
    c0 = stats()
    t0 = time.perf_counter()
    local = src
    if src.startswith("http") and L.access == "download":
        tmp = Path(tempfile.mkdtemp(dir=cfg.data_dir / "tmp"))
        local = str(tmp / Path(src).name)
        _download(src, Path(local))
        rec["download_s"] = time.perf_counter() - t0
    elif not src.startswith("http"):
        evict(Path(src))
    t1 = time.perf_counter()
    L.open(local, threads)
    t2 = time.perf_counter()
    batches = workload(int(mb["_n_cells"]), L.batch_size, L.block_size, int(mb["n_batches"]) + int(mb["warmup_batches"]) + 1, seed)
    x = L.read(batches[0])
    t3 = time.perf_counter()
    tens = L.to_tensor(x)
    t4 = time.perf_counter()
    rec.update({"open_s": t2 - t1 + rec.get("download_s", 0.0), "first_read_s": t3 - t2, "to_tensor_s": t4 - t3,
                "first_batch_s": t4 - t0, "batch_rows": int(tens.size(0))})
    if cuda:  # reported separately: host-to-device copy of the first minibatch, after it is ready on the CPU
        t5 = time.perf_counter()
        tens.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        rec.update({"to_cuda_s": time.perf_counter() - t5, "device": torch.cuda.get_device_name(0)})
    rec.update({f"first_{k}": v for k, v in L.split().items()})
    c1 = stats()
    if c0 is not None and c1 is not None:   # bytes and requests the store served for the open and the first minibatch
        rec.update({"first_proxy_bytes": c1[0] - c0[0], "first_proxy_requests": c1[1] - c0[1]})
    if mb.get("_first_only"):
        rec["sustained_batches"] = 0
        return rec
    warm = int(mb["warmup_batches"])
    if hasattr(L, "iterate"):  # a loader that reads ahead on its own (the .s1 reader, as S1Cells uses it)
        it = L.iterate(batches[1:])
        for _ in range(warm):
            L.to_tensor(next(it))
        c2, t5 = stats(), time.perf_counter()
        cells, per, last = 0, [], t5
        for x in it:
            cells += int(L.to_tensor(x).size(0))
            now = time.perf_counter()
            per.append(now - last)
            last = now
    else:
        for b in batches[1:1 + warm]:
            L.to_tensor(L.read(b))
        c2, t5 = stats(), time.perf_counter()
        cells, per = 0, []
        for b in batches[1 + warm:]:
            t = time.perf_counter()
            cells += int(L.to_tensor(L.read(b)).size(0))
            per.append(time.perf_counter() - t)
    t6 = time.perf_counter()
    c3 = stats()
    rec["batch_s"] = ",".join(f"{v:.5f}" for v in per)   # seconds per sustained minibatch, for the latency CDFs
    if c2 is not None and c3 is not None:
        nb = max(1, len(per))
        rec.update({"proxy_bytes_per_batch": (c3[0] - c2[0]) / nb, "proxy_requests_per_batch": (c3[1] - c2[1]) / nb})
    rec.update({"sustained_cells_per_s": cells / (t6 - t5), "sustained_batches": len(batches) - 1 - int(mb["warmup_batches"])})
    try:
        import resource

        rec["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        pass
    return rec


def _proxy_reader(path):
    """Counters of bwproxy (bytes sent to clients, requests seen), or None when the store has no counting proxy. The
    timed region excludes these reads: they happen between the clock readings that bound each phase."""
    if not path:
        return lambda: None

    def read():
        time.sleep(0.25)   # the proxy writes its counters every 0.1 s
        try:
            b, r = Path(path).read_text().split()
            return int(b), int(r)
        except (OSError, ValueError):
            return None

    return read


def _child(cfg: Config, overrides, unit_key, loader, src, seed, threads, n_cells) -> dict:
    first_only = seed not in cfg["minibatch"].get("sustained_seeds", cfg["minibatch"]["seeds"])
    cmd = [sys.executable, "-m", "s1bench", "--config", str(cfg.path), *sum((["--set", o] for o in overrides), []),
           "--set", f"minibatch._n_cells={n_cells}", "--set", f"minibatch._first_only={str(first_only).lower()}",
           "minibatch-one", unit_key, loader, src, str(seed), str(threads)]
    from .decode import child_env

    r = subprocess.run(cmd, capture_output=True, text=True, cwd=BENCH_DIR, timeout=3 * 3600,
                       env=child_env(LOADERS[loader].fmt if loader in LOADERS else ""))
    if r.returncode != 0:
        return {"status": "error", "note": (r.stderr.strip().splitlines() or ["failed"])[-1]}
    return {"status": "ok", **json.loads(r.stdout.strip().splitlines()[-1])}


def _bpcells(cfg: Config, path: Path, seed: int, threads: int, n_cells: int, src_url: Optional[str]) -> dict:
    mb = cfg["minibatch"]
    wl = workload(n_cells, int(mb["batch_size"]), int(mb["block_size"]), int(mb["n_batches"]) + int(mb["warmup_batches"]) + 1, seed)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(wl, f)
    import os

    cores = ",".join(map(str, sorted(os.sched_getaffinity(0))[:threads]))   # pinned like the other loaders' processes
    args = ["taskset", "-c", cores, "Rscript", str(BENCH_DIR / "R" / "minibatch_bpcells.R"), str(path), f.name, str(threads),
            str(mb["warmup_batches"])]
    if src_url:
        args.append(src_url)
    r = subprocess.run(args, capture_output=True, text=True, cwd=BENCH_DIR)
    if r.returncode != 0:
        return {"status": "error", "note": (r.stderr.strip().splitlines() or ["failed"])[-1]}
    return {"status": "ok", "access": "download" if src_url else "local", **json.loads(r.stdout.strip().splitlines()[-1])}


# ----------------------------------------------------------------------------------------------- driver
_TIER_ORDER = {"S": 0, "M": 1, "L": 2, "XL": 3}


def _skip_download(mb: dict, tier: dict, loader: str, u: Unit) -> bool:
    """Download-first loaders on a slow link above the configured size tier are estimated, not run."""
    limit = (mb.get("skip_download_above_tier") or {}).get(tier["name"])
    access = "download" if loader == "bpcells" else LOADERS[loader].access
    return bool(limit) and access == "download" and _TIER_ORDER[u.dataset["tier"]] > _TIER_ORDER[limit]


def _path_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def run(cfg: Config, units: list[Unit], force: bool = False, overrides=(), tiers: Optional[list[str]] = None) -> None:
    from . import formats, netem
    from .util import read_tsv

    mb = cfg["minibatch"]
    threads = int(cfg["threads"]["headline"])
    tier_list = [t for t in mb["tiers"] if not tiers or t["name"] in tiers]
    loaders = list(mb["loaders"])
    for u in units:
        out = result_path(cfg, "minibatch", u)
        if out.exists() and not force:
            continue
        fdir = format_dir(cfg, u)
        from .util import _expand

        stats = read_tsv(result_path(cfg, "stats", u))
        n_cells = int(stats[0]["cells"])
        # a loader whose format the write stage found not applicable (BPCells above 2^31 non-zeros) gets a row saying so
        written = {r["format"]: r for r in read_tsv(result_path(cfg, "write", u))}
        na, files = {}, {}
        for ld in loaders:
            fmt = "bpcells" if ld == "bpcells" else LOADERS[ld].fmt
            if written.get(fmt, {}).get("status") == "not_applicable":
                na[ld] = written[fmt].get("note", "")
            else:
                files[ld] = formats.get([fmt])[0].path(fdir)
        run_loaders = [ld for ld in loaders if ld not in na]
        # a forced rerun of some loaders (minibatch.loaders) or tiers (--tiers) replaces those rows and keeps the others
        run_tiers = {t["name"] for t in tier_list}
        rows = [r for r in read_tsv(out) if r.get("loader") not in loaders or r.get("tier") not in run_tiers]

        def record(tier, ld, seed, res):
            rows.append({"dataset": u.id, "matrix": u.name, "tier": tier["name"], "gbit": tier.get("gbit", ""),
                         "latency_ms": tier.get("latency_ms", ""), "loader": ld, "seed": seed, "threads": threads, **res})
            log(f"{u.key} {tier['name']} {ld} seed {seed}: {res.get('first_batch_s', res.get('note'))}")

        for tier in tier_list:
            for ld, note in na.items():
                record(tier, ld, "", {"status": "not_applicable", "note": note})
            if not tier.get("gbit"):
                # a file tier: data_dir (node-local disk in the timed jobs) or a root of its own (the shared filesystem)
                root = _expand(str(tier["root"])) if tier.get("root") else None
                if tier.get("root") and not root:
                    log(f"{u.key}: tier {tier['name']} skipped (its root variable is not set)")
                    continue
                tdir = (Path(root) / "formats" / u.key) if root else fdir
                for ld in run_loaders:
                    path = formats.get(["bpcells" if ld == "bpcells" else LOADERS[ld].fmt])[0].path(tdir)
                    for seed in mb["seeds"]:
                        res = (_bpcells(cfg, path, seed, threads, n_cells, None) if ld == "bpcells"
                               else _child(cfg, overrides, u.key, ld, str(path), seed, threads, n_cells))
                        record(tier, ld, seed, res)
                continue
            work = cfg.data_dir / "tmp" / f"netem_{u.key}"
            keys = {f"{u.key}/{p.name}": p for p in set(files.values())}
            keys["_calib/blob.bin"] = netem.calib_blob(cfg, cfg.data_dir / "tmp")
            with netem.object_store(cfg, keys, work, gbit=tier.get("gbit")) as base:
                ceiling = netem.calibrate(cfg, f"{base}/_calib/blob.bin", 0)
                ceiling_par = netem.calibrate(cfg, f"{base}/_calib/blob.bin", 0, streams=threads)
                # one byte of every object first: rclone's first request into a directory waits about a second while it
                # lists it, a cost of the test setup that a real object store does not have and that would otherwise
                # fall on whichever loader runs first
                for k, f in keys.items():
                    if Path(f).is_file():
                        netem._get(f"{base}/{k}", (0, 1))
                netem.set_tier(cfg, tier)
                for ld in run_loaders:
                    url = f"{base}/{u.key}/{files[ld].name}"
                    if _skip_download(mb, tier, ld, u):
                        nbytes = _path_bytes(files[ld])
                        res = {"status": "skipped", "access": "download", "bytes_file": nbytes,
                               "est_download_s": nbytes * 8 / (float(tier["gbit"]) * 1e9),
                               "note": f"download-first loader above tier {mb['skip_download_above_tier'][tier['name']]}"}
                        record(tier, ld, "", res)
                        continue
                    for seed in mb["seeds"]:
                        res = (_bpcells(cfg, files[ld], seed, threads, n_cells, url) if ld == "bpcells"
                               else _child(cfg, overrides, u.key, ld, url, seed, threads, n_cells))
                        res["proxy_ceiling_gbit"] = ceiling            # one connection, under the tier's total limit
                        res["proxy_ceiling_par_gbit"] = ceiling_par    # `threads` connections sharing that limit
                        # the setup cannot deliver the tier even over `threads` connections (10 Gbit/s on loopback)
                        res["capped"] = bool(tier.get("gbit") and tier["gbit"] > 1.05 * ceiling_par)
                        record(tier, ld, seed, res)
        write_tsv(rows, out)
