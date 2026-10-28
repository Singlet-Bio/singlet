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

    def open(self, src, threads):
        import anndata as ad
        from scdataset import BlockShuffling, scDataset

        self.a = ad.read_h5ad(src, backed="r")
        self.ds = scDataset(self.a, BlockShuffling(block_size=self.block_size), batch_size=self.batch_size,
                            fetch_factor=self.fetch_factor)
        self.it = iter(self.ds)

    def read(self, blocks):
        b = next(self.it)
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
        ds = sml.ExperimentDataset(self.q, layer_name="data", batch_size=self.batch_size, shuffle=True, seed=self.seed,
                                   return_sparse_X=True)
        self.it = iter(sml.experiment_dataloader(ds))

    def read(self, blocks):
        x, _obs = next(self.it)
        return x

    def to_tensor(self, x):
        import scipy.sparse as sp

        return x if hasattr(x, "is_sparse_csr") else _csr_to_tensor(sp.csr_matrix(x))


class S1StreamLoader(Loader):
    fmt = "s1"
    access = "range"
    returns_tensor = True
    order = "native"
    modules = ("singlet._pz2", "singlet.s1.container", "requests")

    def open(self, src, threads):
        from .s1_loader import S1Stream, file_fetch, http_fetch

        fetch, size = http_fetch(src) if src.startswith("http") else file_fetch(src)
        self.s = S1Stream(fetch, size=size, threads=threads, order=self.order)

    def read(self, blocks):
        return self.s.batch(blocks)

    def to_tensor(self, x):
        return x

    def split(self):
        return {"fetch_s": self.s.fetch_s, "decode_s": self.s.decode_s, "bytes": self.s.bytes_fetched, "requests": self.s.requests}


class S1StreamCanonical(S1StreamLoader):
    order = "canonical"


LOADERS = {
    "anndata_inmem": AnndataInmem,
    "anndata_backed": AnndataBacked,
    "scdataset": ScDataset,
    "anndata_zarr_lazy": AnndataZarrLazy,
    "soma_ml": SomaMl,
    "s1_stream": S1StreamLoader,
    "s1_stream_canonical": S1StreamCanonical,
}


# ----------------------------------------------------------------------------------------------- one measurement (child process)
def _download(url: str, dst: Path) -> None:
    with urllib.request.urlopen(url) as r, dst.open("wb") as f:
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

    L = LOADERS[loader]()
    import importlib

    for m in L.modules:
        importlib.import_module(m)
    L.batch_size, L.block_size, L.seed = int(mb["batch_size"]), int(mb["block_size"]), seed
    L.fetch_factor = int(mb.get("fetch_factor", 16))
    rec: dict = {"access": L.access if src.startswith("http") else "local"}
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
    rec.update({f"first_{k}": v for k, v in L.split().items()})
    if mb.get("_first_only"):
        rec["sustained_batches"] = 0
        return rec
    for b in batches[1:1 + int(mb["warmup_batches"])]:
        L.to_tensor(L.read(b))
    t5 = time.perf_counter()
    cells = 0
    for b in batches[1 + int(mb["warmup_batches"]):]:
        cells += int(L.to_tensor(L.read(b)).size(0))
    t6 = time.perf_counter()
    rec.update({"sustained_cells_per_s": cells / (t6 - t5), "sustained_batches": len(batches) - 1 - int(mb["warmup_batches"])})
    try:
        import resource

        rec["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        pass
    return rec


def _child(cfg: Config, overrides, unit_key, loader, src, seed, threads, n_cells) -> dict:
    first_only = seed not in cfg["minibatch"].get("sustained_seeds", cfg["minibatch"]["seeds"])
    cmd = [sys.executable, "-m", "s1bench", "--config", str(cfg.path), *sum((["--set", o] for o in overrides), []),
           "--set", f"minibatch._n_cells={n_cells}", "--set", f"minibatch._first_only={str(first_only).lower()}",
           "minibatch-one", unit_key, loader, src, str(seed), str(threads)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=BENCH_DIR, timeout=3 * 3600)
    if r.returncode != 0:
        return {"status": "error", "note": (r.stderr.strip().splitlines() or ["failed"])[-1]}
    return {"status": "ok", **json.loads(r.stdout.strip().splitlines()[-1])}


def _bpcells(cfg: Config, path: Path, seed: int, threads: int, n_cells: int, src_url: Optional[str]) -> dict:
    mb = cfg["minibatch"]
    wl = workload(n_cells, int(mb["batch_size"]), int(mb["block_size"]), int(mb["n_batches"]) + int(mb["warmup_batches"]) + 1, seed)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(wl, f)
    args = ["Rscript", str(BENCH_DIR / "R" / "minibatch_bpcells.R"), str(path), f.name, str(threads), str(mb["warmup_batches"])]
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
        stats = read_tsv(result_path(cfg, "stats", u))
        n_cells = int(stats[0]["cells"])
        files = {}
        for ld in loaders:
            fmt = "bpcells" if ld == "bpcells" else LOADERS[ld].fmt
            files[ld] = formats.get([fmt])[0].path(fdir)
        rows = []

        def record(tier, ld, seed, res):
            rows.append({"dataset": u.id, "matrix": u.name, "tier": tier["name"], "gbit": tier.get("gbit", ""),
                         "latency_ms": tier.get("latency_ms", ""), "loader": ld, "seed": seed, "threads": threads, **res})
            log(f"{u.key} {tier['name']} {ld} seed {seed}: {res.get('first_batch_s', res.get('note'))}")

        for tier in tier_list:
            if tier["name"] == "nvme":
                for ld in loaders:
                    for seed in mb["seeds"]:
                        res = (_bpcells(cfg, files[ld], seed, threads, n_cells, None) if ld == "bpcells"
                               else _child(cfg, overrides, u.key, ld, str(files[ld]), seed, threads, n_cells))
                        record(tier, ld, seed, res)
                continue
            work = cfg.data_dir / "tmp" / f"netem_{u.key}"
            keys = {f"{u.key}/{p.name}": p for p in set(files.values())}
            with netem.object_store(cfg, keys, work) as base:
                ceiling = netem.calibrate(cfg, f"{base}/{u.key}/{files['s1_stream'].name}", 0)
                netem.set_tier(cfg, tier)
                for ld in loaders:
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
                        res["proxy_ceiling_gbit"] = ceiling
                        res["capped"] = bool(tier.get("gbit") and tier["gbit"] > ceiling)
                        record(tier, ld, seed, res)
        write_tsv(rows, out)
