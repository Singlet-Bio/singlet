"""Shared plumbing: configuration, paths, the canonical matrix, TSV results, timing and page-cache eviction."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import yaml

BENCH_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = BENCH_DIR.parent


# ----------------------------------------------------------------------------------------------- configuration
def _expand(s: str) -> str:
    """Expand ${VAR} and ${VAR:-default}."""

    def rep(m):
        name, _, default = m.group(1).partition(":-")
        return os.environ.get(name, default)

    return re.sub(r"\$\{([^}]+)\}", rep, s)


def _set(cfg: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


@dataclass
class Config:
    raw: dict
    path: Path

    def __getitem__(self, k):
        return self.raw[k]

    def get(self, k, default=None):
        return self.raw.get(k, default)

    def _dir(self, key: str) -> Path:
        p = Path(_expand(str(self.raw[key])))
        return p if p.is_absolute() else (self.path.parent / p).resolve()

    @property
    def data_dir(self) -> Path:
        return self._dir("data_dir")

    @property
    def results_dir(self) -> Path:
        return self._dir("results_dir")

    @property
    def generated_dir(self) -> Path:
        return self._dir("generated_dir")

    @property
    def datasets_path(self) -> Path:
        return self._dir("datasets")


def load_config(path: Optional[str] = None, overrides: Iterable[str] = ()) -> Config:
    p = Path(path) if path else BENCH_DIR / "config.yaml"
    raw = yaml.safe_load(p.read_text())
    for o in overrides:
        k, _, v = o.partition("=")
        _set(raw, k, v)
    return Config(raw, p.resolve())


# ----------------------------------------------------------------------------------------------- datasets
@dataclass
class Unit:
    """One count matrix of one dataset: the unit every stage works on."""

    dataset: dict
    matrix: dict

    @property
    def id(self) -> str:
        return self.dataset["id"]

    @property
    def name(self) -> str:
        return self.matrix["name"]

    @property
    def key(self) -> str:
        return f"{self.id}.{self.name}"


def load_datasets(cfg: Config, include_extensions: bool = False) -> list[dict]:
    doc = yaml.safe_load(cfg.datasets_path.read_text())
    ds = list(doc["datasets"])
    if include_extensions:
        ds += list(doc.get("extensions", []))
    return ds


def select_units(cfg: Config, which: str, include_extensions: bool = False, loader_panel: bool = False) -> list[Unit]:
    """`which` is 'all', a dataset id, a comma-separated list of ids, or 'tier:S,M'."""
    ds = load_datasets(cfg, include_extensions)
    if loader_panel:
        ds = [d for d in ds if d.get("loader_panel")]
    if which.startswith("tier:"):
        tiers = set(which[5:].split(","))
        ds = [d for d in ds if d["tier"] in tiers]
    elif which != "all":
        ids = which.split(",")
        known = {d["id"] for d in load_datasets(cfg, True)}
        bad = [i for i in ids if i not in known]
        if bad:
            raise SystemExit(f"unknown dataset id(s): {', '.join(bad)}")
        ds = [d for d in load_datasets(cfg, True) if d["id"] in ids]
    dropped = [d["id"] for d in ds if d.get("excluded")]
    if dropped:
        log(f"excluded (see datasets.yaml): {', '.join(dropped)}")
    return [Unit(d, m) for d in ds if not d.get("excluded") for m in d["matrices"]]


# ----------------------------------------------------------------------------------------------- canonical matrix
@dataclass
class Canon:
    """Genes x cells CSC with uint32 indices (strictly ascending per column) and uint32 counts > 0."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    genes: np.ndarray
    barcodes: np.ndarray

    @property
    def m(self) -> int:
        return len(self.genes)

    @property
    def n(self) -> int:
        return len(self.barcodes)

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])

    def to_scipy(self):
        import scipy.sparse as sp

        return sp.csc_matrix((self.data, self.indices, self.indptr), shape=(self.m, self.n))

    def check(self) -> None:
        if self.indptr[0] != 0 or len(self.indptr) != self.n + 1:
            raise ValueError("bad indptr")
        if self.indices.dtype != np.uint32 or self.data.dtype != np.uint32:
            raise ValueError("indices and data must be uint32")
        if self.nnz and (self.data == 0).any():
            raise ValueError("explicit zeros")
        if self.nnz and int(self.indices.max()) >= self.m:
            raise ValueError("row index out of range")
        if self.nnz < 2:
            return
        is_start = np.zeros(self.nnz, dtype=bool)
        s = self.indptr[:-1]
        is_start[s[s < self.nnz]] = True
        bad = ~is_start[1:] & (self.indices[1:] <= self.indices[:-1])
        if bad.any():
            raise ValueError("row indices not strictly ascending within a column")

    def equals(self, other: "Canon") -> bool:
        return (
            np.array_equal(self.indptr, other.indptr)
            and np.array_equal(self.indices, other.indices)
            and np.array_equal(self.data, other.data)
            and list(map(str, self.genes)) == list(map(str, other.genes))
            and list(map(str, self.barcodes)) == list(map(str, other.barcodes))
        )


def canon_from_scipy(x, genes, barcodes, genes_by_cells: bool = True) -> Canon:
    """Build a Canon from any SciPy sparse or dense array; rejects non-integer or negative values."""
    import scipy.sparse as sp

    x = sp.csc_matrix(x if genes_by_cells else sp.csr_matrix(x).T, copy=True)
    x.sum_duplicates()
    x.eliminate_zeros()
    x.sort_indices()
    d = x.data
    if d.size and (np.any(d < 0) or np.any(np.floor(d) != d)):
        raise ValueError("matrix has negative or non-integer values; this source holds normalised data, not counts")
    if d.size and d.max() > np.iinfo(np.uint32).max:
        raise ValueError("count above uint32")
    c = Canon(
        indptr=x.indptr.astype(np.int64),
        indices=x.indices.astype(np.uint32),
        data=d.astype(np.uint32),
        genes=np.asarray([str(g) for g in genes]),
        barcodes=np.asarray([str(b) for b in barcodes]),
    )
    if c.m != x.shape[0] or c.n != x.shape[1]:
        raise ValueError("names do not match the matrix shape")
    c.check()
    return c


def canon_path(cfg: Config, unit: Unit) -> Path:
    return cfg.data_dir / "canon" / f"{unit.key}.npz"


def save_canon(c: Canon, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, indptr=c.indptr, indices=c.indices, data=c.data, genes=c.genes, barcodes=c.barcodes)
    tmp.replace(path)


def load_canon(cfg: Config, unit: Unit) -> Canon:
    z = np.load(canon_path(cfg, unit), allow_pickle=False)
    return Canon(z["indptr"], z["indices"], z["data"], z["genes"], z["barcodes"])


# ----------------------------------------------------------------------------------------------- results
def result_path(cfg: Config, stage: str, unit: Unit, suffix: str = "") -> Path:
    return cfg.results_dir / stage / f"{unit.key}{suffix}.tsv"


def write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in cols})
    tmp.replace(path)


def _fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "NA" if not np.isfinite(v) else repr(round(float(v), 6))
    return v


def read_tsv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_stage(cfg: Config, stage: str) -> list[dict]:
    rows: list[dict] = []
    d = cfg.results_dir / stage
    for p in sorted(d.glob("*.tsv")) if d.exists() else []:
        rows += read_tsv(p)
    return rows


# ----------------------------------------------------------------------------------------------- timing
def now() -> float:
    return time.perf_counter()


@contextmanager
def timed(out: dict, key: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        out[key] = out.get(key, 0.0) + time.perf_counter() - t0


def repeat(fn, repeats: int, warmup: int, before=None) -> list[float]:
    """Run fn() warmup + repeats times, calling before() untimed ahead of each run; return the timed seconds."""
    out = []
    for i in range(warmup + repeats):
        if before:
            before()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if i >= warmup:
            out.append(dt)
    return out


def summary(ts: list[float]) -> dict:
    a = np.asarray(ts, dtype=float)
    return {"median_s": float(np.median(a)), "q1_s": float(np.quantile(a, 0.25)), "q3_s": float(np.quantile(a, 0.75)), "n": len(a)}


def evict(path: Path) -> None:
    """Drop a file (or every file under a directory) from the page cache. No root needed."""
    paths = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.fsync(fd) if os.access(p, os.W_OK) else None
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def disk_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def pin_cores(n: int) -> None:
    """Restrict this process (and threads it starts) to the first n cores it is allowed to use."""
    if hasattr(os, "sched_getaffinity"):
        cores = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, cores[: max(1, n)])
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


# ----------------------------------------------------------------------------------------------- environment record
def env_record() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return None

    rec = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": run(["bash", "-c", "lscpu | sed -n 's/^Model name:[ ]*//p'"]),
        "cores": os.cpu_count(),
        "isa_flags": run(["bash", "-c", "grep -o -w -E 'avx2|avx512f|avx512bw' /proc/cpuinfo | sort -u | tr '\\n' ' '"]),
        "mem_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
        "governor": run(["bash", "-c", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null"]),
        "git_commit": run(["git", "-C", str(PAPER_DIR), "rev-parse", "HEAD"]),
        "git_dirty": bool(run(["git", "-C", str(PAPER_DIR), "status", "--porcelain"])),
        "packages": {},
    }
    for mod in ("numpy", "scipy", "anndata", "h5py", "zarr", "loompy", "pyarrow", "zstandard", "torch", "tiledbsoma",
                "tiledbsoma_ml", "scdataset", "scanpy", "singlet"):
        try:
            m = __import__(mod)
            rec["packages"][mod] = getattr(m, "__version__", "unknown")
        except Exception:
            rec["packages"][mod] = None
    try:
        from singlet import _pz2  # type: ignore

        rec["pz2_isas"] = {i: bool(_pz2.isa_supported(i)) for i in ("scalar", "avx2", "avx512")}
    except Exception:
        rec["pz2_isas"] = None
    rec["R"] = run(["Rscript", "-e", "cat(R.version.string, ' BPCells', as.character(packageVersion('BPCells')))"])
    return rec


def write_env(cfg: Config) -> Path:
    p = cfg.results_dir / "env.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(env_record(), indent=2, sort_keys=True) + "\n")
    return p


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)
