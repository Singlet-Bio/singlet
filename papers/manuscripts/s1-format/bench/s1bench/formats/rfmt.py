"""R formats (saveRDS of a dgCMatrix, BPCells), driven through `Rscript R/formats.R`.

Python and R exchange the canonical matrix as flat binary files in a directory (`export_for_r`): p.f64 (column
pointers as float64, exact below 2^53), i.i32 (0-based row ids), x.i32 (counts), genes.txt, barcodes.txt. R's
dgCMatrix holds at most 2^31 - 1 non-zeros; larger matrices are reported as not applicable for `rds_*`, which is a real
limit of that format. BPCells has no such limit and is written from the uncompressed h5ad when the matrix is that large.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..util import BENCH_DIR, Canon, canon_from_scipy
from . import GROUP_DOMAIN, GROUP_GENERAL, Format

R_SCRIPT = BENCH_DIR / "R" / "formats.R"
INT32_MAX = 2**31 - 1


def export_for_r(c: Canon, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    c.indptr.astype(np.float64).tofile(d / "p.f64")
    c.indices.astype(np.int32).tofile(d / "i.i32")
    if c.nnz and int(c.data.max()) > INT32_MAX:
        raise ValueError("count above 2^31 - 1 cannot be passed to R")
    c.data.astype(np.int32).tofile(d / "x.i32")
    (d / "genes.txt").write_text("\n".join(map(str, c.genes)) + "\n")
    (d / "barcodes.txt").write_text("\n".join(map(str, c.barcodes)) + "\n")


def import_from_r(d: Path) -> Canon:
    import scipy.sparse as sp

    g = (d / "genes.txt").read_text().splitlines()
    b = (d / "barcodes.txt").read_text().splitlines()
    p = np.fromfile(d / "p.f64", dtype=np.float64).astype(np.int64)
    x = sp.csc_matrix((np.fromfile(d / "x.i32", dtype=np.int32), np.fromfile(d / "i.i32", dtype=np.int32), p), shape=(len(g), len(b)))
    return canon_from_scipy(x, g, b)


def rscript(*args: str) -> str:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(BENCH_DIR), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run(["Rscript", str(R_SCRIPT), *map(str, args)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Rscript {' '.join(map(str, args))} failed:\n{r.stderr[-4000:]}")
    return r.stdout


class RFormat(Format):
    lang = "r"

    def __init__(self, fid: str, label: str, ext: str, is_dir: bool, group: str):
        self.id, self.label, self.ext, self.is_dir, self.group = fid, label, ext, is_dir, group

    def applicable(self, c, cfg):
        if self.id.startswith("rds") and c.nnz > INT32_MAX:
            return f"dgCMatrix holds at most 2^31-1 non-zeros; this matrix has {c.nnz}"
        return None

    def write(self, c, path, threads):
        with tempfile.TemporaryDirectory(dir=path.parent) as t:
            if c.nnz > INT32_MAX and self.id == "bpcells":
                src = path.parent / "h5ad_none.h5ad"
                if not src.exists():
                    raise RuntimeError("bpcells on a >2^31 nnz matrix needs h5ad_none written first")
                rscript("write_from_h5ad", self.id, src, path, threads)
            else:
                export_for_r(c, Path(t))
                rscript("write", self.id, t, path, threads)

    def read(self, path, threads):
        with tempfile.TemporaryDirectory(dir=path.parent) as t:
            rscript("read", self.id, path, t, threads)
            return import_from_r(Path(t))

    def time_read(self, path: Path, threads: int, repeats: int, warmup: int, cold: bool) -> list[float]:
        """Seconds per read, timed inside R (R start-up excluded); cold runs evict the files first."""
        out = rscript("time", self.id, path, threads, repeats, warmup, int(cold))
        return json.loads(out.strip().splitlines()[-1])


FORMATS: list[Format] = [
    RFormat("rds_gz", "rds (gzip)", ".rds", False, GROUP_GENERAL),
    RFormat("rds_xz", "rds (xz)", ".rds", False, GROUP_GENERAL),
    RFormat("bpcells", "BPCells", ".bpcells", True, GROUP_DOMAIN),
]
