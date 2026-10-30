"""Python-side formats. Values are stored as 32-bit integers wherever the format allows, which is the smallest
reasonable choice for counts and so the most favourable setting for every competitor."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..util import Canon, Config, canon_from_scipy
from . import GROUP_DOMAIN, GROUP_GENERAL, Format


def _cells_by_genes_csr(c: Canon) -> sp.csr_matrix:
    """The canonical genes x cells CSC is, array for array, the cells x genes CSR that AnnData stores."""
    return sp.csr_matrix((c.data.astype(np.int32), c.indices.astype(np.int32), c.indptr), shape=(c.n, c.m))


def _from_cells_by_genes(x, genes, barcodes) -> Canon:
    x = sp.csr_matrix(x)
    return canon_from_scipy(sp.csc_matrix((x.data, x.indices, x.indptr), shape=(x.shape[1], x.shape[0])), genes, barcodes)


def _write_names(path: Path, names, compress: bool = False) -> None:
    txt = "\n".join(map(str, names)) + "\n"
    if compress:
        with gzip.open(path, "wt", compresslevel=6) as f:
            f.write(txt)
    else:
        path.write_text(txt)


def _read_names(path: Path) -> list[str]:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        return f.read().splitlines()


# ----------------------------------------------------------------------------------------------- plain arrays
class CscI32(Format):
    id, label, is_dir = "csc_i32", "CSC, 32-bit integers", True

    def write(self, c, path, threads):
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "indptr.npy", c.indptr.astype(np.int64))
        np.save(path / "indices.npy", c.indices.astype(np.int32))
        np.save(path / "data.npy", c.data.astype(np.int32))
        _write_names(path / "genes.txt", c.genes)
        _write_names(path / "barcodes.txt", c.barcodes)

    def read(self, path, threads):
        x = sp.csc_matrix((np.load(path / "data.npy"), np.load(path / "indices.npy"), np.load(path / "indptr.npy")))
        g, b = _read_names(path / "genes.txt"), _read_names(path / "barcodes.txt")
        x.resize((len(g), len(b)))
        return canon_from_scipy(x, g, b)


class CscZstd(Format):
    id, label, is_dir = "csc_zstd19", "CSC + zstd-19", True
    level = 19

    def write(self, c, path, threads):
        import zstandard as zstd

        path.mkdir(parents=True, exist_ok=True)
        z = zstd.ZstdCompressor(level=self.level, threads=threads)
        parts = {
            "indptr": c.indptr.astype(np.int64).tobytes(),
            "indices": c.indices.astype(np.int32).tobytes(),
            "data": c.data.astype(np.int32).tobytes(),
            "genes": ("\n".join(map(str, c.genes)) + "\n").encode(),
            "barcodes": ("\n".join(map(str, c.barcodes)) + "\n").encode(),
        }
        for k, v in parts.items():
            (path / f"{k}.zst").write_bytes(z.compress(v))
        (path / "shape.json").write_text(json.dumps([c.m, c.n]))

    def read(self, path, threads):
        import zstandard as zstd

        d = zstd.ZstdDecompressor()
        get = lambda k: d.decompress((path / f"{k}.zst").read_bytes())
        m, n = json.loads((path / "shape.json").read_text())
        x = sp.csc_matrix(
            (np.frombuffer(get("data"), np.int32), np.frombuffer(get("indices"), np.int32), np.frombuffer(get("indptr"), np.int64)),
            shape=(m, n),
        )
        return canon_from_scipy(x, get("genes").decode().splitlines(), get("barcodes").decode().splitlines())


# ----------------------------------------------------------------------------------------------- text and HDF5
class MtxGz(Format):
    id, label, is_dir = "mtx_gz", "MTX.gz", True

    def write(self, c, path, threads):
        import scipy.io

        path.mkdir(parents=True, exist_ok=True)
        with gzip.open(path / "matrix.mtx.gz", "wb", compresslevel=6) as f:
            scipy.io.mmwrite(f, c.to_scipy().astype(np.int64), field="integer")
        _write_names(path / "features.tsv.gz", c.genes, compress=True)
        _write_names(path / "barcodes.tsv.gz", c.barcodes, compress=True)

    def read(self, path, threads):
        import scipy.io

        x = scipy.io.mmread(str(path / "matrix.mtx.gz"))
        return canon_from_scipy(x, _read_names(path / "features.tsv.gz"), _read_names(path / "barcodes.tsv.gz"))


class TenxH5(Format):
    id, label, ext = "tenx_h5", "10x HDF5", ".h5"

    def write(self, c, path, threads):
        import h5py

        kw = dict(compression="gzip", compression_opts=4, shuffle=True)
        with h5py.File(path, "w") as f:
            g = f.create_group("matrix")
            g.create_dataset("data", data=c.data.astype(np.int32), **kw)
            g.create_dataset("indices", data=c.indices.astype(np.int64), **kw)
            g.create_dataset("indptr", data=c.indptr.astype(np.int64), **kw)
            g.create_dataset("shape", data=np.array([c.m, c.n], dtype=np.int32))
            g.create_dataset("barcodes", data=np.asarray(c.barcodes, dtype="S"), **kw)
            ft = g.create_group("features")
            ft.create_dataset("id", data=np.asarray(c.genes, dtype="S"), **kw)
            ft.create_dataset("name", data=np.asarray(c.genes, dtype="S"), **kw)
            ft.create_dataset("feature_type", data=np.full(c.m, b"Gene Expression"), **kw)
            ft.create_dataset("genome", data=np.full(c.m, b"NA"), **kw)

    def read(self, path, threads):
        from ..canon import parse_tenx_h5

        return parse_tenx_h5(path)


class H5ad(Format):
    ext, group = ".h5ad", GROUP_GENERAL

    def __init__(self, fid: str, label: str, compression: Optional[str]):
        self.id, self.label, self.compression = fid, label, compression

    def write(self, c, path, threads):
        import anndata as ad
        import pandas as pd

        a = ad.AnnData(X=_cells_by_genes_csr(c), obs=pd.DataFrame(index=c.barcodes.astype(str)), var=pd.DataFrame(index=c.genes.astype(str)))
        opts = 4 if self.compression == "gzip" else None
        a.write_h5ad(path, compression=self.compression, compression_opts=opts)

    def read(self, path, threads):
        import anndata as ad

        a = ad.read_h5ad(path)
        return _from_cells_by_genes(a.X, list(a.var.index), list(a.obs.index))


class Zarr(Format):
    id, label, ext, is_dir = "zarr", "AnnData Zarr", ".zarr", True

    def write(self, c, path, threads):
        import anndata as ad
        import pandas as pd

        a = ad.AnnData(X=_cells_by_genes_csr(c), obs=pd.DataFrame(index=c.barcodes.astype(str)), var=pd.DataFrame(index=c.genes.astype(str)))
        a.write_zarr(path)

    def read(self, path, threads):
        import anndata as ad

        try:
            from numcodecs import blosc

            blosc.set_nthreads(threads)
        except Exception:
            pass
        a = ad.read_zarr(path)
        return _from_cells_by_genes(a.X, list(a.var.index), list(a.obs.index))


class Loom(Format):
    id, label, ext = "loom", "loom", ".loom"

    def applicable(self, c, cfg):
        lim = float(cfg.get("loom_max_entries", 2e9))
        return None if c.m * c.n <= lim else f"loom stores dense chunks; {c.m}x{c.n} entries exceeds {lim:g}"

    def write(self, c, path, threads):
        import loompy

        loompy.create(str(path), c.to_scipy().tocoo(), {"Gene": c.genes.astype(str)}, {"CellID": c.barcodes.astype(str)})

    def read(self, path, threads):
        import loompy

        with loompy.connect(str(path), "r") as ds:
            x = ds.sparse()
            g, b = list(ds.ra["Gene"]), list(ds.ca["CellID"])
        return canon_from_scipy(x, g, b)


class Soma(Format):
    id, label, ext, is_dir, group = "soma", "TileDB-SOMA", ".soma", True, GROUP_DOMAIN

    def write(self, c, path, threads):
        import anndata as ad
        import pandas as pd
        import tiledbsoma.io

        a = ad.AnnData(X=_cells_by_genes_csr(c), obs=pd.DataFrame(index=c.barcodes.astype(str)), var=pd.DataFrame(index=c.genes.astype(str)))
        if path.exists():
            shutil.rmtree(path)
        tiledbsoma.io.from_anndata(str(path), a, measurement_name="RNA")

    def read(self, path, threads):
        import tiledbsoma
        import tiledbsoma.io

        ctx = tiledbsoma.SOMATileDBContext(tiledb_config={"sm.compute_concurrency_level": threads, "sm.io_concurrency_level": threads})
        a = tiledbsoma.io.to_anndata(tiledbsoma.Experiment.open(str(path), context=ctx), measurement_name="RNA", X_layer_name="data")
        obs_ids = a.obs["obs_id"] if "obs_id" in a.obs else a.obs.index
        var_ids = a.var["var_id"] if "var_id" in a.var else a.var.index
        return _from_cells_by_genes(a.X, list(map(str, var_ids)), list(map(str, obs_ids)))


class ParquetCoo(Format):
    id, label, is_dir = "parquet_coo", "Parquet COO (zstd)", True

    def write(self, c, path, threads):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path.mkdir(parents=True, exist_ok=True)
        col = np.repeat(np.arange(c.n, dtype=np.uint32), np.diff(c.indptr))
        t = pa.table({"cell": col, "gene": c.indices, "count": c.data})
        pq.write_table(t, path / "counts.parquet", compression="zstd")
        pq.write_table(pa.table({"gene": c.genes.astype(str)}), path / "genes.parquet", compression="zstd")
        pq.write_table(pa.table({"barcode": c.barcodes.astype(str)}), path / "barcodes.parquet", compression="zstd")

    def read(self, path, threads):
        import pyarrow.parquet as pq

        t = pq.read_table(path / "counts.parquet", use_threads=threads > 1)
        g = pq.read_table(path / "genes.parquet").column("gene").to_pylist()
        b = pq.read_table(path / "barcodes.parquet").column("barcode").to_pylist()
        x = sp.csc_matrix(
            (t.column("count").to_numpy(), (t.column("gene").to_numpy(), t.column("cell").to_numpy())), shape=(len(g), len(b))
        )
        return canon_from_scipy(x, g, b)


FORMATS: list[Format] = [
    CscI32(),
    CscZstd(),
    MtxGz(),
    TenxH5(),
    H5ad("h5ad_none", "h5ad (no compression)", None),
    H5ad("h5ad_gzip", "h5ad gzip", "gzip"),
    H5ad("h5ad_lzf", "h5ad lzf", "lzf"),
    Zarr(),
    Loom(),
    Soma(),
    ParquetCoo(),
]
