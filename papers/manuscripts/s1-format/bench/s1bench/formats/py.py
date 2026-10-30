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


def _anndata_to_canon(a) -> Canon:
    return _from_cells_by_genes(a.X, list(map(str, a.var.index)), list(map(str, a.obs.index)))


class _ScipyTriple:
    """load() returns (matrix genes x cells, gene ids, barcodes), the form these libraries give a user."""

    def to_canon(self, obj) -> Canon:
        x, g, b = obj
        return canon_from_scipy(x, g, b)


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
class CscI32(_ScipyTriple, Format):
    id, label, is_dir = "csc_i32", "CSC, 32-bit integers", True

    def write(self, c, path, threads):
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "indptr.npy", c.indptr.astype(np.int64))
        np.save(path / "indices.npy", c.indices.astype(np.int32))
        np.save(path / "data.npy", c.data.astype(np.int32))
        _write_names(path / "genes.txt", c.genes)
        _write_names(path / "barcodes.txt", c.barcodes)

    def load(self, path, threads):
        g, b = _read_names(path / "genes.txt"), _read_names(path / "barcodes.txt")
        x = sp.csc_matrix((np.load(path / "data.npy"), np.load(path / "indices.npy"), np.load(path / "indptr.npy")),
                          shape=(len(g), len(b)))
        return x, g, b


class CscZstd(_ScipyTriple, Format):
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

    def load(self, path, threads):
        import zstandard as zstd

        d = zstd.ZstdDecompressor()
        get = lambda k: d.decompress((path / f"{k}.zst").read_bytes())
        m, n = json.loads((path / "shape.json").read_text())
        x = sp.csc_matrix(
            (np.frombuffer(get("data"), np.int32), np.frombuffer(get("indices"), np.int32), np.frombuffer(get("indptr"), np.int64)),
            shape=(m, n),
        )
        return x, get("genes").decode().splitlines(), get("barcodes").decode().splitlines()


# ----------------------------------------------------------------------------------------------- text and HDF5
class MtxGz(_ScipyTriple, Format):
    id, label, is_dir = "mtx_gz", "MTX.gz", True

    def write(self, c, path, threads):
        import scipy.io

        path.mkdir(parents=True, exist_ok=True)
        with gzip.open(path / "matrix.mtx.gz", "wb", compresslevel=6) as f:
            scipy.io.mmwrite(f, c.to_scipy().astype(np.int64), field="integer")
        _write_names(path / "features.tsv.gz", c.genes, compress=True)
        _write_names(path / "barcodes.tsv.gz", c.barcodes, compress=True)

    def load(self, path, threads):
        import scipy.io

        x = sp.csc_matrix(scipy.io.mmread(str(path / "matrix.mtx.gz")))
        return x, _read_names(path / "features.tsv.gz"), _read_names(path / "barcodes.tsv.gz")


class TenxH5(_ScipyTriple, Format):
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

    def load(self, path, threads):
        import h5py

        with h5py.File(path, "r") as f:
            g = f["matrix"]
            x = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
            genes = [s.decode() for s in g["features"]["id"][:]]
            barcodes = [s.decode() for s in g["barcodes"][:]]
        return x, genes, barcodes


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

    def load(self, path, threads):
        import anndata as ad

        return ad.read_h5ad(path)

    def to_canon(self, obj):
        return _anndata_to_canon(obj)


class Zarr(Format):
    id, label, ext, is_dir = "zarr", "AnnData Zarr", ".zarr", True

    def write(self, c, path, threads):
        import anndata as ad
        import pandas as pd

        a = ad.AnnData(X=_cells_by_genes_csr(c), obs=pd.DataFrame(index=c.barcodes.astype(str)), var=pd.DataFrame(index=c.genes.astype(str)))
        a.write_zarr(path)

    def load(self, path, threads):
        import anndata as ad

        try:
            from numcodecs import blosc

            blosc.set_nthreads(threads)
        except Exception:
            pass
        return ad.read_zarr(path)

    def to_canon(self, obj):
        return _anndata_to_canon(obj)


class Loom(_ScipyTriple, Format):
    id, label, ext = "loom", "loom", ".loom"

    def applicable(self, c, cfg):
        lim = float(cfg.get("loom_max_entries", 2e9))
        return None if c.m * c.n <= lim else f"loom stores dense chunks; {c.m}x{c.n} entries exceeds {lim:g}"

    def write(self, c, path, threads):
        import loompy

        loompy.create(str(path), c.to_scipy().tocoo(), {"Gene": c.genes.astype(str)}, {"CellID": c.barcodes.astype(str)})

    def load(self, path, threads):
        import loompy

        with loompy.connect(str(path), "r") as ds:
            x = ds.sparse()
            g, b = list(ds.ra["Gene"]), list(ds.ca["CellID"])
        return x, g, b


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

    def load(self, path, threads):
        import tiledbsoma
        import tiledbsoma.io

        ctx = tiledbsoma.SOMATileDBContext(tiledb_config={"sm.compute_concurrency_level": threads, "sm.io_concurrency_level": threads})
        return tiledbsoma.io.to_anndata(tiledbsoma.Experiment.open(str(path), context=ctx), measurement_name="RNA", X_layer_name="data")

    def to_canon(self, a):
        obs_ids = a.obs["obs_id"] if "obs_id" in a.obs else a.obs.index
        var_ids = a.var["var_id"] if "var_id" in a.var else a.var.index
        return _from_cells_by_genes(a.X, list(map(str, var_ids)), list(map(str, obs_ids)))


class ParquetCoo(_ScipyTriple, Format):
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

    def load(self, path, threads):
        import pyarrow.parquet as pq

        t = pq.read_table(path / "counts.parquet", use_threads=threads > 1)
        g = pq.read_table(path / "genes.parquet").column("gene").to_pylist()
        b = pq.read_table(path / "barcodes.parquet").column("barcode").to_pylist()
        x = sp.csc_matrix(
            (t.column("count").to_numpy(), (t.column("gene").to_numpy(), t.column("cell").to_numpy())), shape=(len(g), len(b))
        )
        return x, g, b


ZARR3_CHUNK = 49_152          # elements per inner chunk: the .s1 library's default chunk size in non-zeros
ZARR3_SHARD_CHUNKS = 64       # inner chunks per shard object


def zarr3_dtypes(c: Canon) -> tuple:
    """The narrowest unsigned types that hold the gene ids and the counts: what a careful writer of plain arrays picks."""
    idx = np.uint16 if c.m <= 1 << 16 else np.uint32
    vmax = int(c.data.max()) if c.nnz else 0
    val = np.uint8 if vmax < 1 << 8 else np.uint16 if vmax < 1 << 16 else np.uint32
    return idx, val


class Zarr3Shard(_ScipyTriple, Format):
    """The generic-codec control: cells x genes CSR in Zarr v3, data and indices in inner chunks of ZARR3_CHUNK elements
    (the .s1 chunk size) packed into shard objects, Blosc zstd with bit shuffling, narrowest integer types; indptr whole.
    It has the random-access granularity of .s1 and a general-purpose codec, so the difference between the two is the
    codec. Needs zarr-python 3, which the benchmark env (zarr 2 for AnnData) does not have: these processes run with a
    side install first on PYTHONPATH (S1_ZARR3_SITE; see slurm/clipper/timed.sbatch)."""

    id, label, ext, is_dir = "zarr3", "Zarr v3 sharded (Blosc zstd)", ".zarr3", True

    def write(self, c, path, threads):
        import zarr
        from zarr.codecs import BloscCodec

        zarr.config.set({"async.concurrency": max(4, threads), "threading.max_workers": threads})
        x = _cells_by_genes_csr(c)
        idt, vdt = zarr3_dtypes(c)
        if path.exists():
            shutil.rmtree(path)
        g = zarr.open_group(str(path), mode="w", zarr_format=3)
        blosc = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
        n = max(1, c.nnz)
        for name, arr in (("data", x.data.astype(vdt)), ("indices", x.indices.astype(idt))):
            a = g.create_array(name, shape=(c.nnz,), dtype=arr.dtype, chunks=(min(ZARR3_CHUNK, n),),
                               shards=(min(ZARR3_CHUNK * ZARR3_SHARD_CHUNKS, -(-n // ZARR3_CHUNK) * ZARR3_CHUNK),), compressors=blosc)
            a[:] = arr
        ip = g.create_array("indptr", shape=(c.n + 1,), dtype=np.int64, chunks=(c.n + 1,), compressors=blosc)
        ip[:] = x.indptr.astype(np.int64)
        g.attrs.update({"shape": [int(c.n), int(c.m)], "layout": "csr cells x genes"})
        _write_names(path / "genes.txt.gz", c.genes, compress=True)
        _write_names(path / "barcodes.txt.gz", c.barcodes, compress=True)

    def load(self, path, threads):
        import zarr

        zarr.config.set({"async.concurrency": max(4, threads), "threading.max_workers": threads})
        g = zarr.open_group(str(path), mode="r")
        n, m = g.attrs["shape"]
        x = sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=(n, m))
        return x.T, _read_names(path / "genes.txt.gz"), _read_names(path / "barcodes.txt.gz")


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
    Zarr3Shard(),
]
