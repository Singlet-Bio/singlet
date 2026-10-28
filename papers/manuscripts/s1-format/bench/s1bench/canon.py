"""Stage `canon`: parse each source into the canonical genes x cells uint32 CSC (util.Canon) and record descriptors.

Parsers are chosen by `source.format` in datasets.yaml. Dataset-specific steps (dropping mouse donors, filtering an assay,
removing control probes) are in HOOKS at the bottom. When an entry is verified on its portal and the file layout differs
from what a parser expects, fix the parser here, not the data.
"""

from __future__ import annotations

import gzip
import io
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import scipy.io
import scipy.sparse as sp

from .util import Canon, Config, Unit, canon_from_scipy, canon_path, log, result_path, save_canon, write_tsv


# ----------------------------------------------------------------------------------------------- helpers
def _raw_dir(cfg: Config, d: dict) -> Path:
    return cfg.data_dir / "raw" / d["id"]


def _extract(path: Path) -> Path:
    """Unpack a tar or zip once next to itself; return the directory."""
    out = path.parent / (path.name.split(".")[0] + "_x")
    if out.exists():
        return out
    tmp = out.with_suffix(".part")
    tmp.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            t.extractall(tmp, filter="data")
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
    else:
        raise ValueError(f"{path} is not a tar or zip archive")
    tmp.replace(out)
    return out


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def _read_lines(path: Path, col: int = 0) -> list[str]:
    with _open(path) as f:
        return [ln.rstrip("\n").split("\t")[col] for ln in f if ln.strip()]


def _first(root: Path, *patterns: str) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(root.rglob(pat))
        if hits:
            return hits[0]
    return None


def _main_file(cfg: Config, d: dict) -> Path:
    url = d["source"]["url"]
    return _raw_dir(cfg, d) / url.rstrip("/").split("/")[-1]


# ----------------------------------------------------------------------------------------------- parsers
def parse_mtx_dir(root: Path, feature_type: Optional[str] = None) -> Canon:
    """Cell Ranger MTX output: matrix.mtx(.gz), genes.tsv or features.tsv(.gz), barcodes.tsv(.gz)."""
    mtx = _first(root, "matrix.mtx.gz", "matrix.mtx", "*.mtx.gz", "*.mtx")
    feat = _first(root, "features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv")
    bc = _first(root, "barcodes.tsv.gz", "barcodes.tsv")
    if not (mtx and feat and bc):
        raise ValueError(f"no MTX triplet under {root}")
    x = scipy.io.mmread(str(mtx)).tocsc()
    genes = _read_lines(feat, 0)
    if feature_type:
        types = _read_lines(feat, 2)
        keep = np.flatnonzero(np.asarray(types) == feature_type)
        x, genes = x[keep, :], [genes[i] for i in keep]
    return canon_from_scipy(x, genes, _read_lines(bc, 0))


def parse_tenx_h5(path: Path, feature_type: Optional[str] = None) -> Canon:
    """Cell Ranger HDF5, v3 layout (/matrix with features) or v2 layout (one group per genome)."""
    import h5py

    with h5py.File(path, "r") as f:
        g = f["matrix"] if "matrix" in f else f[list(f.keys())[0]]
        shape = tuple(g["shape"][:])
        x = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
        if "features" in g:
            genes = [s.decode() for s in g["features"]["id"][:]]
            types = [s.decode() for s in g["features"]["feature_type"][:]]
        else:
            genes = [s.decode() for s in g["genes"][:]]
            types = ["Gene Expression"] * len(genes)
        barcodes = [s.decode() for s in g["barcodes"][:]]
    if feature_type:
        keep = np.flatnonzero(np.asarray(types) == feature_type)
        x, genes = x[keep, :], [genes[i] for i in keep]
    return canon_from_scipy(x, genes, barcodes)


def parse_h5ad(path: Path, x_location: str = "X") -> Canon:
    """Counts from X, raw/X or layers/<name>; genes from var (or raw/var) index, cells from obs index."""
    import anndata as ad

    a = ad.read_h5ad(path, backed=None)
    if x_location == "X":
        x, var = a.X, a.var
    elif x_location == "raw/X":
        x, var = a.raw.X, a.raw.var
    elif x_location.startswith("layers/"):
        x, var = a.layers[x_location.split("/", 1)[1]], a.var
    else:
        raise ValueError(f"unknown x_location {x_location}")
    return canon_from_scipy(sp.csr_matrix(x), list(var.index), list(a.obs.index), genes_by_cells=False)


def parse_dense_table(paths: list[Path], sep: str, genes_by_cells: bool, chunk: int = 2000) -> Canon:
    """Dense text tables (first column = row names, header = column names), read in chunks into sparse blocks and
    concatenated along cells. Several files are concatenated in the order given (cells must not repeat)."""
    import pandas as pd

    blocks, genes, barcodes = [], None, []
    for p in paths:
        rows, names = [], []
        for df in pd.read_csv(p, sep=sep, index_col=0, chunksize=chunk):
            names += list(map(str, df.index))
            rows.append(sp.csr_matrix(df.to_numpy()))
        x = sp.vstack(rows).tocsc()
        cols = list(map(str, df.columns))
        if genes_by_cells:
            g, b = names, cols
        else:
            x, g, b = x.T.tocsc(), cols, names
        if genes is None:
            genes = g
        elif g != genes:
            raise ValueError(f"{p}: gene list differs from the first file; align before concatenating")
        blocks.append(x)
        barcodes += b
    return canon_from_scipy(sp.hstack(blocks).tocsc(), genes, barcodes)


def parse_gct(path: Path) -> Canon:
    import pandas as pd

    df = pd.read_csv(path, sep="\t", skiprows=2, index_col=0)
    df = df.drop(columns=[c for c in ("Description",) if c in df.columns])
    return canon_from_scipy(sp.csc_matrix(df.to_numpy()), list(df.index), list(df.columns))


def parse_uci_bow(path: Path) -> Canon:
    """UCI bag of words: header D, W, NNZ then 'docID wordID count' (1-based); documents become columns."""
    with _open(path) as f:
        n_docs, n_words, _ = (int(f.readline()) for _ in range(3))
        t = np.loadtxt(f, dtype=np.int64)
    x = sp.csc_matrix((t[:, 2], (t[:, 1] - 1, t[:, 0] - 1)), shape=(n_words, n_docs))
    return canon_from_scipy(x, [f"w{i}" for i in range(n_words)], [f"d{j}" for j in range(n_docs)])


def parse_triplet_txt(path: Path, genes_path: Path, cells_path: Path) -> Canon:
    """Matrix Market or plain 'gene cell count' triplets (1-based), with separate gene and cell annotation CSVs."""
    import pandas as pd

    try:
        x = scipy.io.mmread(str(path)).tocsc()
    except Exception:
        t = pd.read_csv(path, sep=r"[\s,]+", header=None, engine="python").to_numpy()
        x = sp.csc_matrix((t[:, 2], (t[:, 0] - 1, t[:, 1] - 1)))
    genes = list(pd.read_csv(genes_path).iloc[:, 0].astype(str))
    cells = list(pd.read_csv(cells_path).iloc[:, 0].astype(str))
    x = x[: len(genes), : len(cells)] if x.shape != (len(genes), len(cells)) else x
    return canon_from_scipy(x, genes, cells)


def synthetic(src: dict) -> Canon:
    """Gamma-Poisson counts with Zipf-like gene means and log-normal cell depths, for the smoke test
    (tests/smoke.py) and for checking a new cluster install without downloading anything."""
    rng = np.random.default_rng(int(src.get("seed", 0)))
    G, C = int(src["genes"]), int(src["cells"])
    k = int(src.get("types", 5))
    base = 1.0 / np.arange(1, G + 1) ** 1.1
    rng.shuffle(base)
    prof = base[None, :] * rng.lognormal(0, 1.0, size=(k, G))
    prof /= prof.sum(axis=1, keepdims=True)
    depth = rng.lognormal(np.log(float(src.get("depth", 3000))), 0.5, size=C)
    t = rng.integers(0, k, size=C)
    theta = float(src.get("theta", 5.0))
    blocks = []
    for s0 in range(0, C, 2000):
        sl = slice(s0, min(C, s0 + 2000))
        lam = prof[t[sl]] * depth[sl, None]
        x = rng.poisson(rng.gamma(theta, lam / theta))
        blocks.append(sp.csc_matrix(x.T))
    return canon_from_scipy(sp.hstack(blocks).tocsc(), [f"g{i}" for i in range(G)], [f"c{j}" for j in range(C)])


# ----------------------------------------------------------------------------------------------- dispatch
def parse(cfg: Config, u: Unit) -> Canon:
    d, src, mat = u.dataset, u.dataset["source"], u.matrix
    hook = HOOKS.get((d["id"], mat["name"])) or HOOKS.get((d["id"], "*"))
    if hook:
        return hook(cfg, u)
    fmt = src["format"]
    if fmt == "synthetic":
        return synthetic(src)
    main = _main_file(cfg, d)
    ft = mat.get("feature_type")
    if fmt in ("tenx_h5", "tenx_h5_v2"):
        return parse_tenx_h5(main, ft)
    if fmt == "mtx_tar":
        return parse_mtx_dir(_extract(main), ft)
    if fmt == "h5ad":
        return parse_h5ad(main, src.get("x_location", "X"))
    if fmt in ("dense_tsv_genes_by_cells", "dense_txt_genes_by_cells"):
        return parse_dense_table([main], "\t", True)
    if fmt == "dense_csv_genes_by_cells":
        return parse_dense_table([main], ",", True)
    if fmt == "gct":
        return parse_gct(main)
    if fmt == "uci_bow":
        return parse_uci_bow(main)
    raise SystemExit(
        f"{u.key}: no generic parser for source.format={fmt!r}. Add a hook in canon.HOOKS once the file layout is verified."
    )


def descriptors(c: Canon, u: Unit) -> dict:
    x = c.to_scipy()
    per_cell = np.asarray(x.sum(axis=0)).ravel()
    per_cell_nnz = np.diff(c.indptr)
    return {
        "dataset": u.id,
        "matrix": u.name,
        "protocol_class": u.dataset["protocol_class"],
        "tier": u.dataset["tier"],
        "value": u.matrix["value"],
        "genes": c.m,
        "cells": c.n,
        "nnz": c.nnz,
        "density": c.nnz / max(1, c.m * c.n),
        "counts_per_cell_mean": float(per_cell.mean()) if c.n else None,
        "counts_per_cell_median": float(np.median(per_cell)) if c.n else None,
        "nnz_per_cell_median": float(np.median(per_cell_nnz)) if c.n else None,
        "frac_ones": float((c.data == 1).mean()) if c.nnz else None,
        "max_value": int(c.data.max()) if c.nnz else 0,
        "mean_value": float(c.data.mean()) if c.nnz else None,
    }


def run(cfg: Config, units: list[Unit], force: bool = False) -> None:
    for u in units:
        out = canon_path(cfg, u)
        if out.exists() and not force:
            log(f"{u.key}: canonical matrix exists")
            continue
        log(f"{u.key}: parsing")
        c = parse(cfg, u)
        save_canon(c, out)
        write_tsv([descriptors(c, u)], result_path(cfg, "stats", u))
        log(f"{u.key}: {c.m} genes x {c.n} cells, nnz {c.nnz}")


# ----------------------------------------------------------------------------------------------- dataset hooks
def _baron(cfg: Config, u: Unit) -> Canon:
    """Four human donor CSVs (cells x [barcode, assigned_cluster, genes...]) inside GSE84133_RAW.tar."""
    import pandas as pd

    root = _extract(_main_file(cfg, u.dataset))
    blocks, genes, cells = [], None, []
    for p in sorted(root.rglob("*human*_umifm_counts.csv.gz")):
        df = pd.read_csv(p, index_col=0)
        g = [c for c in df.columns if c not in ("barcode", "assigned_cluster")]
        genes = genes or g
        blocks.append(sp.csc_matrix(df[g].to_numpy().T))
        cells += [f"{p.name.split('_')[1]}:{i}" for i in df.index]
    return canon_from_scipy(sp.hstack(blocks).tocsc(), genes, cells)


def _tabula_sapiens(cfg: Config, u: Unit) -> Canon:
    """Keep 10x 3' v3 cells only, counts from raw/X."""
    import anndata as ad

    a = ad.read_h5ad(_main_file_or_glob(cfg, u, "*.h5ad"))
    keep = (a.obs["assay"].astype(str) == "10x 3' v3").to_numpy()
    x = a.raw.X[keep] if a.raw is not None else a.X[keep]
    return canon_from_scipy(sp.csr_matrix(x), list((a.raw or a).var.index), list(a.obs.index[keep]), genes_by_cells=False)


def _xenium(cfg: Config, u: Unit) -> Canon:
    """cell_feature_matrix.h5 inside the outs zip; keep Gene Expression rows (drop control probes and codewords)."""
    root = _extract(_main_file(cfg, u.dataset))
    return parse_tenx_h5(_first(root, "cell_feature_matrix.h5"), "Gene Expression")


def _hao(cfg: Config, u: Unit) -> Canon:
    """GSE164378 3' data: sc.GEX and sc.ADT MTX triplets (file prefixes confirmed on download)."""
    root = _raw_dir(cfg, u.dataset)
    tag = {"rna": "GEX", "adt": "ADT"}[u.name]
    files = sorted(root.rglob(f"*sc.{tag}*"))
    mtx = next(p for p in files if "matrix" in p.name or p.name.endswith(".mtx.gz"))
    feat = next(p for p in files if "features" in p.name or "genes" in p.name)
    bc = next(p for p in files if "barcodes" in p.name)
    return canon_from_scipy(scipy.io.mmread(str(mtx)).tocsc(), _read_lines(feat), _read_lines(bc))


def _cao(cfg: Config, u: Unit) -> Canon:
    root = _raw_dir(cfg, u.dataset)
    return parse_triplet_txt(
        _first(root, "*gene_count.txt.gz"), _first(root, "*gene_annotate.csv.gz"), _first(root, "*cell_annotate.csv.gz")
    )


def _main_file_or_glob(cfg: Config, u: Unit, pattern: str) -> Path:
    d = u.dataset
    if d["source"].get("url"):
        return _main_file(cfg, d)
    hit = _first(_raw_dir(cfg, d), pattern)
    if not hit:
        raise SystemExit(f"{u.key}: no file matching {pattern} under {_raw_dir(cfg, d)}")
    return hit


HOOKS: dict[tuple[str, str], Callable[[Config, Unit], Canon]] = {
    ("baron_pancreas", "*"): _baron,
    ("tabula_sapiens", "*"): _tabula_sapiens,
    ("xenium_breast_rep1", "*"): _xenium,
    ("hao_citeseq", "*"): _hao,
    ("cao_moca", "*"): _cao,
}
