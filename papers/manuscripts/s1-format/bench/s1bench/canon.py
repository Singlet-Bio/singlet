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


def _glob(root: Path, pattern: str) -> list[Path]:
    """Sorted recursive matches, without macOS archive debris (__MACOSX/, ._* resource forks)."""
    return sorted(p for p in root.rglob(pattern) if not p.name.startswith("._") and "__MACOSX" not in p.parts)


def _first(root: Path, *patterns: str) -> Optional[Path]:
    for pat in patterns:
        hits = _glob(root, pat)
        if hits:
            return hits[0]
    return None


def _main_file(cfg: Config, d: dict) -> Path:
    from .fetch import local_name

    return _raw_dir(cfg, d) / local_name(d["source"], d["source"]["url"], 0)


def _raw_file(cfg: Config, d: dict, i: int) -> Path:
    """The i-th downloaded file of a dataset (0 = source.url, then source.extra in order)."""
    from .fetch import _urls, local_name

    return _raw_dir(cfg, d) / local_name(d["source"], _urls(d)[i], i)


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


def _read_elem(node):
    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem
    return read_elem(node)


def _h5ad_rows(node, keep: Optional[np.ndarray] = None, block: int = 100_000) -> sp.csr_matrix:
    """cells x genes CSR from one h5ad matrix element. Sparse groups are read whole; dense datasets (some atlases store
    counts as dense float32) are read in row blocks so the dense array is never in memory at once."""
    import h5py

    if not isinstance(node, h5py.Dataset):
        x = sp.csr_matrix(_read_elem(node))
        return x if keep is None else x[keep]
    blocks = []
    for s in range(0, node.shape[0], block):
        a = node[s : s + block]
        if keep is not None:
            a = a[keep[s : s + block]]
        blocks.append(sp.csr_matrix(a))
    return sp.vstack(blocks, format="csr")


def parse_h5ad(path: Path, x_location: str = "X", obs_filter: Optional[tuple[str, str]] = None) -> Canon:
    """Counts from X, raw/X or layers/<name>; genes from var (or raw/var) index, cells from obs index. Reads only the
    element named, so a normalised X next to raw counts is never loaded. obs_filter = (column, value) keeps those cells."""
    import h5py

    if not (x_location in ("X", "raw/X") or x_location.startswith("layers/")):
        raise ValueError(f"unknown x_location {x_location}")
    with h5py.File(path, "r") as f:
        obs = _read_elem(f["obs"])
        var = _read_elem(f["raw/var" if x_location == "raw/X" else "var"])
        keep = None
        if obs_filter:
            keep = (obs[obs_filter[0]].astype(str) == obs_filter[1]).to_numpy()
            log(f"{path.name}: keeping {int(keep.sum())} of {len(keep)} cells with {obs_filter[0]} == {obs_filter[1]!r}")
        x = _h5ad_rows(f[x_location], keep)
    cells = list(obs.index if keep is None else obs.index[keep])
    return canon_from_scipy(x, list(var.index), cells, genes_by_cells=False)


def parse_dense_table(paths: list[Path], sep: str, genes_by_cells: bool, chunk: int = 2000,
                      index_col: Optional[int] = 0) -> Canon:
    """Dense text tables (first column = row names, header = column names), read in chunks into sparse blocks and
    concatenated along cells. Several files are concatenated in the order given (cells must not repeat).
    index_col=None is for tables whose header has no field for the row names (R write.table, zUMIs): pandas then takes
    the extra first column of the body as the index."""
    import pandas as pd

    blocks, genes, barcodes = [], None, []
    for p in paths:
        rows, names = [], []
        for df in pd.read_csv(p, sep=sep, index_col=index_col, chunksize=chunk):
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
        flt = src.get("obs_filter")
        return parse_h5ad(main, src.get("x_location", "X"), (flt["column"], flt["value"]) if flt else None)
    if fmt in ("dense_tsv_genes_by_cells", "dense_txt_genes_by_cells"):
        return parse_dense_table([main], "\t", True)
    if fmt == "dense_tsv_genes_by_cells_noindexhdr":
        return parse_dense_table([main], "\t", True, index_col=None)
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
        if c.n == 0 or c.m == 0 or c.nnz == 0:
            raise SystemExit(f"{u.key}: parsed an empty matrix ({c.m} x {c.n}, nnz {c.nnz}); the parser does not fit the file")
        save_canon(c, out)
        write_tsv([descriptors(c, u)], result_path(cfg, "stats", u))
        log(f"{u.key}: {c.m} genes x {c.n} cells, nnz {c.nnz}")


# ----------------------------------------------------------------------------------------------- dataset hooks
def _baron(cfg: Config, u: Unit) -> Canon:
    """Four human donor CSVs (cells x [barcode, assigned_cluster, genes...]) inside GSE84133_RAW.tar."""
    import pandas as pd

    root = _extract(_main_file(cfg, u.dataset))
    blocks, genes, cells = [], None, []
    for p in _glob(root, "*human*_umifm_counts.csv.gz"):
        df = pd.read_csv(p, index_col=0)
        g = [c for c in df.columns if c not in ("barcode", "assigned_cluster")]
        genes = genes or g
        blocks.append(sp.csc_matrix(df[g].to_numpy().T))
        cells += [f"{p.name.split('_')[1]}:{i}" for i in df.index]
    return canon_from_scipy(sp.hstack(blocks).tocsc(), genes, cells)


def _hao(cfg: Config, u: Unit) -> Canon:
    """GSE164378 3' data, one 10x MTX triplet per GEO sample: GSM5008737_RNA_3P-* and GSM5008738_ADT_3P-*."""
    root = _raw_dir(cfg, u.dataset)
    tag = {"rna": "RNA_3P", "adt": "ADT_3P"}[u.name]
    files = sorted(root.glob(f"*_{tag}-*"))
    mtx = next(p for p in files if "matrix" in p.name)
    feat = next(p for p in files if "features" in p.name)
    bc = next(p for p in files if "barcodes" in p.name)
    return canon_from_scipy(scipy.io.mmread(str(mtx)).tocsc(), _read_lines(feat), _read_lines(bc))


def _cao(cfg: Config, u: Unit) -> Canon:
    root = _raw_dir(cfg, u.dataset)
    return parse_triplet_txt(
        _first(root, "*gene_count.txt.gz"), _first(root, "*gene_annotate.csv.gz"), _first(root, "*cell_annotate.csv.gz")
    )


def _union_genes(parts: list[tuple[str, Canon]]) -> Canon:
    """Concatenate matrices whose gene lists differ (per-batch tables): rows are the union of genes in order of first
    appearance, absent genes are zero, and barcodes are prefixed with the part name so they stay unique."""
    order: dict[str, int] = {}
    for _, c in parts:
        for g in c.genes:
            order.setdefault(str(g), len(order))
    blocks, cells = [], []
    for name, c in parts:
        x = c.to_scipy().tocoo()
        rows = np.fromiter((order[str(g)] for g in c.genes), dtype=np.int64, count=c.m)[x.row]
        blocks.append(sp.csc_matrix((x.data, (rows, x.col)), shape=(len(order), c.n)))
        cells += [f"{name}:{b}" for b in c.barcodes]
    return canon_from_scipy(sp.hstack(blocks).tocsc(), list(order), cells)


def _mca(cfg: Config, u: Unit) -> Canon:
    """GSE108097_RAW.tar: one dense genes x cells DGE table per tissue batch (GSM*_<Tissue>_dge.txt.gz). Two layouts
    occur: R write.table (space separated, quoted, header without a gene field) and Drop-seq (tab separated, GENE column).
    The cell-line and species-mixing controls (293T, 3T3, mixed species) are left out."""
    import re

    root = _extract(_main_file(cfg, u.dataset))
    files = [p for p in _glob(root, "*_dge.txt.gz") if not re.search(r"293T|3T3|[Mm]ix|GSM2906480", p.name)]
    log(f"{u.key}: {len(files)} tissue DGE tables")
    parts = []
    for p in files:
        with gzip.open(p, "rt") as f:
            head, body = f.readline().rstrip("\n"), f.readline().rstrip("\n")
        sep = "\t" if "\t" in head else " "
        index_col = None if len(body.split(sep)) == len(head.split(sep)) + 1 else 0
        parts.append((p.name.split("_", 1)[1].replace("_dge.txt.gz", ""), parse_dense_table([p], sep, True, index_col=index_col)))
    return _union_genes(parts)


def _mat_strings(a) -> list[str]:
    """Names from a MATLAB v5 variable: a char matrix (one padded row per name) or a cell array."""
    a = np.asarray(a)
    if a.dtype.kind == "U":
        return [str(s).strip() for s in a.ravel()]
    return [str(np.asarray(s).ravel()[0]).strip() if np.asarray(s).size else "" for s in a.ravel()]


def _rosenberg(cfg: Config, u: Unit) -> Canon:
    """GSM3017261_150000_CNS_nuclei.mat.gz (MATLAB v5): the sparse count matrix plus gene and barcode names."""
    path = _main_file(cfg, u.dataset)
    mat = path.with_suffix("")  # .mat
    if not mat.exists():
        with gzip.open(path, "rb") as fi, mat.open("wb") as fo:
            import shutil

            shutil.copyfileobj(fi, fo, 1 << 22)
    m = scipy.io.loadmat(str(mat))
    keys = [k for k in m if not k.startswith("__")]
    log(f"{u.key}: .mat variables {keys}")
    xk = next(k for k in keys if sp.issparse(m[k]))
    genes = _mat_strings(m[next(k for k in keys if "gene" in k.lower())])
    cells = _mat_strings(m[next(k for k in keys if "barcode" in k.lower())])
    x = m[xk]
    if x.shape == (len(cells), len(genes)):
        return canon_from_scipy(sp.csr_matrix(x), genes, cells, genes_by_cells=False)
    return canon_from_scipy(sp.csc_matrix(x), genes, cells)


def _visium_hd(cfg: Config, u: Unit) -> Canon:
    """binned_outputs.tar.gz (gzip, not seekable): stream it once and keep only the 8 um filtered matrix."""
    member = "square_008um/filtered_feature_bc_matrix.h5"
    out = _raw_dir(cfg, u.dataset) / "square_008um_filtered_feature_bc_matrix.h5"
    if not out.exists():
        import shutil

        with tarfile.open(_main_file(cfg, u.dataset), "r|gz") as t:
            for m in t:
                if m.name.endswith(member):
                    with t.extractfile(m) as fi, out.with_suffix(".part").open("wb") as fo:
                        shutil.copyfileobj(fi, fo, 1 << 22)
                    out.with_suffix(".part").replace(out)
                    break
        if not out.exists():
            raise SystemExit(f"{u.key}: {member} not found in the archive")
    return parse_tenx_h5(out, u.matrix.get("feature_type"))


def _slideseq(cfg: Config, u: Unit) -> Canon:
    """Zenodo tarball holding Puck_200115_08_counts_no_cutoff.zip, a 10x-style MTX triplet of every bead (no UMI cutoff)."""
    root = _extract(_main_file(cfg, u.dataset))
    return parse_mtx_dir(_extract(_first(root, "*counts_no_cutoff.zip")))


def _tabula_muris(cfg: Config, u: Unit) -> Canon:
    """FACS.zip: one genes x cells CSV of read counts per tissue (FACS/<Tissue>-counts.csv), same gene list in each."""
    root = _extract(_main_file(cfg, u.dataset))
    files = _glob(root, "*-counts.csv")
    log(f"{u.key}: {len(files)} tissue tables")
    return parse_dense_table(files, ",", True)


def _paul(cfg: Config, u: Unit) -> Canon:
    """GSE72857_umitab.txt.gz holds every well (cells, empty wells, controls). Keep the 2,730 cells of Paul et al., named
    by the wells in the scanpy paul15 file (extra file 1); its own values are batch-corrected floats and are not used."""
    import h5py

    c = parse_dense_table([_main_file(cfg, u.dataset)], "\t", True, index_col=None)
    with h5py.File(_raw_file(cfg, u.dataset, 1), "r") as f:
        names = []
        for k in f.keys():
            if k.startswith("data.debatched_") and "names" in k:
                names.append({s.decode() if isinstance(s, bytes) else str(s) for s in f[k][()].ravel()})
    have = {str(b) for b in c.barcodes}
    wells = max(names, key=lambda s: len(s & have)) if names else set()
    keep = np.flatnonzero([str(b) in wells for b in c.barcodes])
    log(f"{u.key}: {len(keep)} of {c.n} wells are paul15 cells")
    if len(keep) < 2700:
        raise SystemExit(f"{u.key}: only {len(keep)} wells match the paul15 cell names; check the paul15 key names")
    x = c.to_scipy()[:, keep]
    return canon_from_scipy(x, list(c.genes), [c.barcodes[i] for i in keep])


HOOKS: dict[tuple[str, str], Callable[[Config, Unit], Canon]] = {
    ("baron_pancreas", "*"): _baron,
    ("hao_citeseq", "*"): _hao,
    ("cao_moca", "*"): _cao,
    ("mca_microwell", "*"): _mca,
    ("rosenberg_splitseq", "*"): _rosenberg,
    ("visium_hd_crc", "*"): _visium_hd,
    ("slideseq2_hippocampus", "*"): _slideseq,
    ("tabula_muris_facs", "*"): _tabula_muris,
    ("paul_marsseq", "*"): _paul,
}
