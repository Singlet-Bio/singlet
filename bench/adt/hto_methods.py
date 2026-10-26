# SPDX-License-Identifier: MIT
"""Hashtag (HTO) demultiplexing methods behind one interface, for benchmarking on identical counts.

Every ``run_*`` function takes the same inputs and returns the same output:

* ``hto``: DataFrame, rows = hashtag names, columns = barcodes of called cells, integer counts.
* ``empties``: same rows, columns = empty-droplet barcodes (or None).
* ``rna``: AnnData (cells x genes, raw counts) whose ``obs_names`` cover ``hto.columns`` and ``empties.columns``;
  only demuxEM needs it.
* ``workdir``: scratch directory for methods that shell out (a temporary directory when None).

Output: DataFrame indexed by ``hto.columns`` (same order) with ``call`` (a name in ``hto.index``, ``"Doublet"`` or
``"Negative"``) and ``score`` (float, the method's confidence or NaN); the wall time of the method itself is in
``df.attrs["seconds"]``.

Methods: singlet (``singlet.adt.hto.hashed_drops``), Seurat ``HTODemux`` and ``MULTIseqDemux`` (via Rscript),
scanpy's ``hashsolo``, and ``demuxEM``. Heavy dependencies are imported inside each function.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

DOUBLET = "Doublet"
NEGATIVE = "Negative"


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------

def _finish(hto: pd.DataFrame, call, score, seconds: float) -> pd.DataFrame:
    out = pd.DataFrame({"call": pd.Series(list(call), dtype=object).astype(str).to_numpy(),
                        "score": np.asarray(score, dtype=float)}, index=hto.columns)
    out.attrs["seconds"] = float(seconds)
    return out


def _workdir(workdir):
    """(Path, TemporaryDirectory or None)."""
    if workdir is None:
        td = tempfile.TemporaryDirectory(prefix="hto_methods_")
        return Path(td.name), td
    p = Path(workdir)
    p.mkdir(parents=True, exist_ok=True)
    return p, None


def _name_map(names):
    """Map every spelling a tool may return (original, '_' -> '-') back to the original hashtag name."""
    m = {}
    for n in names:
        m[n] = n
        m[n.replace("_", "-")] = n
    return m


def _normalise(values, names):
    """Map raw labels to hashtag names / Doublet / Negative; unknown labels become Negative."""
    m = _name_map(names)
    out = []
    for v in values:
        v = str(v)
        if v in m:
            out.append(m[v])
        elif v.lower() == "doublet":
            out.append(DOUBLET)
        else:
            out.append(NEGATIVE)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# singlet
# ---------------------------------------------------------------------------------------------------------------------

def run_singlet(hto: pd.DataFrame, empties: pd.DataFrame | None = None, rna=None, workdir=None,
                confident_min: float = 2.0) -> pd.DataFrame:
    """singlet's hashedDrops-style rule; ambient = empty-droplet row sums, else estimated from the cells (bimodal)."""
    from singlet.adt.hto import hashed_drops

    t0 = time.perf_counter()
    X = hto.to_numpy(dtype=float)
    ambient = empties.to_numpy(dtype=float).sum(axis=1) if empties is not None and empties.shape[1] else None
    call, lfc_top, _ = hashed_drops(X, ambient, confident_min=confident_min)
    names = list(hto.index)
    labels = [NEGATIVE if c == -1 else DOUBLET if c == -2 else names[int(c)] for c in call]
    return _finish(hto, labels, lfc_top, time.perf_counter() - t0)


def run_singlet_mixture(hto: pd.DataFrame, empties=None, rna=None, workdir=None, **kw) -> pd.DataFrame:
    """singlet's per-hashtag mixture-of-regressions caller (``singlet.adt.hto.mixture_calls``); cells only."""
    from singlet.adt.hto import mixture_calls

    t0 = time.perf_counter()
    call, score, _ = mixture_calls(hto.to_numpy(dtype=float), **kw)
    names = list(hto.index)
    labels = [NEGATIVE if c == -1 else DOUBLET if c == -2 else names[int(c)] for c in call]
    return _finish(hto, labels, score, time.perf_counter() - t0)


def run_singlet_hybrid(hto: pd.DataFrame, empties=None, rna=None, workdir=None, confident_min=0.5, **kw):
    """hashedDrops singlets (``confident_min``) unless the mixture finds two positive hashtags (then doublet)."""
    from singlet.adt.hto import hashed_drops, mixture_calls

    t0 = time.perf_counter()
    X = hto.to_numpy(dtype=float)
    ambient = empties.to_numpy(dtype=float).sum(axis=1) if empties is not None and empties.shape[1] else None
    call, lfc, _ = hashed_drops(X, ambient, confident_min=confident_min)
    mcall, _, _ = mixture_calls(X, **kw)
    call = np.where(mcall == -2, -2, call)
    names = list(hto.index)
    labels = [NEGATIVE if c == -1 else DOUBLET if c == -2 else names[int(c)] for c in call]
    return _finish(hto, labels, lfc, time.perf_counter() - t0)


def _variant(fn, **fixed):
    def run(hto, empties=None, rna=None, workdir=None):
        return fn(hto, empties, rna, workdir, **fixed)
    return run


# the fixed grid of Singlet variants compared by leave-one-dataset-out selection (bench/adt/hto_zenodo.py)
VARIANTS = {}
for _cov in ("others", "total", "none"):
    for _pos in (0.5, 0.8):
        for _sc in (False, True):
            _tag = f"mix_{_cov}_p{int(_pos * 10)}{'_amb' if _sc else ''}"
            VARIANTS[_tag] = _variant(run_singlet_mixture, covariate=_cov, positive=_pos, scale_ambient=_sc)
            VARIANTS["hyb_" + _tag[4:]] = _variant(run_singlet_hybrid, covariate=_cov, positive=_pos, scale_ambient=_sc)
VARIANTS["hd_cm05"] = _variant(run_singlet, confident_min=0.5)
VARIANTS["hd_cm2"] = _variant(run_singlet, confident_min=2.0)


# ---------------------------------------------------------------------------------------------------------------------
# Seurat (HTODemux, MULTIseqDemux) via Rscript
# ---------------------------------------------------------------------------------------------------------------------

_R_HTODEMUX = r"""
args <- commandArgs(trailingOnly = TRUE)
counts_csv <- args[1]; out_csv <- args[2]; q <- as.numeric(args[3])
suppressMessages(library(Seurat))
m <- as.matrix(read.csv(counts_csv, row.names = 1, check.names = FALSE))
hto_matrix <- m
obj <- CreateSeuratObject(counts = hto_matrix, assay = "HTO")
obj <- NormalizeData(obj, assay = "HTO", normalization.method = "CLR", margin = 2)
obj <- HTODemux(obj, assay = "HTO", positive.quantile = q)
md <- obj@meta.data
res <- data.frame(barcode = colnames(obj),
                  global = as.character(md$HTO_classification.global),
                  id = as.character(md$hash.ID),
                  score = as.numeric(md$HTO_margin), stringsAsFactors = FALSE)
write.csv(res, out_csv, row.names = FALSE)
"""

_R_MULTISEQ = r"""
args <- commandArgs(trailingOnly = TRUE)
counts_csv <- args[1]; out_csv <- args[2]
suppressMessages(library(Seurat))
m <- as.matrix(read.csv(counts_csv, row.names = 1, check.names = FALSE))
hto_matrix <- m
obj <- CreateSeuratObject(counts = hto_matrix, assay = "HTO")
obj <- NormalizeData(obj, assay = "HTO", normalization.method = "CLR", margin = 2)
obj <- MULTIseqDemux(obj, assay = "HTO", autoThresh = TRUE)
md <- obj@meta.data
res <- data.frame(barcode = colnames(obj), id = as.character(md$MULTI_ID),
                  stringsAsFactors = FALSE)
write.csv(res, out_csv, row.names = FALSE)
"""


def _run_rscript(hto, script_name, script, extra_args, workdir, rscript):
    wd, td = _workdir(workdir)
    try:
        counts_csv = wd / f"{script_name}_counts.csv"
        out_csv = wd / f"{script_name}_out.csv"
        script_path = wd / f"{script_name}.R"
        hto.astype(np.int64).to_csv(counts_csv)
        script_path.write_text(script.lstrip(), encoding="utf-8", newline="\n")
        if out_csv.exists():
            out_csv.unlink()
        proc = subprocess.run([rscript, str(script_path), str(counts_csv), str(out_csv), *map(str, extra_args)],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not out_csv.exists():
            raise RuntimeError(f"{script_name} failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
        return pd.read_csv(out_csv, dtype={"barcode": str})
    finally:
        if td is not None:
            td.cleanup()


def run_htodemux(hto, empties=None, rna=None, workdir=None, positive_quantile=0.99, rscript="Rscript"):
    """Seurat ``HTODemux`` (CLR over hashtags per cell); score = ``HTO_margin``."""
    t0 = time.perf_counter()
    res = _run_rscript(hto, "htodemux", _R_HTODEMUX, [positive_quantile], workdir, rscript)
    res = res.set_index("barcode").reindex(hto.columns)
    names = list(hto.index)
    labels = []
    for g, i in zip(res["global"], res["id"]):
        g = str(g)
        labels.append(DOUBLET if g == "Doublet" else NEGATIVE if g == "Negative" else _normalise([i], names)[0])
    return _finish(hto, labels, res["score"].to_numpy(dtype=float), time.perf_counter() - t0)


def run_multiseqdemux(hto, empties=None, rna=None, workdir=None, rscript="Rscript"):
    """Seurat ``MULTIseqDemux(autoThresh = TRUE)``; no score."""
    t0 = time.perf_counter()
    res = _run_rscript(hto, "multiseqdemux", _R_MULTISEQ, [], workdir, rscript)
    res = res.set_index("barcode").reindex(hto.columns)
    labels = _normalise(res["id"].fillna(NEGATIVE), list(hto.index))
    return _finish(hto, labels, np.full(hto.shape[1], np.nan), time.perf_counter() - t0)


# ---------------------------------------------------------------------------------------------------------------------
# HashSolo (scanpy.external)
# ---------------------------------------------------------------------------------------------------------------------

def run_hashsolo(hto, empties=None, rna=None, workdir=None):
    """``scanpy.external.pp.hashsolo``; score = probability of the most likely hypothesis when reported."""
    import anndata as ad
    import scanpy.external as sce

    t0 = time.perf_counter()
    names = list(hto.index)
    # hashsolo takes obs columns; use safe column names and map back afterwards
    cols = [f"hto{i}" for i in range(len(names))]
    obs = pd.DataFrame(hto.to_numpy(dtype=np.float64).T, index=hto.columns.astype(str), columns=cols)
    adata = ad.AnnData(X=np.zeros((obs.shape[0], 1), dtype=np.float32), obs=obs)
    sce.pp.hashsolo(adata, cols, inplace=True)
    cls = adata.obs["Classification"].astype(str)
    back = dict(zip(cols, names))
    labels = [back.get(c, DOUBLET if c.lower() == "doublet" else NEGATIVE) for c in cls]
    score = np.full(len(cls), np.nan)
    if "most_likely_hypothesis" in adata.obs:
        hyp = adata.obs["most_likely_hypothesis"].to_numpy()
        prob_cols = {0: "negative_hypothesis_probability", 1: "singlet_hypothesis_probability",
                     2: "doublet_hypothesis_probability"}
        if all(c in adata.obs for c in prob_cols.values()):
            P = np.column_stack([adata.obs[prob_cols[k]].to_numpy(dtype=float) for k in (0, 1, 2)])
            score = P[np.arange(len(hyp)), np.asarray(hyp, dtype=int)]
    return _finish(hto, labels, score, time.perf_counter() - t0)


# ---------------------------------------------------------------------------------------------------------------------
# demuxEM
# ---------------------------------------------------------------------------------------------------------------------

def run_demuxem(hto, empties, rna, workdir=None, threads=4):
    """demuxEM (0.1.7): background model from all droplets, then EM on the called cells.

    Follows the package's pipeline: an RNA object of all barcodes (``n_genes`` / ``n_counts`` in obs), an HTO object
    of the same barcodes (``estimate_background_probs`` fits the ambient profile from the low-count droplets),
    then ``demultiplex`` writes ``demux_type`` and ``assignment`` into the RNA object. Needs ``demuxEM`` and
    ``pegasusio``. ``score`` is NaN (demuxEM reports no single per-cell confidence)."""
    import scipy.sparse as sp
    import demuxEM.tools as dt
    import pegasusio as io

    if empties is None or rna is None:
        raise ValueError("demuxEM needs the empty droplets and an RNA AnnData covering all droplets")
    t0 = time.perf_counter()
    names = list(hto.index)
    all_hto = pd.concat([hto, empties], axis=1)
    barcodes = [str(b) for b in all_hto.columns]
    missing = [b for b in barcodes if b not in set(map(str, rna.obs_names))]
    if missing:
        raise ValueError(f"{len(missing)} barcodes are missing from rna.obs_names (e.g. {missing[:3]})")
    r = rna[barcodes]
    Xr = sp.csr_matrix(r.X)
    n_counts = np.asarray(Xr.sum(axis=1)).ravel()
    n_genes = np.asarray((Xr > 0).sum(axis=1)).ravel()
    rna_data = io.UnimodalData(
        barcode_metadata=pd.DataFrame({"n_genes": n_genes, "n_counts": n_counts}, index=barcodes),
        feature_metadata=pd.DataFrame(index=[str(g) for g in r.var_names]),
        matrices={"X": Xr}, genome="unknown", modality="rna")
    adt_data = io.UnimodalData(
        barcode_metadata=pd.DataFrame(index=barcodes),
        feature_metadata=pd.DataFrame(index=names),
        matrices={"X": sp.csr_matrix(all_hto.to_numpy(dtype=np.float64).T)}, genome="unknown", modality="citeseq")
    dt.estimate_background_probs(adt_data, random_state=0)
    dt.demultiplex(rna_data, adt_data, n_threads=threads)
    obs = rna_data.obs.loc[[str(b) for b in hto.columns]]
    labels = []
    for t, a in zip(obs["demux_type"].astype(str), obs["assignment"].astype(str)):
        if t == "singlet":
            labels.append(_normalise([a], names)[0])
        elif t == "doublet":
            labels.append(DOUBLET)
        else:
            labels.append(NEGATIVE)  # "unknown"
    return _finish(hto, labels, np.full(len(labels), np.nan), time.perf_counter() - t0)


METHODS = {
    "singlet": run_singlet,
    "singlet_mixture": run_singlet_mixture,
    "htodemux": run_htodemux,
    "multiseqdemux": run_multiseqdemux,
    "hashsolo": run_hashsolo,
    "demuxem": run_demuxem,
}
