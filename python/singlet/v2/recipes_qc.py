# SPDX-License-Identifier: MIT
"""Per-cell QC metrics recomputed from the exon and intron matrices (owner: W3).

Numpy port of ``include/singlet/pileup/cell_qc_metrics.h:compute_cell_qc`` for the columns of ``cell_qc_metrics.tsv``:

  n_umi         exon + intron column sum
  n_genes       genes (row-name prefix before the first '_') whose exon+intron sum in the cell is > 0
  pct_mt        float32(mt_umis * (100.0 / n_umi)); mt gene: chromosome chrM/MT/chrMT or symbol [Mm][Tt]-*
  pct_ribo      the same for symbols [Rr][Pp][LlSs]*
  intronic_pct  the same for the intron matrix

The gene symbol and chromosome are parsed from the row names (``<gid>_<symbol>_<chrom>:<s>-<e>``, introns
``<gid>_<symbol>_intron_<chrom>:<s>-<e>``).  The values printed by the pipeline are float32 values written with
``%.4f``; this module returns the float32 arrays.
"""
from __future__ import annotations

import numpy as np


def split_row_names(names, intron: bool):
    """-> (gene_id[list], symbol[list], chrom[list]) for exon or intron row names."""
    gid, sym, chrom = [], [], []
    for n in names:
        rest, _, coord = n.rpartition("_")
        c = coord.split(":", 1)[0]
        if intron:
            if rest.endswith("_intron"):
                rest = rest[: -len("_intron")]
        g, _, s = rest.partition("_")
        gid.append(g)
        sym.append(s)
        chrom.append(c)
    return gid, sym, chrom


def is_mt_gene(sym: str, chrom: str) -> bool:
    if chrom in ("chrM", "MT", "chrMT"):
        return True
    return len(sym) >= 3 and sym[0] in "Mm" and sym[1] in "Tt" and sym[2] == "-"


def is_ribo_gene(sym: str) -> bool:
    return len(sym) >= 3 and sym[0] in "Rr" and sym[1] in "Pp" and sym[2] in "LlSs"


def _masked_col_sums(M, mask):
    if M.nnz == 0:
        return np.zeros(M.shape[1], dtype=np.float64)
    w = np.where(mask[M.indices], M.data, 0).astype(np.float64)
    return np.bincount(M.col_ids(), weights=w, minlength=M.shape[1])


def compute_qc(ex, it):
    """dict of float32/int arrays: n_umi (u64), n_genes (i64), pct_mt, pct_ribo, intronic_pct (float32)."""
    n = ex.shape[1]
    egid, esym, echr = split_row_names(ex.rownames, False)
    has_it = it is not None and it.shape[1] == n and it.shape[0] > 0
    if has_it:
        igid, isym, ichr = split_row_names(it.rownames, True)
    else:
        igid, isym, ichr = [], [], []
    genes = {}
    eg = np.array([genes.setdefault(g, len(genes)) for g in egid], dtype=np.int64)
    ig = np.array([genes.setdefault(g, len(genes)) for g in igid], dtype=np.int64)
    G = max(len(genes), 1)
    emt = np.array([is_mt_gene(s, c) for s, c in zip(esym, echr)], dtype=bool)
    erb = np.array([is_ribo_gene(s) for s in esym], dtype=bool)
    etot = ex.col_sums().astype(np.float64)
    mt = _masked_col_sums(ex, emt)
    rb = _masked_col_sums(ex, erb)
    itot = np.zeros(n, dtype=np.float64)
    if has_it:
        imt = np.array([is_mt_gene(s, c) for s, c in zip(isym, ichr)], dtype=bool)
        irb = np.array([is_ribo_gene(s) for s in isym], dtype=bool)
        itot = it.col_sums().astype(np.float64)
        mt = mt + _masked_col_sums(it, imt)
        rb = rb + _masked_col_sums(it, irb)
    tot = etot + itot
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(tot > 0, 100.0 / np.where(tot > 0, tot, 1.0), 0.0)
    out = {
        "n_umi": tot.astype(np.uint64),
        "pct_mt": (mt * inv).astype(np.float32),
        "pct_ribo": (rb * inv).astype(np.float32),
        "intronic_pct": (itot * inv).astype(np.float32),
    }
    # distinct genes with a positive exon+intron sum
    keys = [ex.col_ids() * G + eg[ex.indices]]
    wts = [ex.data.astype(np.float64)]
    if has_it:
        keys.append(it.col_ids() * G + ig[it.indices])
        wts.append(it.data.astype(np.float64))
    key = np.concatenate(keys)
    w = np.concatenate(wts)
    if len(key):
        uq, inv_ix = np.unique(key, return_inverse=True)
        s = np.bincount(inv_ix, weights=w, minlength=len(uq))
        ng = np.bincount(uq[s > 0] // G, minlength=n)
    else:
        ng = np.zeros(n, dtype=np.int64)
    out["n_genes"] = ng.astype(np.int64)
    return out


def get_qc(ctx):
    """compute_qc for a context, cached on the context object."""
    c = getattr(ctx, "_w3_qc", None)
    if c is None:
        ex = ctx.matrix("exon_counts.1pz")
        it = ctx.matrix("intron_counts.1pz")
        c = compute_qc(ex, it) if ex is not None else None
        try:
            ctx._w3_qc = c
        except Exception:
            pass
    return c
