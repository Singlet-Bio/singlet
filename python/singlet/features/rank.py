# SPDX-License-Identifier: MIT
"""Cell-intrinsic rank statistics for gene sets.

Per cell, genes are ranked by count (descending; tied counts share the average of their ranks; undetected genes rank after every
detected gene). A gene at rank ``r`` carries the signal ``s = max(0, 1 - (r - 1) / max_rank)``, so only the cell's top ``max_rank``
genes contribute. For a gene set with ``n`` genes present in the reference:

    auc = mean(s over the set) / (1 - (n - 1) / (2 * max_rank))

which is the Mann-Whitney U of the set's ranks against the rest of the cell's top ``max_rank`` genes, scaled to [0, 1]: 0 when no
set gene is among the cell's top ``max_rank`` genes, 1 when the set's genes are the cell's ``n`` most abundant. UCell (Andreatta &
Carmona 2021) computes ``U = 1 - (sum min(r, max_rank + 1) - n(n + 1)/2) / (n * max_rank) = mean(s) + (n - 1)/(2 * max_rank)``;
``auc`` orders cells exactly as U does. The value depends only on the cell's own counts, so it means the same in every study.

The background a set would reach by chance depends on the abundance of its genes (sets of abundant genes score high in every
cell) and on the cell's depth. :func:`draw_null_sets` draws, for every set, random sets of the same size whose genes are drawn from
the same abundance bin of a fixed per-organism reference (``data/abundance_<organism>.parquet``, not the sample), so the null is a
constant of the method and scoring it in a cell is still cell-intrinsic. :func:`zscore` measures a set's auc against its null sets in that
cell; the written score is that z divided by 10 and clipped to [0, 1] (``registry.Z_SCALE``). In the pilot it tracked the cell's
detected-gene count far less than the raw auc (median |Spearman| 0.06-0.23 vs 0.36-0.52; docs/cell_features.md).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sps

MAX_RANK = 1500


def rank_signal(X, max_rank: int = MAX_RANK) -> sps.csr_matrix:
    """cells x genes CSR of the rank signal ``s`` (float32): non-zero only for detected genes ranked <= ``max_rank``."""
    X = sps.csr_matrix(X)
    X.sum_duplicates()
    n = X.shape[0]
    nnz = np.diff(X.indptr)
    row = np.repeat(np.arange(n, dtype=np.int64), nnz)
    v = X.data.astype(np.float64)
    order = np.lexsort((-v, row))  # by row, then count descending
    rs, vs = row[order], v[order]
    pos = (
        np.arange(len(order), dtype=np.int64) - X.indptr[:-1].astype(np.int64)[rs]
    )  # 0-based rank within the row
    # ties: runs of equal (row, value) get the mean of their 1-based ranks
    new = np.ones(len(order), bool)
    if len(order):
        new[1:] = (rs[1:] != rs[:-1]) | (vs[1:] != vs[:-1])
    gid = np.cumsum(new) - 1
    first = pos[new]
    cnt = np.bincount(gid)
    avg = first + (cnt - 1) / 2.0 + 1.0
    r = avg[gid]
    s = np.maximum(0.0, 1.0 - (r - 1.0) / max_rank).astype(np.float32)
    keep = s > 0
    out = sps.csr_matrix(
        (s[keep], (rs[keep], X.indices[order][keep])), shape=X.shape, dtype=np.float32
    )
    return out


def set_weights(sets: Sequence[np.ndarray], n_genes: int) -> sps.csr_matrix:
    """genes x sets matrix with 1/n on each set's genes (a set's column of S @ W is its mean signal)."""
    rows, cols, vals = [], [], []
    for j, g in enumerate(sets):
        g = np.unique(np.asarray(g, np.int64))
        if not len(g):
            continue
        rows.append(g)
        cols.append(np.full(len(g), j))
        vals.append(np.full(len(g), 1.0 / len(g)))
    if not rows:
        return sps.csr_matrix((n_genes, len(sets)), dtype=np.float32)
    return sps.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_genes, len(sets)),
        dtype=np.float32,
    ).tocsr()


def auc_scale(n: np.ndarray, max_rank: int = MAX_RANK) -> np.ndarray:
    """The largest mean signal an ``n``-gene set can reach (its genes on ranks 1..n)."""
    n = np.asarray(n, np.float64)
    return 1.0 - (np.maximum(n, 1) - 1) / (2.0 * max_rank)


def score_sets(
    X,
    sets: Sequence[np.ndarray],
    *,
    null_sets: Optional[Sequence[Sequence[np.ndarray]]] = None,
    max_rank: int = MAX_RANK,
    chunk: int = 20000,
    S: Optional[sps.csr_matrix] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Score every set in every cell.

    Returns ``(auc, null_mean, null_sd)``: cells x sets float32. ``null_sets[j]`` is a list of index arrays (random sets matched to
    set ``j``); their auc is averaged per cell (``null_mean``) with its standard deviation. Without ``null_sets`` both are None."""
    X = sps.csr_matrix(X)
    G = X.shape[1]
    P = len(sets)
    n = np.array([len(np.unique(g)) for g in sets], np.float64)
    cols: List[np.ndarray] = list(sets)
    owner: List[int] = []
    nn: List[float] = []
    if null_sets is not None:
        for j, ns in enumerate(null_sets):
            for g in ns:
                cols.append(g)
                owner.append(j)
                nn.append(len(np.unique(g)))
    W = set_weights(cols, G)
    scale = auc_scale(np.r_[n, np.array(nn, np.float64)], max_rank).astype(np.float32)
    nc = X.shape[0]
    auc = np.empty((nc, P), np.float32)
    nm = sd = None
    if owner:
        owner_a = np.array(owner)
        k_per = np.bincount(owner_a, minlength=P).astype(np.float32)
        nm = np.zeros((nc, P), np.float32)
        sd = np.zeros((nc, P), np.float32)
    for a in range(0, nc, chunk):
        b = min(a + chunk, nc)
        Sa = rank_signal(X[a:b], max_rank) if S is None else S[a:b]
        M = np.asarray((Sa @ W).todense(), np.float32) / scale
        np.minimum(M, 1.0, out=M)
        auc[a:b] = M[:, :P]
        if owner:
            Z = M[:, P:]
            s1 = np.zeros((b - a, P), np.float32)
            s2 = np.zeros((b - a, P), np.float32)
            for j in range(P):
                idx = np.flatnonzero(owner_a == j)
                if len(idx):
                    s1[:, j] = Z[:, idx].sum(1)
                    s2[:, j] = (Z[:, idx] ** 2).sum(1)
            kk = np.maximum(k_per, 1)
            m = s1 / kk
            var = np.maximum(s2 / kk - m * m, 0) * (kk / np.maximum(kk - 1, 1))
            nm[a:b] = m
            sd[a:b] = np.sqrt(var)
    return auc, nm, sd


def zscore(
    auc: np.ndarray, null_mean: np.ndarray, null_sd: np.ndarray, floor: float = 0.002
) -> np.ndarray:
    """(auc - null mean) / null sd per cell and set; the sd is floored at max(floor, 0.1 * null mean), both from the cell
    itself, so cells whose null sets are all near zero do not get extreme values."""
    den = np.maximum(null_sd, np.maximum(floor, 0.1 * null_mean))
    return ((auc - null_mean) / den).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------- null sets
def abundance_bins(mean: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """Equal-count bins (0..n_bins-1) by rank of the reference mean expression."""
    from scipy.stats import rankdata

    r = rankdata(mean, method="ordinal") - 1
    return np.minimum((r * n_bins) // max(len(mean), 1), n_bins - 1).astype(np.int64)


def draw_null_sets(
    members: Sequence[str],
    ref_ids: Sequence[str],
    ref_bins: np.ndarray,
    k: int,
    seed: int,
) -> List[List[str]]:
    """``k`` random sets for one gene set: each member gene of the reference replaced by a random reference gene of the same
    abundance bin, never a member. Members missing from the reference are skipped. Deterministic for a given seed and reference."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(ref_ids, dtype=object)
    pos = {g: i for i, g in enumerate(ids)}
    idx = np.array([pos[g] for g in members if g in pos], np.int64)
    if not len(idx):
        return [[] for _ in range(k)]
    member = np.zeros(len(ids), bool)
    member[idx] = True
    by_bin = {b: np.flatnonzero((ref_bins == b) & ~member) for b in np.unique(ref_bins[idx])}
    out = []
    for _ in range(k):
        pick = []
        for b in ref_bins[idx]:
            pool = by_bin[b]
            pick.append(ids[pool[rng.integers(len(pool))]] if len(pool) else None)
        out.append([p for p in dict.fromkeys(pick) if p is not None])
    return out
