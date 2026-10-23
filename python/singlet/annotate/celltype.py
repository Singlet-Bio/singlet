# SPDX-License-Identifier: MIT
"""Per-cell cell type calls against the cell guide, with optional per-sample self-training.

``guide``      Marker scores for every guide node in one sparse product over a rank signal (UCell-like, cell-intrinsic: a gene's
               signal is ``1 - (rank - 1) / max_rank`` within its own cell, 0 beyond ``max_rank`` or when not detected), smoothed
               over a kNN graph of the sample, then gated top-down: a cell enters the best child when it scores at least
               ``min_score`` and beats the runner-up sibling by ``min_margin``, otherwise it stops at the parent. Stopping early is
               the abstention: a cell with no evidence below the root gets no call. No reference data, cost linear in cells.
``selftrain``  The guide's confident calls (the upper half by margin of every node at depth >= 2) train a logistic regression
               on the whole transcriptome of the same sample, which then relabels every cell, voted within over-clustered
               neighbourhoods. Where the regression and the guide disagree the call backs off to their deepest common ancestor,
               and a regression probability below ``min_prob`` backs off to the guide call.

Both are the methods benchmarked in Singlet-Bio/singlet-agents ``engine/docs/phenotype/REPORT.md`` (guide v2: depth 2/3/4
accuracy 0.890/0.807/0.563 over 15 datasets; with self-training 0.880/0.745/0.570 on 5 held-out datasets), re-implemented here
in NumPy so the atlas run needs no engine build. ``docs/cell_annotation.md`` has the accuracy on Singlet's own counts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sps

from .guide import Guide, ontology

MAX_RANK = 1500
NEG_WEIGHT = 0.5
ALPHA = 0.8  # kNN smoothing weight
BETA = 0.8  # look-ahead strength
MIN_SCORE = 0.02
MIN_MARGIN = 0.01
TEMPERATURE = 0.05
MIN_GENES = 100  # cells detecting fewer genes get no call


# --------------------------------------------------------------------------------------------------------------- signals
def lognorm(X: sps.csr_matrix, target_sum: float = 1e4) -> sps.csr_matrix:
    """log1p(CP10k), CSR float32, sharing the index arrays of ``X``."""
    X = sps.csr_matrix(X)
    tot = np.asarray(X.sum(1), dtype=np.float64).ravel()
    f = (target_sum / np.maximum(tot, 1)).astype(np.float32)
    data = np.log1p(X.data.astype(np.float32) * np.repeat(f, np.diff(X.indptr)))
    return sps.csr_matrix((data, X.indices, X.indptr), shape=X.shape)


def row_ranks(X: sps.csr_matrix) -> np.ndarray:
    """Average-tie descending rank (1 = highest) of every stored value within its row, aligned with ``X.data``."""
    n = X.shape[0]
    nnz = np.diff(X.indptr)
    row = np.repeat(np.arange(n, dtype=np.int64), nnz)
    order = np.lexsort((-X.data, row))
    v = X.data[order]
    r = row[order]
    pos = np.arange(len(v)) - X.indptr[r]
    new = np.ones(len(v), bool)
    new[1:] = (r[1:] != r[:-1]) | (v[1:] != v[:-1])
    gid = np.cumsum(new) - 1
    gstart = np.flatnonzero(new)
    gend = np.r_[gstart[1:], len(v)] - 1
    avg = (pos[gstart] + pos[gend]) / 2.0 + 1.0
    out = np.empty(len(v), np.float32)
    out[order] = avg[gid]
    return out


def node_scores(guide: Guide, X: sps.csr_matrix, chunk: int = 20000) -> np.ndarray:
    """Cells x nodes marker scores (float32; column 0, the root, is 0)."""
    W = sps.csr_matrix(guide.Wpos - NEG_WEIGHT * guide.Wneg, dtype=np.float32)
    rows = np.flatnonzero(np.diff(W.indptr))
    Wr = W[rows]
    out = np.zeros((X.shape[0], W.shape[1]), np.float32)
    for s in range(0, X.shape[0], chunk):
        Xc = sps.csr_matrix(X[s : s + chunk], dtype=np.float32)
        Xc.sort_indices()
        rk = row_ranks(Xc)
        sig = np.clip(1.0 - (rk - 1.0) / MAX_RANK, 0, None).astype(np.float32)
        S = sps.csr_matrix((sig, Xc.indices, Xc.indptr), shape=Xc.shape)[:, rows]
        out[s : s + chunk] = np.asarray((S @ Wr).todense(), np.float32)
    out[:, 0] = 0
    return out


def neighbours(
    X: sps.csr_matrix, n_hvg: int = 2000, n_pcs: int = 30, k: int = 15, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """PCA of log1p CP10k over the most dispersed genes (z-scored, clipped at 10) and exact kNN. Returns (pcs, knn index)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import NearestNeighbors

    Xn = lognorm(X)
    mu = np.asarray(Xn.mean(0)).ravel()
    m2 = np.asarray(Xn.multiply(Xn).mean(0)).ravel()
    var = np.maximum(m2 - mu**2, 0)
    disp = np.where(mu > 0, var / np.maximum(mu, 1e-12), 0)
    z = np.zeros_like(disp)
    if (mu > 0).any():
        bins = np.digitize(mu, np.quantile(mu[mu > 0], np.linspace(0, 1, 21)[1:-1]))
        for b in np.unique(bins):
            m = (bins == b) & (mu > 0)
            if m.sum() > 1:
                z[m] = (disp[m] - disp[m].mean()) / (disp[m].std() + 1e-12)
    hv = np.argsort(-z)[:n_hvg]
    Y = Xn[:, hv].toarray()
    Y -= mu[hv].astype(np.float32)
    Y /= np.sqrt(np.maximum(var[hv], 1e-12)).astype(np.float32)
    np.clip(Y, -10, 10, out=Y)
    n_pcs = max(1, min(n_pcs, Y.shape[0] - 1, Y.shape[1] - 1))
    pcs = (
        TruncatedSVD(n_pcs, random_state=seed, algorithm="randomized", n_iter=4)
        .fit_transform(Y)
        .astype(np.float32)
    )
    k = min(k, X.shape[0])
    idx = (
        NearestNeighbors(n_neighbors=k)
        .fit(pcs)
        .kneighbors(pcs, return_distance=False)
        .astype(np.int32)
    )
    return pcs, idx


def overcluster(
    knn_idx: np.ndarray, resolution: float = 5.0, seed: int = 0
) -> Optional[np.ndarray]:
    """Leiden communities on the kNN graph at high resolution (for majority voting); None when leidenalg is not installed."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        return None
    n, k = knn_idx.shape
    src = np.repeat(np.arange(n), k)
    g = ig.Graph(
        n=n, edges=list(zip(src.tolist(), knn_idx.ravel().tolist())), directed=False
    ).simplify()
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=seed
    )
    return np.asarray(part.membership, np.int32)


# --------------------------------------------------------------------------------------------------------------- gating
def lookahead(
    guide: Guide, scores: np.ndarray, beta: float = BETA, max_depth: int = 2
) -> np.ndarray:
    """``eff[j] = max(s[j], beta * max over children eff[c])`` bottom-up for nodes at depth 1..max_depth, so a heterogeneous parent
    (e.g. haematopoietic cells, which include PTPRC-negative erythroid cells) wins when one of its children clearly does."""
    eff = scores.copy()
    for j in reversed(guide._topo()):
        ch = guide.gateable(j)
        if ch and 0 < guide.depth[j] <= max_depth:
            eff[:, j] = np.maximum(eff[:, j], beta * eff[:, ch].max(1))
    return eff


def gate(
    guide: Guide,
    scores: np.ndarray,
    min_score: float = MIN_SCORE,
    min_margin: float = MIN_MARGIN,
    temperature: float = TEMPERATURE,
):
    """Walk every cell from the root. Returns (node per cell, per-step softmax confidence, per-step margin), steps = depth."""
    n = scores.shape[0]
    D = int(guide.depth.max()) + 1
    cur = np.zeros(n, np.int32)
    active = np.ones(n, bool)
    conf = np.zeros((n, D), np.float32)
    marg = np.zeros((n, D), np.float32)
    for d in range(D):
        nxt = cur.copy()
        for p in np.unique(cur[active]):
            ch = guide.gateable(int(p))
            idx = np.flatnonzero(active & (cur == p))
            if not ch:
                active[idx] = False
                continue
            s = scores[np.ix_(idx, ch)]
            o = np.argsort(-s, 1)
            r = np.arange(len(idx))
            best = s[r, o[:, 0]]
            second = s[r, o[:, 1]] if len(ch) > 1 else np.full(len(idx), -np.inf, np.float32)
            pr = 1.0 / np.exp((s - best[:, None]) / temperature).sum(1)
            ok = (best >= min_score) & ((best - second) >= min_margin)
            chosen = np.asarray(ch, np.int32)[o[:, 0]]
            nxt[idx[ok]] = chosen[ok]
            conf[idx[ok], d] = pr[ok]
            marg[idx[ok], d] = np.minimum(best - second, 1e3)[ok]
            active[idx[~ok]] = False
        cur = nxt
        if not active.any():
            break
    return cur, conf, marg


def lca(guide: Guide, a: int, b: int) -> int:
    pa, pb = [0] + guide.path(a), [0] + guide.path(b)
    out = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        out = x
    return out


# --------------------------------------------------------------------------------------------------------------- self-training
def _standardise(Xl: sps.csr_matrix):
    n = Xl.shape[0]
    mean = np.asarray(Xl.mean(0)).ravel().astype(np.float64)
    sq = Xl.multiply(Xl).sum(0)
    var = np.maximum(np.asarray(sq).ravel() / n - mean**2, 0)
    sd = np.sqrt(var)
    sd[sd == 0] = 1.0
    return mean.astype(np.float32), sd.astype(np.float32), var


def fit_lr(
    X: sps.csr_matrix,
    y: np.ndarray,
    n_classes: int,
    C: float = 1.0,
    max_genes: int = 6000,
    clip: float = 10.0,
    max_iter: int = 100,
):
    """One-vs-rest L2 logistic regression on standardised log1p CP10k (CellTypist's model family), dense L-BFGS.
    Returns a model dict (gene columns, mean, sd, coef, intercept)."""
    from scipy.optimize import minimize
    from scipy.special import expit

    Xl = lognorm(X)
    mean, sd, var = _standardise(Xl)
    keep = np.flatnonzero(np.asarray(Xl.sum(0)).ravel() > 0)
    if max_genes and len(keep) > max_genes:
        keep = np.sort(keep[np.argsort(-var[keep])[:max_genes]])
    Z = Xl[:, keep].toarray()
    Z -= mean[keep]
    Z /= sd[keep]
    np.minimum(Z, clip, out=Z)
    n, g = Z.shape
    K = n_classes
    Y = np.zeros((n, K), np.float32)
    Y[np.arange(n), y] = 1
    lam = 1.0 / (C * n)

    def f(p):
        W = p[: g * K].reshape(g, K)
        b = p[g * K :]
        S = (Z @ W.astype(np.float32) + b.astype(np.float32)).astype(np.float64)
        loss = (np.logaddexp(0, S) - Y * S).sum() / n
        R = ((expit(S) - Y) / n).astype(np.float32)
        gW = (Z.T @ R).astype(np.float64) + lam * W
        return loss + 0.5 * lam * (W * W).sum(), np.concatenate(
            [gW.ravel(), R.sum(0).astype(np.float64)]
        )

    p0 = np.zeros(g * K + K)
    pri = np.clip(np.bincount(y, minlength=K) / n, 1e-6, 1 - 1e-6)
    p0[g * K :] = np.log(pri / (1 - pri))
    r = minimize(
        f,
        p0,
        jac=True,
        method="L-BFGS-B",
        options=dict(
            maxiter=max_iter, maxfun=int(max_iter * 1.5) + 10, gtol=1e-5, ftol=1e-10, maxcor=10
        ),
    )
    return dict(
        cols=keep,
        mean=mean[keep],
        sd=sd[keep],
        coef=r.x[: g * K].reshape(g, K).astype(np.float32),
        intercept=r.x[g * K :].astype(np.float32),
        clip=clip,
    )


def predict_lr(model: dict, X: sps.csr_matrix, block: int = 8192) -> np.ndarray:
    """Cells x classes one-vs-rest probabilities."""
    from scipy.special import expit

    Xl = lognorm(X)[:, model["cols"]]
    out = np.empty((X.shape[0], model["coef"].shape[1]), np.float32)
    for s in range(0, X.shape[0], block):
        Z = Xl[s : s + block].toarray()
        Z -= model["mean"]
        Z /= model["sd"]
        np.minimum(Z, model["clip"], out=Z)
        out[s : s + block] = expit(Z @ model["coef"] + model["intercept"])
    return out


def _majority(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    gu, gi = np.unique(groups, return_inverse=True)
    lu, li = np.unique(labels, return_inverse=True)
    T = np.zeros((len(gu), len(lu)), np.int64)
    np.add.at(T, (gi, li), 1)
    return lu[T.argmax(1)][gi]


# --------------------------------------------------------------------------------------------------------------- result
@dataclass
class CellTypeResult:
    """Per-cell calls. ``node`` indexes ``guide`` (0 = no call). ``score`` is in [0, 1] (NaN where there is no call)."""

    guide: Guide
    node: np.ndarray
    score: np.ndarray
    method: str
    guide_node: np.ndarray
    notes: Dict[str, object] = field(default_factory=dict)

    @property
    def cl(self) -> np.ndarray:
        out = self.guide.cl[self.node].astype(object)
        out[self.node == 0] = None
        return out

    @property
    def label(self) -> np.ndarray:
        onto = ontology()
        names = np.array([onto.name(c) if c else None for c in self.guide.cl], dtype=object)
        out = names[self.node]
        out[self.node == 0] = None
        return out


def annotate_cell_types(
    X,
    gene_names: Sequence[str],
    *,
    species: str = "human",
    tissue: Optional[str] = None,
    disease: Optional[str] = None,
    method: str = "selftrain",
    k: int = 15,
    min_prob: float = 0.5,
    seed: int = 0,
    per_node_cap: int = 2000,
    total_cap: int = 30000,
    min_genes: int = MIN_GENES,
) -> CellTypeResult:
    """Cell type calls for one sample. ``X``: raw counts, cells x genes (exon + intron at gene level is what Singlet serves).
    ``tissue``/``disease``: free text from the sample metadata (restricts the guide to plausible nodes; unknown words are ignored).
    ``method``: ``"guide"`` or ``"selftrain"`` (falls back to the guide call when fewer than two confident nodes exist)."""
    if method not in ("guide", "selftrain"):
        raise ValueError("method must be 'guide' or 'selftrain'")
    t0 = time.time()
    X = sps.csr_matrix(X, dtype=np.float32)
    n = X.shape[0]
    notes: Dict[str, object] = {"n_cells": n}
    g = Guide(gene_names, species=species, tissue=tissue, disease=disease)
    notes["n_guide_nodes"] = len(g.nodes) - 1
    ngenes = np.diff(X.indptr)
    if n == 0:
        z = np.zeros(0, np.int32)
        return CellTypeResult(g, z, np.zeros(0, np.float32), method, z, notes)
    sc = node_scores(g, X)
    t1 = time.time()
    idx = None
    if n > 50:
        _, idx = neighbours(X, k=min(k, n - 1), seed=seed)
        nb = sc[idx].mean(1)
        sc = ((1 - ALPHA) * sc + ALPHA * nb).astype(np.float32)
    t2 = time.time()
    sc = lookahead(g, sc)
    node, conf, marg = gate(g, sc)
    depth = g.depth[node]
    D = conf.shape[1]
    last = np.clip(depth - 1, 0, D - 1)
    gconf = np.where(depth > 0, conf[np.arange(n), last], np.nan).astype(np.float32)
    lowq = ngenes < min_genes
    node[lowq] = 0
    gconf[lowq] = np.nan
    notes.update(
        score_s=t1 - t0,
        neighbours_s=t2 - t1,
        gate_s=time.time() - t2,
        n_low_quality=int(lowq.sum()),
    )
    guide_node = node.copy()
    if method == "guide" or idx is None:
        notes["method_used"] = "guide"
        return CellTypeResult(g, node, gconf, "guide", guide_node, notes)

    # ---- self-training on the confident guide calls
    t3 = time.time()
    margin = np.where(depth > 0, marg[np.arange(n), last], -np.inf)
    rng = np.random.default_rng(seed)
    picks: Dict[int, np.ndarray] = {}
    for j in np.unique(node):
        if j == 0 or g.depth[j] < 2:
            continue
        ii = np.flatnonzero(node == j)
        ii = ii[margin[ii] >= np.median(margin[ii])]
        if len(ii) >= 5:
            picks[int(j)] = ii
    if len(picks) < 2:
        notes["method_used"] = "guide"
        notes["selftrain_skipped"] = "fewer than two confident nodes"
        return CellTypeResult(g, node, gconf, "guide", guide_node, notes)
    cap = per_node_cap
    while sum(min(len(v), cap) for v in picks.values()) > total_cap and cap > 50:
        cap = int(cap * 0.8)
    sel = np.concatenate(
        [rng.choice(v, cap, replace=False) if len(v) > cap else v for v in picks.values()]
    )
    classes = np.array(sorted(picks), np.int32)
    cpos = {c: i for i, c in enumerate(classes)}
    y = np.array([cpos[int(node[i])] for i in sel], np.int64)
    model = fit_lr(X[sel], y, len(classes))
    P = predict_lr(model, X)
    lab = classes[P.argmax(1)]
    groups = overcluster(idx, 5.0, seed)
    if groups is None:  # no leidenalg: vote over each cell's kNN neighbourhood instead
        votes = lab[idx]
        lab = np.array([np.bincount(v).argmax() for v in votes], np.int32)
        notes["vote"] = "knn"
    else:
        lab = _majority(lab, groups).astype(np.int32)
        notes["vote"] = "leiden_r5"
    prob = P[np.arange(n), np.searchsorted(classes, lab)]
    final = lab.copy()
    sc_out = prob.astype(np.float32)
    weak = prob < min_prob
    final[weak] = guide_node[weak]
    sc_out[weak] = gconf[weak]
    # disagreement with the guide's path: back off to the deepest common ancestor
    for i in np.flatnonzero(~weak & (guide_node > 0)):
        a, b = int(final[i]), int(guide_node[i])
        if a != b and a not in g.path(b) and b not in g.path(a):
            final[i] = lca(g, a, b)
    final[lowq] = 0
    sc_out[final == 0] = np.nan
    notes.update(
        method_used="selftrain",
        selftrain_s=time.time() - t3,
        n_train_cells=int(len(sel)),
        n_train_nodes=len(picks),
        frac_weak=float(weak.mean()),
    )
    return CellTypeResult(g, final.astype(np.int32), sc_out, "selftrain", guide_node, notes)


def at_depth_labels(res: CellTypeResult, depth: int) -> List[Optional[str]]:
    """CL id of each call cut to guide depth ``depth`` (None where there is no call)."""
    anc = res.guide.ancestor_at_depth(res.node, depth)
    return [res.guide.cl[j] if j else None for j in anc]
