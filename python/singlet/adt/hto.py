# SPDX-License-Identifier: MIT
"""Hashtag demultiplexing calls (per called cell): singlet with its hashtag, doublet, or negative.

Method: ``DropletUtils::hashedDrops`` (Lun et al.; Bioconductor), the ambient-aware rule:

1. Ambient proportions per hashtag from the empty droplets; each cell's counts are divided by them, so a hashtag that
   is only ambient in the cell sits at the cell's ambient level (the median of the hashtags below the top two).
2. Log2 fold change between the largest and second-largest scaled count (``lfc_top``) and between the second-largest
   and the ambient level (``lfc_second``), with pseudocount 5.
3. Doublet if ``lfc_second`` is an upper outlier (> median + ``nmads`` MAD among cells) and above ``doublet_min``;
   confident singlet if ``lfc_top >= confident_min`` and it is not a lower outlier; otherwise negative (unassigned).

Output per cell: ``call`` (hashtag index, -1 negative, -2 doublet), ``lfc_top``, ``lfc_second``.
"""

from __future__ import annotations

import numpy as np

CONFIDENT_MIN = 2.0  # hashedDrops default (log2 fold)
DOUBLET_MIN = 2.0
DOUBLET_NMADS = 3.0
CONFIDENT_NMADS = 3.0
PSEUDO = 5.0


def _mad(x):
    m = np.median(x)
    return m, 1.4826 * np.median(np.abs(x - m))


def ambient_bimodal(cells: np.ndarray) -> np.ndarray:
    """Ambient profile from called cells alone (no empty droplets), as ``DropletUtils::ambientProfileBimodal``: each hashtag's
    log counts split into two groups (1-D k-means, k = 2); the ambient level is the mean count of the lower group."""
    X = np.asarray(cells, dtype=float)
    out = np.zeros(X.shape[0])
    for i, row in enumerate(X):
        v = np.log1p(row)
        lo, hi = v.min(), v.max()
        if hi - lo < 1e-9:
            out[i] = row.mean()
            continue
        c = np.array([lo, hi])
        for _ in range(100):
            lab = np.abs(v[:, None] - c[None, :]).argmin(axis=1)
            new = np.array([v[lab == j].mean() if (lab == j).any() else c[j] for j in (0, 1)])
            if np.allclose(new, c):
                break
            c = new
        low = lab == int(np.argmin(c))
        out[i] = row[low].mean() if low.any() else row.mean()
    return out


def hashed_drops(cells: np.ndarray, ambient: np.ndarray = None, confident_min: float = CONFIDENT_MIN):
    """cells: hashtags x cells counts; ambient: per-hashtag counts in empty droplets (None: estimated from the cells with
    :func:`ambient_bimodal`). ``confident_min``: the log2 fold change for a confident singlet (DropletUtils default 2)."""
    X = np.asarray(cells, dtype=float)
    k, n = X.shape
    if ambient is None:
        ambient = ambient_bimodal(X)
    amb = np.asarray(ambient, dtype=float) + 1e-9
    amb = amb / amb.sum()
    # counts relative to the ambient proportions (an ambient-only hashtag then sits at the cell's ambient level)
    adj = X / (amb[:, None] * k)
    o = np.argsort(-adj, axis=0, kind="stable")
    srt = np.take_along_axis(adj, o, axis=0)
    top = srt[0]
    sec = srt[1] if k > 1 else np.zeros(n)
    # the cell's ambient level: the hashtags below the top two (a doublet's second hashtag must not set it)
    amb_level = np.median(srt[2:], axis=0) if k > 2 else np.zeros(n)
    lfc_top = np.log2((top + PSEUDO) / (sec + PSEUDO))
    lfc_second = np.log2((sec + PSEUDO) / (amb_level + PSEUDO))
    call = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return call, lfc_top, lfc_second
    md, mad = _mad(lfc_second)
    doublet = (lfc_second > md + DOUBLET_NMADS * mad) & (lfc_second >= DOUBLET_MIN)
    mt, madt = _mad(lfc_top[~doublet]) if (~doublet).any() else (0.0, 0.0)
    confident = (~doublet) & (lfc_top >= confident_min) & (lfc_top >= mt - CONFIDENT_NMADS * madt)
    call[confident] = o[0][confident]
    call[doublet] = -2
    return call, lfc_top, lfc_second


SIDECAR_SCHEMA = "1"
FEATURE_SET = "hto"
METHOD_VERSION = "singlet-adt-hto/1 (hashedDrops rule; confident 2, doublet 2 + 3 MAD)"


def feature_sidecar(gsm: str, cols, call, lfc_top, hashtag_names):
    """Per-cell hashtag calls as a feature sidecar (``docs/CELL-ARCHITECTURE.md`` section 3), called cells only.

    Columns: ``gsm``, ``col`` (int64), ``hto/call`` (uint16 category: 0 negative, 1 doublet, 2 + k hashtag k of
    ``hashtag_names``), ``hto/margin`` (float32 in [0, 1]: ``1 - 2**-lfc_top`` = (top - second) / top on the
    ambient-scaled counts; 0 for negatives). Returns (pyarrow table, category names)."""
    import pyarrow as pa

    call = np.asarray(call)
    cat = np.where(call == -1, 0, np.where(call == -2, 1, call + 2)).astype(np.uint16)
    margin = np.clip(1 - np.power(2.0, -np.asarray(lfc_top, dtype=float)), 0, 1).astype(np.float32)
    margin[call == -1] = 0
    names = ["negative", "doublet"] + list(hashtag_names)
    t = pa.table({"gsm": pa.array([gsm] * len(cat), pa.string()), "col": pa.array(np.asarray(cols, np.int64)),
                  "hto/call": pa.array(cat), "hto/margin": pa.array(margin)})
    md = {b"singlet_features_schema": SIDECAR_SCHEMA.encode(), b"feature_set": FEATURE_SET.encode(),
          b"method_version": METHOD_VERSION.encode(), b"categories": "|".join(names).encode()}
    return t.replace_schema_metadata(md), names


# ----------------------------------------------------------------------------------------------------------- mixture
MIX_POSITIVE = 0.5  # posterior for a hashtag to be positive in a cell (fixed a priori, not tuned on benchmarks)
MIX_ITERS = 200


def _mixture_one(y, z):
    """Two-component mixture of linear regressions of y on z (EM). Returns the posterior of the higher component."""
    n = len(y)
    Z = np.column_stack([np.ones(n), z]) if np.std(z) > 0 else np.ones((n, 1))
    # init: split y at its 1-D 2-means boundary
    c = np.quantile(y, [0.25, 0.95])
    for _ in range(50):
        lab = np.abs(y[:, None] - c[None, :]).argmin(axis=1)
        new = np.array([y[lab == k].mean() if (lab == k).any() else c[k] for k in (0, 1)])
        if np.allclose(new, c):
            break
        c = new
    r = (lab == int(np.argmax(c))).astype(float)
    r = np.clip(r, 0.02, 0.98)
    ll_old = -np.inf
    for _ in range(MIX_ITERS):
        params = []
        for w in (1 - r, r):
            W = w + 1e-9
            beta = np.linalg.lstsq(Z * np.sqrt(W)[:, None], y * np.sqrt(W), rcond=None)[0]
            res = y - Z @ beta
            var = max(float((W * res ** 2).sum() / W.sum()), 1e-4)
            params.append((beta, var, float(w.mean())))
        dens = []
        for beta, var, pi in params:
            mu = Z @ beta
            dens.append(np.log(max(pi, 1e-9)) - 0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mu) ** 2 / var)
        dens = np.vstack(dens)
        m = dens.max(axis=0)
        ll = float((m + np.log(np.exp(dens - m).sum(axis=0))).sum())
        r = np.exp(dens[1] - m) / np.exp(dens - m).sum(axis=0)
        if abs(ll - ll_old) < 1e-6 * max(1.0, abs(ll)):
            break
        ll_old = ll
    # the positive component is the one with the higher mean at the median covariate
    zrow = np.array([1.0, np.median(z)])[:Z.shape[1]]
    mu0 = float(params[0][0] @ zrow)
    mu1 = float(params[1][0] @ zrow)
    return r if mu1 >= mu0 else 1 - r


def mixture_calls(cells: np.ndarray, positive: float = MIX_POSITIVE, covariate: str = "others",
                  scale_ambient: bool = False):
    """Hashtag calls from per-hashtag mixtures (cells only, no empty droplets needed).

    For each hashtag, log1p counts are modelled as a two-component mixture of regressions on the cell's log1p count of all
    the other hashtags (a proxy for the cell's nonspecific binding, as demuxmix uses detected genes). A hashtag is positive
    in a cell when its posterior >= ``positive``. One positive hashtag: singlet; two or more: doublet; none: negative.
    ``covariate``: ``others`` (default), ``total`` (log1p of all hashtags) or ``none``; ``scale_ambient`` divides each hashtag
    by its ambient level (:func:`ambient_bimodal`) first.
    Returns (call: hashtag index / -1 negative / -2 doublet, score: posterior of the call, posteriors hashtags x cells)."""
    X = np.asarray(cells, dtype=float)
    k, n = X.shape
    if scale_ambient:
        X = X / np.maximum(ambient_bimodal(X), 1e-3)[:, None]
    tot = X.sum(axis=0)
    post = np.zeros_like(X)
    for j in range(k):
        y = np.log1p(X[j])
        if covariate == "others":
            z = np.log1p(tot - X[j])
        elif covariate == "total":
            z = np.log1p(tot)
        else:
            z = np.zeros(n)
        post[j] = _mixture_one(y, z) if n >= 20 and y.std() > 0 else 0.0
    pos = post >= positive
    npos = pos.sum(axis=0)
    call = np.full(n, -1, dtype=np.int64)
    one = npos == 1
    call[one] = post[:, one].argmax(axis=0)
    call[npos >= 2] = -2
    srt = np.sort(post, axis=0)
    score = np.where(one, srt[-1] * (1 - srt[-2]), np.where(npos >= 2, srt[-2], 1 - srt[-1]))
    return call, score, post
