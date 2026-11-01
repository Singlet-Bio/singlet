# SPDX-License-Identifier: MIT
"""Prototype of ``called_ed``: an EmptyDrops cell call computed from the counts a Singlet sample already stores.

Why: the stored ``called`` flag is STARsolo's default ``CellRanger2.2 3000 0.99 10`` (one UMI threshold), which misses low-RNA
cells (docs/empty-droplets.md). ``called_ed`` keeps every ``called`` barcode and adds the uncalled barcodes whose expression
profile rejects "ambient RNA only" (EmptyDrops, Lun et al. 2019) -- the same construction as Cell Ranger 3 (OrdMag cells plus
EmptyDrops on the rest), but with the ambient pool and the tested range set from the stored data. See docs/called-ed.md.

Inputs are only what a store sample object holds: the exon and intron matrices of every stored barcode (aggregated to genes) and
the stored ``called`` bitmap. The call is deterministic (fixed seed, fixed number of Monte Carlo draws), so it can be recomputed
and checked byte for byte.

Usage as a library::

    res = call_ed(exon, intron, called, protocol="10x")   # genes x barcodes CSC matrices, bool vector
    res["called_ed"], res["fdr"], res["params"]

or on a fetched sample (bench/empty_droplets/fetch.py npz): ``python -P call_ed.py SAMPLE.npz [--protocol 10x]``.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "empty_droplets"))
import callers as C  # noqa: E402

METHOD = "singlet-emptydrops-v2.0"

# Per-protocol defaults (decided in docs/called-ed.md). ``lower``: ambient pool = barcodes with 0 < total <= lower (an int, or
# "plateau" = derived from the barcode-rank curve, see ambient_lower). ``test_min``: barcodes are tested only above this total.
# ``fdr``: BH threshold. ``retain``: barcodes at or above the knee are called without a test (DropletUtils default) -- "knee" or
# None. ``keep_called``: the stored ``called`` set is always part of ``called_ed`` (Cell Ranger 3 semantics).
DEFAULTS = {"matrix": "sum", "lower": "plateau", "test_min": 50, "fdr": 0.001, "retain": "knee", "keep_called": True,
            "niters": 10000, "seed": 20261023}
PROTOCOLS = {
    # droplet protocols with a whitelist and an ambient plateau among the stored barcodes
    "10x": {},
    "10x_nuclei": {},
    "dropseq": {},
    "indrop": {},
    "seqwell": {},          # wells, not droplets, but a free-RNA background exists and the model applies
    "bd_rhapsody": {},
    # combinatorial indexing: no droplets and no ambient pool in the stored barcodes; keep the stored call
    "parse": {"method": "none"},
    "splitseq": {"method": "none"},
}
PROTOCOL_ALIASES = {
    "10x_v2": "10x", "10x_v3": "10x", "10x_v3.1": "10x", "10x_5p": "10x", "10x_multiome": "10x_nuclei", "chromium": "10x",
    "drop-seq": "dropseq", "dropseq": "dropseq", "indrop": "indrop", "seq-well": "seqwell", "seqwell": "seqwell",
    "bd": "bd_rhapsody", "rhapsody": "bd_rhapsody", "parse": "parse", "split-seq": "splitseq", "splitseq": "splitseq",
}


def protocol_family(protocol: str | None) -> str:
    p = (protocol or "").strip().lower().replace(" ", "_")
    if p in PROTOCOLS:
        return p
    for k, v in PROTOCOL_ALIASES.items():
        if k in p:
            return v
    return "10x"


def gene_matrix(exon: sp.csc_matrix, intron: sp.csc_matrix, how: str) -> sp.csc_matrix:
    """Counts the call runs on. ``sum`` = exon + intron (what ``n_umi_exon + n_umi_intron`` counts, double-counted molecules
    included), ``exon`` = exon only, ``max`` = per gene max(exon, intron) (a lower bound on distinct molecules)."""
    if how == "exon":
        X = exon.tocsc()
    elif how == "sum":
        X = (exon + intron).tocsc()
    elif how == "max":
        X = exon.maximum(intron).tocsc()
    else:
        raise ValueError(how)
    X.data = X.data.astype(np.int32)
    X.eliminate_zeros()
    return X


def ambient_lower(tot: np.ndarray, called: np.ndarray, rule) -> int:
    """Upper total of the ambient pool.

    * int: that value.
    * ``"adaptive"``: clip(0.5 x 5th percentile of the called totals, 5, 100) (the empty-droplet study's rule).
    * ``"plateau"``: 3 x the mode of log10(total) among barcodes below half the called 5th percentile, clipped to [5, 100]:
      the ambient plateau of the barcode-rank curve, so that low-RNA cells (which sit between the plateau and the called cells)
      stay out of the ambient pool.
    """
    if isinstance(rule, (int, np.integer)):
        return int(rule)
    q05 = float(np.quantile(tot[called], 0.05)) if called.any() else 200.0
    if rule == "adaptive":
        return int(np.clip(0.5 * q05, 5, 100))
    if rule == "plateau":
        t = tot[(tot > 0) & (tot < max(4.0, 0.5 * q05))]
        if len(t) < 100:
            return int(np.clip(0.5 * q05, 5, 100))
        lt = np.log10(t)
        h, e = np.histogram(lt, bins=np.arange(0, lt.max() + 0.1, 0.1))
        h = np.convolve(h, np.ones(3) / 3, mode="same")
        mode = 10 ** (0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1]))
        return int(np.clip(round(3 * mode), 5, 100))
    raise ValueError(rule)


def emptydrops_pvalues(X: sp.csc_matrix, ambient: np.ndarray, test: np.ndarray, niters: int, seed: int) -> dict:
    """Monte Carlo p-values of every tested barcode against the Dirichlet-multinomial ambient model (no retain, no BH)."""
    ed = C.emptydrops(X, ambient_mask=ambient, test_mask=test, niters=niters, seed=seed)
    return ed


def finish(tot, called, pvalue, tested, fdr, retain_total=None, keep_called=True, test_min=0):
    """Apply test floor, retain and BH to raw p-values -> (called_ed, fdr vector over all barcodes)."""
    t = tested & (tot > test_min) & np.isfinite(pvalue)
    p = pvalue.copy()
    if retain_total is not None:
        p[t & (tot >= retain_total)] = 0.0
    q = np.full(len(tot), np.nan)
    q[t] = C.bh(p[t])
    cv2 = np.nan_to_num(q, nan=1.0) <= fdr
    if keep_called:
        cv2 |= called
    return cv2, q


def call_ed(exon: sp.csc_matrix, intron: sp.csc_matrix, called: np.ndarray, protocol: str | None = None, **over) -> dict:
    """``called_ed`` for one sample. Returns ``called_ed`` (bool per barcode), ``fdr`` (float per barcode, NaN where not
    tested), ``params`` (everything needed to reproduce the call, incl. the fitted alpha and the ambient pool size)."""
    fam = protocol_family(protocol)
    P = {**DEFAULTS, **PROTOCOLS.get(fam, {}), **over}
    called = np.asarray(called, bool)
    n = exon.shape[1]
    params = {"method": METHOD, "protocol_family": fam, **{k: P[k] for k in DEFAULTS}}
    if P.get("method") == "none":
        params["method"] = "stored-call"
        return {"called_ed": called.copy(), "fdr": np.full(n, np.nan), "params": params}
    X = gene_matrix(exon, intron, P["matrix"])
    tot = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
    lower = ambient_lower(tot, called, P["lower"])
    ambient = (tot > 0) & (tot <= lower) & ~called
    test = (tot > max(lower, int(P["test_min"]))) & ~called
    retain = None
    if P["retain"] == "knee":
        knee, _ = C.barcode_ranks(tot, lower=lower)
        retain = float(knee) if np.isfinite(knee) and knee > lower else None
    params.update({"lower_value": lower, "retain_value": retain, "n_ambient": int(ambient.sum()),
                   "ambient_umi": int(tot[ambient].sum()), "n_tested": int(test.sum())})
    ed = emptydrops_pvalues(X, ambient, test, int(P["niters"]), int(P["seed"]))
    if "error" in ed:
        params["error"] = ed["error"]
        return {"called_ed": called.copy(), "fdr": np.full(n, np.nan), "params": params}
    params["alpha"] = float(ed["alpha"])
    cv2, q = finish(tot, called, ed["pvalue"], test, P["fdr"], retain, P["keep_called"], 0)
    params["n_called_ed"] = int(cv2.sum())
    params["n_added"] = int((cv2 & ~called).sum())
    return {"called_ed": cv2, "fdr": q, "params": params, "total": tot}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="sample npz from bench/empty_droplets/fetch.py")
    ap.add_argument("--protocol", default=None, help="protocol (default: from the sample's metadata)")
    a = ap.parse_args(argv)
    z = np.load(a.npz, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    shape = (len(z["genes"]), len(z["barcodes"]))
    E = sp.csc_matrix((z["exon_data"], z["exon_indices"], z["exon_indptr"]), shape=shape)
    I = sp.csc_matrix((z["intron_data"], z["intron_indices"], z["intron_indptr"]), shape=shape)
    prot = a.protocol or (meta.get("gsm_meta") or {}).get("protocol") or (meta.get("summary") or {}).get("protocol_id")
    res = call_ed(E, I, z["called"].astype(bool), prot)
    print(json.dumps(res["params"], indent=1))


if __name__ == "__main__":
    sys.exit(main())
