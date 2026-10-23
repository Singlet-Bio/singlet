# SPDX-License-Identifier: MIT
"""CellBender I/O for the empty-droplet study.

``export``: a fetch.py npz -> 10x-style MatrixMarket directory (matrix.mtx.gz genes x barcodes of exon+intron counts,
features.tsv.gz, barcodes.tsv.gz) that ``cellbender remove-background --input`` reads.
``collect``: CellBender's output h5 -> ``<GSM>.cellbender.json`` with the cell probabilities compared with Singlet's call.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, q  # noqa: E402


def export(npz, out_dir):
    d = load(npz)
    X = (d["exon"] + d["intron"]).tocsc()
    os.makedirs(out_dir, exist_ok=True)
    with gzip.open(os.path.join(out_dir, "matrix.mtx.gz"), "wb") as fh:
        scipy.io.mmwrite(fh, X.tocoo().astype(np.int32))
    with gzip.open(os.path.join(out_dir, "features.tsv.gz"), "wt") as fh:
        for g in d["genes"]:
            fh.write(f"{g}\t{g}\tGene Expression\n")
    with gzip.open(os.path.join(out_dir, "barcodes.tsv.gz"), "wt") as fh:
        for b in d["barcodes"]:
            fh.write(f"{b}\n")
    tot = np.asarray(X.sum(axis=0)).ravel()
    print(json.dumps({"n_barcodes": int(X.shape[1]), "n_called": int(d["called"].sum()), "min_total": int(tot.min())}))


def collect(npz, h5, out_json, analysis_npz=None):
    import h5py

    d = load(npz)
    called = d["called"]
    X = (d["exon"] + d["intron"]).tocsc()
    tot = np.asarray(X.sum(axis=0)).ravel()
    with h5py.File(h5, "r") as f:
        grp = f["droplet_latent_probabilities"]
        idx = grp["barcode_indices_for_latents"][()]
        prob = grp["cell_probability"][()]
        all_bcs = [b.decode() if isinstance(b, bytes) else str(b) for b in f["matrix/barcodes"][()]]
    p = np.zeros(len(all_bcs))
    p[idx] = prob
    pos = {b: i for i, b in enumerate(all_bcs)}
    cp = np.array([p[pos[b]] if b in pos else np.nan for b in d["barcodes"]])
    cb = cp > 0.5
    unc = ~called
    R = {"gsm": d["meta"]["gsm"], "n_cellbender": int(cb.sum()), "n_ours": int(called.sum()),
         "theirs_not_ours": int((cb & unc).sum()), "ours_not_theirs": int((called & ~cb).sum()),
         "theirs_not_ours_umi": q(tot[cb & unc]), "ours_not_theirs_umi": q(tot[called & ~cb]),
         "jaccard": float((cb & called).sum() / max(1, (cb | called).sum())),
         "n_scored": int(np.isfinite(cp).sum())}
    if analysis_npz and os.path.exists(analysis_npz):
        z = np.load(analysis_npz)
        if "fdr_cell" in z.files:
            sig = unc & (np.nan_to_num(z["fdr_cell"], nan=1) < 0.01)
            R["theirs_not_ours_profile_sig"] = int((cb & sig).sum())
            R["profile_sig_called_by_cellbender"] = float(cb[sig].mean()) if sig.any() else None
    with open(out_json, "w") as fh:
        json.dump(R, fh, indent=1)
    print(json.dumps(R))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("npz")
    e.add_argument("out_dir")
    c = sub.add_parser("collect")
    c.add_argument("npz")
    c.add_argument("h5")
    c.add_argument("out_json")
    c.add_argument("--analysis-npz")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.npz, a.out_dir)
    else:
        collect(a.npz, a.h5, a.out_json, a.analysis_npz)
