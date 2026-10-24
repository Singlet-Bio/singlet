#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Refit a trained network's temperature so that the call's probability is reliable: among calls made with probability p, a
fraction p has the truth inside the call's subtree. Fitted on the validation collections of the corpus (clean and augmented
counts, equally weighted), never on the benchmark.

``train.py`` fits the temperature on the closure likelihood. That likelihood is dominated by cells whose author label conflicts
with the labels of other studies (another branch, probability near 0), which pushes the temperature up and leaves the 0.5 roll-up
call underconfident and too coarse. This script picks the temperature from a grid by the expected calibration error (10 bins)
of the call itself, the quantity the output promises. Calls finer than the author label cannot be checked and are left out, as
in ``bench.py``.

  python scripts/celltype_nn/calibrate.py --data $D --model $D/models/m1   # rewrites model.pt's temperature, writes calibrate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
sys.path.insert(0, str(HERE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True, help="model directory")
    ap.add_argument("--grid", default="0.3,3.0,55")
    a = ap.parse_args(argv)

    import pandas as pd
    import torch

    from singlet.celltype_nn import model as M
    from singlet.celltype_nn.features import AugmentConfig, Augmenter, transform
    from singlet.celltype_nn.ontology import TermSpace

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, meta = M.load(a.model / "model.pt", map_location="cpu")
    net = net.to(dev)
    sp = TermSpace(meta["terms"])
    X = sps.load_npz(a.data / "corpus_counts.npz").tocsr()
    obs = pd.read_parquet(a.data / "corpus_obs.parquet", columns=["split"])
    cells = pd.read_parquet(a.data / "space_cells.parquet", columns=["author_term"])
    au = cells["author_term"].to_numpy(object)
    va = np.flatnonzero((obs["split"].to_numpy() == "val") & pd.notna(au))
    targets = sorted(set(au[va].tolist()))
    tix = {t: i for i, t in enumerate(targets)}
    # outcome of calling closure term j for target t: 1 truth inside, 0 another branch, NaN finer than the truth (unknown)
    out = np.zeros((len(targets), len(sp.closure)), np.float32)
    for i, t in enumerate(targets):
        at = sp.ancestors(t)
        for j, c in enumerate(sp.closure):
            out[i, j] = 1.0 if c in at else (np.nan if t in sp.ancestors(c) else 0.0)
    out = torch.from_numpy(out).to(dev)
    ti = torch.tensor([tix[t] for t in au[va]], device=dev)
    ir = np.load(a.data / "intron_ratio.npy") if (a.data / "intron_ratio.npy").exists() else None
    aug = Augmenter(X.shape[1], ir, AugmentConfig(), dev, seed=1)
    Rup = torch.from_numpy(sp.R).to(dev)
    key_c = torch.from_numpy(sp.ic_closure + 1e-6 * sp.depth_closure).to(dev)
    net.eval()
    logits = {"clean": [], "aug": []}
    with torch.no_grad():
        for s in range(0, len(va), 8192):
            c = torch.from_numpy(X[va[s : s + 8192]].toarray().astype(np.float32)).to(dev)
            logits["clean"].append(net(transform(c, net.transform)))
            logits["aug"].append(net(transform(aug(c), net.transform)))
    logits = {k: torch.cat(v) for k, v in logits.items()}

    def evaluate(T: float, L):
        P = torch.softmax(L / T, 1) @ Rup
        k = torch.where(P >= 0.5, key_c[None, :] + 1e-3 * P, torch.full_like(P, -1e9)).argmax(1)
        p = P.gather(1, k[:, None]).ravel()
        y = out[ti].gather(1, k[:, None]).ravel()
        m = ~torch.isnan(y)
        p, y = p[m], y[m]
        ece = 0.0
        for lo in np.linspace(0.5, 0.95, 10):
            s = (p >= lo) & (p < lo + 0.05 + (1e-6 if lo >= 0.95 else 0))
            if s.any():
                ece += float(s.float().mean()) * abs(float(p[s].mean()) - float(y[s].mean()))
        depth = float(torch.from_numpy(sp.depth_closure).to(dev)[k].mean())
        return dict(T=T, ece=ece, checked=float(m.float().mean()), accuracy=float(y.mean()), mean_prob=float(p.mean()), call_depth=depth)

    lo, hi, n = a.grid.split(",")
    grid = np.exp(np.linspace(np.log(float(lo)), np.log(float(hi)), int(n)))
    rows = []
    for T in grid:
        r = {k: evaluate(float(T), L) for k, L in logits.items()}
        rows.append(dict(T=float(T), ece=(r["clean"]["ece"] + r["aug"]["ece"]) / 2, clean=r["clean"], aug=r["aug"]))
    best = min(rows, key=lambda r: r["ece"])
    old = float(net.temperature)
    net.temperature.fill_(best["T"])
    meta_out = {k: v for k, v in meta.items() if k not in ("state", "config")}
    meta_out["calibration"] = dict(method="call reliability (ECE, 10 bins) on validation collections, clean + augmented",
                                   temperature=best["T"], closure_nll_temperature=old)
    keep = a.model / "model_nll.pt"  # the closure-likelihood temperature as trained
    if not keep.exists():
        import shutil

        shutil.copy2(a.model / "model.pt", keep)
    M.save(net.cpu(), a.model / "model.pt", meta_out)
    (a.model / "calibrate.json").write_text(json.dumps(dict(best=best, closure_nll_temperature=old, grid=rows), indent=1))
    print(json.dumps(dict(best=best, closure_nll_temperature=old), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
