#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Scale check of the doublet stage on one large real sample read from data.singlet.bio (.s1): wall, CPU seconds and peak RSS per
stage, and the same on random subsamples to see how time and memory grow with cells.  Needs the singlet readers (pip package or an
in-place build) next to the repo's python/ on PYTHONPATH.   python scripts/doublet_scale.py GSE GSM --out OUT [--engine-tsv F]"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def hwm_reset():
    """Reset the kernel's peak-RSS counter so the next peak is the stage's own (Linux; False if not permitted)."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def hwm_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024.0
    return rss()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gse")
    ap.add_argument("gsm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="")
    a = ap.parse_args()
    from singlet.celllabels import io as A

    t = time.perf_counter()
    s = (getattr(A, "read_s1", None) or A.read_v2)(a.gse, a.gsm)
    print("read", s.X.shape, s.species, f"{time.perf_counter() - t:.0f}s", "rss", round(rss()), "MB", flush=True)
    # the readers may come from another build of singlet: put this checkout's features package in front of theirs
    import singlet.features as F0

    F0.__path__.insert(0, str(ROOT / "python" / "singlet" / "features"))
    for m in [m for m in sys.modules if m.startswith("singlet.features.")]:
        del sys.modules[m]
    from singlet.features import doublet as D

    X = s.X.tocsr()
    ids = list(s.gene_ids) or list(s.gene_names)
    # the engine's doublet score is not part of the stored sample here: use the sample's own per-cell column when the cells table has it
    prod = None
    try:
        try:
            from singlet.s1 import client as C
        except ImportError:
            from singlet.v2 import client as C

        obs = C.cells(a.gse, where="called", samples=[a.gsm]).obs
        for c in ("doublet_score", "dbl_doublet_score"):
            if c in obs.columns:
                prod = obs[c].to_numpy(np.float64)
                break
    except Exception as e:  # noqa: BLE001
        print("no engine score in the cells table:", e)
    standin = prod is None
    if standin:  # cost does not depend on the values: log library size stands in (reported)
        prod = np.log1p(np.asarray(X.sum(1)).ravel())
    D.stacker(), D.atlas()
    rep = []
    sizes = [int(x) for x in a.sizes.replace("+", ",").split(",") if x] + [X.shape[0]]
    rng = np.random.default_rng(0)
    for n in sizes:
        idx = np.sort(rng.choice(X.shape[0], n, replace=False)) if n < X.shape[0] else np.arange(X.shape[0])
        ok = hwm_reset()
        base = hwm_mb()
        w0, c0 = time.perf_counter(), time.process_time()
        tim = {}
        F = D.compute_inputs(X[idx], ids, s.species, prod[idx], timings=tim)
        t1 = time.perf_counter()
        sc = D.score_inputs(F)
        tim["stacker_s"] = time.perf_counter() - t1
        rep.append(dict(n_cells=int(n), nnz=int(X[idx].nnz), wall_s=round(time.perf_counter() - w0, 2), cpu_s=round(time.process_time() - c0, 2),
                        stage_wall_s={k: round(v, 2) for k, v in tim.items()}, rss_before_mb=round(base), stage_peak_rss_mb=round(hwm_mb()), peak_reset=ok, prod_is_standin=standin, finite=float(np.isfinite(sc).mean())))
        print(rep[-1], flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"scale_{a.gsm}.json").write_text(json.dumps(dict(gse=a.gse, gsm=a.gsm, species=s.species, runs=rep), indent=1))


if __name__ == "__main__":
    main()
