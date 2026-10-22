#!/usr/bin/env python
"""E1 validation on synthetic cells with KNOWN state information (null / leakage test and a calibration test).

    python -P tools/e1_synth.py  -> prints pooled ladder numbers and writes results/e1_synth.tsv

Scenario "null": every cell is Multinomial(N_c, p) from ONE profile p  -> state information is exactly 0; every gain
  over L1 must be <= ~0 (any positive gain would mean leakage of B into the predictor).
Scenario "types": K cell types with profiles p_k = normalise(p * exp(sigma z_k)); cells are Multinomial(N_c, p_k).
  The oracle gain (cost of B under p_type versus under the population mixture) is computed exactly and compared with
  what L2/L3 recover from the A half.
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1_split as E


def make(F, n, K, sigma, med_N, seed):
    rng = np.random.default_rng(seed)
    base = np.exp(rng.normal(0, 2.0, F))
    base /= base.sum()
    if K == 0:
        prof = base[None, :]
        typ = np.zeros(n, dtype=int)
    else:
        z = rng.normal(0, 1, (K, F))
        prof = base[None, :] * np.exp(sigma * z)
        prof /= prof.sum(1, keepdims=True)
        typ = rng.integers(0, K, n)
    N = np.maximum(50, np.round(np.exp(rng.normal(np.log(med_N), 0.5, n)))).astype(int)
    ptr = [0]
    ind, val = [], []
    for c in range(n):
        x = rng.multinomial(N[c], prof[typ[c]])
        nz = np.flatnonzero(x)
        ind.append(nz)
        val.append(x[nz])
        ptr.append(ptr[-1] + len(nz))
    return (np.array(ptr, dtype=np.int64), np.concatenate(ind).astype(np.int64), np.concatenate(val).astype(np.int64),
            prof, typ, N)


def run(name, F, n, K, sigma, med_N, cap, rows):
    t0 = time.time()
    ptr, ind, val, prof, typ, N = make(F, n, K, sigma, med_N, 7)
    log = lambda m: None
    cols = np.arange(n)
    S = E.Subset("cells", ptr, ind, val, F, cols, E.N_MIN_CELL, cap, 11, log, True)
    S.tune()
    S.neighbours()
    S.score()
    o = S.out
    v = ~np.isnan(o["N"])
    UB, nB = o["NB"][v].sum(), o["nnzB"][v].sum()
    # oracle: cost of B under the cell's true profile vs under the population mixture (both multinomial, same B)
    mix = np.bincount(typ, minlength=max(K, 1)) / n @ prof
    gain_or = 0.0
    uniq = np.flatnonzero(np.bincount(ind, minlength=F) > 0)
    for ii, p in enumerate(S.sidx):
        s, e = S.pptr[p], S.pptr[p + 1]
        fi = S.pidx[s:e]
        # map compact -> original feature id: compact index = rank among features present in pop
        orig = uniq[fi]
        b = S.pval[s:e] - S.Afull[s:e]
        gain_or += (b * (np.log2(prof[typ[p]][orig]) - np.log2(mix[orig]))).sum()
    ms = lambda a: np.nansum(a[v])
    L1 = ms(o["ms_L1"])
    def best(a):  # pooled best candidate (optimistic; just for the check)
        return np.nanmin(np.nansum(a[v].reshape(v.sum(), -1), 0))
    res = dict(name=name, cells=int(v.sum()), L0=ms(o["ms_L0"]) / nB, L1=L1 / nB,
               gain_L2=(L1 - best(o["ms_L2"])) / nB, gain_L3seq=(L1 - np.nanmin(np.nansum(o["ms_L3"][v][:, :, 0], 0))) / nB,
               gain_L3=(L1 - best(o["ms_L3"])) / nB, gain_L3dm=(L1 - best(o["ms_L3dm"])) / nB,
               oracle_gain=gain_or / nB)
    print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in res.items()}, f"{time.time() - t0:.0f}s", flush=True)
    rows.append(res)


if __name__ == "__main__":
    rows = []
    run("null_F20k_N700", 20000, 3000, 0, 0.0, 700, 1500, rows)
    run("types4_s1.0_N700", 20000, 3000, 4, 1.0, 700, 1500, rows)
    run("types4_s0.5_N700", 20000, 3000, 4, 0.5, 700, 1500, rows)
    run("types16_s0.5_N700", 20000, 3000, 16, 0.5, 700, 1500, rows)
    import csv
    with open(os.path.join(E.RES, "e1_synth.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
