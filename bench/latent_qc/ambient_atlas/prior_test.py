"""Atlas-prior test: predict a held-out sample's ambient profile (half B of its pool) from priors learned on OTHER studies,
and from n UMIs of the other half (A) alone / shrunk toward a prior. Grouped K-fold by study (never tests on a study used to
learn the prior or the hyper-parameters; hyper-parameters are cross-fitted inside the training studies).

usage: prior_test.py REC_DIR SAMPLES_TSV OUT_DIR [n_folds]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from aa_load import load_records

rec_dir, meta_tsv, out = sys.argv[1], sys.argv[2], sys.argv[3]
NF = int(sys.argv[4]) if len(sys.argv) > 4 else 20
os.makedirs(out, exist_ok=True)
S, M, genes, sym, gcl = load_records(rec_dir, meta_tsv, org=os.environ.get("AA_ORG", "human"))
use = ((S.poolA_umi >= 1000) & (S.poolB_umi >= 1000) & S.tissue_coarse.notna()).values
print("samples", len(S), "usable", int(use.sum()), flush=True)
dist = {q: {k: float(np.percentile(S[k], q)) for k in ("pool1_umi", "pool1_n", "pool2_umi", "pool2_n", "n_called", "med_called_umi")} for q in (10, 25, 50, 75, 90)}
dist["n_samples_total"] = len(S)
dist["frac_pool1_umi_ge_1e4"] = float((S.pool1_umi >= 1e4).mean())
dist["frac_pool1_umi_ge_1e5"] = float((S.pool1_umi >= 1e5).mean())
dist["frac_pool1_umi_lt_2000"] = float((S.pool1_umi < 2000).mean())
dist["frac_pool2_umi_ge_1e4"] = float((S.pool2_umi >= 1e4).mean())
json.dump(dist, open(f"{out}/prior_pool_dist.json", "w"), indent=1)
S = S[use].reset_index(drop=True)
M = {k: v[use].astype(np.float64) for k, v in M.items()}
n, Gc = len(S), len(genes)
norm = lambda X: X / X.sum(1, keepdims=True)
Cf, Af = norm(M["called"]), norm(M["pool1"])
Acnt, Bcnt = M["poolA"], M["poolB"]
Bf = norm(Bcnt)
tis, gse = S.tissue_coarse.values, S.gse.values
nz = [np.flatnonzero(Bcnt[i]) for i in range(n)]
tv = [Bf[i, nz[i]] for i in range(n)]
Hent = np.array([-(t * np.log2(t)).sum() for t in tv])
EPS = 1e-7
LAMS = [0.25, 0.5, 0.75, 1.0]
KAPS = np.array([0.1, 0.3, 1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 1e5])
NS = [100, 300, 1000, 3000, 10000, 30000]
REPS = 3
rng = np.random.default_rng(7)

# pre-drawn subsamples of half A (sparse)
draws = {}
for i in range(n):
    a = Acnt[i].astype(np.int64)
    idx = np.flatnonzero(a)
    tot = a.sum()
    for nn in NS:
        if tot < nn:
            continue
        for r in range(REPS):
            x = rng.multivariate_hypergeometric(a[idx], nn, method="marginals")
            draws[(i, nn, r)] = (idx, x)
    draws[(i, 0, 0)] = (idx, a[idx])  # n = all of half A
print("draws", len(draws), flush=True)


def build(tr, te, lams):
    abar, cbar = Af[tr].mean(0), Cf[tr].mean(0)
    r = (abar + EPS) / (cbar + EPS)
    out = {"cell": Cf[te]}
    for lam in lams:
        q = Cf[te] * r ** lam
        out[f"rw{lam}"] = q / q.sum(1, keepdims=True)
    out["glob"] = np.tile(abar / abar.sum(), (len(te), 1))
    tm = {}
    for t in np.unique(tis[te]):
        m = tr[tis[tr] == t]
        tm[t] = Af[m].mean(0) if len(m) else abar
    out["tissue"] = np.stack([tm[t] / tm[t].sum() for t in tis[te]])
    return out


def gather(P, te):
    return np.concatenate([P[j, nz[i]] for j, i in enumerate(te)])


def fit_mix(comps, t, iters=200):
    K = len(comps)
    pi = np.ones(K) / K
    for _ in range(iters):
        mix = sum(p * c for p, c in zip(pi, comps))
        pn = np.array([(t * p * c / mix).sum() for p, c in zip(pi, comps)])
        pn /= pn.sum()
        if np.abs(pn - pi).max() < 1e-7:
            pi = pn
            break
        pi = pn
    mix = sum(p * c for p, c in zip(pi, comps))
    return pi, -(t * np.log2(mix)).sum()


def fit_hparams(tr):
    """Cross-fitted (inner grouped 5-fold) predictions for the training samples, then lambda, mixture weights, kappas."""
    ng = min(5, len(np.unique(gse[tr])))
    comps = {k: [None] * len(tr) for k in ["cell", "glob", "tissue"] + [f"rw{l}" for l in LAMS]}
    for itr, ival in GroupKFold(ng).split(tr, groups=gse[tr]):
        P = build(tr[itr], tr[ival], LAMS)
        for k in comps:
            for r_, (j, i) in enumerate(zip(ival, tr[ival])):
                comps[k][j] = P[k][r_, nz[i]]
    flat = {k: np.concatenate(v) for k, v in comps.items()}
    t = np.concatenate([tv[i] for i in tr])
    u = np.full(len(t), 1.0 / Gc)
    sid = np.concatenate([np.full(len(nz[i]), j) for j, i in enumerate(tr)])
    hp = {"pi": {}}
    best = None
    for l in LAMS:
        pi, ce = fit_mix([flat[f"rw{l}"], u], t)
        if best is None or ce < best[1]:
            best = (l, ce, pi)
    lam = best[0]
    hp["lam"] = lam
    for k in ("cell", "glob", "tissue"):
        hp["pi"][k] = fit_mix([flat[k], u], t)[0]
    hp["pi"]["rw"] = best[2]
    pim, cem = fit_mix([flat["cell"], flat[f"rw{lam}"], flat["tissue"], flat["glob"], u], t)
    hp["pi"]["mix"] = pim
    hp["ce_train"] = {"mix": cem / len(tr), "rw": best[1] / len(tr)}
    # prior vectors at nz for kappa tuning
    pri = {"unif": u, "tissue": hp["pi"]["tissue"][0] * flat["tissue"] + hp["pi"]["tissue"][1] * u,
           "mix": pim[0] * flat["cell"] + pim[1] * flat[f"rw{lam}"] + pim[2] * flat["tissue"] + pim[3] * flat["glob"] + pim[4] * u}
    hp["kappa"] = {}
    off = np.cumsum([0] + [len(nz[i]) for i in tr])
    for kind, pv in pri.items():
        for nn in NS + [0]:
            tot = np.zeros(len(KAPS)); cnt = 0
            for j, i in enumerate(tr):
                d = draws.get((i, nn, 0))
                if d is None:
                    continue
                dense = np.zeros(Gc); dense[d[0]] = d[1]
                x = dense[nz[i]]
                nobs = d[1].sum()
                p = pv[off[j]:off[j + 1]]
                for q, kap in enumerate(KAPS):
                    tot[q] += -(tv[i] * np.log2((x + kap * p) / (nobs + kap))).sum()
                cnt += 1
            hp["kappa"][(kind, nn)] = float(KAPS[int(np.argmin(tot))]) if cnt else 10.0
    return hp


rows = []
folds = list(GroupKFold(min(NF, len(np.unique(gse)))).split(np.arange(n), groups=gse))
hp_log = []
t0 = time.time()
for fi, (tr, te) in enumerate(folds):
    hp = fit_hparams(tr)
    P = build(tr, te, [hp["lam"]])
    lam = hp["lam"]
    u = np.full(Gc, 1.0 / Gc)
    pi = hp["pi"]
    prior = {
        "cell": pi["cell"][0] * P["cell"] + pi["cell"][1] * u,
        "rw": pi["rw"][0] * P[f"rw{lam}"] + pi["rw"][1] * u,
        "tissue": pi["tissue"][0] * P["tissue"] + pi["tissue"][1] * u,
        "glob": pi["glob"][0] * P["glob"] + pi["glob"][1] * u,
        "mix": pi["mix"][0] * P["cell"] + pi["mix"][1] * P[f"rw{lam}"] + pi["mix"][2] * P["tissue"] + pi["mix"][3] * P["glob"] + pi["mix"][4] * u,
    }
    hp_log.append(dict(fold=fi, n_train=len(tr), n_test=len(te), lam=lam, pi_mix=pi["mix"].tolist(), pi_cell=pi["cell"].tolist(),
                       pi_rw=pi["rw"].tolist(), pi_tissue=pi["tissue"].tolist(), pi_glob=pi["glob"].tolist(),
                       kappa={f"{k[0]}_{k[1]}": v for k, v in hp["kappa"].items()}))

    def score(j, i, p, method, nn, rep):
        ce = -(tv[i] * np.log2(p[nz[i]])).sum()
        rows.append(dict(gsm=S.gsm[i], gse=gse[i], tissue=tis[i], method=method, n=nn, rep=rep, ce=ce, kl=ce - Hent[i],
                         tv=0.5 * np.abs(p - Bf[i]).sum(), spearman=spearmanr(p[nz[i]], Bcnt[i, nz[i]])[0]))

    for j, i in enumerate(te):
        for m, pv in prior.items():
            score(j, i, pv[j], f"prior_{m}", 0, 0)
        score(j, i, P[f"rw{lam}"][j], "raw_rw", 0, 0)
        score(j, i, P["cell"][j], "raw_cell", 0, 0)
        for nn in NS + [0]:
            for r in range(REPS):
                d = draws.get((i, nn, r))
                if d is None:
                    continue
                x = np.zeros(Gc); x[d[0]] = d[1]
                nobs = d[1].sum()
                for kind, pv in (("alone", u), ("tissue", prior["tissue"][j]), ("mix", prior["mix"][j])):
                    kap = hp["kappa"][({"alone": "unif"}.get(kind, kind), nn)]
                    score(j, i, (x + kap * pv) / (nobs + kap), f"emp_{kind}", nn if nn else int(nobs), r)
    print(f"fold {fi + 1}/{len(folds)} done, {time.time() - t0:.0f}s, lam={lam}, pi_mix={np.round(pi['mix'], 3)}", flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{out}/prior_test.csv", index=False, float_format="%.5g")
json.dump(hp_log, open(f"{out}/prior_hparams.json", "w"))
# summary: average reps within sample first
g = R.groupby(["method", "n", "gsm", "gse"], as_index=False)[["kl", "tv", "spearman"]].mean()
summ = g.groupby(["method", "n"]).agg(n_samples=("kl", "size"), mean_kl=("kl", "mean"), median_kl=("kl", "median"), mean_tv=("tv", "mean"),
                                       mean_spearman=("spearman", "mean")).reset_index()
summ.to_csv(f"{out}/prior_summary.csv", index=False, float_format="%.4g")
print(summ.to_string())
# paired contrasts with study-cluster bootstrap CI
pairs = []
brng = np.random.default_rng(3)
for nn in sorted(g.n.unique()):
    if nn == 0:
        continue
    a = g[(g.method == "emp_alone") & (g.n == nn)].set_index("gsm")
    for other in ("emp_tissue", "emp_mix"):
        b = g[(g.method == other) & (g.n == nn)].set_index("gsm")
        common = a.index.intersection(b.index)
        d = (b.kl[common] - a.kl[common])
        st = a.gse[common]
        studies = st.unique()
        bs = []
        for _ in range(500):
            pick = brng.choice(studies, len(studies))
            bs.append(np.mean(np.concatenate([d[st == s].values for s in pick])))
        pairs.append(dict(n=nn, contrast=f"{other} - emp_alone (KL bits)", n_samples=len(common), mean_delta=d.mean(),
                          ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), frac_prior_better=float((d < 0).mean())))
    # prior alone vs empirical alone (does the prior-only predictor beat n UMIs?)
    pm = g[(g.method == "prior_mix")].set_index("gsm")
    common = a.index.intersection(pm.index)
    d = pm.kl[common] - a.kl[common]
    pairs.append(dict(n=nn, contrast="prior_mix (no data from sample's pool) - emp_alone", n_samples=len(common), mean_delta=d.mean(),
                      ci_lo=np.nan, ci_hi=np.nan, frac_prior_better=float((d < 0).mean())))
pd.DataFrame(pairs).to_csv(f"{out}/prior_contrasts.csv", index=False, float_format="%.4g")
print(pd.DataFrame(pairs).to_string())
