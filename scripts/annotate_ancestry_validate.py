#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Out-of-sample validation of ``singlet.annotate.ancestry.infer_ancestry`` on simulated donors from public 1000 Genomes genotypes.

The pipeline's real allele depths are restricted, so donors are simulated from 1KGP phase 3 (GRCh38) genotypes:

  1. ``extract``  stream per-chromosome phased VCFs, keep biallelic SNVs with INFO AF >= 0.05 (the pipeline panel rule) and store
                  an int8 genotype matrix (alt-allele count, sites x 2504) per chromosome.
  2. ``run``      split the 2504 individuals 80/20 stratified by the 26 populations (fixed seed); per-super-population alternate
                  allele frequencies come from the 80% only; each held-out individual is turned into scRNA-like depths
                  (below) and passed to ``infer_ancestry``. Raw per-donor q / calls are saved to ``results/run.npz``.
  3. ``report``   confusion matrices, per-population accuracy, calibration, q_min trade-off, bootstrap stability, leakage and
                  runtime tables -> ``results/results.json`` (+ markdown on stdout).

Simulation (per held-out individual x condition; seeded by (seed, individual, n_sites, rho), so conditions are paired):
  * ``n_sites`` panel sites are drawn uniformly without replacement (grid 300 ... 300000; the panel here is only chr18-22, so 300000 is a large share of it);
  * depth = 1 + Poisson(mu), mu ~ LogNormal(ln 0.5, 1.5) per site (zero-truncated; median 1, ~25% of sites >= 3 reads, tail to
    hundreds), matching real coverage tables (136k-917k covered panel sites per donor, median depth 1);
  * alternate reads ~ Beta-Binomial(depth, p, rho): p = 0.5 for genotype 1, p = eps = 0.01 for 0 and 1 - eps for 2; rho is the
    intra-class correlation (a + b = (1 - rho) / rho), rho in {0.02, 0.1} mimicking allele-specific expression.
Leakage check: the same simulated data scored with frequencies computed from ALL individuals (test individuals included).

  python scripts/annotate_ancestry_validate.py extract --raw DIR --out DIR --chroms 18 19 20 21 22
  python scripts/annotate_ancestry_validate.py run --raw DIR --out DIR
  python scripts/annotate_ancestry_validate.py report --out DIR
``--raw`` holds the VCFs and the sample ``.panel`` file (downloaded from the EBI FTP); ``--out`` holds the derived files.
Genotypes never leave ``--out``; nothing is printed per individual.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import gzip
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

SUPERS = ("AFR", "AMR", "EAS", "EUR", "SAS")
SEED = 20261023
EPS_SIM = 0.01
N_SITES_GRID = (300, 1000, 3000, 10000, 30000, 100000, 300000)
DEPTH_MU, DEPTH_SIGMA = (
    0.5,
    1.5,
)  # depth = 1 + Poisson(LogNormal): median 1, ~25% of sites >= 3 reads, tail to hundreds
RHOS = (0.02, 0.1)
LEAK_NSITES = (300, 1000, 10000, 100000)
BOOT_CONDS = ((300, 0.1), (1000, 0.1), (1000, 0.02), (10000, 0.1), (100000, 0.1))
N_BOOT_LARGE = 16  # donors with n_boot=20 at >= 100000 sites (cost)
N_LARGE = 100  # donors evaluated at >= 100000 sites (one fit takes 7-25 s there)
LARGE_MAX = 300000
N_BOOT_DONORS = 120
SUBGROUPS = (
    "AFR_ASW_ACB",
    "AFR_other(YRI,LWK,GWD,MSL,ESN)",
    "AMR_MXL",
    "AMR_PUR",
    "AMR_CLM",
    "AMR_PEL",
    "non_admixed_pops",
)
Q_MINS = (0.6, 0.7, 0.8, 0.9)
CAL_BINS = ((0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.0001))
VCF = "ALL.chr{c}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"
PANEL = "integrated_call_samples_v3.20130502.ALL.panel"


# ---------------------------------------------------------------- extract
def _extract_chrom(args):
    raw, out, chrom, min_af = args
    rows, pos, ref, alt, af = [], [], [], [], []
    n_lines = 0
    with gzip.open(Path(raw) / VCF.format(c=chrom), "rb") as fh:
        for line in fh:
            if line[0] == 35:
                if line.startswith(b"#CHROM"):
                    samples = line.rstrip(b"\n").split(b"\t")[9:]
                continue
            n_lines += 1
            f = line.split(b"\t", 9)
            if (
                len(f[3]) != 1 or len(f[4]) != 1
            ):  # SNV, biallelic (multi-allelic ALT contains a comma)
                continue
            a = None
            for item in f[7].split(b";"):
                if item.startswith(b"AF="):
                    a = float(item[3:])
                    break
            if a is None or a < min_af:
                continue
            gt = np.frombuffer(f[9].rstrip(b"\n") + b"\t", np.uint8)
            if gt.size != 4 * len(samples) or not (gt[1::4] == 124).all():
                continue  # unphased / unexpected layout
            rows.append((gt[0::4] - 48 + gt[2::4] - 48).astype(np.int8))
            pos.append(int(f[1]))
            ref.append(f[3])
            alt.append(f[4])
            af.append(a)
    gt = np.stack(rows)
    np.savez(
        Path(out) / f"panel_chr{chrom}.npz",
        gt=gt,
        pos=np.array(pos, np.int64),
        ref=np.array(ref, "S1"),
        alt=np.array(alt, "S1"),
        af=np.array(af, np.float32),
        samples=np.array([s.decode() for s in samples]),
        chrom=np.full(len(pos), int(chrom), np.int8),
    )
    return chrom, n_lines, gt.shape[0]


def cmd_extract(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Pool(min(a.procs, len(a.chroms))) as pool:
        for chrom, n_lines, n_keep in pool.imap_unordered(
            _extract_chrom, [(a.raw, a.out, c, a.min_af) for c in a.chroms]
        ):
            print(
                f"chr{chrom}: {n_lines} variants read, {n_keep} biallelic SNVs with AF >= {a.min_af}",
                flush=True,
            )


# ---------------------------------------------------------------- data
def load_panel(raw: Path, out: Path, chroms):
    import pandas as pd

    parts = [np.load(out / f"panel_chr{c}.npz") for c in chroms]
    samples = list(parts[0]["samples"])
    assert all(list(p["samples"]) == samples for p in parts)
    gt = np.concatenate([p["gt"] for p in parts])
    in_panel = pd.read_csv(raw / PANEL, sep="	", usecols=["sample"])["sample"]
    keep = np.flatnonzero(
        np.isin(samples, in_panel)
    )  # the VCF carries 2548 samples, the call set 2504
    gt, samples = gt[:, keep], [samples[i] for i in keep]
    meta = {k: np.concatenate([p[k] for p in parts]) for k in ("chrom", "pos", "af")}
    panel = (
        pd.read_csv(raw / PANEL, sep="\t", usecols=["sample", "pop", "super_pop"])
        .set_index("sample")
        .loc[samples]
    )
    return gt, meta, panel.index.to_numpy(), panel["pop"].to_numpy(), panel["super_pop"].to_numpy()


def split_individuals(pops: np.ndarray, test_frac=0.2, seed=SEED):
    """80/20 split stratified by population: round(0.2 n) test individuals per population (>= 1)."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(pops), bool)
    for p in sorted(set(pops)):
        idx = np.flatnonzero(pops == p)
        k = max(1, int(round(test_frac * len(idx))))
        test[rng.choice(idx, k, replace=False)] = True
    return test


def super_freqs(gt: np.ndarray, supers: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Alternate-allele frequency (sites x 5) from the individuals in ``rows`` (boolean mask over individuals)."""
    F = np.empty((gt.shape[0], len(SUPERS)), np.float64)
    for k, s in enumerate(SUPERS):
        cols = np.flatnonzero(rows & (supers == s))
        F[:, k] = gt[:, cols].sum(1, dtype=np.int64) / (2.0 * len(cols))
    return F


def simulate(g: np.ndarray, rng: np.random.Generator, n_sites: int, rho: float):
    """Draw n_sites covered sites for one individual (genotype column g over all panel sites) -> (site index, ad, dp)."""
    idx = np.sort(rng.choice(g.shape[0], n_sites, replace=False))
    gi = g[idx]
    dp = 1 + rng.poisson(rng.lognormal(np.log(DEPTH_MU), DEPTH_SIGMA, n_sites))
    p = np.where(gi == 1, 0.5, np.where(gi == 2, 1 - EPS_SIM, EPS_SIM))
    k = (1 - rho) / rho
    ad = rng.binomial(dp, np.clip(rng.beta(p * k, (1 - p) * k), 0.0, 1.0))
    return idx, ad, dp


# ---------------------------------------------------------------- run
_W: dict = {}


def _init_worker(out: str, chroms, seed):
    from singlet.annotate import ancestry

    d = np.load(Path(out) / "run_inputs.npz")
    _W.update(
        anc=ancestry, gt=d["gt_test"], F={"train": d["F_train"], "all": d["F_all"]}, seed=seed
    )


def _do(task):
    ci, j, cond = task
    idx, ad, dp = simulate(
        _W["gt"][:, j],
        np.random.default_rng([_W["seed"], j, cond["n_sites"], int(cond["rho"] * 1000)]),
        cond["n_sites"],
        cond["rho"],
    )
    t0 = time.perf_counter()
    r = _W["anc"].infer_ancestry(ad, dp, _W["F"][cond["freq"]][idx], n_boot=cond["n_boot"], seed=j)
    dt = time.perf_counter() - t0
    q = [r.q.get(s, np.nan) for s in SUPERS]
    return (
        ci,
        j,
        q,
        r.group or "None",
        r.n_sites,
        np.nan if r.stability is None else r.stability,
        dt,
        int(np.median(dp)),
        int(dp.max()),
        float((dp >= 3).mean()),
    )


def grid(n_panel=10**9):
    return tuple(min(n, int(0.75 * n_panel)) if n > 100000 else n for n in N_SITES_GRID)


def conditions(n_panel=10**9):
    """The condition grid; a site count above 75% of the panel is replaced by 75% of the panel (the matrix is smaller than the
    real panel, so e.g. 300000 may not fit)."""
    c = []
    cap = int(0.75 * n_panel)
    for n in grid(n_panel):
        for rho in RHOS:
            if n > 100000 and rho != 0.1:
                continue  # large conditions: rho 0.1 only (cost)
            c.append(dict(n_sites=n, rho=rho, freq="train", n_boot=0, kind="main"))
    for n in LEAK_NSITES:
        for rho in RHOS:
            c.append(dict(n_sites=n, rho=rho, freq="all", n_boot=0, kind="leak"))
    for n, rho in BOOT_CONDS:
        c.append(dict(n_sites=n, rho=rho, freq="train", n_boot=20, kind="boot"))
    return c


def cmd_run(a):
    raw, out = Path(a.raw), Path(a.out)
    gt, meta, samples, pops, supers = load_panel(raw, out, a.chroms)
    test = split_individuals(pops)
    F_train = super_freqs(gt, supers, ~test)
    F_all = super_freqs(gt, supers, np.ones(len(supers), bool))
    tcols = np.flatnonzero(test)
    np.savez(out / "run_inputs.npz", gt_test=gt[:, tcols], F_train=F_train, F_all=F_all)
    print(
        f"{gt.shape[0]} panel sites; {len(tcols)} test / {int((~test).sum())} train individuals",
        flush=True,
    )
    rng = np.random.default_rng(SEED + 1)
    boot_set = set(rng.choice(len(tcols), N_BOOT_DONORS, replace=False).tolist())
    conds = conditions(gt.shape[0])
    large_set = set(
        np.random.default_rng(SEED + 2).choice(len(tcols), N_LARGE, replace=False).tolist()
    )

    def want(c, j):
        if c["n_sites"] >= 100000 and j not in large_set:
            return False
        if c["kind"] != "boot":
            return True
        return j in (sorted(large_set)[:N_BOOT_LARGE] if c["n_sites"] >= 100000 else boot_set)

    tasks = [(ci, j, c) for ci, c in enumerate(conds) for j in range(len(tcols)) if want(c, j)]
    tasks.sort(key=lambda t: -t[2]["n_sites"] * (21 if t[2]["n_boot"] else 1))
    nt, nc = len(tcols), len(conds)
    Q = np.full((nc, nt, 5), np.nan)
    G = np.full((nc, nt), "", dtype="U8")
    S = np.full((nc, nt), np.nan)
    NS = np.zeros((nc, nt), int)
    DMED, DMAX, F3 = np.zeros((nc, nt)), np.zeros((nc, nt)), np.zeros((nc, nt))
    t0 = time.time()
    with Pool(a.procs, initializer=_init_worker, initargs=(str(out), a.chroms, SEED)) as pool:
        for n, (ci, j, q, grp, ns, stab, _dt, dm, dx, f3) in enumerate(
            pool.imap_unordered(_do, tasks, chunksize=4)
        ):
            Q[ci, j], G[ci, j], S[ci, j], NS[ci, j], DMED[ci, j], DMAX[ci, j], F3[ci, j] = (
                q,
                grp,
                stab,
                ns,
                dm,
                dx,
                f3,
            )
            if n % 2000 == 0:
                print(f"  {n}/{len(tasks)} fits, {time.time() - t0:.0f}s", flush=True)
    # serial timing (no parallel load): infer_ancestry only
    _init_worker(str(out), a.chroms, SEED)
    timing = []
    for n in grid(gt.shape[0]):
        for nb in (0, 20) if n in (1000, 10000, 100000, 300000) else (0,):
            cond = dict(n_sites=n, rho=0.1, freq="train", n_boot=nb)
            js = range((15 if n < 100000 else 6) if nb == 0 else (4 if n < 100000 else 2))
            dts = [_do((0, j, cond))[6] for j in js]
            timing.append(
                dict(
                    n_sites=n,
                    n_boot=nb,
                    n_donors=len(js),
                    mean_s=float(np.mean(dts)),
                    max_s=float(np.max(dts)),
                )
            )
            print("timing", timing[-1], flush=True)
    np.savez(
        out / "run.npz",
        Q=Q,
        G=G,
        S=S,
        NS=NS,
        DMED=DMED,
        DMAX=DMAX,
        F3=F3,
        test_samples=samples[tcols].astype(str),
        test_pop=pops[tcols].astype(str),
        test_super=supers[tcols].astype(str),
        conds=json.dumps(conds),
        timing=json.dumps(timing),
        n_sites_panel=gt.shape[0],
        n_train=int((~test).sum()),
    )
    print(f"done in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- report
def _acc(true, call):
    return float(np.mean(true == call)) if len(true) else float("nan")


def cmd_report(a):
    from singlet.annotate import ancestry

    out = Path(a.out)
    d = np.load(out / "run.npz", allow_pickle=False)
    conds = json.loads(str(d["conds"]))
    Q, G, S = d["Q"], d["G"], d["S"]
    sup, pop = d["test_super"], d["test_pop"]
    cols = list(SUPERS) + ["admixed", "None"]
    res: dict = dict(
        n_panel_sites=int(d["n_sites_panel"]),
        n_train=int(d["n_train"]),
        n_test=int(len(sup)),
        seed=SEED,
        test_per_super={s: int((sup == s).sum()) for s in SUPERS},
        timing=json.loads(str(d["timing"])),
        conditions={},
    )
    md = []

    def key(c):
        return f"{c['kind']}_n{c['n_sites']}_rho{c['rho']}_{c['freq']}_boot{c['n_boot']}"

    def summarize(ci, mask=None):
        c = conds[ci]
        m = (
            G[ci] != ""
        )  # donors evaluated in this condition (boot and very large conditions use a subset)
        if mask is not None:
            m = m & mask
        g, t, q = G[ci][m], sup[m], Q[ci][m]
        o = dict(n=int(m.sum()))
        o["confusion"] = {s: {k: int((g[t == s] == k).sum()) for k in cols} for s in SUPERS}
        o["accuracy_per_super"] = {s: _acc(t[t == s], g[t == s]) for s in SUPERS}
        o["accuracy_overall"] = _acc(t, g)
        o["frac_admixed"] = float(np.mean(g == "admixed"))
        called = np.isin(g, SUPERS)
        o["frac_named"] = float(called.mean())
        o["accuracy_among_named"] = _acc(t[called], g[called])
        qmax = np.where(np.isnan(q).all(1), np.nan, np.nanmax(q, 1)) if len(q) else q
        o["calibration"] = {}
        for lo, hi in CAL_BINS:
            b = called & (qmax >= lo) & (qmax < hi)
            o["calibration"][f"{lo}-{min(hi, 1.0)}"] = dict(
                n=int(b.sum()),
                accuracy=_acc(t[b], g[b]),
                mean_qmax=float(np.mean(qmax[b])) if b.any() else None,
            )
        o["qmin_tradeoff"] = {}
        for qm in Q_MINS:
            gg = np.array([ancestry._group(x, qm) if not np.isnan(x).any() else "None" for x in q])
            nm = np.isin(gg, SUPERS)
            o["qmin_tradeoff"][str(qm)] = dict(
                frac_admixed=float(np.mean(gg == "admixed")),
                accuracy_among_named=_acc(t[nm], gg[nm]),
                correct_over_all=_acc(t, gg),
            )
        sub = {}
        for name, mm in (
            ("AFR_ASW_ACB", np.isin(pop[m], ["ASW", "ACB"])),
            ("AFR_other(YRI,LWK,GWD,MSL,ESN)", (t == "AFR") & ~np.isin(pop[m], ["ASW", "ACB"])),
            *[(f"AMR_{p}", pop[m] == p) for p in ("MXL", "PUR", "CLM", "PEL")],
            ("non_admixed_pops", ~np.isin(pop[m], ["ASW", "ACB", "MXL", "PUR", "CLM", "PEL"])),
        ):
            if mm.any():
                lab = {k: int((g[mm] == k).sum()) for k in cols}
                sub[name] = dict(
                    n=int(mm.sum()),
                    accuracy=_acc(t[mm], g[mm]),
                    frac_admixed=float(np.mean(g[mm] == "admixed")),
                    calls=lab,
                    mean_q_true=float(
                        np.nanmean([q[i, SUPERS.index(t[i])] for i in np.flatnonzero(mm)])
                    ),
                )
        o["subgroups"] = sub
        o["per_pop_accuracy"] = {p: _acc(t[pop[m] == p], g[pop[m] == p]) for p in sorted(set(pop))}
        o["mean_q_true"] = float(np.nanmean([q[i, SUPERS.index(t[i])] for i in range(len(t))]))
        if c["n_boot"]:
            s = S[ci][m]
            o["stability"] = {}
            for name, lo, hi in (("<0.8", 0, 0.8), ("0.8-<1", 0.8, 1.0), ("=1", 1.0, 1.01)):
                b = (s >= lo) & (s < hi)
                o["stability"][name] = dict(
                    n=int(b.sum()),
                    accuracy=_acc(t[b], g[b]),
                    frac_admixed=float(np.mean(g[b] == "admixed")) if b.any() else None,
                )
            o["stability"]["mean_stability_correct"] = (
                float(np.nanmean(s[g == t])) if (g == t).any() else None
            )
            o["stability"]["mean_stability_wrong_named"] = (
                float(np.nanmean(s[called & (g != t)])) if (called & (g != t)).any() else None
            )
            o["stability"]["mean_stability_admixed"] = (
                float(np.nanmean(s[g == "admixed"])) if (g == "admixed").any() else None
            )
        return o

    for ci, c in enumerate(conds):
        res["conditions"][key(c)] = dict(
            cond=c,
            depth_median_of_medians=float(np.median(d["DMED"][ci][d["DMED"][ci] > 0])),
            depth_max=float(d["DMAX"][ci].max()),
            frac_sites_dp_ge3=float(d["F3"][ci][d["F3"][ci] > 0].mean()),
            **summarize(ci),
        )
    # leakage: paired held-out-frequency vs all-individual-frequency
    leak = {}
    for ci, c in enumerate(conds):
        if c["kind"] != "leak":
            continue
        base = next(
            i
            for i, b in enumerate(conds)
            if b["kind"] == "main" and b["n_sites"] == c["n_sites"] and b["rho"] == c["rho"]
        )
        both = (G[ci] != "") & (G[base] != "")  # paired donors (the large conditions use a subset)
        h, l = summarize(base, both), summarize(ci, both)
        leak[f"n{c['n_sites']}_rho{c['rho']}"] = dict(
            held_out=dict(
                acc=h["accuracy_overall"],
                admixed=h["frac_admixed"],
                mean_q_true=h["mean_q_true"],
                named_acc=h["accuracy_among_named"],
            ),
            in_sample=dict(
                acc=l["accuracy_overall"],
                admixed=l["frac_admixed"],
                mean_q_true=l["mean_q_true"],
                named_acc=l["accuracy_among_named"],
            ),
            delta=dict(
                acc=l["accuracy_overall"] - h["accuracy_overall"],
                admixed=l["frac_admixed"] - h["frac_admixed"],
                mean_q_true=l["mean_q_true"] - h["mean_q_true"],
            ),
            n=int(both.sum()),
            n_flip_to_correct=int(((G[ci] == sup) & (G[base] != sup) & both).sum()),
            n_flip_to_wrong=int(((G[ci] != sup) & (G[base] == sup) & both).sum()),
        )
    res["leakage"] = leak
    (out / "results.json").write_text(json.dumps(res, indent=1))

    # ---- markdown
    def conf_md(k):
        r = res["conditions"][k]
        md.append(
            f"\n### Confusion matrix {k} (n={r['n']}; rows true, cols called)\n| true | "
            + " | ".join(cols)
            + " | acc |\n|"
            + "---|" * (len(cols) + 2)
        )
        for s in SUPERS:
            md.append(
                f"| {s} | "
                + " | ".join(str(r["confusion"][s][x]) for x in cols)
                + f" | {r['accuracy_per_super'][s]:.3f} |"
            )

    for n in (1000, 10000, 100000, max(c["n_sites"] for c in conds)):
        for rho in RHOS:
            if f"main_n{n}_rho{rho}_train_boot0" in res["conditions"]:
                conf_md(f"main_n{n}_rho{rho}_train_boot0")
    md.append(
        "\n### Accuracy vs n_sites (held-out frequencies; correct/all, admixed rate, accuracy among named calls)\n| n_sites | rho | acc | admixed | named acc | "
        + " | ".join(SUPERS)
        + " |\n|"
        + "---|" * (5 + 5)
    )
    for c in conds:
        if c["kind"] == "main":
            r = res["conditions"][key(c)]
            md.append(
                f"| {c['n_sites']} | {c['rho']} | {r['accuracy_overall']:.3f} | {r['frac_admixed']:.3f} | {r['accuracy_among_named']:.3f} | "
                + " | ".join(f"{r['accuracy_per_super'][s]:.3f}" for s in SUPERS)
                + " |"
            )
    md.append(
        "\n### Calibration (accuracy of named call by q_max bin; n)\n| n_sites | rho | "
        + " | ".join(f"{lo}-{min(hi, 1.0)}" for lo, hi in CAL_BINS)
        + " |\n|---|---|---|---|---|---|"
    )
    for c in conds:
        if c["kind"] == "main":
            r = res["conditions"][key(c)]
            md.append(
                f"| {c['n_sites']} | {c['rho']} | "
                + " | ".join(
                    f"{v['accuracy']:.3f} ({v['n']})" if v["n"] else "- (0)"
                    for v in r["calibration"].values()
                )
                + " |"
            )
    md.append(
        "\n### Subgroups (accuracy / admixed rate; n)\n| n_sites | rho | "
        + " | ".join(SUBGROUPS)
        + " |\n|"
        + "---|" * 10
    )
    for c in conds:
        if c["kind"] == "main":
            r = res["conditions"][key(c)]
            md.append(
                f"| {c['n_sites']} | {c['rho']} | "
                + " | ".join(
                    (
                        f"{r['subgroups'][k]['accuracy']:.2f}/{r['subgroups'][k]['frac_admixed']:.2f} ({r['subgroups'][k]['n']})"
                        if k in r["subgroups"]
                        else "-"
                    )
                    for k in SUBGROUPS
                )
                + " |"
            )
    md.append(
        "\n### q_min trade-off (admixed rate; accuracy among named; correct/all)\n| n_sites | rho | "
        + " | ".join(f"q_min={q}" for q in Q_MINS)
        + " |\n|---|---|---|---|---|---|"
    )
    for c in conds:
        if c["kind"] == "main":
            r = res["conditions"][key(c)]
            md.append(
                f"| {c['n_sites']} | {c['rho']} | "
                + " | ".join(
                    f"{v['frac_admixed']:.3f} / {v['accuracy_among_named']:.3f} / {v['correct_over_all']:.3f}"
                    for v in r["qmin_tradeoff"].values()
                )
                + " |"
            )
    md.append(
        "\n### Bootstrap stability (n_boot=20, subset of donors)\n| n_sites | rho | n | acc | <0.8 (n, acc) | 0.8-<1 | =1 | mean stab correct / wrong-named / admixed |\n|---|---|---|---|---|---|---|---|"
    )
    for c in conds:
        if c["kind"] == "boot":
            r = res["conditions"][key(c)]
            st = r["stability"]
            f = lambda v: "-" if v is None else f"{v:.2f}"
            md.append(
                f"| {c['n_sites']} | {c['rho']} | {r['n']} | {r['accuracy_overall']:.3f} | "
                + " | ".join(
                    f"{st[k]['n']}, {f(st[k]['accuracy'])}" for k in ("<0.8", "0.8-<1", "=1")
                )
                + f" | {f(st['mean_stability_correct'])} / {f(st['mean_stability_wrong_named'])} / {f(st['mean_stability_admixed'])} |"
            )
    md.append(
        "\n### Leakage (frequencies from all individuals vs 80% train)\n| cond | held-out acc / admixed / mean q_true | in-sample acc / admixed / mean q_true | delta acc / admixed / q_true | flips to correct / to wrong |\n|---|---|---|---|---|"
    )
    for k, v in leak.items():
        md.append(
            f"| {k} | {v['held_out']['acc']:.3f} / {v['held_out']['admixed']:.3f} / {v['held_out']['mean_q_true']:.3f} | {v['in_sample']['acc']:.3f} / {v['in_sample']['admixed']:.3f} / {v['in_sample']['mean_q_true']:.3f} | {v['delta']['acc']:+.3f} / {v['delta']['admixed']:+.3f} / {v['delta']['mean_q_true']:+.4f} | {v['n_flip_to_correct']} / {v['n_flip_to_wrong']} |"
        )
    md.append(
        "\n### Runtime (seconds per donor, infer_ancestry only, single process)\n| n_sites | n_boot | mean | max |\n|---|---|---|---|"
    )
    for t in res["timing"]:
        md.append(f"| {t['n_sites']} | {t['n_boot']} | {t['mean_s']:.3f} | {t['max_s']:.3f} |")
    text = "\n".join(md)
    (out / "report.md").write_text(text)
    print(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["extract", "run", "report"])
    ap.add_argument("--raw", help="directory with the VCFs and the sample .panel file")
    ap.add_argument("--out", required=True, help="directory for derived files")
    ap.add_argument("--chroms", nargs="+", default=["18", "19", "20", "21", "22"])
    ap.add_argument("--min-af", type=float, default=0.05)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    {"extract": cmd_extract, "run": cmd_run, "report": cmd_report}[a.step](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
