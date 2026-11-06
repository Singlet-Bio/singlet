#!/usr/bin/env python3
"""State-of-the-art ceilings on the mixed AD/DP matrices of one mixture: Vireo and souporcell (+ troublet).

usage: ceilings.py vireo|souporcell MIXDIR PRIVDIR K [--seeds 42,7,123] [--threads 4,16] [--kmax 6]

MIXDIR   public results of one mixture (truth.tsv, mix.json). Receives runs-<tool>.jsonl and runs/<tag>/assign.tsv
         (labels and probabilities only, the same format as the engine).
PRIVDIR  holds mix_ad.mtx and mix_dp.mtx (SNP x cell counts). These are per-SNP counts and never leave the cluster.

Tags are <tool>_<mode>_t<threads>_s<seed>, mode k (K given) or auto.
K given: every seed at every thread count. Auto: every seed at the largest thread count over K = 2..kmax.
  vireo        auto K = argmax of Vireo's own ELBO over K.
  souporcell   has no K selection of its own: auto K = argmax of BIC, total log probability - 0.5 ln(N) L K,
               L = loci souporcell uses (>= 10 cells with alt and >= 10 cells with ref).
Both read the standard cellSNP-style site filter: >= 20 reads in total and minor allele fraction >= 0.1 in the pool.
Needs numpy, scipy, vireoSNP. souporcell and troublet run from the apptainer image in $SOUP_SIF.
"""
import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmread

BUDGET_S = 3 * 3600 + 1800


def load(priv):
    ad = sparse.csc_matrix(mmread(str(Path(priv) / "mix_ad.mtx"))).astype(np.float64)
    dp = sparse.csc_matrix(mmread(str(Path(priv) / "mix_dp.mtx"))).astype(np.float64)
    tot = np.asarray(dp.sum(1)).ravel()
    alt = np.asarray(ad.sum(1)).ravel()
    af = np.divide(alt, tot, out=np.zeros_like(tot), where=tot > 0)
    maf = np.minimum(af, 1 - af)
    keep = (tot >= 20) & (maf >= 0.1)
    if keep.sum() < 2000:
        keep = (tot >= 10) & (maf >= 0.05)
    ad, dp = ad[keep], dp[keep]
    print(f"[ceil] {int(keep.sum())} sites kept, {ad.shape[1]} cells", flush=True)
    return sparse.csc_matrix(ad), sparse.csc_matrix(dp)


def cpu_children():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    s = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime + s.ru_utime + s.ru_stime


def write_assign(mixdir, tag, labels, pmax, pdbl):
    d = Path(mixdir) / "runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "assign.tsv", "w") as f:
        f.write("cell\tdonor_id\tprob_max\tprob_doublet\tlabel\n")
        for i, l in enumerate(labels):
            did = int(l[5:]) if l.startswith("donor") else -1
            f.write(f"c{i}\t{did}\t{pmax[i]:.6g}\t{pdbl[i]:.6g}\t{l}\n")


def run_vireo(ad, dp, K, threads, seed):
    from vireoSNP.utils.vireo_wrap import vireo_wrap
    RV = vireo_wrap(ad, dp, n_donor=K, n_init=50, random_seed=seed, check_doublet=True, nproc=threads)
    idp, dbp = np.asarray(RV["ID_prob"]), np.asarray(RV["doublet_prob"])
    pmax = idp.max(1)
    pdbl = dbp.max(1) if dbp.ndim == 2 and dbp.shape[1] > 0 else np.zeros(len(pmax))
    nvar = np.asarray((dp > 0).sum(0)).ravel()
    lab = np.array([f"donor{k}" for k in idp.argmax(1)], dtype=object)
    lab[pmax < 0.9] = "unassigned"
    lab[pdbl >= 0.9] = "doublet"
    lab[nvar < 10] = "unassigned"
    return lab, pmax, pdbl, float(RV["LB_doublet"])


def write_mtx(path, m):
    m = sparse.coo_matrix(m)
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        f.write(f"{m.shape[0]} {m.shape[1]} {m.nnz}\n")
        for r, c, v in zip(m.row, m.col, m.data):
            f.write(f"{r + 1} {c + 1} {int(v)}\n")


def lse(a):
    m = a.max(1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(1, keepdims=True))).ravel()


def run_souporcell(work, K, threads, seed, sif):
    """Clustering with souporcell, then troublet. Returns labels, prob_max, prob_doublet, total log probability."""
    n = int(open(work / "barcodes.tsv").read().count("\n"))
    bind = f"--bind {work}:{work}"
    base = f"apptainer exec {bind} {sif}"
    cmd = (f"{base} souporcell -a {work}/alt.mtx -r {work}/ref.mtx -b {work}/barcodes.tsv -k {K} -t {threads} "
           f"--min_alt 10 --min_ref 10 --restarts 100 --seed {seed}")
    for use_seed in (True, False):
        c = cmd if use_seed else cmd.rsplit(" --seed", 1)[0]
        with open(work / "clusters_tmp.tsv", "w") as so, open(work / "souporcell.err", "w") as se:
            p = subprocess.run(c, shell=True, stdout=so, stderr=se, timeout=2400)
        if p.returncode == 0:
            break
    if p.returncode != 0:
        raise RuntimeError("souporcell failed: " + open(work / "souporcell.err").read()[-400:])
    err = open(work / "souporcell.err").read()
    m = re.findall(r"best total log probability\s*=\s*(-?[0-9.eE+-]+)", err)
    rows = [l.rstrip("\n").split("\t") for l in open(work / "clusters_tmp.tsv") if l.strip()]
    rows = [r for r in rows if r[0] != "barcode"]
    total = float(m[-1]) if m else float(sum(max(float(x) for x in r[2:2 + K]) for r in rows))
    for flags in ("--alts {a} --refs {r} --clusters {c}", "-a {a} -r {r} --clusters {c}"):
        t = f"{base} troublet " + flags.format(a=f"{work}/alt.mtx", r=f"{work}/ref.mtx", c=f"{work}/clusters_tmp.tsv")
        with open(work / "clusters.tsv", "w") as so, open(work / "troublet.err", "w") as se:
            p = subprocess.run(t, shell=True, stdout=so, stderr=se, timeout=2400)
        if p.returncode == 0:
            break
    if p.returncode != 0:
        raise RuntimeError("troublet failed: " + open(work / "troublet.err").read()[-400:])
    lines = [l.rstrip("\n").split("\t") for l in open(work / "clusters.tsv") if l.strip()]
    head = lines[0]
    body = lines[1:] if head[0] == "barcode" else lines
    if head[0] != "barcode":
        head = ["barcode", "status", "assignment"] + [f"log_prob_singleton_{i}" for i in range(K)] + ["log_prob_doublet_x"] * (len(head) - 3 - K)
    si = [i for i, h in enumerate(head) if h.startswith("log_prob_singleton")]
    di = [i for i, h in enumerate(head) if h.startswith("log_prob_doublet")]
    L = np.array([[float(r[i]) for i in si + di] for r in body])
    allp = lse(L)
    pmax = np.exp(L[:, :len(si)].max(1) - allp)
    pdbl = np.exp(lse(L[:, len(si):]) - allp) if di else np.zeros(len(body))
    lab = []
    for r in body:
        st, asg = r[1], r[2]
        lab.append("doublet" if st == "doublet" else ("unassigned" if st == "unassigned" else f"donor{int(asg)}"))
    if len(lab) != n:
        raise RuntimeError(f"troublet returned {len(lab)} cells, expected {n}")
    return np.array(lab, dtype=object), pmax, pdbl, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", choices=["vireo", "souporcell"])
    ap.add_argument("mixdir")
    ap.add_argument("privdir")
    ap.add_argument("k", type=int)
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--threads", default="4,16")
    ap.add_argument("--kmax", type=int, default=6)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    threads = [int(x) for x in a.threads.split(",")]
    tmax = max(threads)
    t_start = time.time()
    ad, dp = load(a.privdir)
    N = ad.shape[1]
    sif = os.environ.get("SOUP_SIF", "")
    work = Path(a.privdir) / f"{a.tool}_work"
    L_loci = None
    if a.tool == "souporcell":
        work.mkdir(parents=True, exist_ok=True)
        (work / "barcodes.tsv").write_text("".join(f"c{i}\n" for i in range(N)))
        write_mtx(work / "alt.mtx", ad)
        write_mtx(work / "ref.mtx", dp - ad)
        alt_cells = np.asarray((ad > 0).sum(1)).ravel()
        ref_cells = np.asarray(((dp - ad) > 0).sum(1)).ravel()
        L_loci = int(((alt_cells >= 10) & (ref_cells >= 10)).sum())
        print(f"[ceil] souporcell loci with >= 10 alt and >= 10 ref cells: {L_loci}", flush=True)
    log = open(Path(a.mixdir) / f"runs-{a.tool}.jsonl", "a")
    fail = open(Path(a.mixdir) / f"failures-{a.tool}.txt", "a")

    def one(K, t, seed, tag, mode):
        if time.time() - t_start > BUDGET_S:
            fail.write(f"{tag}: skipped, time budget\n"); fail.flush()
            return None
        w0, c0 = time.time(), cpu_children()
        try:
            if a.tool == "vireo":
                lab, pmax, pdbl, obj = run_vireo(ad, dp, K, t, seed)
            else:
                lab, pmax, pdbl, obj = run_souporcell(work, K, t, seed, sif)
        except Exception as e:  # keep going: one failed fit must not stop the other runs
            fail.write(f"{tag}: {type(e).__name__}: {str(e)[:500]}\n"); fail.flush()
            print(f"[ceil] {tag} failed: {e}", flush=True)
            return None
        wall, cpu = time.time() - w0, cpu_children() - c0
        write_assign(a.mixdir, tag, lab, pmax, pdbl)
        rec = {"tool": a.tool, "tag": tag, "k_arg": K if mode == "k" else -1, "threads": t, "seed": seed, "wall_s": wall,
               "cpu_s": cpu, "objective": obj, "n_cells": N, "k_fit": K}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[ceil] {tag}: K={K} wall {wall:.1f}s objective {obj:.4g}", flush=True)
        return lab, pmax, pdbl, obj

    for seed in seeds:
        sweep = {}
        r = one(a.k, tmax, seed, f"{a.tool}_k_t{tmax}_s{seed}", "k")
        if r:
            sweep[a.k] = r
        for K in range(2, a.kmax + 1):
            if K != a.k:
                r = one(K, tmax, seed, f"{a.tool}_sweep{K}_t{tmax}_s{seed}", "sweep")
                if r:
                    sweep[K] = r
        for t in threads:
            if t != tmax:
                one(a.k, t, seed, f"{a.tool}_k_t{t}_s{seed}", "k")
        if sweep:
            if a.tool == "vireo":
                score = {K: v[3] for K, v in sweep.items()}
            else:
                score = {K: v[3] - 0.5 * np.log(N) * L_loci * K for K, v in sweep.items()}
            kb = max(score, key=score.get)
            lab, pmax, pdbl, obj = sweep[kb]
            tag = f"{a.tool}_auto_t{tmax}_s{seed}"
            write_assign(a.mixdir, tag, lab, pmax, pdbl)
            log.write(json.dumps({"tool": a.tool, "tag": tag, "k_arg": -1, "threads": tmax, "seed": seed, "wall_s": None,
                                  "cpu_s": None, "objective": obj, "n_cells": N, "k_fit": kb,
                                  "sweep_score": {str(k): float(v) for k, v in score.items()}}) + "\n")
            log.flush()
            print(f"[ceil] auto K seed {seed}: {kb} from {score}", flush=True)
    # souporcell work files hold per-SNP matrices: they stay in PRIVDIR, which is private to the cluster


if __name__ == "__main__":
    sys.exit(main())
