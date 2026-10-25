"""Ambient RNA truth in real tissue: Kang 2018 lane 2.1 (8 pooled PBMC donors), Singlet prod/modality-fixes end to end.

A read in a donor's singlet that carries an allele the donor does not have came from another donor's RNA, i.e. from the ambient
pool. At a site where the donor is homozygous reference, the expected alt fraction of a cell's reads is
    rho * f_s + e        (f_s = alt fraction of the ambient pool at the site, e = sequencing / genotyping error)
and symmetrically for homozygous-alt sites. So rho_c = (X_c - e N_c) / Q_c with X = foreign reads, N = reads, Q = sum n_s q_s.
Donor genotypes come from the pooled reads of the donor's confident singlets, f_s from the pooled reads of the ambient barcodes.
Only cross-donor ambient is visible: rho_true estimates total contamination because the pool's allele frequencies are measured
on the pool itself (the cell's own donor contributes to f_s and that is accounted for).

Usage: kang_ambient.py dump OUT_DIR          gene-level exon/intron dumps for harness_cells: all barcodes, and barcodes with
                                             >= T total UMIs (T = 5, 10, 20, 30, 50; called cells always kept) to mimic the
                                             legacy atlas matrices (barcodes with >= 100 reads only)
       kang_ambient.py score OUT_DIR         genotype truth, production rho per variant, calibration tables -> OUT_DIR/*.csv, .md
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.stats import spearmanr

W = "/mnt/projects/debruinz_project/debruinz/singlet"
RUN = f"{W}/modality-bench/genotype_pool/singlet_modfix"
TRUTH = f"{W}/modality-bench/genotype_pool/truth/GSE96583_batch2.total.tsne.df.tsv.gz"
THRESH = [0, 5, 10, 20, 30, 50]


def mtx(path):
    with gzip.open(path, "rb") as f:
        return scipy.io.mmread(f).tocsc()


def lines(path):
    with gzip.open(path, "rt") as f:
        return [l.rstrip("\n").split("\t")[0] for l in f]


def dump(out):
    E, I = mtx(f"{RUN}/exon_counts.mtx.gz"), mtx(f"{RUN}/intron_counts.mtx.gz")
    fe, fi = lines(f"{RUN}/exon_counts_features.tsv.gz"), lines(f"{RUN}/intron_counts_features.tsv.gz")
    be, bi = lines(f"{RUN}/exon_counts_barcodes.tsv.gz"), lines(f"{RUN}/intron_counts_barcodes.tsv.gz")
    assert be == bi
    ge, gi = [f.split("_", 1)[0] for f in fe], [f.split("_", 1)[0] for f in fi]
    genes = sorted(set(ge) | set(gi))
    ix = {g: i for i, g in enumerate(genes)}

    def agg(M, g):
        r = np.array([ix[x] for x in g])
        A = sp.csr_matrix((np.ones(len(r)), (r, np.arange(len(r)))), shape=(len(genes), len(r)))
        out_ = (A @ M.astype(np.float64)).tocsc()
        out_.data = np.rint(out_.data)
        return out_

    Eg, Ig = agg(E, ge), agg(I, gi)
    if os.environ.get("CELLS") == "run":  # the production run's own cell set (ambient_contamination.tsv = cc_result.cell_indices)
        called = set(pd.read_csv(f"{RUN}/ambient_contamination.tsv", sep="\t").barcode.astype(str))
    else:
        cc = pd.read_csv(f"{RUN}/cell_calls.tsv", sep="\t")
        called = set(cc.loc[cc.is_cell.astype(str).str.upper() == "TRUE", "barcode"].astype(str))
    bc = np.array(be)
    is_cell = np.array([b in called for b in bc])
    tot = np.asarray(Eg.sum(0)).ravel() + np.asarray(Ig.sum(0)).ravel()
    print("genes", len(genes), "barcodes", len(bc), "called", is_cell.sum(), flush=True)
    for T in [int(x) for x in os.environ.get("THRESH", " ".join(map(str, THRESH))).split()]:
        keep = np.flatnonzero(is_cell | (tot >= T)) if T else np.arange(len(bc))
        d = f"{out}/in_T{T}{os.environ.get('TAG', '')}"
        os.makedirs(d, exist_ok=True)
        for tag, M in (("exon", Eg[:, keep]), ("intron", Ig[:, keep])):
            M = M.tocsc()
            M.sort_indices()
            M.indptr.astype(np.int64).tofile(f"{d}/{tag}.indptr")
            M.indices.astype(np.int32).tofile(f"{d}/{tag}.indices")
            M.data.astype(np.uint32).tofile(f"{d}/{tag}.data")
        cols = np.flatnonzero(is_cell[keep]).astype(np.uint32)
        cols.tofile(f"{d}/cells.idx")
        nE, nI = Eg[:, keep].nnz, Ig[:, keep].nnz
        open(f"{d}/dims.txt", "w").write(f"{len(genes)} {len(keep)} {nE} {nI} {len(cols)} 100\n")
        pd.Series(bc[keep][cols]).to_csv(f"{d}/cells.txt", index=False, header=False)
        print(f"T={T}: {len(keep)} barcodes, pool candidates (non-cell, <=100 UMI): "
              f"{int(((~is_cell[keep]) & (tot[keep] <= 100)).sum())}", flush=True)


def score(out):
    AD, DP = mtx(f"{RUN}/private/snp_ad.mtx.gz").tocsc(), mtx(f"{RUN}/private/snp_dp.mtx.gz").tocsc()
    bcs = np.array(lines(f"{RUN}/private/snp_ad_barcodes.tsv.gz"))
    A = pd.read_csv(f"{RUN}/donor/donor_assignments.tsv", sep="\t").set_index("cell").reindex(bcs)
    t = pd.read_csv(TRUTH, sep="\t", index_col=0)
    t = t[t["stim"] == "ctrl"]
    t.index = [i.split("-")[0] for i in t.index]
    short = np.array([b.split("-")[0] for b in bcs])
    tm = t.reindex(short)
    conf = (A.donor_id.astype(str).str.startswith("donor") & (A.prob_max >= 0.99) & (A.cell_status == "cell")).values
    singlet = conf & (tm.multiplets.values == "singlet")
    amb = (A.cell_status == "ambient").values
    donor = A.donor_id.astype(str).values
    print("confident truth singlets", singlet.sum(), "ambient barcodes", amb.sum(), flush=True)
    # ambient pool allele frequency per site
    ADr, DPr = AD.tocsr(), DP.tocsr()
    a_amb = np.asarray(ADr[:, amb].sum(1)).ravel()
    d_amb = np.asarray(DPr[:, amb].sum(1)).ravel()
    f = np.where(d_amb >= 10, a_amb / np.maximum(d_amb, 1), np.nan)
    # donor pseudobulk genotypes
    donors = sorted(set(donor[singlet]))
    G = np.full((AD.shape[0], len(donors)), np.nan)
    for j, dn in enumerate(donors):
        m = singlet & (donor == dn)
        a = np.asarray(ADr[:, m].sum(1)).ravel()
        dd = np.asarray(DPr[:, m].sum(1)).ravel()
        af = np.where(dd >= 8, a / np.maximum(dd, 1), np.nan)
        G[:, j] = np.where(af <= 0.1, 0.0, np.where(af >= 0.9, 1.0, np.nan))
    print("sites with pool AF", np.isfinite(f).sum(), "hom calls per donor", np.isfinite(G).sum(0), flush=True)
    cells = np.flatnonzero(singlet)
    dj = {dn: j for j, dn in enumerate(donors)}
    X = np.zeros(len(cells)); N = np.zeros(len(cells)); Q = np.zeros(len(cells))
    Xe = 0.0; Ne = 0.0
    rng = np.random.default_rng(0)
    half = rng.random(AD.shape[0]) < 0.5
    Xh = np.zeros((len(cells), 2)); Qh = np.zeros((len(cells), 2)); Nh = np.zeros((len(cells), 2))
    for k, c in enumerate(cells):
        s0, s1 = DP.indptr[c], DP.indptr[c + 1]
        sites = DP.indices[s0:s1]
        n = DP.data[s0:s1].astype(float)
        a = np.zeros(len(sites))
        a0, a1 = AD.indptr[c], AD.indptr[c + 1]
        pos = np.searchsorted(sites, AD.indices[a0:a1])
        a[pos] = AD.data[a0:a1]
        g = G[sites, dj[donor[c]]]
        fs = f[sites]
        ok = np.isfinite(g) & np.isfinite(fs)
        x = np.where(g == 0, a, n - a)
        q = np.where(g == 0, fs, 1 - fs)
        inf = ok & (q >= 0.05)
        err = ok & (q < 0.01)
        Xe += x[err].sum(); Ne += n[err].sum()
        X[k], N[k], Q[k] = x[inf].sum(), n[inf].sum(), (n * q)[inf].sum()
        for h in (0, 1):
            m = inf & (half[sites] == h)
            Xh[k, h], Nh[k, h], Qh[k, h] = x[m].sum(), n[m].sum(), (n * q)[m].sum()
    e = Xe / max(Ne, 1)
    print(f"error rate e = {e:.5f} from {Ne:.0f} reads at sites with pool q < 0.01", flush=True)
    rho_t = (X - e * N) / np.maximum(Q, 1e-9)
    D = pd.DataFrame(dict(barcode=bcs[cells], donor=donor[cells], X=X, N=N, Q=Q, rho_truth=rho_t,
                          celltype=tm.cell.values[cells]))
    # production rho: the run's own output and the harness variants
    P = pd.read_csv(f"{RUN}/ambient_contamination.tsv", sep="\t").set_index("barcode")
    D["prod_run"] = P.rho.reindex(D.barcode).values
    D["n_umi"] = P.n_umi.reindex(D.barcode).values
    for T in THRESH:
        rb = f"{out}/rho_T{T}.f64"
        if os.path.exists(rb):
            r = np.fromfile(rb, np.float64)
            cb = pd.read_csv(f"{out}/in_T{T}/cells.txt", header=None)[0].values
            D[f"prod_T{T}"] = pd.Series(r, index=cb).reindex(D.barcode).values
    D.to_csv(f"{out}/cells.csv.gz", index=False)

    Xa, Na, Qa = D.X.values, D.N.values, D.Q.values

    def pooled(m):
        idx = np.flatnonzero(m) if m.dtype == bool else m
        return (Xa[idx].sum() - e * Na[idx].sum()) / Qa[idx].sum()

    def boot(m, B=300):
        idx = np.flatnonzero(m)
        v = [pooled(rng.choice(idx, len(idx))) for _ in range(B)]
        return np.percentile(v, [2.5, 97.5])

    est = [c for c in D.columns if c.startswith("prod_")]
    rows = []
    allm = np.ones(len(D), bool)
    lo, hi = boot(allm)
    r = dict(group="all cells", cells=len(D), truth=pooled(allm), truth_lo=lo, truth_hi=hi)
    for c in est:
        r[c + "_mean"] = D[c].mean()
        r[c + "_Qweighted"] = np.average(D[c].fillna(0), weights=D.Q)
    rows.append(r)
    for ct, m in D.groupby("celltype").groups.items():
        m = D.index.isin(m)
        if m.sum() < 100:
            continue
        lo, hi = boot(m, 200)
        r = dict(group=f"celltype: {ct}", cells=int(m.sum()), truth=pooled(m), truth_lo=lo, truth_hi=hi)
        for c in est:
            r[c + "_mean"] = D[c][m].mean()
            r[c + "_Qweighted"] = np.average(D[c][m].fillna(0), weights=D.Q[m])
        rows.append(r)
    dec = pd.qcut(D.prod_run.rank(method="first"), 5, labels=False)
    for b in range(5):
        m = (dec == b).values
        lo, hi = boot(m, 200)
        r = dict(group=f"production rho quintile {b + 1}", cells=int(m.sum()), truth=pooled(m), truth_lo=lo, truth_hi=hi)
        for c in est:
            r[c + "_mean"] = D[c][m].mean()
            r[c + "_Qweighted"] = np.average(D[c][m].fillna(0), weights=D.Q[m])
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(f"{out}/calibration.csv", index=False)
    # per-cell agreement and the truth's own split-half reliability
    rh = (Xh - e * Nh) / np.maximum(Qh, 1e-9)
    okh = (Qh > 0).all(1)
    sh = spearmanr(rh[okh, 0], rh[okh, 1])[0]
    corr = {c: spearmanr(D[c], D.rho_truth, nan_policy="omit")[0] for c in est}
    summ = dict(error_rate=e, cells=len(D), median_informative_reads=float(np.median(N)), median_Q=float(np.median(Q)),
                split_half_spearman=sh, split_half_spearman_brown=2 * sh / (1 + sh), per_cell_spearman=corr,
                harness_vs_run_spearman=spearmanr(D.prod_run, D.get("prod_T0"), nan_policy="omit")[0] if "prod_T0" in D else None)
    for T in THRESH:
        j = f"{out}/run_T{T}.json"
        if os.path.exists(j):
            summ[f"harness_T{T}"] = json.load(open(j))
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(json.dumps(summ, indent=1, default=float))
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    {"dump": dump, "score": score}[sys.argv[1]](sys.argv[2])
