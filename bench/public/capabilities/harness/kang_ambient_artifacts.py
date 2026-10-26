# SPDX-License-Identifier: MIT
"""Real-tissue ambient truth from donor genotypes (Kang 2018 pool) and every tool's per-cell removal, as one compact artifact.

    python -I kang_ambient_artifacts.py ID RUN_DIR WORK_DIR ARTIFACT_ROOT TRUTH_TSV LANE_FILTER

Truth construction follows bench/latent_qc/ambient_kang/kang_ambient.py (branch study/latent-qc, PR #43), re-implemented here on this
benchmark's own Singlet run: in a donor's confident demuxlet singlet, a read carrying an allele the donor does not have (at a site where
the donor is homozygous in its own pooled reads) came from another donor's RNA through the ambient pool. With f_s the alt fraction of the
ambient pool (barcodes called ambient by Singlet) and e the sequencing error measured where the pool is monomorphic (q < 0.01),
    rho_c = (X_c - e N_c) / Q_c,   X = foreign reads, N = informative reads, Q = sum_s n_s q_s   (sites with q >= 0.05)
The artifact keeps only per-cell scalars (rho_truth, weight Q): no genotype, no per-site allele count leaves the run directory.
Weighted mean of rho_truth with weights Q equals the pooled estimate (sum X - e sum N) / sum Q.
Writes ARTIFACT_ROOT/ambient-rna/<id>_tissue.tsv.gz and <id>_tissue.json.
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barnyard_artifacts import read_10x_named  # noqa: E402


def mtx(path):
    with gzip.open(path, "rb") as f:
        return scipy.io.mmread(f).tocsc()


def lines(path):
    with gzip.open(path, "rt") as f:
        return [l.rstrip("\n").split("\t")[0] for l in f]


def main(ds, run, work, art, truth_tsv, lane):
    cells = pd.read_csv(f"{work}/cells.csv")
    cbc = cells.barcode.astype(str).values

    AD, DP = mtx(f"{run}/private/snp_ad.mtx.gz"), mtx(f"{run}/private/snp_dp.mtx.gz")
    bcs = np.array([b.rsplit("-", 1)[0] for b in lines(f"{run}/private/snp_ad_barcodes.tsv.gz")])
    A = pd.read_csv(f"{run}/donor/donor_assignments.tsv", sep="\t")
    A["cell"] = A.cell.astype(str).str.replace(r"-1$", "", regex=True)
    A = A.drop_duplicates("cell").set_index("cell").reindex(bcs)
    t = pd.read_csv(truth_tsv, sep="\t", index_col=0)
    col, val = lane.split("=")
    t = t[t[col].astype(str) == val]
    t.index = [i.rsplit("-", 1)[0] for i in t.index]
    tm = t.reindex(bcs)
    conf = (A.donor_id.astype(str).str.startswith("donor") & (A.prob_max >= 0.99) & (A.cell_status == "cell")).values
    in_called = np.isin(bcs, cbc)
    singlet = conf & (tm.multiplets.values == "singlet") & in_called
    amb = (A.cell_status == "ambient").values
    donor = A.donor_id.astype(str).values
    print("confident truth singlets", int(singlet.sum()), "ambient barcodes", int(amb.sum()), flush=True)

    ADr, DPr = AD.tocsr(), DP.tocsr()
    a_amb = np.asarray(ADr[:, amb].sum(1)).ravel()
    d_amb = np.asarray(DPr[:, amb].sum(1)).ravel()
    f = np.where(d_amb >= 10, a_amb / np.maximum(d_amb, 1), np.nan)
    donors = sorted(set(donor[singlet]))
    G = np.full((AD.shape[0], len(donors)), np.nan)
    for j, dn in enumerate(donors):
        m = singlet & (donor == dn)
        a = np.asarray(ADr[:, m].sum(1)).ravel()
        dd = np.asarray(DPr[:, m].sum(1)).ravel()
        af = np.where(dd >= 8, a / np.maximum(dd, 1), np.nan)
        G[:, j] = np.where(af <= 0.1, 0.0, np.where(af >= 0.9, 1.0, np.nan))
    dj = {dn: j for j, dn in enumerate(donors)}
    idx = np.flatnonzero(singlet)
    X = np.zeros(len(idx)); N = np.zeros(len(idx)); Q = np.zeros(len(idx))
    Xe = Ne = 0.0
    for k, c in enumerate(idx):
        s0, s1 = DP.indptr[c], DP.indptr[c + 1]
        sites = DP.indices[s0:s1]
        n = DP.data[s0:s1].astype(float)
        a = np.zeros(len(sites))
        a0, a1 = AD.indptr[c], AD.indptr[c + 1]
        a[np.searchsorted(sites, AD.indices[a0:a1])] = AD.data[a0:a1]
        g = G[sites, dj[donor[c]]]
        fs = f[sites]
        ok = np.isfinite(g) & np.isfinite(fs)
        x = np.where(g == 0, a, n - a)
        q = np.where(g == 0, fs, 1 - fs)
        inf, err = ok & (q >= 0.05), ok & (q < 0.01)
        Xe += x[err].sum(); Ne += n[err].sum()
        X[k], N[k], Q[k] = x[inf].sum(), n[inf].sum(), (n * q)[inf].sum()
    e = Xe / max(Ne, 1)
    rho_t = (X - e * N) / np.maximum(Q, 1e-9)
    D = pd.DataFrame({"barcode": bcs[idx], "celltype": tm["cell"].values[idx], "rho_truth": rho_t, "weight": Q})
    D = D[D.weight > 0]
    print(f"error rate {e:.5f}; cells {len(D)}; pooled truth {np.average(D.rho_truth, weights=D.weight):.4f}", flush=True)

    cells = cells.set_index("barcode")
    D["n_umi"] = cells.n_umi.reindex(D.barcode).values
    # Singlet: UMIs after its gene-level correction, and its native rho
    C, cid, _, cb = read_10x_named(run, "ambient_corrected_counts")
    after = pd.Series(np.asarray(C.sum(0)).ravel(), index=cb)
    D["singlet_after"] = after.reindex(D.barcode).values
    a_ = pd.read_csv(f"{run}/ambient_contamination.tsv", sep="\t")
    a_["barcode"] = a_.barcode.astype(str).str.replace(r"-1$", "", regex=True)
    D["singlet_rho"] = a_.set_index("barcode").rho.reindex(D.barcode).values
    for tname in ("soupx", "decontx", "decontx_bg"):
        p = f"{work}/tools/{tname}.csv"
        if os.path.exists(p):
            d = pd.read_csv(p).set_index("barcode")
            D[f"{tname}_after"] = (d.human_after + d.mouse_after).reindex(D.barcode).values
        else:
            print(f"[warn] {tname}: no output")
    for cbn in ("cellbender", "cellbender_rec"):
        p = f"{work}/{cbn}/cb.h5"
        if not os.path.exists(p):
            print(f"[warn] {cbn}: no output")
            continue
        import h5py
        with h5py.File(p, "r") as fh:
            g = fh["matrix"] if "matrix" in fh else fh
            hb = np.array([b.decode().rsplit("-", 1)[0] for b in g["barcodes"][:]])
            M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
        D[f"{cbn}_after"] = pd.Series(np.asarray(M.sum(0)).ravel(), index=hb).reindex(D.barcode).values
    os.makedirs(f"{art}/ambient-rna", exist_ok=True)
    D.to_csv(f"{art}/ambient-rna/{ds}_tissue.tsv.gz", sep="\t", index=False, float_format="%.6g")
    json.dump({"error_rate": float(e), "n_truth_singlets": int(len(D)), "median_informative_reads": float(np.median(N)),
               "lane_filter": lane, "truth": "genotype foreign-allele reads (Kang 2018 donors); see script docstring"},
              open(f"{art}/ambient-rna/{ds}_tissue.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:7])
