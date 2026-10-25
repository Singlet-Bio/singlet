"""Build the doublet datasets (ds.npz, see dbl_common).  Usage: prep.py xili|kang|hgmm|homsim

xili    the 16 Xi & Li 2021 real datasets ($LQC/dbl/data/<name>/matrix.mtx from xili_convert.R)
kang    Kang 2018 lane 2.1 as Singlet counts it (prod/modality-fixes, gene level, called cells) with demuxlet truth; extras:
        vireo genotype doublet probability, Singlet's genotype doublet probability (fixed demux), Singlet's expression score
hgmm    hgmm1k v3 (Singlet counts) + 44 synthetic human+mouse doublets (ambient benchmark)
homsim  Kang truth singlets + 5% homotypic and 5% heterotypic synthetic doublets (same / different coarse cluster)
"""
import glob
import gzip
import os
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from dbl_common import DATA, load, save, simulate_doublets, lognorm, hvg

W = "/mnt/projects/debruinz_project/debruinz/singlet"
MB = f"{W}/modality-bench"


def species_of(genes) -> str:
    g = pd.Series(genes).astype(str)
    hum = g.str.match(r"^(ENSG\d|hg19_|GRCh38_|hg38_)").mean()
    mou = g.str.match(r"^(ENSMUSG\d|mm10_|mm9_|GRCm3)").mean()
    if hum > 0.2 and mou > 0.2:
        return "mixed"
    if mou > 0.2:
        return "mouse"
    if hum > 0.2:
        return "human"
    # symbols: mouse symbols are Capitalised (Gapdh), human upper case (GAPDH)
    up = g.str.match(r"^[A-Z][A-Z0-9\-]+$").mean()
    cap = g.str.match(r"^[A-Z][a-z0-9][a-z0-9\-]*$").mean()
    return "mouse" if cap > up else "human"


def xili():
    for d in sorted(glob.glob(f"{DATA}/*/matrix.mtx")):
        d = os.path.dirname(d)
        name = os.path.basename(d)
        M = scipy.io.mmread(f"{d}/matrix.mtx").tocsc()  # genes x cells
        genes = [l.rstrip("\n") for l in open(f"{d}/genes.txt")]
        cells = [l.rstrip("\n") for l in open(f"{d}/cells.txt")]
        lab = np.array([l.strip() == "doublet" for l in open(f"{d}/labels.txt")], np.int8)
        X = sp.csr_matrix(M.T).astype(np.int32)
        sp_ = species_of(genes)
        save(name, X, genes, cells, lab, sp_, meta=dict(source="Xi & Li 2021 (Zenodo 4062232)", genes_example=genes[:5]))


def read_bin(pre):
    nr, nc, nnz = np.fromfile(pre + ".hdr", np.uint32)
    ip = np.fromfile(pre + ".indptr", np.int32)
    ix = np.fromfile(pre + ".indices", np.int32)
    dv = np.fromfile(pre + ".data", np.uint16)
    return sp.csc_matrix((dv.astype(np.int32), ix, ip), shape=(nr, nc))


def kang():
    K = f"{MB}/doublet_fix/kang/k8g"
    G = read_bin(K)  # genes x cells
    bcs = np.array([l.strip().split("-")[0] for l in open(K + ".barcodes.tsv")])
    with gzip.open(f"{MB}/genotype_pool/singlet_k8/filtered_feature_bc_matrix/features.tsv.gz", "rt") as f:
        genes = np.array([l.split("\t")[0] for l in f])
    assert len(genes) == G.shape[0], (len(genes), G.shape)
    t = pd.read_csv(f"{MB}/genotype_pool/truth/GSE96583_batch2.total.tsne.df.tsv.gz", sep="\t", index_col=0)
    t = t[t["stim"] == "ctrl"].copy()
    t.index = [i.split("-")[0] for i in t.index]
    S = pd.read_csv(f"{MB}/genotype_pool/results/doublets_k8_scores.csv.gz", index_col=0)
    D = pd.read_csv(f"{MB}/demux_fix/final/k8/donor_assignments.tsv", sep="\t", index_col=0)
    D.index = [i.split("-")[0] for i in D.index]
    P = pd.read_csv(f"{MB}/doublet_fix/kang/k8g_final.tsv", sep="\t")
    df = pd.DataFrame(dict(bc=bcs, col=np.arange(len(bcs)), prod_score=P.score.values)).set_index("bc")
    df = df.join(t[["multiplets", "cell"]], how="inner")
    df = df[df.multiplets.isin(["singlet", "doublet"])]
    df = df.join(S[["vireo_geno", "scrublet"]], how="left").join(D[["prob_doublet"]], how="left")
    X = sp.csr_matrix(G[:, df.col.values].T)
    save("kang21", X, genes, df.index.values, (df.multiplets == "doublet").values, "human",
         meta=dict(source="Kang 2018 GSE96583 lane 2.1, Singlet prod/modality-fixes counts, demuxlet truth"),
         x_vireo_geno=df.vireo_geno.fillna(0).values, x_prod_geno=df.prob_doublet.fillna(0).values,
         x_prod_score_e2e=df.prod_score.values, x_author_celltype=df.cell.astype(str).values)


def hgmm():
    A = f"{MB}/ambient/work"
    M = scipy.io.mmread(f"{A}/dbl/gene_synth.mtx").tocsc()
    genes = pd.read_csv(f"{A}/genes.csv").id.values
    assert len(genes) == M.shape[0]
    tr = pd.read_csv(f"{A}/dbl/truth.csv")
    save("hgmm1k_synth", sp.csr_matrix(M.T).astype(np.int32), genes, tr.col.astype(str).values, tr.truth.values.astype(int),
         "mixed", meta=dict(source="hgmm1k v3 Singlet counts, 1 natural + 44 synthetic human+mouse doublets"),
         x_synthetic=tr.synthetic.values.astype(int))


def homsim():
    import scanpy as sc

    d = load("kang21")
    keep = d["label"] == 0
    X = d["X"][keep]
    rng = np.random.default_rng(7)
    ad = sc.AnnData(lognorm(X)[:, hvg(X, 2000)])
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, n_comps=30)
    sc.pp.neighbors(ad, n_neighbors=15)
    sc.tl.leiden(ad, resolution=0.3, flavor="igraph", n_iterations=2, random_state=0)
    grp = ad.obs.leiden.astype(int).values
    n = int(0.05 * X.shape[0])
    Sh, ph = simulate_doublets(X, n, rng, grp, True)
    Se, pe = simulate_doublets(X, n, rng, grp, False)
    Xa = sp.vstack([X, Sh, Se]).tocsr()
    typ = np.r_[np.zeros(X.shape[0]), np.ones(n), np.full(n, 2)].astype(np.int8)
    cells = np.r_[d["cells"][keep], [f"hom{i}" for i in range(n)], [f"het{i}" for i in range(n)]]
    ct = d["x_author_celltype"][keep]
    parents_ct = np.r_[ct, [f"{ct[a]}+{ct[b]}" for a, b in ph], [f"{ct[a]}+{ct[b]}" for a, b in pe]]
    print("coarse clusters", np.bincount(grp))
    save("kang21_homsim", Xa, d["genes"], cells, (typ > 0).astype(int), "human",
         meta=dict(source="Kang lane 2.1 truth singlets + synthetic doublets within / across leiden(0.3) clusters"),
         x_dbl_type=typ, x_author_celltype=parents_ct, x_cluster=np.r_[grp, grp[ph[:, 0]], grp[pe[:, 0]]])


if __name__ == "__main__":
    {"xili": xili, "kang": kang, "hgmm": hgmm, "homsim": homsim}[sys.argv[1]]()
