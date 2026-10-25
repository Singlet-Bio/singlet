"""Three figures from results/ (run locally; small CSVs only). usage: make_figs.py RESULTS_DIR OUT_DIR"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

res, out = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# 1. structure: PCA of ambient and called-cell profiles, coloured by tissue
lab = pd.read_csv(f"{res}/pc_labels.csv")
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
tis = sorted(lab.tissue_coarse.unique())
cm = plt.get_cmap("tab20")
for k, (nm, f) in enumerate((("ambient pool profiles", "pc_ambient.npy"), ("called-cell profiles", "pc_cells.npy"))):
    Z = np.load(f"{res}/{f}")
    for i, t in enumerate(tis):
        m = (lab.tissue_coarse == t).values
        ax[k].scatter(Z[m, 0], Z[m, 1], s=9, color=cm(i), label=t, alpha=.8)
    ax[k].set_title(nm + " (PC1, PC2 of log2 CPM)")
ax[0].legend(fontsize=6, ncol=2, frameon=False)
st = pd.read_csv(f"{res}/structure_knn.csv")
st = st[st.label.isin(["tissue", "chemistry", "nuclei"])]
x = np.arange(3)
for j, (p, c) in enumerate((("ambient", "#c0504d"), ("cells", "#4f81bd"))):
    s = st[st.profile == p].set_index("label").loc[["tissue", "chemistry", "nuclei"]]
    ax[2].bar(x + j * 0.35 - 0.175, s.knn5_loso_acc, 0.33, color=c, label=p)
    if j == 0:
        ax[2].scatter(x, s.majority_frac, color="k", marker="_", s=300, label="majority class")
ax[2].set_xticks(x); ax[2].set_xticklabels(["tissue", "chemistry", "nuclei vs cells"]); ax[2].set_ylabel("5-NN accuracy, other studies only")
ax[2].legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{out}/fig1_structure.png", dpi=130); plt.close()

# 2. prior test: KL vs number of UMIs from the held-out sample's own pool
R = pd.read_csv(f"{res}/prior_test.csv")
NS = [100, 300, 1000, 3000, 10000, 30000]
ok = R[(R.method == "emp_alone") & (R.n == 30000)].gsm.unique()
R = R[R.gsm.isin(ok)]
R.loc[R.method.str.startswith("raw_"), "kl"] = np.nan
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, (met, yl) in enumerate((("kl", "KL(held-out half-pool || prediction), bits"), ("tv", "total variation"))):
    for m, c, nm in (("emp_alone", "k", "n UMIs, empirical (+tuned pseudocount)"), ("emp_tissue", "#e69f00", "n UMIs + tissue-mean prior"), ("emp_mix", "#c0504d", "n UMIs + best atlas prior")):
        y = [R[(R.method == m) & (R.n == n)].groupby("gsm")[met].mean().mean() for n in NS]
        ax[k].plot(NS, y, "-o", color=c, label=nm)
    for m, ls, nm in (("prior_cell", ":", "own cells only"), ("prior_mix", "--", "atlas prior (cells x enrichment + tissue + global)"), ("prior_tissue", "-.", "tissue mean ambient (other studies)")):
        ax[k].axhline(R[R.method == m].groupby("gsm")[met].mean().mean(), color="gray", ls=ls, label="no pool UMIs: " + nm)
    ax[k].set_xscale("log"); ax[k].set_xlabel("UMIs drawn from the sample's own ambient pool (half A)"); ax[k].set_ylabel(yl)
ax[0].set_yscale("log")
ax[0].legend(fontsize=6.5, frameon=False)
ax[0].set_title(f"{R.gsm.nunique()} samples, study-grouped 20-fold")
plt.tight_layout(); plt.savefig(f"{out}/fig2_prior_test.png", dpi=130); plt.close()

# 3. production rho by tissue and nuclei
D = pd.read_csv(f"{res}/rho_by_sample.csv")
D = D[D.prod_ok == 1]
order = D.groupby("tissue_coarse").prod_mean_rho.median().sort_values().index
fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]})
ax[0].boxplot([D[D.tissue_coarse == t].prod_mean_rho for t in order], labels=order, showfliers=True, flierprops={"markersize": 2})
ax[0].set_ylabel("mean rho per sample (production estimator)"); ax[0].tick_params(axis="x", rotation=60)
ax[1].boxplot([D[D.nuclei == t].prod_mean_rho for t in ("cells", "nuclei")], labels=[f"cells (n={int((D.nuclei == 'cells').sum())})", f"nuclei (n={int((D.nuclei == 'nuclei').sum())})"])
plt.tight_layout(); plt.savefig(f"{out}/fig3_rho.png", dpi=130); plt.close()
