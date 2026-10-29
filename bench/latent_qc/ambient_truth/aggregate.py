"""Aggregate results/raw/*.csv (pulled from Clipper) into results/*.csv and print the markdown tables used in RESULTS.md.
Local, pandas only:  python aggregate.py"""
import glob
import os

import pandas as pd

D = os.path.dirname(os.path.abspath(__file__)) + "/results"
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)


def cat(suffix):
    fs = sorted(glob.glob(f"{D}/raw/*__{suffix}.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


est = pd.concat([cat(s) for s in ("est_full-full", "est_A-B", "est_B-A", "ext_full-full", "ext_A-B", "ext_B-A")], ignore_index=True)
est["tag"] = est["tag"].fillna("ext")
cor = pd.concat([cat("corr_full"), cat("extcorr_full")], ignore_index=True)
est.to_csv(f"{D}/estimates_all.csv", index=False)
cor.to_csv(f"{D}/correction_all.csv", index=False)
ceil = pd.read_csv(f"{D}/raw/ceiling.csv")
ceil.to_csv(f"{D}/truth_noise.csv", index=False)

LIBS = ["hgmm1k_v3_singlet", "hgmm1k_v3", "hgmm5k_v3", "hgmm10k_v3", "hgmm1k_v2", "hgmm6k_v2", "hgmm12k_v2"]
main = est[est.tag.isin(["eval", "ext"])].copy()
main["method"] = main.method.replace({"production_t1": "production", "production_t16": "production_t16"})
main = main[main.method != "production_t16"]
full = main[main.view == "full->full"]
split = main[main.view.isin(["A->B", "B->A"])].groupby(["lib", "method"])[["spearman", "pearson", "ratio"]].mean().reset_index()
for name, d, val in (("full-gene truth: Spearman", full, "spearman"), ("full-gene truth: Pearson", full, "pearson"),
                     ("full-gene truth: mean ratio to truth", full, "ratio"),
                     ("gene-split truth (mean of A->B, B->A): Spearman", split, "spearman"),
                     ("gene-split truth: Pearson", split, "pearson"), ("gene-split truth: ratio", split, "ratio")):
    print(f"\n### {name}")
    print(d.pivot_table(index="method", columns="lib", values=val)[LIBS].round(3).to_string())
print("\n### truth noise")
print(ceil.round(3).to_string())
print("\n### correction (full-gene)")
print(cor.round(3).to_string())
print("\n### depth thirds (full-gene)")
print(full[["lib", "method", "ratio", "est_low", "truth_low", "est_mid", "truth_mid", "est_high", "truth_high"]].round(4).to_string())
cost = est[est.tag == "cost"]
print("\n### cost (1 thread, own process)\n", cost[["lib", "method", "seconds", "maxrss_gb", "n_cells", "n_barcodes"]].round(2).to_string())
ext = est[(est.tag == "ext") & (est.view == "full->full")]
print("\n### external cost\n", ext[["lib", "method", "seconds", "maxrss_gb", "n_cells", "n_barcodes"]].round(2).to_string())

# ---- compact tables for RESULTS.md ----
print("\n\n===== COMPACT =====")
cor = cor.drop_duplicates(["lib", "method"], keep="last")
KEEP = ["production", "decontx_background", "decontx_default", "knn:k=30:tau=10000:kappa=1000:hier=1", "knn:k=50:tau=10000:kappa=1000:hier=1",
        "knn:k=30:tau=1000:kappa=1000:hier=0", "cluster:tau=100:kappa=1000", "knn:k=30:tau=10000:kappa=1000:hier=1:latent=vae"]
c = cor[cor.method.isin(KEEP)]
print("\ncross-species residual % (lower = more removed)\n", c.pivot_table(index="method", columns="lib", values="cross_species_residual_pct")[LIBS].round(1).to_string())
print("\nown-species removed / expected own-ambient\n", c.pivot_table(index="method", columns="lib", values="own_removed_over_expected")[LIBS].round(2).to_string())
print("\ntotal UMIs removed %\n", c.pivot_table(index="method", columns="lib", values="total_removed_pct")[LIBS].round(2).to_string())
f = full[full.method.isin(KEEP)]
for b in ("low", "mid", "high"):
    print(f"\nmean estimate / mean truth in the {b} depth third\n", (f.assign(r=f[f"est_{b}"] / f[f"truth_{b}"]).pivot_table(index="method", columns="lib", values="r")[LIBS]).round(2).to_string())
print("\nmean truth (%)", (full.groupby("lib").mean_truth.first()[LIBS] * 100).round(2).to_dict())
print("\nheld-out mean over the 5 libs not used for tuning (5k_v3, 10k_v3, 1k_v2, 6k_v2, 12k_v2)")
HO = ["hgmm5k_v3", "hgmm10k_v3", "hgmm1k_v2", "hgmm6k_v2", "hgmm12k_v2"]
for nm, d in (("full", full), ("split", split)):
    h = d[d.lib.isin(HO)].groupby("method")[["spearman", "pearson", "ratio"]].mean().round(3)
    print(nm, "\n", h.to_string())
cost = est[est.tag == "cost"]
print("\ncost\n", cost[["lib", "method", "seconds", "maxrss_gb", "n_cells", "n_barcodes"]].round(2).to_string())
