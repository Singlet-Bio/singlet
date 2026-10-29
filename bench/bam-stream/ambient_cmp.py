"""Does the ambient fit agree between two runs (stream vs sorted)? (ws-w3b update i, order-free-rules R10)

Usage: python -I ambient_cmp.py <A out dir> <B out dir> <out.json>
Compares ambient_contamination.json (pool size, pool UMIs, mean/median rho), ambient_profile.tsv (L1 distance and
correlation of the profiles) and ambient_contamination.tsv (per-cell rho over the cells both runs called, by barcode).
"""
import json, sys
import numpy as np, pandas as pd

a, b, out = sys.argv[1:4]
res = {}
ja = json.load(open(f"{a}/ambient_contamination.json")); jb = json.load(open(f"{b}/ambient_contamination.json"))
keys = ["method", "n_cells", "n_clusters", "mean_rho", "median_rho", "umi_weighted_rho", "umis_removed",
        "ambient_profile_n_barcodes", "ambient_profile_total_umis", "ambient_profile_n_genes_detected",
        "ambient_pool_min_umi", "ambient_pool_max_umi"]
res["json"] = {k: [ja.get(k), jb.get(k)] for k in keys}
pa = pd.read_csv(f"{a}/ambient_profile.tsv", sep="\t").set_index("feature")["ambient_fraction"]
pb = pd.read_csv(f"{b}/ambient_profile.tsv", sep="\t").set_index("feature")["ambient_fraction"]
pa, pb = pa.align(pb, fill_value=0.0)
res["profile"] = {"l1": float((pa - pb).abs().sum()), "max_abs": float((pa - pb).abs().max()),
                  "pearson": float(np.corrcoef(pa, pb)[0, 1]), "identical": bool((pa == pb).all())}
ca = pd.read_csv(f"{a}/ambient_contamination.tsv", sep="\t").set_index("barcode")
cb = pd.read_csv(f"{b}/ambient_contamination.tsv", sep="\t").set_index("barcode")
both = ca.index.intersection(cb.index)
ra, rb = ca.loc[both, "rho"], cb.loc[both, "rho"]
d = (ra - rb).abs()
res["cells"] = {"a": len(ca), "b": len(cb), "both": len(both),
                "rho_identical": int((d == 0).sum()), "rho_abs_diff_quantiles_50_90_99_max":
                [float(d.quantile(q)) for q in (0.5, 0.9, 0.99)] + [float(d.max())],
                "rho_pearson": float(np.corrcoef(ra, rb)[0, 1]) if len(both) > 1 else None,
                "cluster_same_share": float((ca.loc[both, "cluster"] == cb.loc[both, "cluster"]).mean())}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
