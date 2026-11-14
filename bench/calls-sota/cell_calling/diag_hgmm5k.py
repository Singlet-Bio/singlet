# SPDX-License-Identifier: MIT
"""Post-hoc diagnostic (not part of the gate): where Singlet and EmptyDrops_CR differ on hgmm5k_v3. python -P diag_hgmm5k.py TRUTH_DIR CV2_PARAMS"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/jobs/cellcall_truth")
import score as SC
td, pp = sys.argv[1:3]
t = pd.read_csv(f"{td}/hgmm5k_v3.truth.tsv.gz", sep="\t", dtype={"barcode": str}, low_memory=False)
_, p = SC.mixture_score(t, nb=2)
u = t.merge(p, on="barcode")
print(open(pp).read())
S, E, C = u.called_v2 == 1, u.edcr == 1, u.cellbender == 1
for nm, m in (("edcr_only", E & ~S), ("singlet_only", S & ~E), ("both", S & E), ("cellbender_not_singlet", C & ~S)):
    g = u[m]
    print(nm, len(g), "umi q10/50/90", np.percentile(g.umi_total, [10, 50, 90]).round(0) if len(g) else None, "mean post", round(g.post.mean(), 3) if len(g) else None,
          "truth", g.truth.value_counts().to_dict(), "knee", int(g.knee.sum()))
print("post mass of cells not called by singlet by UMI bin:")
b = pd.cut(u.umi_total, [50, 100, 200, 300, 500, 1000, 2000, 1e9], right=False)
print(u.assign(b=b, miss=(~S) * u.post, e=E * 1).groupby("b", observed=True).agg(n=("post", "size"), post=("post", "sum"), miss=("miss", "sum"), singlet=("called_v2", "sum"), edcr=("e", "sum")).round(0))
