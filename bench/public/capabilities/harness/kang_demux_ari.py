# SPDX-License-Identifier: MIT
"""Donor-label agreement of Singlet's automatic donor count with demuxlet truth, and which truth donor Singlet split or merged.

    python -I kang_demux_ari.py RUN_DIR TRUTH_TSV LANE_FILTER      (e.g. stim=ctrl, batch=C)

ARI over the truth singlets that Singlet called (unassigned and doublet calls count as their own labels), plus the cross-tabulation of
Singlet donor labels against truth donors. Labels only; no genotypes.
"""
import sys

import pandas as pd
from sklearn.metrics import adjusted_rand_score

run, truth_tsv, lane = sys.argv[1:4]
a = pd.read_csv(f"{run}/donor/donor_assignments.tsv", sep="\t")
a["cell"] = a.cell.astype(str).str.replace(r"-1$", "", regex=True)
a = a.drop_duplicates("cell").set_index("cell")
t = pd.read_csv(truth_tsv, sep="\t", index_col=0)
col, val = lane.split("=")
t = t[t[col].astype(str) == val]
t.index = [i.rsplit("-", 1)[0] for i in t.index]
j = a.join(t[["multiplets", "ind"]], how="inner")
j = j[(j.multiplets == "singlet") & (j.cell_status == "cell")]
j = j[j["ind"].notna()].copy()
j["ind"] = j["ind"].astype(int).astype(str)
print("truth singlets scored:", len(j), " Singlet donors:", sorted(set(j.donor_id) - {"doublet", "unassigned"}))
print("ARI (all truth singlets, doublet/unassigned as labels):", round(adjusted_rand_score(j.ind, j.donor_id), 4))
k = j[j.donor_id.str.startswith("donor")]
print("ARI (assigned only):", round(adjusted_rand_score(k.ind, k.donor_id), 4), " assigned fraction:", round(len(k) / len(j), 4))
ct = pd.crosstab(j.ind, j.donor_id)
print(ct.to_string())
split = {d: ct.loc[d][ct.loc[d] > 0.05 * ct.loc[d].sum()].to_dict() for d in ct.index}
print("truth donors whose cells land in >1 Singlet cluster (each with > 5% of the donor):",
      {d: v for d, v in split.items() if len([x for x in v if x.startswith('donor')]) > 1})
