#!/bin/bash
#SBATCH --job-name=hto-i1diag
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Why is the top tag unrelated to the vireo donor on I1? (diagnostic only; no caller)
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/adt-bench/bin:$PATH
D=$W/expression-sota/hto/data/stoeckius2018
python -W ignore - <<'EOF'
import pandas as pd, numpy as np
D = "/mnt/projects/debruinz_project/debruinz/singlet/expression-sota/hto/data/stoeckius2018/"
t = pd.read_csv(D + "vireo_full/vireo/donor_ids.tsv", sep="\t")
print("vireo full: n_vars quantiles", t.n_vars.quantile([.1, .5, .9]).to_dict(), "prob_max median", t.prob_max.median())
print(t.donor_id.value_counts().to_dict())
cells = pd.read_csv(D + "singlet_full/cell_calls.tsv", sep="\t")
called = set(cells[cells.is_cell.astype(str).str.upper() == "TRUE"].barcode)
H = pd.read_csv(D + "geo/GSM2895283_Hashtag-HTO-count.csv.gz", index_col=0)
H = H.loc[[r for r in H.index if r.startswith("Batch")]]
H.columns = [c.replace("-1", "") for c in H.columns]
print("HTO table barcodes", H.shape[1], "called", len(called), "overlap", len(called & set(H.columns)))
tt = t.assign(c=t["cell"].str.replace("-1", "", regex=False)).set_index("c")
common = [c for c in H.columns if c in tt.index and tt.loc[c, "donor_id"].startswith("donor")]
X = H[common]
mx = X.max(0); srt = np.sort(X.values, 0)
clean = (mx >= 300) & (srt[-2] < 0.2 * srt[-1])
print("cells with donor and a clean dominant tag:", int(clean.sum()), "of", len(common))
ct = pd.crosstab(X.idxmax(0)[clean], tt.loc[X.columns[clean.values], "donor_id"])
print(ct.to_string())
print("max-per-row fraction", ct.max(1).sum() / ct.values.sum())
# same, but vireo donors vs RNA UMI knee: does the HTO table barcodes include most called cells with strong tags?
print("called cells in table with max tag>=300:", int(((H.columns.isin(called)) & (H.max(0) >= 300)).sum()))
EOF
