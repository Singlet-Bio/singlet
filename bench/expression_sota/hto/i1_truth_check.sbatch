#!/bin/bash
#SBATCH --job-name=hto-i1truth
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# I1 truth only (no caller): vireo outputs and the loader of eval_independent.py.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/adt-bench/bin:$PATH
D=$W/expression-sota/hto/data/stoeckius2018
cat $D/vireo/vireo/summary.tsv
python -W ignore - <<'EOF'
import pandas as pd, numpy as np
D = "/mnt/projects/debruinz_project/debruinz/singlet/expression-sota/hto/data/stoeckius2018/vireo/vireo/"
t = pd.read_csv(D + "donor_ids.tsv", sep="\t")
print(t.donor_id.value_counts().to_dict())
print("prob_doublet quantiles", t.prob_doublet.quantile([.5, .75, .9, .95, .99]).round(4).to_dict())
for th in (0.1, 0.2, 0.3, 0.5, 0.9):
    print("prob_doublet >=", th, int((t.prob_doublet >= th).sum()))
print("n_vars quantiles", t.n_vars.quantile([.1, .5, .9]).to_dict())
print("doublet_logLikRatio >0:", int((t.doublet_logLikRatio > 0).sum()))
print("unassigned by n_vars", t[t.donor_id == "unassigned"].n_vars.median(), "assigned", t[t.donor_id != "unassigned"].n_vars.median())
EOF
