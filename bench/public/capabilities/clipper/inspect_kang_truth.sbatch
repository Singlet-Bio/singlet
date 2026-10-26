#!/bin/bash
#SBATCH --job-name=bp-ktruth
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Structure of the Kang 2018 demuxlet truth tables (columns, lanes, donors, multiplet labels).
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/ambient/bin:$PATH
python -I - <<'EOF'
import pandas as pd
D = "/mnt/projects/debruinz_project/debruinz/singlet/bench-public/data/kang_truth"
for b in ("batch1", "batch2"):
    t = pd.read_csv(f"{D}/GSE96583_{b}.total.tsne.df.tsv.gz", sep="\t", index_col=0)
    print("=====", b, t.shape); print(t.head(3).to_string()); print(t.dtypes.to_string())
    for c in t.columns:
        if t[c].nunique() < 40: print(c, t[c].value_counts(dropna=False).to_dict())
    print("barcode suffix:", pd.Series([i.split("-")[-1] if "-" in i else "" for i in t.index]).value_counts().to_dict())
    print("duplicated barcodes (ignoring suffix):", pd.Series([i.split("-")[0] for i in t.index]).duplicated().sum())
    print(pd.crosstab(t.get("stim", t.iloc[:, 0]), t.multiplets))
    if "ind" in t: print(pd.crosstab(t.get("stim", t.iloc[:, 0]), t.ind.fillna(-1)))
EOF
