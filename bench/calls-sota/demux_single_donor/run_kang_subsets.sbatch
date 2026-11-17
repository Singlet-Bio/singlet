#!/bin/bash
#SBATCH --job-name=kang_single
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Single-donor test from the Kang pool: run the v1.3.7 demultiplexer (auto-K) on the truth singlets of one donor at a time.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
G=$W/modality-bench/genotype_pool
O=$W/bench/calls-sota/demux_single_donor/kang_subsets; mkdir -p $O
export PATH=$W/envs/pbench/bin:$PATH
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
git clone -q --depth 1 --branch v1.3.7 https://github.com/Singlet-Bio/singlet.git $S/eng || git clone -q --depth 1 --branch v1.3.7 git@github.com:Singlet-Bio/singlet.git $S/eng
echo "donor_demux.h blob: $(git hash-object $S/eng/include/singlet/pileup/donor_demux.h)"
P=$W/envs/pbench
x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -pthread -I$S/eng/include -I$P/include $S/eng/bench/modalities/demux_fix/demux_harness.cpp -o $S/harness -L$P/lib -Wl,-rpath,$P/lib -lzstd -lz
PY=$W/envs/gpool/venv/bin/python
$PY - <<PYEOF
import pandas as pd
t = pd.read_csv("$G/truth/GSE96583_batch2.total.tsne.df.tsv.gz", sep="\t", index_col=0)
t = t[(t["stim"] == "ctrl") & (t["multiplets"] == "singlet")]
t["bc"] = [i.split("-")[0] for i in t.index]; t["ind"] = t["ind"].astype("Int64").astype(str)
cc = pd.read_csv("$G/singlet_k8/cell_calls.tsv", sep="\t")
print(t["ind"].value_counts().to_dict())
for d, g in t.groupby("ind"):
    pd.DataFrame({"barcode": g["bc"], "is_cell": "TRUE"}).to_csv("$S/calls_%s.tsv" % d, sep="\t", index=False)
PYEOF
for f in $S/calls_*.tsv; do
  d=$(basename $f .tsv); d=${d#calls_}; mkdir -p $O/d$d
  CALLS_COL=is_cell $S/harness $G/singlet_k8/donor/snp_ad.mtx.gz $G/singlet_k8/donor/snp_dp.mtx.gz $G/singlet_k8/donor/snp_ad_barcodes.tsv.gz $f $O/d$d 8 sub_$d -1 2>&1 | grep -E "harness|K=|n_donors" | tail -4 || true
done
args=(); for d in $O/d*/; do p=$(ls $d/*/donor_assignments.tsv 2>/dev/null | head -1); [ -n "$p" ] && args+=("$p:kang_$(basename $d)"); done
apptainer exec -B /mnt $W/images/python-3.12-bookworm.sif $W/venvs/singlet/bin/python -I $W/bench/calls-sota/demux_single_donor/score_single_split.py $O/kang_single_split.json "${args[@]}"
