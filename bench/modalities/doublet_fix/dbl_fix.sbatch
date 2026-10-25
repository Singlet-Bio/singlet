#!/bin/bash
#SBATCH --job-name=dbl-fix
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# New detect_doublets() (doublet_detect.h compiled in the job from $J/inc) on the Kang pool and on the hgmm synthetic-doublet matrix.
#   sbatch --export=ALL,LABEL=default,ARGS="rate=0.1 hvg=2000" dbl_fix.sbatch      (ARGS are dbl_run key=value pairs)
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/modfix-doublet; O=$W/modality-bench/doublet_fix
K=$W/modality-bench/genotype_pool; A=$W/modality-bench/ambient/work/dbl
PY=$W/envs/gpool/venv/bin/python
export PATH=$W/envs/pbench/bin:$PATH
LABEL=${LABEL:-default}; ARGS=${ARGS:-}
mkdir -p $O/kang $O/hgmm $O/build $O/results
x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -pthread -I$J/inc $J/dbl_run.cpp -o $O/build/dbl_run_$LABEL

# ---- Kang pool (8 donors, demuxlet truth), called cells of singlet_k8 ----
[ -f $O/kang/k8.hdr ] || $PY $J/kang_prep.py $K/singlet_k8 $O/kang/k8
echo "== Kang  [$LABEL] $ARGS"
$O/build/dbl_run_$LABEL $O/kang/k8 $O/kang/$LABEL.tsv threads=16 $ARGS
$PY $J/kang_eval.py $K/truth/GSE96583_batch2.total.tsne.df.tsv.gz $O/kang/k8 $O/kang/$LABEL.tsv $K/results/doublets_k8_scores.csv.gz $O/results/kang_$LABEL.json singlet_new

# ---- hgmm1k + 44 synthetic doublets (matrix written by ambient/dbl_synth.py) ----
if [ ! -f $O/hgmm/exon.f2g ]; then
  for e in hdr indptr indices data; do ln -sf $A/exon.$e $O/hgmm/exon.$e; done
  $PY $J/hgmm_prep.py $W/modality-bench/ambient/singlet/exon_counts_features.tsv.gz $O/hgmm/exon.f2g
fi
echo "== hgmm  [$LABEL] $ARGS"
$O/build/dbl_run_$LABEL $O/hgmm/exon $O/hgmm/$LABEL.tsv threads=16 $ARGS
$PY $J/hgmm_eval.py $A $O/results/hgmm_$LABEL.csv $O/hgmm/$LABEL.tsv="Singlet new ($LABEL)"
