#!/bin/bash
#SBATCH --job-name=dbl-sweep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Parameter sweep of the new detect_doublets() on Kang (exon+map, and gene-level matrix) and hgmm synthetic.
#   sbatch --export=ALL,SWEEP=$J/sweep.txt dbl_sweep.sbatch      SWEEP lines: "label key=value key=value ..."
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/modfix-doublet; O=$W/modality-bench/doublet_fix
K=$W/modality-bench/genotype_pool; A=$W/modality-bench/ambient/work/dbl
PY=$W/envs/gpool/venv/bin/python
export PATH=$W/envs/pbench/bin:$PATH
SWEEP=${SWEEP:-$J/sweep.txt}; THR=${SLURM_CPUS_PER_TASK:-8}
mkdir -p $O/kang $O/hgmm $O/build $O/results
x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -pthread -I$J/inc $J/dbl_run.cpp -o $O/build/dbl_run_sweep
[ -f $O/kang/k8_in.hdr ] || $PY $J/kang_prep.py $K/singlet_k8 $O/kang/k8
if [ ! -f $O/hgmm/exon.f2g ]; then
  for e in hdr indptr indices data; do ln -sf $A/exon.$e $O/hgmm/exon.$e; done
  $PY $J/hgmm_prep.py $W/modality-bench/ambient/singlet/exon_counts_features.tsv.gz $O/hgmm/exon.f2g
fi
while read -r LABEL ARGS; do
  [ -z "$LABEL" ] && continue
  case "$LABEL" in \#*) continue;; esac
  echo "######## $LABEL  $ARGS"
  for D in ${DSETS:-k8 k8g k8i}; do
    P=$D; EXTRA=""; if [ "$D" = k8x ]; then P=k8; EXTRA="second=$O/kang/k8_in"; fi   # k8x: exon + intron source, as export.h
    $O/build/dbl_run_sweep $O/kang/$P $O/kang/${D}_$LABEL.tsv threads=$THR $EXTRA $ARGS 2> $O/kang/${D}_$LABEL.log
    echo "$D: $(cat $O/kang/${D}_$LABEL.log)"
    $PY $J/kang_eval.py $K/truth/GSE96583_batch2.total.tsne.df.tsv.gz $O/kang/$P $O/kang/${D}_$LABEL.tsv $K/results/doublets_k8_scores.csv.gz $O/results/kang_${D}_$LABEL.json new \
      | $PY -c "import sys,json; d=json.load(sys.stdin)['new']; print('  KANG $D', {k:round(v,4) for k,v in d.items()})"
  done
  HL=$LABEL${HG_TAG:-}
  $O/build/dbl_run_sweep $O/hgmm/exon $O/hgmm/$HL.tsv threads=$THR $ARGS ${HG_ARGS:-} 2> $O/hgmm/$HL.log
  echo "hgmm: $(cat $O/hgmm/$HL.log)"
  $PY $J/hgmm_eval.py $A $O/results/hgmm_$HL.csv $O/hgmm/$HL.tsv="new_$HL" | tail -1
done < "$SWEEP"
