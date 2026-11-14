#!/bin/bash
#SBATCH --job-name=cc-cb
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:tesla_v100s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:55:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CellBender 0.3.2 (one V100S, owner-allowed) on the held-out libraries and the Kang b1 lanes. Input dirs from cb_prep.py (held-out) or the STARsolo raw matrices (Kang).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; S=$W/bench/calls-sota/cell_calling; O=$S/cellbender; mkdir -p $O
SIF=$W/images/cellbender-0.3.2.sif; export APPTAINER_CACHEDIR=$W/cache/apptainer XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
run() { # name inputdir exp total
  n=$1; d=$2; e=$3; t=$4; mkdir -p $O/$n $SCR/$n; cd $SCR/$n
  CMD=(cellbender remove-background --cuda --input $d --output $SCR/$n/cb.h5 --expected-cells $e --total-droplets-included $t --fpr 0.01 --epochs 150)
  echo "$n: ${CMD[*]}" | tee $O/$n/params.txt
  /usr/bin/time -v -o $O/$n/time.txt apptainer exec --nv --bind $W --bind /tmp $SIF "${CMD[@]}" > $O/$n/run.log 2>&1
  if [ ! -s $SCR/$n/cb_cell_barcodes.csv ]; then CMD+=(--learning-rate 0.00005); echo "retried: ${CMD[*]}" >> $O/$n/params.txt
    apptainer exec --nv --bind $W --bind /tmp $SIF "${CMD[@]}" > $O/$n/run.log 2>&1; fi
  cp $SCR/$n/cb_cell_barcodes.csv $SCR/$n/cb_metrics.csv $O/$n/ 2>/dev/null; wc -l $O/$n/cb_cell_barcodes.csv
}
for n in cellplex10k nuclei_poc; do read e t < $S/cb_in/$n/params.txt; run $n $S/cb_in/$n $e $t; done
read e t < $S/nuc_fair/cb_in/params.txt; run nucleus_hash $S/nuc_fair/cb_in $e $t
for ID in kang_b1A kang_b1B kang_b1C; do
  RAW=$W/state/cellcall_truth/starsolo/$ID/Solo.out/Gene/raw; mkdir -p $SCR/in_$ID
  for f in matrix.mtx barcodes.tsv features.tsv; do gzip -c $RAW/$f > $SCR/in_$ID/$f.gz; done
  read e t < <($W/envs/pbench/bin/python -I - "$RAW/matrix.mtx" <<'PY'
import sys, numpy as np, scipy.io
m = scipy.io.mmread(sys.argv[1]).tocsc()
tot = np.sort(np.asarray(m.sum(axis=0)).ravel())[::-1]
exp = int((tot >= np.quantile(tot[:3000], 0.99) / 10.0).sum()); print(exp, max(3 * exp, 10000))
PY
)
  run $ID $SCR/in_$ID $e $t
done
