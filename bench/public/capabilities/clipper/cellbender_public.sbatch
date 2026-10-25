#!/bin/bash
#SBATCH --job-name=pub-cellbender
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Singlet 1.0.0 public benchmark, cell-calling: CellBender 0.3.2 remove-background (its cell calls) on STARsolo's raw Gene matrix
# (every barcode with a UMI), array index = dataset (same order as starsolo_public.sbatch). Needs starsolo_public.sbatch MODE=cr.
#   sbatch --array=0-7 --dependency=afterok:<starsolo job> cellbender_public.sbatch
# Parameters come from the raw UMI curve only (no truth): --expected-cells = barcodes at or above 1/10 of the 99th percentile of the
# top 3000 barcodes (the Cell Ranger 2.2 rule, which is STARsolo's documented default and what CellBender's docs suggest reading off the
# UMI curve); --total-droplets-included = max(3 x expected, 10000). Everything else at the CellBender defaults for 0.3.2
# (--fpr 0.01, --epochs 150 as in the CellBender tutorial).
# Output: $PB/runs/<dataset>/cellbender-public/ (cb_cell_barcodes.csv, cb_metrics.csv, time.txt, params.txt).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
SIF=$W/images/cellbender-0.3.2.sif
DSS=(pbmc1k_v3 pbmc1k_v2 neuron1k_v3 hgmm1k_v3 pbmc1k_cite_v3 nuclei2k_v2 pbmc5p_v1 kang2018_ctrl)
ID=${DSS[$SLURM_ARRAY_TASK_ID]}
RAW=$PB/runs/$ID/starsolo-public-cr/Solo.out/Gene/raw
O=$PB/runs/$ID/cellbender-public; rm -rf $O; mkdir -p $O
export APPTAINER_CACHEDIR=$W/cache/apptainer XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
read EXP TOT < <($W/envs/pbench/bin/python -I - "$RAW/matrix.mtx" <<'PY'
import sys, numpy as np, scipy.io
m = scipy.io.mmread(sys.argv[1]).tocsc()
tot = np.sort(np.asarray(m.sum(axis=0)).ravel())[::-1]
top = tot[:3000]
cut = np.quantile(top, 0.99) / 10.0
exp = int((tot >= cut).sum())
print(exp, max(3 * exp, 10000))
PY
)
echo "$ID expected-cells=$EXP total-droplets-included=$TOT" | tee $O/params.txt
nvidia-smi -L || true
mkdir -p $SCR/raw; for f in matrix.mtx barcodes.tsv features.tsv; do gzip -c $RAW/$f > $SCR/raw/$f.gz; done  # 10x v3 layout
cd $SCR
CMD=(cellbender remove-background --cuda --input $SCR/raw --output $SCR/cb.h5 --expected-cells $EXP --total-droplets-included $TOT --fpr 0.01 --epochs 150)
echo "${CMD[*]}" >> $O/params.txt
/usr/bin/time -v -o $O/time.txt apptainer exec --nv --bind $W --bind /tmp $SIF "${CMD[@]}" > $O/run.log 2>&1
cp $SCR/cb_cell_barcodes.csv $SCR/cb_metrics.csv $O/ 2>/dev/null || true
tail -3 $O/run.log; ls -la $O; wc -l $O/cb_cell_barcodes.csv
