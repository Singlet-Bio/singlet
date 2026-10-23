#!/bin/bash
# CellBender remove-background on fetched samples of the empty-droplet study (GPU). Image pulled once from the Broad's registry.
#   sbatch --array=0-9 --export=ALL,LIST=<file of GSE/GSM lines> cellbender.sbatch
#SBATCH --job-name=edrop-cb
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
SRC=$WORK/src/singlet-py/claude__empty-droplet-check-624070
VENV=$WORK/venvs/singlet
SIF=$WORK/images/python-3.12-bookworm.sif
CB_SIF=$WORK/images/cellbender-0.3.2.sif
DATA=$WORK/data/emptydrop
OUT=$WORK/state/emptydrop/results
TASK=${SLURM_ARRAY_TASK_ID:-0}
LINE=$(sed -n "$((TASK + 1))p" "${LIST:?}")
GSE=${LINE%%/*}; GSM=${LINE##*/}
[ -n "$GSM" ] || { echo "no line $TASK"; exit 0; }
export XDG_CACHE_HOME=$WORK/cache APPTAINER_CACHEDIR=$WORK/cache/apptainer NUMBA_CACHE_DIR=$WORK/cache/numba-edrop MPLCONFIGDIR=$WORK/cache/mpl
SCRATCH=/tmp/${USER:-x}-${SLURM_JOB_ID:-x}-$TASK; mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
(
  flock -w 3600 9
  [ -f "$CB_SIF" ] || apptainer pull "$CB_SIF" docker://us.gcr.io/broad-dsde-methods/cellbender:0.3.2
) 9>"$WORK/state/edrop-cb-pull.lock"
echo "$GSE $GSM on $(hostname)"; nvidia-smi -L || true
apptainer exec --bind "$WORK" --bind /tmp "$SIF" "$VENV/bin/python" -P "$SRC/bench/empty_droplets/cellbender_io.py" export "$DATA/$GSE/$GSM.npz" "$SCRATCH/in" | tee "$SCRATCH/export.json"
NCALL=$(apptainer exec "$SIF" python3 -c "import json,sys; print(json.load(open('$SCRATCH/export.json'))['n_called'])")
NTOT=$(apptainer exec "$SIF" python3 -c "import json,sys; print(json.load(open('$SCRATCH/export.json'))['total_droplets'])")
mkdir -p "$OUT/$GSE"
cd "$SCRATCH"
apptainer exec --nv --bind "$WORK" --bind /tmp "$CB_SIF" cellbender remove-background --cuda --input "$SCRATCH/in" \
  --output "$SCRATCH/cb.h5" --expected-cells "$NCALL" --total-droplets-included "$NTOT" --epochs 150 ${CB_ARGS:-} 2>&1 | tail -40
cp "$SCRATCH"/cb_cell_barcodes.csv "$OUT/$GSE/$GSM.cellbender_cells.csv" 2>/dev/null || true
cp "$SCRATCH"/cb.h5 "$OUT/$GSE/$GSM.cellbender.h5"   # kept on Clipper so collect can be rerun after a re-analysis
apptainer exec --bind "$WORK" --bind /tmp "$SIF" "$VENV/bin/python" -P "$SRC/bench/empty_droplets/cellbender_io.py" collect \
  "$DATA/$GSE/$GSM.npz" "$SCRATCH/cb.h5" "$OUT/$GSE/$GSM.cellbender.json" --analysis-npz "$OUT/$GSE/$GSM.bc.npz"   --cells-csv "$OUT/$GSE/$GSM.cellbender_cells.csv"
