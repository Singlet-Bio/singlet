#!/bin/bash
#SBATCH --job-name=cc-nucfair
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# MODE=prep (default): prep + emptyDrops + score without CellBender; MODE=score: rescore with CellBender output if present.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; S=$W/bench/calls-sota/cell_calling; O=$S/nuc_fair; PY=$W/envs/gpool/venv/bin/python
if [ "${MODE:-prep}" = prep ]; then
  NUMBA_NUM_THREADS=8 NUMBA_CACHE_DIR=/tmp/$USER-nb-$SLURM_JOB_ID $PY -P $S/nuc_fair.py prep $W/expression-sota/nucleus_hash/singlet $W/expression-sota/nucleus_hash/hashtags.counts.tsv $S/cellcall_v22_rank.py $O
  $W/envs/ambient/bin/Rscript $S/emptydrops_ref.R $O
  $PY -P $S/merge_ed.py $O
fi
$PY -P $S/nuc_fair.py score $O $S/cellbender/nucleus_hash/cb_cell_barcodes.csv $O/scores.json
