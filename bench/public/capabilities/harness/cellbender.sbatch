#!/bin/bash
#SBATCH --job-name=bp-cellbender
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:tesla_v100s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch cellbender.sbatch <cohort-id> <tag: cellbender | cellbender_rec> [extra cellbender args]
# CellBender 0.3.2 remove-background on the raw droplet matrix written by barnyard_prep.py (one V100S: the 0.3.2 image's PyTorch has no
# kernels for the H100, sm_90). Two invocations:
#   cellbender      the documented minimal call, `--cuda` only (cell and droplet counts estimated by CellBender)
#   cellbender_rec  the authors' recommended call: --expected-cells and --total-droplets-included given (the tutorial asks for them),
#                   here the number of cells Singlet called and 6x that number, as the first CellBender runs on this data used
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; BP=$W/bench-public; ID=$1; TAG=$2; shift 2 || true
SIF=$W/images/cellbender-0.3.2.sif; D=$BP/work/$ID; O=$D/$TAG; mkdir -p $O
export APPTAINER_CACHEDIR=$W/cache/apptainer XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
nvidia-smi -L; nvidia-smi -L | head -1 > $O/gpu.txt; cd $SCR
echo "cellbender remove-background --input RAW --output cb.h5 --cuda $*" | tee $O/cmd.txt
/usr/bin/time -v -o $O/time.txt apptainer exec --nv --bind $W --bind /tmp $SIF cellbender remove-background \
  --input $D/raw --output $SCR/cb.h5 --cuda "$@" > $O/run.log 2>&1 || { grep -v "^  File\|^    " $O/run.log | tail -30; exit 1; }
cp $SCR/cb.h5 $SCR/cb_cell_barcodes.csv $SCR/cb_metrics.csv $O/ 2>/dev/null || true
grep -E "Elapsed \(wall|Maximum resident" $O/time.txt; ls -la $O
