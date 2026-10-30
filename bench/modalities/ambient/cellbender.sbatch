#!/bin/bash
#SBATCH --job-name=amb-cb
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch cellbender.sbatch [EXPECTED_CELLS] [TOTAL_DROPLETS]
# CellBender 0.3.2 remove-background on Singlet's raw barcode x gene matrix of hgmm_1k_v3 (GPU, container).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; A=$W/modality-bench/ambient
SIF=$W/images/cellbender-0.3.2.sif; EXP=${1:-1000}; TOT=${2:-6000}
export APPTAINER_CACHEDIR=$W/cache/apptainer XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
mkdir -p $A/cellbender; nvidia-smi -L || true
cd $SCR
/usr/bin/time -v -o $A/cellbender/time.txt apptainer exec --nv --bind $W --bind /tmp $SIF cellbender remove-background \
  --cuda --input $A/work/raw --output $SCR/cb.h5 \
  --expected-cells $EXP --total-droplets-included $TOT --epochs 150 --fpr 0.01 > $A/cellbender/run.log 2>&1
cp $SCR/cb.h5 $SCR/cb_filtered.h5 $SCR/cb_cell_barcodes.csv $SCR/cb_metrics.csv $A/cellbender/ 2>/dev/null || true
cp $SCR/*.pdf $SCR/*.log $A/cellbender/ 2>/dev/null || true
tail -5 $A/cellbender/run.log; ls -la $A/cellbender
