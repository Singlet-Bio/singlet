#!/bin/bash
# CPU doublet scores per dataset (array over $LQC/dbl/datasets.txt): export, Scrublet, production scorer, NNLS two-type,
# scDblFinder.   sbatch --array=0-19%10 cpu.sbatch     (METHODS to restrict)
#SBATCH --job-name=lqc-cpu
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/study/latent-qc:refs/heads/study/latent-qc || true
git --git-dir=$W/repos/singlet-py.git show refs/heads/study/latent-qc:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh
source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
B=$SRC/bench/latent_qc/doublets; export PYTHONPATH=$B
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap "rm -rf $SCRATCH" EXIT; export TMPDIR=$SCRATCH
N=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $LQC/dbl/datasets.txt); [ -n "$N" ] || exit 0; echo "dataset $N"
M=${METHODS:-export scrublet prod nnls scdbl}
export DBL_RUN=$SCRATCH/dbl_run
[[ " $M " == *" prod "* ]] && $CXX -O2 -std=c++17 -pthread -I$PROD/include/singlet/pileup $PROD/bench/modalities/doublet_fix/dbl_run.cpp -o $DBL_RUN
for m in $M; do
  case $m in
    scrublet) "$RENV/bin/python" $B/methods_cpu.py $N scrublet ;;
    scdbl) [ -f $LQC/dbl/scores/$N/scdblfinder.tsv ] || R $B/scdbl.R $LQC/dbl/scores $N ;;
    *) rt env PYTHONPATH=$B DBL_RUN=$DBL_RUN python $B/methods_cpu.py $N $m ;;
  esac
done
