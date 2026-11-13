#!/bin/bash
#SBATCH --job-name=identify-panel
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Run `singlet identify` over a source list (id TAB source), part ARRAY_TASK of N.   sbatch run_panel.sbatch SOURCES.tsv OUTNAME [--genome-dir DIR]
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/identify; ENV=$W/envs/pbench
SRC=$1; OUT=$J/out/$2; shift 2
N=$(( ${SLURM_ARRAY_TASK_MAX:-0} + 1 )); I=${SLURM_ARRAY_TASK_ID:-0}
export LD_LIBRARY_PATH=$ENV/lib PATH=$ENV/bin:$PATH
SCR=/tmp/$USER-$SLURM_JOB_ID-$I; mkdir -p $SCR/home $OUT/part$I; trap 'rm -rf $SCR' EXIT
export HOME=$SCR/home        # the archive reader's block cache goes to node-local scratch, not $HOME on the shared quota
awk -v n=$N -v i=$I 'NR%n==i' $SRC | tr -d '\r' | sed "s#\$DATA#$W/data#g; s#\$REF#/mnt/projects/debruinz_project/cellarium/reference#g" > $SCR/part.tsv
$J/bin/singlet identify --list $SCR/part.tsv --out-dir $OUT/part$I --whitelist-dir $J/wl "$@"
