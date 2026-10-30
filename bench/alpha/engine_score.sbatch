#!/bin/bash
#SBATCH --job-name=alpha-escore
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-6
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Score runs/<ds>/singlet-alpha30m with the pipeline benchmark's scorers. Output $WORK/alpha_bench/engine/<ds>.json
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
REF=${1:-claude/alpha-bench}
DS=(pbmc1k_v3 pbmc1k_v2 pbmc5p_v1 pbmc1k_cite_v3 neuron1k_v3 nuclei2k_v2 hgmm1k_v3)
ID=${DS[$SLURM_ARRAY_TASK_ID]}
FLAGS=""; [ $ID = hgmm1k_v3 ] && FLAGS=--barnyard; [ $ID = pbmc1k_cite_v3 ] && FLAGS=--cite
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
git clone -q --depth 1 -b $REF git@github.com:Singlet-Bio/singlet.git $SCR/src
echo "code $(git -C $SCR/src rev-parse HEAD) $ID"
mkdir -p $WORK/alpha_bench/engine
apptainer exec --bind "$WORK,/tmp,/mnt/projects/debruinz_project/cellarium" \
  --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" \
  --env "EDROP=$WORK/src/singlet-py/claude__empty-droplet-check-624070" --env "HDF5_USE_FILE_LOCKING=FALSE" \
  --env "NUMBA_CACHE_DIR=$SCR/numba" --env "MPLCONFIGDIR=$SCR/mpl" $WORK/images/python-3.12-bookworm.sif \
  python -P $SCR/src/bench/alpha/engine_score.py $WORK/pipeline-benchmark \
    $WORK/src/singlet-py/fix__molecule-labels/bench/pipeline/analysis $ID $WORK/alpha_bench/engine/$ID.json $FLAGS
