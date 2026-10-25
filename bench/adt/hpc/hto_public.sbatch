#!/bin/bash
#SBATCH --job-name=hto-pub
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Public HTO benchmark (capability hto) on Howitt et al. 2023: prepare + every method per dataset, then the document.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
SRC=${SRC:-$WORK/src/singlet-py/claude__adt-modality-0637a6}
Z=$WORK/data/adt_bench/zenodo_8304003
PUB=$WORK/state/adt/hto_zenodo/published
OUT=$WORK/bpub/artifacts/hto; mkdir -p $OUT
SCRATCH=/tmp/$USER-htopub-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
export PATH=$WORK/envs/adt-bench/bin:$PATH LD_LIBRARY_PATH=$WORK/envs/adt-bench/lib:$WORK/envs/adt/lib
export PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome SINGLET_COMMIT=$(git -C $SRC rev-parse HEAD)
for D in howitt2023_bal1 howitt2023_bal2 howitt2023_bal3 howitt2023_cl; do
  R="python -P $SRC/bench/adt/hto_public_run.py --dataset $D --zenodo $Z --published $PUB --out $OUT --scratch $SCRATCH"
  $R --tool prepare
  for t in singlet htodemux multiseqdemux hashsolo; do $R --tool $t || echo "tool $t failed on $D"; done
done
