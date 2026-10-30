#!/bin/bash
#SBATCH --job-name=alpha-final
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Census JSON, per-sample table, results documents and figures -> $WORK/alpha_bench/results/
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
A=$WORK/alpha_bench; OUT=$A/results; mkdir -p $OUT
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
git clone -q --depth 1 -b ${1:-claude/alpha-bench} git@github.com:Singlet-Bio/singlet.git $SCR/src
git clone -q --depth 1 -b claude/bench-public-cohort-485cf7 git@github.com:Singlet-Bio/singlet.git $SCR/pub
SHA=$(git -C $SCR/src rev-parse HEAD); echo "code $SHA"
RUN="apptainer exec --bind $WORK,/tmp --env PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin --env CONTRACT=$SCR/pub/bench/public/contract.py --env CODE_SHA=$SHA --env MPLCONFIGDIR=$SCR/mpl $WORK/images/python-3.12-bookworm.sif python -P"
B=$SCR/src/bench/alpha
$RUN $B/census_alpha.py json $A/census $A/census/alpha-policy.json $A/census/alpha_reads.tsv.gz $OUT/alpha_census.json $A/census/summaries_*.tsv > $OUT/census.log
$RUN $B/aggregate.py samples $A/compare $OUT
$RUN $B/aggregate.py docs $OUT
$RUN $B/aggregate.py engine $OUT $A/engine
$RUN $B/figs.py $OUT
cp $SCR/pub/bench/public/contract.py $OUT/.contract_used.py
ls -la $OUT
