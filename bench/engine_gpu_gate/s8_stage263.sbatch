#!/bin/bash
#SBATCH --job-name=s8-stage263
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: encode the deep sample GSM7656149 (.1fq, 263 M reads) as .1fg2 with S9's singlet-stage, for the memory profile of the table-fed route.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/data/atlas-rerun/perf/GSM7656149; F=$W/jobs/s9/fixtures
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
T=/tmp/$USER-st263-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
time $F/singlet-stage --whitelist-dir $SINGLET_WHITELIST_DIR --accessions GSM7656149 --scratch $T -o $D/sample.1fg2 --verify --json $D/sample.stage.json $D/sample.1fq | tail -c 400
ls -l $D
