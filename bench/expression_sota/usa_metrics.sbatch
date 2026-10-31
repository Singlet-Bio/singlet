#!/bin/bash
#SBATCH --job-name=usa-metrics
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# sbatch usa_metrics.sbatch DATASET   (after perf_run.sbatch PERF_TAG=usa wrote perf/<ds>_usa/t8_r1)
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; X=$W/expression-sota; DS=$1
PY=$W/envs/pbench/bin/python; mkdir -p $X/usa
$PY $X/code/usa_metrics.py $PB $DS $PB/runs/$DS/singlet-public $X/usa/$DS.before.json --label alpha-before
$PY $X/code/usa_metrics.py $PB $DS $PB/perf/${DS}_usa/t8_r1 $X/usa/$DS.json --label mc99cpu-default-rules
