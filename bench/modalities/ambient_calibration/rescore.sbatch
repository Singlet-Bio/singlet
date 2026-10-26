#!/bin/bash
#SBATCH --job-name=af-rescore
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch rescore.sbatch TAG LANE   re-scores existing estimates $AF/est/TAG/LANE_* against $AF/dump/LANE truth
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/ambfix/common.sh
af_sync
TAG=$1; L=$2; D=$AF/dump/$L; O=$AF/est/$TAG; B=$SRC/bench/modalities/ambient_calibration
specs=("prod_run=$D/prod_run")
for f in $O/${L}_*.rho.f64; do p=${f%.rho.f64}; specs+=("${p#$O/${L}_}=$p"); done
rm -f $AF/res/${TAG}_kang_$L.jsonl
py $B/score_kang.py score $D $AF/res/${TAG}_kang_$L.jsonl "${specs[@]}"
