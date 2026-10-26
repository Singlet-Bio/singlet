#!/bin/bash
#SBATCH --job-name=af-prep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Per Kang lane: gene-level dump of the stored run (caller cells + knee cells) and the genotype truth of its confident singlets.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/ambfix/common.sh
sleep $((SLURM_ARRAY_TASK_ID * 5)); af_sync
LANES=(ctrl A B C stim); L=${LANES[$SLURM_ARRAY_TASK_ID]}; RUN=$(lane_rundir $L); D=$AF/dump/$L
B=$SRC/bench/modalities/ambient_calibration
[ -f $D/dims.txt ] || py $B/dump_run.py $RUN $D
[ -f $D/truth_meta.json ] || py $B/score_kang.py truth $(lane_truth_run $L) $D $(lane_truth $L) $L
cat $D/dims.txt $D/truth_meta.json
