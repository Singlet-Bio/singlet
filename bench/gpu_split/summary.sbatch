#!/bin/bash
#SBATCH --job-name=gsplit-summary
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Summarizer of the gpu/bold-split chain. Environment: BUILD_JOB IDENT_JOBS (space list) PATHS_JOB TP_JOBS (space list), BR (branch holding the scripts).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export W
BARE=$W/repos/singlet-py.git; BR=${BR:-gpu/bold-split}; URL="$(git -C $BARE config remote.origin.url)"
SCR=/tmp/$USER-gsplit-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf "${SCR:?}"; git -C $BARE update-ref -d refs/gsplit/sum-$SLURM_JOB_ID 2>/dev/null' EXIT
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$BR:refs/gsplit/sum-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/gsplit/sum-$SLURM_JOB_ID bench/gpu_split bench/gpu_bold/analyze_points.py | tar -x -C $SCR
OUT=$W/jobs/gpu-split/results/overnight; mkdir -p $OUT
export ANALYZE=$SCR/bench/gpu_bold/analyze_points.py PATH=$W/envs/pbench/bin:$PATH
python3 $SCR/bench/gpu_split/summarize.py $OUT/summary.md && echo "wrote $OUT/summary.md"
