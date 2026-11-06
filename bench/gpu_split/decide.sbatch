#!/bin/bash
#SBATCH --job-name=gsplit-decide
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Picks the STEPS_B setting of the MPS arm from the first timing job. Environment: TP_JOB, BR (branch with the scripts).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; BARE=$W/repos/singlet-py.git; BR=${BR:-gpu/bold-split}; URL="$(git -C $BARE config remote.origin.url)"
SCR=/tmp/$USER-gsplitd-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf "${SCR:?}"; git -C $BARE update-ref -d refs/gsplit/dec-$SLURM_JOB_ID 2>/dev/null' EXIT
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$BR:refs/gsplit/dec-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/gsplit/dec-$SLURM_JOB_ID bench/gpu_split bench/gpu_bold/analyze_points.py | tar -x -C $SCR
export PATH=$W/envs/pbench/bin:$PATH
python3 $SCR/bench/gpu_split/decide.py $W/jobs/gpu-bold/results/tp-${TP_JOB:?} $W/jobs/gpu-split/results/overnight $SCR/bench/gpu_bold/analyze_points.py
