#!/bin/bash
#SBATCH --job-name=rc-summary
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Read-collapse study, final job: summary.md and summary.json from the per-sample and cross-sample results.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; RC=$W/jobs/read-collapse
RES=${RES:-$RC/results/overnight}
BARE=$W/repos/singlet-py.git; SBR=bench/read-collapse
SCR=/tmp/$USER-rcs-$SLURM_JOB_ID; mkdir -p $SCR/repo; SREF=refs/rc/sum-$SLURM_JOB_ID
trap 'git -C $BARE update-ref -d $SREF 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$SBR:$SREF ) 9>$BARE/.rc-fetch.lock || { echo "fetch failed"; exit 3; }
git -C $BARE archive $SREF bench/read_collapse | tar -x -C $SCR/repo
export PATH=$W/envs/pbench/bin:$PATH
python3 $SCR/repo/bench/read_collapse/rc_summary.py $RES $RES
echo "summary exit $?"; wc -c $RES/summary.md $RES/summary.json
