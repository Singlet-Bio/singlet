#!/bin/bash
#SBATCH --job-name=txf-summary
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Summarizer of both studies. Runs after the others finish, whatever their state (afterany). CPU only.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; RES=$W/jobs/tx-first/results
export PATH=$W/envs/pbench/bin:$PATH
BARE=$W/repos/singlet-py.git; SBR=${SCRIPTS_BRANCH:-bench/tx-first}; SCR=/tmp/$USER-txf-$SLURM_JOB_ID; mkdir -p $SCR/repo; trap 'rm -rf "${SCR:?}"' EXIT
URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$SBR:refs/txf/$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/txf/$SLURM_JOB_ID bench/tx_first | tar -x -C $SCR/repo; git -C $BARE update-ref -d refs/txf/$SLURM_JOB_ID
: > $RES/states.txt
for id in ${JOBIDS:-}; do sacct -j $id -n -X -P -o JobID,JobName,State,ExitCode,Elapsed,NodeList >> $RES/states.txt; done
python3 $SCR/repo/bench/tx_first/summarize.py $RES
{ echo; echo "## Job states"; echo; sed 's/^/    /' $RES/states.txt; } >> $RES/overnight/summary.md
ls -la $RES/overnight
