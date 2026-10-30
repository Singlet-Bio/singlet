#!/bin/bash
#SBATCH --job-name=if-eval
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# sbatch sweep_eval.sbatch TAG OUTNAME [datasets...]  -> jobs/intronfix/results/OUTNAME.json
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; ENV=$W/envs/pbench
BARE=$W/repos/singlet-py.git; BR=bench/public-counts; SRC=$W/src/singlet-py/bench__public-counts
git -C $BARE fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR
[ -d $SRC ] || git -C $BARE worktree add -q -f $SRC $BR
git -C $SRC reset -q --hard refs/heads/$BR; git -C $SRC log --oneline -1
TAG=$1; NAME=$2; shift 2
mkdir -p $W/jobs/intronfix/results
$ENV/bin/python $W/jobs/intronfix/sweep_eval.py $SRC/bench/public/capabilities $PB $TAG $W/jobs/intronfix/results/$NAME.json "$@"
