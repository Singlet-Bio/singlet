#!/bin/bash
#SBATCH --job-name=rc-cross
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Read-collapse study, stage 2: cross-sample hit rates, top-K tables, k-mer classifier, pool growth (rc_cross.py).
# Test mode: --export=ALL,RES=<dir>,SAMPLES_TSV=tmp (a tsv written next to the results).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; RC=$W/jobs/read-collapse
RES=${RES:-$RC/results/overnight}
BARE=$W/repos/singlet-py.git; SBR=bench/read-collapse
SCR=/tmp/$USER-rcx-$SLURM_JOB_ID; mkdir -p $SCR/repo; SREF=refs/rc/cross-$SLURM_JOB_ID
trap 'git -C $BARE update-ref -d $SREF 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$SBR:$SREF ) 9>$BARE/.rc-fetch.lock || { echo "fetch failed"; exit 3; }
git -C $BARE archive $SREF bench/read_collapse | tar -x -C $SCR/repo
RCS=$SCR/repo/bench/read_collapse
export PATH=$W/envs/pbench/bin:$PATH
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) scripts=$(git -C $BARE rev-parse --short $SREF) $(date -Is)"
TSV=${SAMPLES_TSV:-$RCS/samples.tsv}
python3 $RCS/rc_cross.py --samples $TSV --data $RES --out $RES/cross.json
echo "rc_cross exit $?"
