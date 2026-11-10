#!/bin/bash
#SBATCH --job-name=gsota-lclblind
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; BARE=$W/repos/singlet-py.git; E=$W/envs/bpub
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/claude/ancestry-label-rule:refs/s8/ancl
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/s8/ancl python/singlet/celllabels | tar -x -C $SCR/src
$E/bin/python -I $W/jobs/gsota-r12/lcl_blind.py $SCR/src/python/singlet/celllabels/ancestry.py $W/bpub/work/ancestry/af_leaveout.parquet $W/bpub/runs
