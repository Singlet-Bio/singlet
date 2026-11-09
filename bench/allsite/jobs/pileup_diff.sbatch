#!/bin/bash
#SBATCH --job-name=allsite-pdiff
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Read-level check of the all-site AD gap against cellsnp-lite (bench/allsite/pileup_diff.py). Needs NAME (a finished validation library),
# optional NSITES (default 1000). Reads the engine's out/, cellsnp_sites/ and the tagged BAM of that validation; writes counts-only JSON.
set -uo pipefail
: "${NAME:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; O=$W/genotypes/allsite-val/$NAME; R=$W/jobs/allsite/results/$NAME
BARE=$W/repos/singlet-py.git; BR=${BR:-feat/allele-outputs}; SCR=/tmp/$USER-pd-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
( flock 9; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/pd/$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/pd/$SLURM_JOB_ID bench/allsite | tar -x -C $SCR
$W/envs/pbench/bin/python3 -I $SCR/bench/allsite/pileup_diff.py $O/out $O/cellsnp_sites $O/bam/possorted.bam $W/envs/pbench/bin/samtools ${NSITES:-1000} $R/pileup_diff.json
chmod -R go-rwx $R
