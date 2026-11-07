#!/bin/bash
#SBATCH --job-name=rl-stage
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One whole node for the read-level benchmarks' staging: Penter fetch, TE data fetch and TE comparator setup run side by side
# (the node rule: one node at a time, no scattered small jobs). Each part is resumable and logs to $L/<part>.log.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/readlevel; L=$W/logs/rl-stage-$SLURM_JOB_ID; mkdir -p $L
echo "node=$(hostname) cpus=$SLURM_CPUS_ON_NODE $(date -Is)"
pids=()
for p in fetch_penter fetch_te setup_te; do
  bash $J/$p.sbatch > $L/$p.log 2>&1 & pids+=($!); echo "$p pid $!"
done
rc=0; i=0
for p in fetch_penter fetch_te setup_te; do wait ${pids[$i]}; r=$?; echo "$p exit $r $(date -Is)"; tail -3 $L/$p.log; [ $r -eq 0 ] || rc=1; i=$((i+1)); done
exit $rc
