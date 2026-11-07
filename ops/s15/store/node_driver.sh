#!/bin/bash
# S15 store: whole-node driver. P pipelines run side by side on one exclusive node and pop dataset ids from a shared queue (flock).
# Submit:  sbatch --exclusive --nodes=1 --mem=0 --partition=bigmem|cpu --time=5-00:00:00 node_driver.sh P CHAIN
# Idempotent: pipeline.sh skips every tool already in index.tsv, so a successor job (afterany) resumes where this one stopped. CHAIN counts the successors still to submit.
P=${1:-3}; CHAIN=${2:-3}
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; STORE=$W/benchmark-store; Q=${QUEUE:-$D/queue/store.queue}; mkdir -p $D/queue; touch $Q   # QUEUE=big.queue: the bigmem driver only takes work that failed on a cpu node (STAR needing > 450 GB)
echo "== $(date -Is) node driver job=$SLURM_JOB_ID node=$(hostname) P=$P chain=$CHAIN"
if [ "$CHAIN" -gt 0 ]; then
  sbatch --parsable --account=debruinz --job-name=s15-node --partition=${SLURM_JOB_PARTITION:-cpu} --exclusive --nodes=1 --mem=0 --time=5-00:00:00 --dependency=afterany:$SLURM_JOB_ID \
    --output=$W/logs/%x-%j.out $D/node_driver.sh $P $((CHAIN - 1)) > /dev/null
fi
pop() { flock $Q.lock bash -c "id=\$(head -1 $Q); [ -n \"\$id\" ] && sed -i 1d $Q && echo \$id"; }
slot() { n=$1; while :; do id=$(pop)
    if [ -z "$id" ]; then [ -n "$QUEUE_WAIT" ] && squeue -u $USER -h -t R -o "%j %P" | grep -q "^s15-node cpu" && { sleep 600; continue; }; break; fi
    bash $D/pipeline.sh $id >> $W/logs/node-$SLURM_JOB_ID-s$n-$id.log 2>&1; done; echo "slot $n idle $(date -Is)"; }
for n in $(seq 1 $P); do slot $n & sleep 20; done
wait
echo "== $(date -Is) queue empty, driver done"
