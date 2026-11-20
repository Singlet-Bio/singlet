#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 27 --mem=200G --time=02:00:00 --exclude=g050 -J t2-cggate-cg -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-cggate-cg (cggate, from t2-pooladmit-cg.sbatch): ONE H100, --mem=200G by default (override: sbatch --mem=150G / --mem=365G) (not exclusive, so the node's MemFree is far above the job's limit), three d300_3p items in one pool with
# 3 workers, default pool settings, SINGLET_POOL_MEMSAMPLE=2; the cgroup / node sampler t2-cgsamp.sh runs beside it once a second.
# Derived from t2-rt_deep2.sbatch: only the resources, the item list (ITEMS, default "d300_3p d300_3p d300_3p"), WORKERS (3) and the sampler differ; no output hashing.
# Parameters: BIN, ITEMS, WORKERS, TAG, T, SLOTS, MEMGB, LIM, XENV (XENV="SINGLET_POOL_PEAKADMIT=1" for mode 1; empty = today's gate). Adds: the pool's peak lines
# (peak figure / model exceeded / baseline) and memsamp-cg.tsv in the report, the engine peakmodel lines of every item, memory.events at the end.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
ITEMS=${ITEMS:-d300_3p d300_3p d300_3p}; WORKERS=${WORKERS:-3}; TAG=${TAG:-cg}; T=${T:-8}; SLOTS=${SLOTS:-2}; MEMGB=${MEMGB:-31}; LIM=${LIM:-7000}; GD=$REF/GRCh38-2024-A
O=$W/jobs/gpu-runtime/results/t2-cggate-$SLURM_JOB_ID-$TAG; mkdir -p $O/pool0; SCR=/tmp/$USER-cgg-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/run; R=$O/results.txt
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_ALLSITE_TIMING=1
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70
export SINGLET_GPU_SHARED_INDEX=1 SINGLET_ENGINE_GPU_SLOTS=$SLOTS SINGLET_ENGINE_GPU_MEM_GB=$MEMGB SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16 SINGLET_POOL_MEMSAMPLE=2
for kv in ${XENV:-}; do export "$kv"; done
PARGS="--no-hla --hla-t1k $W/jobs/gpu-campaign/agents/hla-speed/t1k-p3b --no-editing"
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
mypids() { find $CG -name cgroup.procs -exec cat {} + 2>/dev/null | sort -u; }
killmine() { local p; for p in $(mypids); do [ "$p" = "$$" ] && continue; tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -aq -e "$2" && kill -$1 $p 2>/dev/null; done; true; }
buskeys() { nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed -E 's/^0000//; s/[:.]/_/g'; }
stopholders() { local b; killmine TERM gpu-holder; sleep 3; for b in $(buskeys); do rm -f /tmp/singlet_gpu_holder.${b}.* 2>/dev/null; done; true; }
cleanup() { killmine TERM "singlet-gpu"; killmine TERM "nvidia-smi"; stopholders; sleep 2; rm -rf "${SCR:?}"; }
trap 'kill $(jobs -p) 2>/dev/null; cleanup' EXIT INT TERM
BIN=${BIN:?}
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG bin=$BIN items='$ITEMS' workers=$WORKERS T=$T slots=$SLOTS memgb=$MEMGB xenv='${XENV:-}' mem=${SLURM_MEM_PER_NODE:-?} $(date -Is)" | tee $R
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | tee -a $R; free -g | head -2 | tee -a $R
echo "node at start: $(awk '/MemFree|^Shmem:|^Cached:/{printf "%s %d GB; ", $1, $2/1048576}' /proc/meminfo); squeue on node: $(squeue -h -w $(hostname -s) -o '%i/%j/%u/%m' | paste -sd' ')" | tee -a $R
echo "cgroup $CG memory.max $(cat $CG/memory.max)" | tee -a $R
t=$(date +%s); for s in $(printf '%s\n' $ITEMS | sort -u); do
  F=$W/data/gpu-deep/$s/$s.1fq; [ -s $F ] || F=$PB/data/$s/$s.1fq; [ -s $F ] || { echo "MISSING INPUT $s" | tee -a $R; exit 2; }
  cp $F $SCR/data/$s.1fq & done; wait; echo "input copy to node-local disk: $(( $(date +%s) - t )) s" | tee -a $R
stopholders
: > $SCR/run/queue0; n=0
for s in $ITEMS; do mkdir -p $SCR/run/c0.$n; echo "$SCR/data/$s.1fq $SCR/run/c0.$n/out/ --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf" >> $SCR/run/queue0; echo -e "$n\t$s" >> $O/items.tsv; n=$((n+1)); done
POOLDIR=$O/pool0 MARK_N=$n $W/jobs/gpu-runtime/t2-cgsamp.sh $O/cg1hz.tsv $O/markers.tsv 2>/dev/null &
SMP=$!
[ -x $W/jobs/gpu-runtime/t2-cgsamp.sh ] || echo "NO SAMPLER" | tee -a $R
st=$(date +%s)
setsid bash -c 'exec "$0" gpu-pool --device 0 --queue '$SCR/run/queue0' --slots '$SLOTS' --workers '$WORKERS' --threads '$T' --report '$O/pool0' --timeout '$LIM' --args "--ref-base '$REF' --pipeline --output-format 1pz '"$PARGS"' --engine gpu"' "$BIN" > $O/pool0/stdout.txt 2>&1
en=$(date +%s); sleep 3; kill $SMP 2>/dev/null
echo "pool wall $(( en - st )) s" | tee -a $R
cat $O/pool0/samples.tsv 2>/dev/null | tee -a $R
for f in memory.events memory.peak memory.max; do echo "cgroup $f: $(cat $CG/$f 2>/dev/null | paste -sd" ")" | tee -a $R; done
grep -a "memory admission\|admission\|phase drop\|peak \|model exceeded\|baseline\|start item\|end item" $O/pool0/pool.log | tee -a $R | head -150
for i in $(seq 0 $((n-1))); do grep -a "peakmodel" $O/pool0/$i.log | sed "s/^/item $i: /" | tee -a $R; done
cp $O/pool0/memsamp-cg.tsv $O/ 2>/dev/null; echo "memory.events (grep oom_kill): $(grep oom_kill $CG/memory.events 2>/dev/null)" | tee -a $R
echo "done $(date -Is)" | tee -a $R
