#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 27 --mem=200G --time=03:59:00 --exclude=g050 -J t2-chimshare-pool -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-chimshare-pool (chimshare task, from t2-integ2-chim.sbatch): the three --chimeric items GSM6250953, d300_5p, pbmc5p_v1 in ONE pool (3 workers, 8 threads, budget 31 GB, 2 slots, --no-hla, --engine gpu,
# --chimeric) on one H100, --mem=200G, run once per arm of ARMS ("<tag>:<env,env>"; default S = sharing on (the default), N = SINGLET_CHIM_SHARE_GENOME=0; add "P:SINGLET_POOL_PEAKADMIT=1" for the peak-model gate) one
# after the other on the same node and the same node-local inputs. Per arm: results dir t2-chimshare-pool-<job>-<tag>.
# Parameters: BIN, ITEMS (default "gsm d300_5p pbmc5p_v1"), ARMS, WORKERS (3), T (8), SLOTS (2), MEMGB (31), LIM
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
ITEMS=${ITEMS:-gsm d300_5p pbmc5p_v1}; WORKERS=${WORKERS:-3}; T=${T:-8}; SLOTS=${SLOTS:-2}; MEMGB=${MEMGB:-31}; LIM=${LIM:-7000}; GD=$REF/GRCh38-2024-A
ARMS=${ARMS:-S: N:SINGLET_CHIM_SHARE_GENOME=0}; CPU1=${CPU1:-}
SCR=/tmp/$USER-chimshare-pool-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/run
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_ALLSITE_TIMING=1
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70
export SINGLET_GPU_SHARED_INDEX=1 SINGLET_ENGINE_GPU_SLOTS=$SLOTS SINGLET_ENGINE_GPU_MEM_GB=$MEMGB SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16 SINGLET_POOL_MEMSAMPLE=2
PARGS="--no-hla"
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
mypids() { find $CG -name cgroup.procs -exec cat {} + 2>/dev/null | sort -u; }
killmine() { local p; for p in $(mypids); do [ "$p" = "$$" ] && continue; tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -aq -e "$2" && kill -$1 $p 2>/dev/null; done; true; }
buskeys() { nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed -E 's/^0000//; s/[:.]/_/g'; }
stopholders() { local b; killmine TERM gpu-holder; sleep 3; for b in $(buskeys); do rm -f /tmp/singlet_gpu_holder.${b}.* 2>/dev/null; done; true; }
cleanup() { killmine TERM "singlet-gpu"; killmine TERM "nvidia-smi"; stopholders; sleep 2; rm -rf "${SCR:?}"; }
trap 'kill $(jobs -p) 2>/dev/null; cleanup' EXIT INT TERM
BIN=${BIN:?}
O0=$W/jobs/gpu-runtime/results/t2-chimshare-pool-$SLURM_JOB_ID; mkdir -p $O0; R0=$O0/results.txt
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$BIN items='$ITEMS' arms='$ARMS' workers=$WORKERS T=$T slots=$SLOTS memgb=$MEMGB mem=${SLURM_MEM_PER_NODE:-?} $(date -Is)" | tee $R0
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | tee -a $R0; free -g | head -2 | tee -a $R0
declare -A FQ
t=$(date +%s); for s in $(printf '%s\n' $ITEMS | sort -u); do
  if [ "$s" = gsm ]; then
    for r in SRR19716870 SRR19716869; do [ -s $W/campaign/dryrun2/stage/$r/$r.sra ] || { echo "MISSING INPUT $r" | tee -a $R0; exit 2; }; done
    ( cd $SCR/data && $BIN download $W/campaign/dryrun2/stage/SRR19716870/SRR19716870.sra -o $SCR/data/gsm0.1fq > $O0/encode-0.txt 2>&1 ) &
    ( cd $SCR/data && $BIN download $W/campaign/dryrun2/stage/SRR19716869/SRR19716869.sra -o $SCR/data/gsm1.1fq > $O0/encode-1.txt 2>&1 ) &
    FQ[gsm]="$SCR/data/gsm0.1fq $SCR/data/gsm1.1fq"
  else
    F=$W/data/gpu-deep/$s/$s.1fq; [ -s $F ] || F=$PB/data/$s/$s.1fq; [ -s $F ] || { echo "MISSING INPUT $s" | tee -a $R0; exit 2; }
    cp $F $SCR/data/$s.1fq & FQ[$s]="$SCR/data/$s.1fq"
  fi
done; wait; echo "input copy / encode on node-local disk: $(( $(date +%s) - t )) s" | tee -a $R0
for arm in $ARMS; do
  TAG=${arm%%:*}; XENV=${arm#*:}; XENV=${XENV//,/ }
  O=$O0-$TAG; mkdir -p $O/pool0; R=$O/results.txt; rm -rf "${SCR:?}/run"; mkdir -p $SCR/run
  echo "ARM $TAG xenv='$XENV' start $(date -Is) node at start: $(awk '/MemFree|^Shmem:|^Cached:/{printf "%s %d GB; ", $1, $2/1048576}' /proc/meminfo); squeue on node: $(squeue -h -w $(hostname -s) -o '%i/%j/%u/%m' | paste -sd' ')" | tee -a $R | tee -a $R0
  stopholders
  : > $SCR/run/queue0; n=0
  for s in $ITEMS; do mkdir -p $SCR/run/c0.$n; set -- ${FQ[$s]}; echo "$1 $SCR/run/c0.$n/out/ ${*:2} --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf --chimeric" >> $SCR/run/queue0; echo -e "$n\t$s" >> $O/items.tsv; n=$((n+1)); done
  POOLDIR=$O/pool0 MARK_N=$n $W/jobs/gpu-runtime/t2-cgsamp.sh $O/cg1hz.tsv $O/markers.tsv 2>/dev/null &
  SMP=$!
  st=$(date +%s)
  ( for kv in $XENV; do export "$kv"; done
    setsid bash -c 'exec "$0" gpu-pool --device 0 --queue '$SCR/run/queue0' --slots '$SLOTS' --workers '$WORKERS' --threads '$T' --report '$O/pool0' --timeout '$LIM' --args "--ref-base '$REF' --pipeline --output-format 1pz '"$PARGS"' --engine gpu"' "$BIN" > $O/pool0/stdout.txt 2>&1 )
  en=$(date +%s); sleep 3; kill $SMP 2>/dev/null
  echo "ARM $TAG pool wall $(( en - st )) s" | tee -a $R | tee -a $R0
  cat $O/pool0/samples.tsv 2>/dev/null | tee -a $R
  for f in memory.events memory.peak memory.max; do echo "cgroup $f (job-wide, cumulative over the arms so far): $(cat $CG/$f 2>/dev/null | paste -sd" ")" | tee -a $R; done
  grep -a "memory admission\|admission\|phase drop\|peak \|model exceeded\|baseline\|start item\|end item" $O/pool0/pool.log | tee -a $R | head -150
  for i in $(seq 0 $((n-1))); do grep -a "peakmodel" $O/pool0/$i.log | sed "s/^/item $i: /" | tee -a $R; done
  for i in $(seq 0 $((n-1))); do s=$(sed -n "$((i+1))p" $O/items.tsv | cut -f2); L=$O/pool0/$i.log
    grep -a -h '\[singlet\] chimeric\|\[singlet\] Total' $L | cut -c1-500 | sed "s/^/item $i $s LOG: /" | tee -a $R
    grep -a -h '\[gpu-pool\] genome' $L | cut -c1-300 | sed "s/^/item $i $s POOL: /" | tee -a $R
    Z=$SCR/run/c0.$i/out/chimeric_junctions.tsv.gz; cp $Z $O/item$i.chimeric_junctions.tsv.gz 2>/dev/null
    echo "item $i $s table md5 (decompressed) $(zcat $Z 2>/dev/null | md5sum | cut -c1-32) rows $(zcat $Z 2>/dev/null | wc -l)" | tee -a $R | tee -a $R0; done
  cp $O/pool0/memsamp-cg.tsv $O/ 2>/dev/null; echo "memory.events (grep oom_kill): $(grep oom_kill $CG/memory.events 2>/dev/null)" | tee -a $R
  echo "ARM $TAG done $(date -Is)" | tee -a $R | tee -a $R0
done
if [ -n "$CPU1" ]; then   # one --chimeric item without a pool (--engine cpu): no sharing line, table md5 as expected
  L=${CPU1}_cpu; dd=$SCR/run/$L/out; mkdir -p $dd; set -- ${FQ[$CPU1]}
  echo "RUN $L (no pool, --engine cpu, 16 threads) start $(date -Is)" | tee -a $R0
  st=$(date +%s)
  ( cd $SCR/run/$L && timeout -k 30 7200 $BIN "$@" --ref-base $REF --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf --out-prefix $dd/ --threads 16 --pipeline --output-format 1pz --no-hla --engine cpu --chimeric 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $O0/$L.run.log.ts )
  echo "RUN $L wall $(( $(date +%s) - st )) s" | tee -a $R0
  grep -a -h '\[singlet\] chimeric\|\[singlet\] Total' $O0/$L.run.log.ts | cut -c1-500 | sed "s/^/RUN $L LOG: /" | tee -a $R0
  echo "RUN $L sharing line count: $(grep -a -c 'genome shared with the pool' $O0/$L.run.log.ts)" | tee -a $R0
  echo "RUN $L table md5 (decompressed) $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | md5sum | cut -c1-32) rows $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | wc -l)" | tee -a $R0
fi
echo "done $(date -Is)" | tee -a $R0
