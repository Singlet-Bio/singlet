#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 27 --mem=300G --time=03:00:00 --exclude=g050 -J t2-rc-item -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-recrc-item (recrc): derived from t2-fallzone-item.sbatch. ONE H100 (not exclusive), WORKERS workers (default 1), the items of ITEMS in one pool (one STAR/engine process each) per arm,
# the arms of ARMS one after the other on the same node and card: 'new' = default switches, 'old' = SINGLET_ENGINE_GPU_REC_CHUNK=0 (today's behaviour). Budget and slots from MEMGB, SLOTS, XENV.
# Per arm and item: wall, the engine's level / budget / record-chunk / tally / fall-back lines, the host peak (cgroup anon+shmem in the item's window), the card peak
# (nvidia-smi, 1 Hz, whole card, in the item's window and over the arm), and the md5 of every non-private file and a set hash over the matrix, junction and .pz/.1pz files.
# NO file under any private/ directory is opened (the hash set leaves them out).
# Parameters: BIN, ITEMS, ARMS, WORKERS, TAG, T, SLOTS, MEMGB, LIM, XENV
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
ITEMS=${ITEMS:-d100s_3p nuclei2k_v2}; WORKERS=${WORKERS:-1}; ARMS=${ARMS:-new}; TAG=${TAG:-rc}; T=${T:-8}; SLOTS=${SLOTS:-2}; MEMGB=${MEMGB:-31}; LIM=${LIM:-3000}; GD=$REF/GRCh38-2024-A
OBASE=$W/jobs/gpu-runtime/results/t2-recrc-$SLURM_JOB_ID-$TAG; mkdir -p $OBASE; SCR=/tmp/$USER-rc-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/run; R=$OBASE/results.txt
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_ALLSITE_TIMING=1
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70
export SINGLET_GPU_SHARED_INDEX=1 SINGLET_ENGINE_GPU_SLOTS=$SLOTS SINGLET_ENGINE_GPU_MEM_GB=$MEMGB SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16 SINGLET_POOL_MEMSAMPLE=2 SINGLET_E16_TRACE=1
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
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG bin=$BIN items='$ITEMS' arms='$ARMS' workers=$WORKERS T=$T slots=$SLOTS memgb=$MEMGB xenv='${XENV:-}' mem=${SLURM_MEM_PER_NODE:-?} $(date -Is)" | tee $R
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | tee -a $R; free -g | head -2 | tee -a $R
echo "node at start: $(awk '/MemFree|^Shmem:|^Cached:/{printf "%s %d GB; ", $1, $2/1048576}' /proc/meminfo); squeue on node: $(squeue -h -w $(hostname -s) -o '%i/%j/%u/%m' | paste -sd' ')" | tee -a $R
echo "cgroup $CG memory.max $(cat $CG/memory.max)" | tee -a $R
t=$(date +%s); for s in $(printf '%s\n' $ITEMS | sort -u); do
  F=$W/data/gpu-deep/$s/$s.1fq; [ -s $F ] || F=$PB/data/$s/$s.1fq; [ -s $F ] || { echo "MISSING INPUT $s" | tee -a $R; exit 2; }
  cp $F $SCR/data/$s.1fq & done; wait; echo "input copy to node-local disk: $(( $(date +%s) - t )) s" | tee -a $R
hashset() { local d=$1 f; (cd $d && find . -maxdepth 3 -type f -not -path './private/*' | sort | while read -r f; do echo "$(md5sum < $f | cut -c1-32) $f"; done); }
coreset() { grep -E ' \./(star_Solo\.out/Gene/(raw|filtered)/(matrix\.mtx|barcodes\.tsv|features\.tsv)|star_SJ\.out\.tab)$| \./.*\.(pz|pz2|1pz)$' "$1"; }
runarm() {
  local arm=$1 k=$2 O=$OBASE/$1$2; mkdir -p $O/pool0
  local AENV=""; [ "$arm" = old ] && AENV="SINGLET_ENGINE_GPU_REC_CHUNK=0"
  [ -n "$AENV" ] && export $AENV || unset SINGLET_ENGINE_GPU_REC_CHUNK
  echo "=== ARM $arm$k ($AENV) $(date -Is) ===" | tee -a $R
  rm -rf "${SCR:?}/run"; mkdir -p $SCR/run
  stopholders
  : > $SCR/run/queue0; local n=0 s
  for s in $ITEMS; do mkdir -p $SCR/run/c0.$n; echo "$SCR/data/$s.1fq $SCR/run/c0.$n/out/ --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf" >> $SCR/run/queue0; echo -e "$n\t$s" >> $O/items.tsv; n=$((n+1)); done
  POOLDIR=$O/pool0 MARK_N=$n $W/jobs/gpu-runtime/t2-cgsamp.sh $O/cg1hz.tsv $O/markers.tsv 2>/dev/null &
  local SMP=$!
  ( while :; do echo "$(date +%s.%N) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"; sleep 1; done > $O/gpumem.tsv 2>/dev/null ) &
  local GSM=$!
  local st=$(date +%s) en
  setsid bash -c 'exec "$0" gpu-pool --device 0 --queue '$SCR/run/queue0' --slots '$SLOTS' --workers '$WORKERS' --threads '$T' --report '$O/pool0' --timeout '$LIM' --args "--ref-base '$REF' --pipeline --output-format 1pz '"$PARGS"' --engine gpu"' "$BIN" > $O/pool0/stdout.txt 2>&1
  en=$(date +%s); sleep 3; kill $SMP $GSM 2>/dev/null
  echo "pool wall $(( en - st )) s (arm $arm$k)" | tee -a $R
  cat $O/pool0/samples.tsv 2>/dev/null | tee -a $R
  awk -F'\t' 'NR>1{print $1"\t"$5"\t"$6"\t"$7"\t"$3}' $O/pool0/samples.tsv > $O/windows.tsv
  local i s0 e0 wl nm dd dev host
  while IFS=$'\t' read -r i s0 e0 wl nm; do
    nm=$(basename $nm .1fq); dd=$SCR/run/c0.$i/out
    cp $dd/star_Log.out $O/item$i.star_Log.out 2>/dev/null; grep -a -i 'engine_gpu\|pileup\|allsite-gpu\|gpu_pileup' $dd/star_Log.out > $O/item$i.engine.log 2>/dev/null
    hashset $dd > $O/files$i.txt; coreset $O/files$i.txt > $O/core$i.txt
    dev=$(awk -v a=$s0 -v b=$e0 '$1>=a && $1<=b && $2+0>m{m=$2+0} END{printf "%.1f", m/1024}' $O/gpumem.tsv)
    host=$(awk -F'\t' -v a=$s0 -v b=$e0 'NR>1 && $1>=a && $1<=b{v=$4+$6; if(v>m)m=v} END{printf "%.1f", m/1073741824}' $O/cg1hz.tsv)
    echo "ITEM $arm$k/$i $nm wall ${wl} s; card peak nvidia-smi ${dev} GiB (whole card, 1 s, item window); host peak cgroup anon+shmem ${host} GiB; files $(wc -l < $O/files$i.txt), core files $(wc -l < $O/core$i.txt), core set hash $(md5sum < $O/core$i.txt | cut -c1-32), all-files list hash $(md5sum < $O/files$i.txt | cut -c1-32)" | tee -a $R
    grep -a -h 'E19: level\|E17 budget\|E17 records by record chunks\|E8 records on the device\|E8 records not built\|read chunks\|FELL BACK\|level 0 (requested\|peak (lanes\|tally peak\|engine_gpu: [0-9]* reads,' $dd/star_Log.out | cut -c1-420 | sed "s/^/ITEM $arm$k\/$i LINE: /" | tee -a $R
  done < $O/windows.tsv
  grep -a -h 'records by record chunks' $O/pool0/*.log $O/pool0/stdout.txt 2>/dev/null | cut -c1-300 | sed "s/^/STDERR-LINE $arm$k: /" | tee -a $R
  awk -v a=$st -v b=$en '$1>=a && $1<=b && $2+0>m{m=$2+0} END{printf "ARM %s card peak nvidia-smi (1 s, whole card) over the pool window: %.1f GiB\n", "'$arm$k'", m/1024}' $O/gpumem.tsv | tee -a $R
  awk -F'\t' 'NR>1{v=$4+$6; if(m==""||v<m)m=v} END{printf "ARM '$arm$k' idle floor (min cgroup anon+shmem): %.1f GiB\n", m/1073741824}' $O/cg1hz.tsv | tee -a $R
}
k=0; for a in $ARMS; do k=$((k+1)); runarm $a $k; done
for f in memory.events memory.peak memory.max; do echo "cgroup $f: $(cat $CG/$f 2>/dev/null | paste -sd" ")" | tee -a $R; done
echo "done $(date -Is)" | tee -a $R
