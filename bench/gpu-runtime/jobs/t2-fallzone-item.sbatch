#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 27 --mem=300G --time=01:30:00 --exclude=g050 -J t2-fz-item -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-fallzone-item (fallzone): ONE H100 (not exclusive), ONE worker, the items of ITEMS one after the other in one pool (one STAR/engine process each), the budget and slot count of the
# arm (MEMGB, SLOTS, XENV), SINGLET_E16_TRACE=1 (size marks), the 1 Hz cgroup sampler and a 1 Hz nvidia-smi memory sampler. Per item: wall, the engine's level / budget / read-chunk / fall-back
# lines (copied: item<N>.engine.log, item<N>.star_Log.out), the device peak (the engine's own "peak" line and nvidia-smi inside the item's window), the host peak (cgroup anon+shmem in the
# window, and the idle floor of the run), and the md5 of every non-private file of the item's output (files<N>.txt) and a set hash over the matrix, junction and .pz/.1pz files.
# NO file under any private/ directory is opened (the hash set leaves them out: they are the sensitive ones). Derived from t2-peakmodel-item.sbatch.
# Parameters: BIN, ITEMS, TAG, T, SLOTS, MEMGB, LIM, XENV
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
ITEMS=${ITEMS:-d100s_3p nuclei2k_v2}; WORKERS=1; TAG=${TAG:-fz}; T=${T:-8}; SLOTS=${SLOTS:-2}; MEMGB=${MEMGB:-31}; LIM=${LIM:-3000}; GD=$REF/GRCh38-2024-A
O=$W/jobs/gpu-runtime/results/t2-fallzone-$SLURM_JOB_ID-$TAG; mkdir -p $O/pool0; SCR=/tmp/$USER-fz-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/run; R=$O/results.txt
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
( while :; do echo "$(date +%s.%N) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"; sleep 1; done > $O/gpumem.tsv 2>/dev/null ) &
GSM=$!
st=$(date +%s)
setsid bash -c 'exec "$0" gpu-pool --device 0 --queue '$SCR/run/queue0' --slots '$SLOTS' --workers '$WORKERS' --threads '$T' --report '$O/pool0' --timeout '$LIM' --args "--ref-base '$REF' --pipeline --output-format 1pz '"$PARGS"' --engine gpu"' "$BIN" > $O/pool0/stdout.txt 2>&1
en=$(date +%s); sleep 3; kill $SMP $GSM 2>/dev/null
echo "pool wall $(( en - st )) s" | tee -a $R
cat $O/pool0/samples.tsv 2>/dev/null | tee -a $R
# per item: the engine's lines, the peaks, the hashes
hashset() { local d=$1 f; (cd $d && find . -maxdepth 3 -type f -not -path './private/*' | sort | while read -r f; do echo "$(md5sum < $f | cut -c1-32) $f"; done); }
coreset() { grep -E ' \./(star_Solo\.out/Gene/(raw|filtered)/(matrix\.mtx|barcodes\.tsv|features\.tsv)|star_SJ\.out\.tab)$| \./.*\.(pz|pz2|1pz)$' "$1"; }
awk -F'\t' 'NR>1{print $1"\t"$5"\t"$6"\t"$7"\t"$3}' $O/pool0/samples.tsv > $O/windows.tsv
while IFS=$'\t' read -r i s0 e0 wl nm; do
  nm=$(basename $nm .1fq); dd=$SCR/run/c0.$i/out
  cp $dd/star_Log.out $O/item$i.star_Log.out 2>/dev/null; grep -a -i 'engine_gpu\|pileup\|allsite-gpu\|gpu_pileup' $dd/star_Log.out > $O/item$i.engine.log 2>/dev/null
  grep -a '\[e16\]' $O/pool0/$i.log > $O/item$i.e16.txt 2>/dev/null
  hashset $dd > $O/files$i.txt; coreset $O/files$i.txt > $O/core$i.txt
  dev=$(awk -v a=$s0 -v b=$e0 '$1>=a && $1<=b && $2+0>m{m=$2+0} END{printf "%.1f", m/1024}' $O/gpumem.tsv)
  host=$(awk -F'\t' -v a=$s0 -v b=$e0 'NR>1 && $1>=a && $1<=b{v=$4+$6; if(v>m)m=v} END{printf "%.1f", m/1073741824}' $O/cg1hz.tsv)
  echo "ITEM $i $nm wall ${wl} s; device peak nvidia-smi ${dev} GiB (whole card, 1 s); host peak cgroup anon+shmem ${host} GiB; files $(wc -l < $O/files$i.txt), core files $(wc -l < $O/core$i.txt), core set hash $(md5sum < $O/core$i.txt | cut -c1-32), all-files list hash $(md5sum < $O/files$i.txt | cut -c1-32)" | tee -a $R
  grep -a -h 'E19: level\|E17 budget\|E8 records not built\|read chunks\|FELL BACK\|level 0 (requested\|peak (lanes\|engine_gpu: [0-9]* reads,' $dd/star_Log.out | cut -c1-420 | sed "s/^/ITEM $i LINE: /" | tee -a $R
done < $O/windows.tsv
awk -F'\t' 'NR>1{v=$4+$6; if(m==""||v<m)m=v} END{printf "idle floor of the run (min cgroup anon+shmem): %.1f GiB\n", m/1073741824}' $O/cg1hz.tsv | tee -a $R
for f in memory.events memory.peak memory.max; do echo "cgroup $f: $(cat $CG/$f 2>/dev/null | paste -sd" ")" | tee -a $R; done
echo "done $(date -Is)" | tee -a $R
