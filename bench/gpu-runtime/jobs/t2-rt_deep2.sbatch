#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 27 --mem=700G --exclusive --time=03:59:00 --exclude=g050 -J rt-deep2 -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-rt_deep2 (hostcut round 3): TWO items of the sample in ONE pool (--workers 2) on an exclusive node; memory admission decides whether they run together.
# rt-deep: one deep sample alone on one H100 (one gpu-pool, one item), device memory sampled every second; derived from rt_node.sbatch.
# Parameters: BIN (singlet-gpu), S (sample dir under data/gpu-deep), TAG, T (threads), XENV ("A=1 B=2"), SLOTS, MEMGB, LIM, KEEP (1: keep the output)
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
S=${S:-d1000_3p}; TAG=${TAG:-base}; T=${T:-16}; SLOTS=${SLOTS:-2}; MEMGB=${MEMGB:-31}; LIM=${LIM:-13000}; GD=$REF/GRCh38-2024-A
O=$W/jobs/gpu-runtime/results/rt-deep-$SLURM_JOB_ID-$TAG; mkdir -p $O; SCR=/tmp/$USER-rtdeep-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/run/c0.0 $SCR/run/c0.1; R=$O/results.txt
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_ALLSITE_TIMING=1
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70
export SINGLET_GPU_SHARED_INDEX=1 SINGLET_ENGINE_GPU_SLOTS=$SLOTS SINGLET_ENGINE_GPU_MEM_GB=$MEMGB SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16
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
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG bin=$BIN sample=$S T=$T slots=$SLOTS memgb=$MEMGB xenv='${XENV:-}' $(date -Is)" | tee $R
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | tee -a $R; free -g | head -2 | tee -a $R
F=$W/data/gpu-deep/$S/$S.1fq; [ -s $F ] || { echo "MISSING INPUT $F" | tee -a $R; exit 2; }
t=$(date +%s); cp $F $SCR/data/$S.1fq || exit 2; echo "input copy: $(( $(date +%s) - t )) s" | tee -a $R
stopholders
mkdir -p $O/pool0; echo "$SCR/data/$S.1fq $SCR/run/c0.0/out/ --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf" > $SCR/run/queue0; echo "$SCR/data/$S.1fq $SCR/run/c0.1/out/ --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf" >> $SCR/run/queue0
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 1000 > $O/gpumem.csv 2>&1 &
( while :; do sleep 10; echo -e "$(date +%s)\t$(awk '$1=="anon"||$1=="shmem"{t+=$2} END{printf "%.1f", t/1073741824}' $CG/memory.stat)"; done >> $O/hostmem.tsv ) &
st=$(date +%s)
setsid bash -c 'exec "$0" gpu-pool --device 0 --queue '$SCR/run/queue0' --slots '$SLOTS' --workers 2 --threads '$T' --report '$O/pool0' --timeout '$LIM' --args "--ref-base '$REF' --pipeline --output-format 1pz '"$PARGS"' --engine gpu"' "$BIN" > $O/pool0/stdout.txt 2>&1
en=$(date +%s)
dd=$SCR/run/c0.0/out; dd1=$SCR/run/c0.1/out
cp $dd/star_Log.out $O/star_Log.out 2>/dev/null; grep -a -i "engine_gpu\|pileup\|allsite-gpu\|gpu_pileup" $dd/star_Log.out > $O/engine.log 2>/dev/null
cp $dd/read_stats.tsv $O/ 2>/dev/null
echo "pool wall $(( en - st )) s" | tee -a $R
cat $O/pool0/samples.tsv 2>/dev/null | tee -a $R
awk -F, 'NR>0{ if($2+0>m)m=$2+0 } END{printf "device memory peak (nvidia-smi, 1 s): %.1f GiB\n", m/1024}' $O/gpumem.csv | tee -a $R
(
 hashset() { local d=$1 f; for f in private/allsite/allele_delta.bin private/allsite/cells.tsv.gz private/allsite/sites.tsv.gz private/allsite/store.json \
      private/genotype/cells.vcf.gz private/genotype/cells_allsite.vcf.gz private/genotype/donors_allsite.vcf.gz star_Solo.out/Gene/raw/matrix.mtx \
      star_Solo.out/Gene/raw/barcodes.tsv star_Solo.out/Gene/filtered/matrix.mtx star_SJ.out.tab; do
    [ -f $d/$f ] && echo "$(md5sum < $d/$f | cut -c1-32) $f" || echo "missing $f"; done
   (cd $d && find . -maxdepth 3 -type f \( -name '*.pz' -o -name '*.pz2' -o -name '*.1pz' \) | sort | while read -r f; do echo "$(md5sum < $f | cut -c1-32) $f"; done); }
 hashset $dd > $O/files.txt; echo "item0 set hash $(md5sum < $O/files.txt | cut -c1-32), files $(wc -l < $O/files.txt), missing $(grep -c '^missing' $O/files.txt)" | tee -a $R
 hashset $dd1 > $O/files1.txt; echo "item1 set hash $(md5sum < $O/files1.txt | cut -c1-32), files $(wc -l < $O/files1.txt), missing $(grep -c '^missing' $O/files1.txt)" | tee -a $R
 for f in memory.events memory.peak memory.max; do echo "cgroup $f: $(cat $CG/$f 2>/dev/null | paste -sd" ")" | tee -a $R; done
 grep -a "memory admission\|item [01]: \|admission" $O/pool0/pool.log | tee -a $R | head -20 )

echo "done $(date -Is)" | tee -a $R
