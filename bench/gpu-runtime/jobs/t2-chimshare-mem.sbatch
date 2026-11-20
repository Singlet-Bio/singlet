#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 27 --mem=200G --time=03:59:00 --exclude=g050 -J t2-chimshare-mem -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-chimshare-mem (chimshare task, throughput track 2; from t2-chimpass.sbatch): the memory of the --chimeric second pass, one item at a time in a gpu-pool of ONE worker, one run per entry
# of RUNS = "<sample>:<threads>:<on|off> ...": the pass's --runThreadN is the item's thread count; on = SINGLET_CHIM_SHARE_GENOME unset (the default), off = 0. Samples: gsm (GSM6250953, two SRA runs
# encoded on the node), or a benchmark .1fq ($W/data/gpu-deep/<s>/<s>.1fq, else $PB/data/<s>/<s>.1fq). Label <sample>_T<threads>_<on|off>.
# The 1 Hz watcher (t2-chimpass-watch.sh) starts at the pass's chim_Log.out: the rise = max(cgroup anon+shmem) - its first sample (as T2-chimpass.md measured it).
# CPU1=<sample> additionally runs that sample once WITHOUT a pool (--engine cpu, 16 threads): the sharing line must be absent and the table md5 as expected.
# Parameters: BIN, RUNS, CPU1, TAG, LIM
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference; JR=$W/jobs/gpu-runtime
BIN=${BIN:?}; TAG=${TAG:-m}; SLOTS=2; MEMGB=31; LIM=${LIM:-3600}; GD=$REF/GRCh38-2024-A; RUNS=${RUNS:?}; CPU1=${CPU1:-}
O=$JR/results/t2-chimshare-mem-$SLURM_JOB_ID-$TAG; mkdir -p $O; SCR=/tmp/$USER-chimshare-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/run; R=$O/results.txt
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_ALLSITE_TIMING=1
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70
export SINGLET_GPU_SHARED_INDEX=1 SINGLET_ENGINE_GPU_SLOTS=$SLOTS SINGLET_ENGINE_GPU_MEM_GB=$MEMGB SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
mypids() { find $CG -name cgroup.procs -exec cat {} + 2>/dev/null | sort -u; }
killmine() { local p; for p in $(mypids); do [ "$p" = "$$" ] && continue; tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -aq -e "$2" && kill -$1 $p 2>/dev/null; done; true; }
buskeys() { nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed -E 's/^0000//; s/[:.]/_/g'; }
stopholders() { local b; killmine TERM gpu-holder; sleep 3; for b in $(buskeys); do rm -f /tmp/singlet_gpu_holder.${b}.* 2>/dev/null; done; true; }
cleanup() { killmine TERM "singlet-gpu"; killmine TERM t2-chimpass-watch; stopholders; sleep 2; rm -rf "${SCR:?}"; }
trap 'kill $(jobs -p) 2>/dev/null; cleanup' EXIT INT TERM
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG bin=$BIN runs='$RUNS' cpu1='$CPU1' $(date -Is)" | tee $R
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | tee -a $R; free -g | head -2 | tee -a $R
echo "cgroup $CG memory.max $(cat $CG/memory.max)" | tee -a $R
echo "node at start: $(awk '/MemFree|^Shmem:|^Cached:/{printf "%s %d GB; ", $1, $2/1048576}' /proc/meminfo); squeue on node: $(squeue -h -w $(hostname -s) -o '%i/%j/%u/%m' | paste -sd' ')" | tee -a $R
declare -A FQ
t=$(date +%s)
for s in $( (for r in $RUNS; do echo ${r%%:*}; done; [ -n "$CPU1" ] && echo $CPU1) | sort -u); do
  if [ "$s" = gsm ]; then
    for r in SRR19716870 SRR19716869; do [ -s $W/campaign/dryrun2/stage/$r/$r.sra ] || { echo "MISSING INPUT $r" | tee -a $R; exit 2; }; done
    ( cd $SCR/data && $BIN download $W/campaign/dryrun2/stage/SRR19716870/SRR19716870.sra -o $SCR/data/gsm0.1fq > $O/encode-0.txt 2>&1 ) &
    ( cd $SCR/data && $BIN download $W/campaign/dryrun2/stage/SRR19716869/SRR19716869.sra -o $SCR/data/gsm1.1fq > $O/encode-1.txt 2>&1 ) &
    FQ[gsm]="$SCR/data/gsm0.1fq $SCR/data/gsm1.1fq"
  else
    F=$W/data/gpu-deep/$s/$s.1fq; [ -s $F ] || F=$PB/data/$s/$s.1fq; [ -s $F ] || { echo "MISSING INPUT $s" | tee -a $R; exit 2; }
    cp $F $SCR/data/$s.1fq & FQ[$s]="$SCR/data/$s.1fq"
  fi
done; wait; echo "input copy / encode on node-local disk: $(( $(date +%s) - t )) s" | tee -a $R
stopholders
for r in $RUNS; do
  s=${r%%:*}; rest=${r#*:}; T=${rest%%:*}; sh=${rest#*:}; L=${s}_T${T}_${sh}
  dd=$SCR/run/$L/out; mkdir -p $SCR/run/$L; set -- ${FQ[$s]}
  echo "$1 $dd/ ${*:2} --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf --chimeric" > $SCR/run/queue.$L; mkdir -p $O/pool.$L
  bash $JR/t2-chimpass-watch.sh $dd $O $L 0 "" &
  WP=$!
  echo "RUN $L start $(date -Is) node memory: $(awk '/MemFree|^Shmem:|^Cached:/{printf "%s %d GB; ", $1, $2/1048576}' /proc/meminfo)" | tee -a $R
  st=$(date +%s)
  ( [ "$sh" = off ] && export SINGLET_CHIM_SHARE_GENOME=0
    setsid bash -c 'exec "$0" gpu-pool --device 0 --queue '$SCR/run/queue.$L' --slots '$SLOTS' --workers 1 --threads '$T' --report '$O/pool.$L' --timeout '$LIM' --args "--ref-base '$REF' --pipeline --output-format 1pz --no-hla --engine gpu"' "$BIN" > $O/pool.$L/stdout.txt 2>&1 )
  en=$(date +%s); sleep 3; kill $WP 2>/dev/null; wait $WP 2>/dev/null
  echo "RUN $L pool wall $(( en - st )) s" | tee -a $R
  cut -f1,7,14 $O/pool.$L/samples.tsv | sed "s/^/RUN $L samples: /" | tee -a $R
  cp $O/pool.$L/0.log $O/$L.pool.log 2>/dev/null
  grep -a -h '\[singlet\] chimeric\|\[singlet\] Total\|\[gpu-pool\] genome' $O/pool.$L/0.log $O/pool.$L/pool.log 2>/dev/null | cut -c1-500 | sed "s/^/RUN $L LOG: /" | tee -a $R
  echo "RUN $L table md5 (decompressed) $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | md5sum | cut -c1-32) rows $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | wc -l)" | tee -a $R
  awk -F'\t' 'BEGIN{OFS=""} /^#/{next} NF>=14 {a=($12+$13)/1073741824; if(n==0){f=a; mx=a}; if(a>mx)mx=a; n++} END{printf "RUN '$L' pass memory (cgroup anon+shmem, GiB): first sample %.2f, max %.2f, RISE %.2f (%d samples)\n", f, mx, mx-f, n}' $O/$L.watch.tsv | tee -a $R
  awk -F'\t' '/^#/{next} NF>=15 {if($8!="" && $8+0>m)m=$8+0; if($7!="" && $7+0>h)h=$7+0} END{printf "RUN '$L' pass child: RssAnon max %.2f GiB, VmHWM max %.2f GiB\n", m/1048576, h/1048576}' $O/$L.watch.tsv | tee -a $R
  rm -rf $dd/chim_* 2>/dev/null
done
if [ -n "$CPU1" ]; then
  L=${CPU1}_cpu; dd=$SCR/run/$L/out; mkdir -p $dd; set -- ${FQ[$CPU1]}
  echo "RUN $L (no pool, --engine cpu, 16 threads) start $(date -Is)" | tee -a $R
  st=$(date +%s)
  ( cd $SCR/run/$L && timeout -k 30 7200 $BIN "$@" --ref-base $REF --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf --out-prefix $dd/ --threads 16 --pipeline --output-format 1pz --no-hla --engine cpu --chimeric 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $O/$L.run.log.ts )
  echo "RUN $L wall $(( $(date +%s) - st )) s" | tee -a $R
  grep -a -h '\[singlet\] chimeric\|\[singlet\] Total' $O/$L.run.log.ts | cut -c1-500 | sed "s/^/RUN $L LOG: /" | tee -a $R
  echo "RUN $L sharing line count: $(grep -a -c 'genome shared with the pool' $O/$L.run.log.ts)" | tee -a $R
  echo "RUN $L table md5 (decompressed) $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | md5sum | cut -c1-32) rows $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | wc -l)" | tee -a $R
fi
echo "done $(date -Is)" | tee -a $R
