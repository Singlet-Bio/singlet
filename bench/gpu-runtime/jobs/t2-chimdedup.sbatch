#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 27 --mem=300G --time=06:00:00 --exclude=g050 -J t2-chimdedup -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-chimdedup (chimdedup task, throughput track 2): the deduplicated --chimeric second pass (SINGLET_CHIM_DEDUP) against today's path, same binary, same job.
# From t2-chimpass.sbatch: ONE H100 (not exclusive), ONE worker, one gpu-pool run per (sample, arm) with the campaign's H100 settings (budget 31 GB, 2 slots,
# --no-hla, --engine gpu, --chimeric), item threads T (default 8).
#   ARMS="<label>:<VAR>=<v>[,<VAR>=<v>...] ..."   environment of each arm, e.g. on:SINGLET_CHIM_DEDUP=1  off:SINGLET_CHIM_DEDUP=0
#         rawoff:SINGLET_CHIM_DEDUP=0,SINGLET_KEEP_CHIMERIC_JUNCTION=1   rawon:SINGLET_CHIM_DEDUP=1,SINGLET_CHIM_DEDUP_DUMP=1
#   SRAS="<staged .sra> ..." : one sample named "gsm" encoded as twin_check.sbatch does (no --protocol), reused by every arm
#   ITEMS="pbmc5p_v1 d300_5p" : benchmark .1fq files ($W/data/gpu-deep/<s>/<s>.1fq, else $PB/data/<s>/<s>.1fq); every arm runs on every item, item-major
# Per run: 1 Hz watcher (t2-chimdedup-watch.sh), the pool log, md5 of the decompressed chimeric_junctions.tsv.gz (plain and sorted). Arms with
# SINGLET_KEEP_CHIMERIC_JUNCTION=1 / SINGLET_CHIM_DEDUP_DUMP=1: the data lines of the raw junction file / of the expanded dump with the read-name column
# replaced by ".", sorted with LC_ALL=C, md5 and line count; the two are compared in the "RAWPROOF" lines (counts and md5 only). NO file under a private/ dir is opened.
# Parameters: BIN, SRAS or ITEMS, ARMS, T (default 8), TAG, LIM (item timeout, s), EXTRA (default --chimeric), ORDER (item-major | arm-major, default item-major)
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference; JR=$W/jobs/gpu-runtime
BIN=${BIN:-$JR/bin/t2-chimdedup/singlet-gpu}; TAG=${TAG:-cd}; SLOTS=2; MEMGB=31; LIM=${LIM:-3600}; GD=$REF/GRCh38-2024-A; EXTRA=${EXTRA:---chimeric}; T=${T:-8}
SRAS=${SRAS:-}; ITEMS=${ITEMS:-}; ARMS=${ARMS:-on:SINGLET_CHIM_DEDUP=1 off:SINGLET_CHIM_DEDUP=0}
O=$JR/results/t2-chimdedup-$SLURM_JOB_ID-$TAG; mkdir -p $O; SCR=/tmp/$USER-chimdedup-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/run; R=$O/results.txt
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_ALLSITE_TIMING=1
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70
export SINGLET_GPU_SHARED_INDEX=1 SINGLET_ENGINE_GPU_SLOTS=$SLOTS SINGLET_ENGINE_GPU_MEM_GB=$MEMGB SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
mypids() { find $CG -name cgroup.procs -exec cat {} + 2>/dev/null | sort -u; }
killmine() { local p; for p in $(mypids); do [ "$p" = "$$" ] && continue; tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -aq -e "$2" && kill -$1 $p 2>/dev/null; done; true; }
buskeys() { nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed -E 's/^0000//; s/[:.]/_/g'; }
stopholders() { local b; killmine TERM gpu-holder; sleep 3; for b in $(buskeys); do rm -f /tmp/singlet_gpu_holder.${b}.* 2>/dev/null; done; true; }
cleanup() { killmine TERM "singlet-gpu"; killmine TERM t2-chimdedup-watch; stopholders; sleep 2; rm -rf "${SCR:?}"; }   # own node-local scratch only
trap 'kill $(jobs -p) 2>/dev/null; cleanup' EXIT INT TERM
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG bin=$BIN sras='$SRAS' items='$ITEMS' arms='$ARMS' T=$T extra='$EXTRA' $(date -Is)" | tee $R
echo "binary sha256 $(sha256sum $BIN | cut -c1-16) commit $(cat $(dirname $BIN)/commit-gpu 2>/dev/null)" | tee -a $R
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | tee -a $R; free -g | head -2 | tee -a $R
echo "nproc $(nproc) CLK_TCK $(getconf CLK_TCK) cgroup $CG memory.max $(cat $CG/memory.max) cpu.max $(cat $CG/cpu.max 2>/dev/null)" | tee -a $R
echo "node at start: $(awk '/MemFree|^Shmem:|^Cached:/{printf "%s %d GB; ", $1, $2/1048576}' /proc/meminfo); squeue on node: $(squeue -h -w $(hostname -s) -o '%i/%j/%u/%m' | paste -sd' ')" | tee -a $R

declare -A FQ; SAMPLES=""
if [ -n "$SRAS" ]; then
  FQS=""; n=0; t=$(date +%s)
  for s in $SRAS; do ( cd $SCR/data && $BIN download $s -o $SCR/data/r$n.1fq > $O/encode-$n.txt 2>&1; echo "rc=$?" >> $O/encode-$n.txt ) & FQS="$FQS $SCR/data/r$n.1fq"; n=$((n+1)); done; wait
  echo "encode of $n runs without --protocol: $(( $(date +%s) - t )) s" | tee -a $R
  for i in $(seq 0 $((n-1))); do echo "encode r$i: $(tail -1 $O/encode-$i.txt)" | tee -a $R; done
  FQ[gsm]="$FQS"; SAMPLES=gsm
else
  t=$(date +%s); for s in $ITEMS; do
    F=$W/data/gpu-deep/$s/$s.1fq; [ -s $F ] || F=$PB/data/$s/$s.1fq; [ -s $F ] || { echo "MISSING INPUT $s" | tee -a $R; exit 2; }
    cp $F $SCR/data/$s.1fq & FQ[$s]="$SCR/data/$s.1fq"; SAMPLES="$SAMPLES $s"; done; wait
  echo "input copy to node-local disk: $(( $(date +%s) - t )) s" | tee -a $R
fi
# the data lines of a raw junction file with the read-name column (10) replaced by ".", sorted: md5 and line count on stdout
norm() { grep -a -v -e '^#' -e '^chr_donorA' "$1" | awk -F'\t' 'BEGIN{OFS="\t"}{$10="."; print}' | LC_ALL=C sort -S 60G --parallel=16 -T $SCR > "$2"; echo "$(md5sum < "$2" | cut -c1-32) $(wc -l < "$2")"; }
stopholders
RUNLIST=""
if [ "${ORDER:-item-major}" = arm-major ]; then for a in $ARMS; do for s in $SAMPLES; do RUNLIST="$RUNLIST $s|$a"; done; done
else for s in $SAMPLES; do for a in $ARMS; do RUNLIST="$RUNLIST $s|$a"; done; done; fi
for RA in $RUNLIST; do
  s=${RA%%|*}; a=${RA#*|}; arm=${a%%:*}; envs=${a#*:}; L=${s}_${arm}
  dd=$SCR/run/$L/out; mkdir -p $SCR/run/$L; set -- ${FQ[$s]}
  echo "$1 $dd/ ${*:2} --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf $EXTRA" > $SCR/run/queue.$L; mkdir -p $O/pool.$L
  bash $JR/t2-chimdedup-watch.sh $dd $O $L &
  WP=$!
  echo "RUN $L env '$envs' threads $T start $(date -Is) node memory: $(awk '/MemFree|^Shmem:|^Cached:/{printf "%s %d GB; ", $1, $2/1048576}' /proc/meminfo)" | tee -a $R
  st=$(date +%s)
  ( IFS=,; for kv in $envs; do export "$kv"; done; unset IFS
    setsid bash -c 'exec "$0" gpu-pool --device 0 --queue '$SCR/run/queue.$L' --slots '$SLOTS' --workers 1 --threads '$T' --report '$O/pool.$L' --timeout '$LIM' --args "--ref-base '$REF' --pipeline --output-format 1pz --no-hla --engine gpu"' "$BIN" > $O/pool.$L/stdout.txt 2>&1 )
  en=$(date +%s); sleep 3; kill $WP 2>/dev/null; wait $WP 2>/dev/null
  echo "RUN $L pool wall $(( en - st )) s" | tee -a $R
  cut -f1,7,14 $O/pool.$L/samples.tsv | sed "s/^/RUN $L samples: /" | tee -a $R
  cp $dd/star_Log.out $O/$L.star_Log.out 2>/dev/null; cp $O/pool.$L/0.log $O/$L.pool.log 2>/dev/null
  grep -a -h 'E19: level\|engine_gpu: [0-9]* reads,\|^runThreadN\|^clip3pAdapterSeq\|SEQCACHE' $dd/star_Log.out | cut -c1-260 | sed "s/^/RUN $L STAR: /" | tee -a $R
  grep -a -h '\[singlet\] chimeric\|\[singlet\] Total' $O/pool.$L/0.log | cut -c1-500 | sed "s/^/RUN $L LOG: /" | tee -a $R
  cp $dd/chimeric_junctions.tsv.gz $O/$L.chimeric_junctions.tsv.gz 2>/dev/null
  echo "RUN $L table md5 (decompressed) $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | md5sum | cut -c1-32) rows $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | wc -l)" | tee -a $R
  echo "RUN $L table md5 (decompressed, sorted lines) $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | LC_ALL=C sort | md5sum | cut -c1-32)" | tee -a $R
  ls $dd | grep -a -e '^chim_' | sed "s/^/RUN $L leftover: /" | tee -a $R
  J=$dd/chim_Chimeric.out.junction; X=$dd/chim_Expanded.junction
  if [ -s $J ]; then echo "RUN $L RAWNORM raw $(norm $J $SCR/raw.sorted)" | tee -a $R; fi
  if [ -s $X ]; then echo "RUN $L RAWNORM dump $(norm $X $SCR/dump.sorted)" | tee -a $R; fi
  [ -s $J ] && [ "$(echo "$envs" | grep -c DUMP)" = 0 ] && cp $SCR/raw.sorted $SCR/raw.keep.$s 2>/dev/null
  if [ -s $X ]; then   # the dump arm: compare with the raw arm of the same sample, if it ran (arms order: raw arm first)
    if [ -s $SCR/raw.keep.$s ]; then
      A=$(md5sum < $SCR/raw.keep.$s | cut -c1-32); B=$(md5sum < $SCR/dump.sorted | cut -c1-32)
      echo "RAWPROOF $s raw-line md5 $A lines $(wc -l < $SCR/raw.keep.$s); dump md5 $B lines $(wc -l < $SCR/dump.sorted); $([ "$A" = "$B" ] && echo IDENTICAL || echo DIFFERENT)" | tee -a $R
      if [ "$A" != "$B" ]; then
        echo "RAWPROOF $s only in raw: $(LC_ALL=C comm -23 $SCR/raw.keep.$s $SCR/dump.sorted | wc -l), only in dump: $(LC_ALL=C comm -13 $SCR/raw.keep.$s $SCR/dump.sorted | wc -l)" | tee -a $R
        echo "RAWPROOF $s only-in-raw classes (junction_type, num_chim_aln, count):" | tee -a $R; LC_ALL=C comm -23 $SCR/raw.keep.$s $SCR/dump.sorted | cut -f7,15 | sort | uniq -c | sort -rn | head -10 | tee -a $R
        echo "RAWPROOF $s only-in-dump classes:" | tee -a $R; LC_ALL=C comm -13 $SCR/raw.keep.$s $SCR/dump.sorted | cut -f7,15 | sort | uniq -c | sort -rn | head -10 | tee -a $R
      fi
    else echo "RAWPROOF $s: no raw arm before the dump arm" | tee -a $R; fi
  fi
  : > $J; : > $X; rm -f $SCR/raw.sorted $SCR/dump.sorted
  : > $dd/star_Unmapped.out.mate1; : > $dd/star_Unmapped.out.mate2
done
echo "done $(date -Is)" | tee -a $R
