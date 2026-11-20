#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 27 --mem=300G --time=03:00:00 --exclude=g050 -J t2-chimpass -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-chimpass (chimpass task, throughput track 2): where the time of the --chimeric second pass goes. ONE H100 (not exclusive), ONE worker, one gpu-pool run per
# entry of RUNS with the campaign's H100 settings (budget 31 GB, 2 slots, --no-hla, --engine gpu, --chimeric). The pass's --runThreadN is the item's thread count
# (singlet.cpp: base_args from star_args.txt), so RUNS "8 16 27" re-runs the same sample with the pass at 8, 16 and 27 threads (the main engine gets them too).
# Two input modes:
#   SRAS="<staged .sra> ..."  : encoded as twin_check.sbatch does (no --protocol), each run uses the same .1fq; the first run keeps the pass's input reads (hard links)
#                               and its raw junction file; after the last run the reads are analysed (t2-chimpass-reads.cpp: counts only).
#   ITEMS="pbmc5p_v1 d300_5p" : benchmark .1fq files ($W/data/gpu-deep/<s>/<s>.1fq, else $PB/data/<s>/<s>.1fq), one run each at the first thread count of RUNS.
# Per run: 1 Hz watcher of the pass (t2-chimpass-watch.sh: STAR child CPU seconds, RSS, cgroup memory, copies of the pass's own logs), the pool log, the md5 of the
# decompressed chimeric_junctions.tsv.gz, and of the raw junction file (plain and sorted). NO file under any private/ directory is opened.
# Parameters: BIN, SRAS or ITEMS, RUNS (default "8 16 27" / "8"), TAG, LIM (item timeout, s), EXTRA (default --chimeric)
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference; JR=$W/jobs/gpu-runtime
BIN=${BIN:-$JR/bin/t2-hostcut5/singlet-gpu}; TAG=${TAG:-cp}; SLOTS=2; MEMGB=31; LIM=${LIM:-3600}; GD=$REF/GRCh38-2024-A; EXTRA=${EXTRA:---chimeric}
SRAS=${SRAS:-}; ITEMS=${ITEMS:-}; RUNS=${RUNS:-$([ -n "$SRAS" ] && echo "8 16 27" || echo 8)}
O=$JR/results/t2-chimpass-$SLURM_JOB_ID-$TAG; mkdir -p $O; SCR=/tmp/$USER-chimpass-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/run $SCR/keep; R=$O/results.txt
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_ALLSITE_TIMING=1
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70
export SINGLET_GPU_SHARED_INDEX=1 SINGLET_ENGINE_GPU_SLOTS=$SLOTS SINGLET_ENGINE_GPU_MEM_GB=$MEMGB SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16
export SINGLET_KEEP_CHIMERIC_JUNCTION=1   # keeps Chimeric.out.junction (the pass's raw junction lines) next to the outputs
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
mypids() { find $CG -name cgroup.procs -exec cat {} + 2>/dev/null | sort -u; }
killmine() { local p; for p in $(mypids); do [ "$p" = "$$" ] && continue; tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -aq -e "$2" && kill -$1 $p 2>/dev/null; done; true; }
buskeys() { nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed -E 's/^0000//; s/[:.]/_/g'; }
stopholders() { local b; killmine TERM gpu-holder; sleep 3; for b in $(buskeys); do rm -f /tmp/singlet_gpu_holder.${b}.* 2>/dev/null; done; true; }
cleanup() { killmine TERM "singlet-gpu"; killmine TERM t2-chimpass-watch; stopholders; sleep 2; rm -rf "${SCR:?}"; }   # own node-local scratch only
trap 'kill $(jobs -p) 2>/dev/null; cleanup' EXIT INT TERM
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG bin=$BIN sras='$SRAS' items='$ITEMS' runs='$RUNS' extra='$EXTRA' $(date -Is)" | tee $R
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | tee -a $R; free -g | head -2 | tee -a $R
df -h /tmp | tail -1 | tee -a $R; echo "nproc $(nproc) CLK_TCK $(getconf CLK_TCK) cgroup $CG memory.max $(cat $CG/memory.max) cpu.max $(cat $CG/cpu.max 2>/dev/null)" | tee -a $R
echo "node at start: $(awk '/MemFree|^Shmem:|^Cached:/{printf "%s %d GB; ", $1, $2/1048576}' /proc/meminfo); squeue on node: $(squeue -h -w $(hostname -s) -o '%i/%j/%u/%m' | paste -sd' ')" | tee -a $R

declare -A FQ   # run label -> fastq list
LABELS=""
if [ -n "$SRAS" ]; then
  FQS=""; n=0; t=$(date +%s)
  for s in $SRAS; do ( cd $SCR/data && $BIN download $s -o $SCR/data/r$n.1fq > $O/encode-$n.txt 2>&1; echo "rc=$?" >> $O/encode-$n.txt ) & FQS="$FQS $SCR/data/r$n.1fq"; n=$((n+1)); done; wait
  echo "encode of $n runs without --protocol: $(( $(date +%s) - t )) s" | tee -a $R
  for i in $(seq 0 $((n-1))); do echo "encode r$i: $(tail -1 $O/encode-$i.txt)" | tee -a $R; done
  i=0; for T in $RUNS; do FQ[T$T]="$FQS"; LABELS="$LABELS T$T"; i=$((i+1)); done
else
  t=$(date +%s); for s in $ITEMS; do
    F=$W/data/gpu-deep/$s/$s.1fq; [ -s $F ] || F=$PB/data/$s/$s.1fq; [ -s $F ] || { echo "MISSING INPUT $s" | tee -a $R; exit 2; }
    cp $F $SCR/data/$s.1fq &
    for T in $RUNS; do FQ[${s}_T$T]="$SCR/data/$s.1fq"; LABELS="$LABELS ${s}_T$T"; done; done; wait   # one run per item and thread count; label <item>_T<threads>
  echo "input copy to node-local disk: $(( $(date +%s) - t )) s" | tee -a $R
fi
stopholders
first=1
for L in $LABELS; do
  T=${L##*T}   # threads: the digits after the last T of the label
  dd=$SCR/run/$L/out; mkdir -p $SCR/run/$L; set -- ${FQ[$L]}
  echo "$1 $dd/ ${*:2} --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf $EXTRA" > $SCR/run/queue.$L; mkdir -p $O/pool.$L
  keep=0; [ -n "$SRAS" ] && [ $first = 1 ] && keep=1
  bash $JR/t2-chimpass-watch.sh $dd $O $L $keep $SCR/keep &
  WP=$!
  echo "RUN $L threads $T keep=$keep start $(date -Is) node memory: $(awk '/MemFree|^Shmem:|^Cached:/{printf "%s %d GB; ", $1, $2/1048576}' /proc/meminfo)" | tee -a $R
  st=$(date +%s)
  setsid bash -c 'exec "$0" gpu-pool --device 0 --queue '$SCR/run/queue.$L' --slots '$SLOTS' --workers 1 --threads '$T' --report '$O/pool.$L' --timeout '$LIM' --args "--ref-base '$REF' --pipeline --output-format 1pz --no-hla --engine gpu"' "$BIN" > $O/pool.$L/stdout.txt 2>&1
  en=$(date +%s); sleep 3; kill $WP 2>/dev/null; wait $WP 2>/dev/null
  echo "RUN $L pool wall $(( en - st )) s" | tee -a $R
  cut -f1,7,14 $O/pool.$L/samples.tsv | sed "s/^/RUN $L samples: /" | tee -a $R
  cp $dd/star_Log.out $O/$L.star_Log.out 2>/dev/null; cp $O/pool.$L/0.log $O/$L.pool.log 2>/dev/null
  grep -a -h 'E19: level\|engine_gpu: [0-9]* reads,\|^runThreadN\|^clip3pAdapter' $dd/star_Log.out | cut -c1-260 | sed "s/^/RUN $L STAR: /" | tee -a $R
  grep -a -h '\[singlet\] chimeric\|\[singlet\] Total' $O/pool.$L/0.log | cut -c1-500 | sed "s/^/RUN $L LOG: /" | tee -a $R
  cp $dd/chimeric_junctions.tsv.gz $O/$L.chimeric_junctions.tsv.gz 2>/dev/null
  echo "RUN $L table md5 (decompressed) $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | md5sum | cut -c1-32) rows $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | wc -l)" | tee -a $R
  echo "RUN $L table md5 (decompressed, sorted lines) $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | LC_ALL=C sort | md5sum | cut -c1-32)" | tee -a $R
  J=$dd/chim_Chimeric.out.junction
  if [ -s $J ]; then
    echo "RUN $L raw junction file: $(stat -c %s $J) bytes, $(wc -l < $J) lines, md5 $(md5sum < $J | cut -c1-32)" | tee -a $R
    echo "RUN $L raw junction file sorted (LC_ALL=C) md5 $(LC_ALL=C sort -S 60G --parallel=16 -T $SCR $J | md5sum | cut -c1-32)" | tee -a $R
    echo "RUN $L raw junction file without its '#' lines: $(grep -a -v '^#' $J | md5sum | cut -c1-32), sorted: $(grep -a -v '^#' $J | LC_ALL=C sort -S 60G --parallel=16 -T $SCR | md5sum | cut -c1-32); '#' lines: $(grep -a -c '^#' $J)" | tee -a $R
    grep -a '^#' $J | cut -c1-400 | sed "s/^/RUN $L comment line: /" | tee -a $R
    [ $first = 1 ] && [ -n "$SRAS" ] || : > $J
  else echo "RUN $L no raw junction file" | tee -a $R; fi
  if [ $keep = 0 ]; then : > $dd/star_Unmapped.out.mate1; : > $dd/star_Unmapped.out.mate2; fi
  first=0
done
if [ -n "$SRAS" ] && [ -s $SCR/keep/star_Softclip.out.mate1 ]; then
  ls -l $SCR/keep | tee -a $R
  AD=$(grep -a '^clip3pAdapterSeq ' $O/T${RUNS%% *}.star_Log.out | head -1 | awk '{print $2}'); MMP=$(grep -a '^clip3pAdapterMMp ' $O/T${RUNS%% *}.star_Log.out | head -1 | awk '{print $2}')
  echo "reads analysis: adapter ${AD:-none} mmp ${MMP:-0.1}" | tee -a $R
  g++ -O2 -fopenmp -std=c++17 -o $SCR/chimreads $JR/t2-chimpass-reads.cpp 2>&1 | head -5 | tee -a $R
  t=$(date +%s)
  $SCR/chimreads --junction $SCR/run/T${RUNS%% *}/out/chim_Chimeric.out.junction --unm $SCR/keep/star_Unmapped.out.mate1 --sc $SCR/keep/star_Softclip.out.mate1 --adapter "${AD:-AAGCAGTGGTATCAACGCAGAGTACATGGG}" --mmp "${MMP:-0.1}" --minseg 10 --threads 24 > $O/reads-analysis.txt 2>&1
  echo "reads analysis: $(( $(date +%s) - t )) s, $(wc -l < $O/reads-analysis.txt) lines -> reads-analysis.txt" | tee -a $R
fi
echo "done $(date -Is)" | tee -a $R
