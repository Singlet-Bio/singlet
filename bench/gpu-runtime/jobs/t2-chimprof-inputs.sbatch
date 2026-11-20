#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 27 --mem=300G --time=03:00:00 --exclude=g050 -J t2-chimprof-in -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-chimprof-inputs (chimprof task, throughput track 2): one normal --chimeric item run per sample on ONE H100 (as t2-chimdedup.sbatch, dedup arm),
# with SINGLET_CHIM_KEEP_INPUT=1: the pass input (chim_dedup.mate1/2) and the pass's STAR argument list (chim_args.txt, one argument per line) are
# kept and copied to $JR/results/t2-chimprof-inputs/<sample>/ so that the STAR pass can be run on its own later (t2-chimprof-run.sbatch).
#   ITEMS="pbmc5p_v1"            benchmark .1fq files ($W/data/gpu-deep/<s>/<s>.1fq, else $PB/data/<s>/<s>.1fq)
#   SRAS="<.sra> <.sra>"         one more sample named "gsm", encoded as in t2-chimdedup.sbatch (no --protocol)
# Checks: the table md5 of every run is printed (known: pbmc5p_v1 a5d71eb6a8b45889fe667bf57f17dce3, GSM6250953 3573f1645c4fc6e920d37c70a0033df7).
# Counts, sizes, md5 only. NO file under a private/ dir is opened.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference; JR=$W/jobs/gpu-runtime
BIN=${BIN:-$JR/bin/t2-chimprof/singlet-gpu}; TAG=${TAG:-in}; SLOTS=2; MEMGB=31; LIM=${LIM:-3600}; GD=$REF/GRCh38-2024-A; T=${T:-8}
ITEMS=${ITEMS:-pbmc5p_v1}; SRAS=${SRAS:-}
O=$JR/results/t2-chimprof-$SLURM_JOB_ID-$TAG; mkdir -p $O; KEEP=$JR/results/t2-chimprof-inputs; mkdir -p $KEEP
SCR=/tmp/$USER-chimprof-in-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/run; R=$O/results.txt
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_ALLSITE_TIMING=1
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70
export SINGLET_GPU_SHARED_INDEX=1 SINGLET_ENGINE_GPU_SLOTS=$SLOTS SINGLET_ENGINE_GPU_MEM_GB=$MEMGB SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
mypids() { find $CG -name cgroup.procs -exec cat {} + 2>/dev/null | sort -u; }
killmine() { local p; for p in $(mypids); do [ "$p" = "$$" ] && continue; tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -aq -e "$2" && kill -$1 $p 2>/dev/null; done; true; }
buskeys() { nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed -E 's/^0000//; s/[:.]/_/g'; }
stopholders() { local b; killmine TERM gpu-holder; sleep 3; for b in $(buskeys); do rm -f /tmp/singlet_gpu_holder.${b}.* 2>/dev/null; done; true; }
cleanup() { killmine TERM "singlet-gpu"; stopholders; sleep 2; rm -rf "${SCR:?}"; }   # own node-local scratch only
trap 'kill $(jobs -p) 2>/dev/null; cleanup' EXIT INT TERM
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG bin=$BIN items='$ITEMS' sras='$SRAS' T=$T $(date -Is)" | tee $R
echo "binary sha256 $(sha256sum $BIN | cut -c1-16) commit $(cat $(dirname $BIN)/commit-gpu 2>/dev/null)" | tee -a $R
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | tee -a $R
echo "nproc $(nproc) squeue on node: $(squeue -h -w $(hostname -s) -o '%i/%j/%u/%m' | paste -sd' ')" | tee -a $R
echo "perf_event_paranoid $(cat /proc/sys/kernel/perf_event_paranoid) perf $(command -v perf)" | tee -a $R

declare -A FQ; SAMPLES=""
t=$(date +%s); for s in $ITEMS; do
  F=$W/data/gpu-deep/$s/$s.1fq; [ -s $F ] || F=$PB/data/$s/$s.1fq; [ -s $F ] || { echo "MISSING INPUT $s" | tee -a $R; exit 2; }
  cp $F $SCR/data/$s.1fq & FQ[$s]="$SCR/data/$s.1fq"; SAMPLES="$SAMPLES $s"; done; wait
echo "input copy to node-local disk: $(( $(date +%s) - t )) s" | tee -a $R
if [ -n "$SRAS" ]; then
  FQS=""; n=0; t=$(date +%s)
  for s in $SRAS; do ( cd $SCR/data && $BIN download $s -o $SCR/data/r$n.1fq > $O/encode-$n.txt 2>&1; echo "rc=$?" >> $O/encode-$n.txt ) & FQS="$FQS $SCR/data/r$n.1fq"; n=$((n+1)); done; wait
  echo "encode of $n runs without --protocol: $(( $(date +%s) - t )) s" | tee -a $R
  FQ[gsm]="$FQS"; SAMPLES="$SAMPLES gsm"
fi
# Q5 (index copy): once the pass of an item has started, one snapshot of ipcs -m and of the smaps of every process of the job's cgroup
memprobe() {
  while [ ! -e $1/chim_Log.out ]; do sleep 2; done
  sleep ${PROBE_DELAY:-25}
  { echo "date $(date -Is) item $2"; echo "ipcs -m (segments): $(ipcs -m | grep -c '^0x')"; ipcs -m
    for p in $(mypids); do
      [ -r /proc/$p/smaps_rollup ] || continue
      echo "PID $p comm $(cat /proc/$p/comm) fds-on-chim_Log.out $(ls -l /proc/$p/fd 2>/dev/null | grep -c chim_Log.out) args: $(tr '\000' ' ' < /proc/$p/cmdline | cut -c1-110)"
      grep -E '^(Rss|Pss|Shared_Clean|Shared_Dirty|Private_Clean|Private_Dirty|Anonymous|AnonHugePages|Swap):' /proc/$p/smaps_rollup | tr -s ' ' | tr '\n' ' '; echo
      awk '/^[0-9a-f]+-[0-9a-f]+ /{ if (rss >= 1048576) printf "  map %s Size_kB %d Rss %d Pss %d Shared_Clean %d Shared_Dirty %d Private_Dirty %d Anon %d AnonHuge %d %s\n", rng, sz, rss, pss, sc, sd, pd, an, ah, nm; rng=$1; nm=$6; rss=pss=sc=sd=pd=an=ah=sz=0 }
           /^Size:/{sz=$2} /^Rss:/{rss=$2} /^Pss:/{pss=$2} /^Shared_Clean:/{sc=$2} /^Shared_Dirty:/{sd=$2} /^Private_Dirty:/{pd=$2} /^Anonymous:/{an=$2} /^AnonHugePages:/{ah=$2}
           END{ if (rss >= 1048576) printf "  map %s Size_kB %d Rss %d Pss %d Shared_Clean %d Shared_Dirty %d Private_Dirty %d Anon %d AnonHuge %d %s\n", rng, sz, rss, pss, sc, sd, pd, an, ah, nm }' /proc/$p/smaps 2>/dev/null
    done; } > $O/memprobe.$2.txt 2>&1
}
stopholders
for s in $SAMPLES; do
  dd=$SCR/run/$s/out; mkdir -p $SCR/run/$s; set -- ${FQ[$s]}
  echo "$1 $dd/ ${*:2} --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf --chimeric" > $SCR/run/queue.$s; mkdir -p $O/pool.$s
  echo "RUN $s start $(date -Is)" | tee -a $R
  st=$(date +%s)
  memprobe $dd $s & MP=$!
  ( export SINGLET_CHIM_DEDUP=1 SINGLET_CHIM_KEEP_INPUT=1
    setsid bash -c 'exec "$0" gpu-pool --device 0 --queue '$SCR/run/queue.$s' --slots '$SLOTS' --workers 1 --threads '$T' --report '$O/pool.$s' --timeout '$LIM' --args "--ref-base '$REF' --pipeline --output-format 1pz --no-hla --engine gpu"' "$BIN" > $O/pool.$s/stdout.txt 2>&1 )
  en=$(date +%s); kill $MP 2>/dev/null
  echo "RUN $s pool wall $(( en - st )) s" | tee -a $R
  cut -f1,7,14 $O/pool.$s/samples.tsv | sed "s/^/RUN $s samples: /" | tee -a $R
  grep -a -h '\[singlet\] chimeric\|\[singlet\] Total' $O/pool.$s/0.log | cut -c1-500 | sed "s/^/RUN $s LOG: /" | tee -a $R
  echo "RUN $s table md5 (decompressed) $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | md5sum | cut -c1-32) rows $(zcat $dd/chimeric_junctions.tsv.gz 2>/dev/null | wc -l)" | tee -a $R
  if [ -s $dd/chim_dedup.mate1 ] && [ -s $dd/chim_args.txt ]; then
    mkdir -p $KEEP/$s
    cp $dd/chim_args.txt $KEEP/$s/args.txt; cp $dd/chim_dedup.mate1 $KEEP/$s/chim_dedup.mate1; cp $dd/chim_dedup.mate2 $KEEP/$s/chim_dedup.mate2
    cp $dd/chim_Log.out $KEEP/$s/chim_Log.out 2>/dev/null
    echo "RUN $s pass Log.out genome lines: $(grep -a -i -E 'preload|genomeLoad|shared memory|Finished loading' $dd/chim_Log.out | head -8 | cut -c1-200 | tr '\n' '|')" | tee -a $R
    echo "RUN $s main Log.out genome lines: $(grep -a -i -E 'Genome preload|genomeLoad' $dd/star_Log.out | head -6 | cut -c1-200 | tr '\n' '|')" | tee -a $R
    echo "RUN $s pool log genome lines: $(grep -a -i -E 'genome ' $O/pool.$s/stdout.txt | head -4 | cut -c1-200 | tr '\n' '|')" | tee -a $R
    echo "RUN $s kept: $(ls -l $KEEP/$s/chim_dedup.mate1 $KEEP/$s/chim_dedup.mate2 | awk '{printf "%s %s; ", $9, $5}')" | tee -a $R
    echo "RUN $s kept lines: mate1 $(wc -l < $KEEP/$s/chim_dedup.mate1) mate2 $(wc -l < $KEEP/$s/chim_dedup.mate2)" | tee -a $R
    echo "RUN $s kept md5: mate1 $(md5sum < $KEEP/$s/chim_dedup.mate1 | cut -c1-32) mate2 $(md5sum < $KEEP/$s/chim_dedup.mate2 | cut -c1-32)" | tee -a $R
    sed "s/^/RUN $s ARGS: /" $KEEP/$s/args.txt | tr '\n' ' ' | tee -a $R; echo | tee -a $R
  else echo "RUN $s: NO KEPT INPUT" | tee -a $R; fi
  ls $dd | grep -a -e '^chim_' | sed "s/^/RUN $s leftover: /" | tee -a $R
  : > $dd/star_Unmapped.out.mate1; : > $dd/star_Unmapped.out.mate2; : > $dd/chim_dedup.mate1; : > $dd/chim_dedup.mate2
done
echo "done $(date -Is)" | tee -a $R
