#!/bin/bash
#SBATCH -A debruinz -p short-gpu --exclude=g050,g051,g052 -c 14 --mem=160G --time=04:00:00 -J t2-chimprof-run -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-chimprof-run (chimprof task, throughput track 2): the --chimeric second pass (the bundled STAR, `singlet star <args>`) run on its own, on the pass
# input kept by t2-chimprof-inputs.sbatch ($JR/results/t2-chimprof-inputs/<sample>/{args.txt,chim_dedup.mate1,chim_dedup.mate2}), CPU only.
# Arms per sample (one after the other, same node, T threads); every arm's stderr tail, exit status and time are kept:
#   base       no instrumentation (CPU-s and wall of the unmodified path of this binary)
#   prof       SINGLET_CHIM_PROF  per-read records  -> t2-chimprof-reads.py  (also the instrumentation overhead against base)
#   samp-<m>   SINGLET_CHIM_SAMPLE wall-clock stack sampler, mode m = fp (frame-pointer walk), bt (backtrace()), pc (interrupted pc only)
#              -> t2-chimprof-samples.py
#   samp-auto  the first of fp, bt, pc (the order of MODES) that runs to the end and writes its meta file
#   perf       only when perf_event_paranoid and a probe allow it: perf record -g --call-graph dwarf
# The sorted raw junction lines of every arm are hashed (instrumentation must not change them). Counts, hashes, seconds, function names only.
#   BIN=<singlet-gpu of the build with the instrumentation>  SAMPLES="pbmc5p_v1 gsm"  T=8  ARMS="base prof samp-auto"  TAG=  MODES="fp bt pc"  SHZ=100
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; JR=$W/jobs/gpu-runtime
BIN=${BIN:-$JR/bin/t2-chimprof-2/singlet-gpu}; T=${T:-8}; SAMPLES=${SAMPLES:-pbmc5p_v1 gsm}; ARMS=${ARMS:-base prof samp-auto}; TAG=${TAG:-run}; MODES=${MODES:-fp bt pc}
KEEP=$JR/results/t2-chimprof-inputs; O=$JR/results/t2-chimprof-$SLURM_JOB_ID-$TAG; mkdir -p $O
SCR=/tmp/$USER-chimprof-run-$SLURM_JOB_ID; mkdir -p $SCR; R=$O/results.txt
PY=$W/envs/pbench/bin/python3; SRC=$JR   # the analysis scripts sit next to the job scripts in $JR
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM   # own node-local scratch only
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG bin=$BIN samples='$SAMPLES' arms='$ARMS' modes='$MODES' T=$T $(date -Is)" | tee $R
echo "binary sha256 $(sha256sum $BIN | cut -c1-16)" | tee -a $R
echo "nproc $(nproc) cpus-per-task ${SLURM_CPUS_PER_TASK:-?} perf_event_paranoid $(cat /proc/sys/kernel/perf_event_paranoid) perf $(command -v perf) ptrace_scope $(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null) ulimit-s $(ulimit -s)" | tee -a $R
echo "node at start: $(awk '/MemFree|^Cached:/{printf "%s %d GB; ", $1, $2/1048576}' /proc/meminfo); squeue on node: $(squeue -h -w $(hostname -s) -o '%i/%j/%u/%m' | paste -sd' ')" | tee -a $R
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|L3' | tee -a $R
PERFOK=0
if command -v perf >/dev/null; then
  if perf record -o $SCR/probe.data -F 99 -- /bin/true > $SCR/probe.txt 2>&1; then PERFOK=1; fi
  echo "perf probe: ok=$PERFOK: $(tr '\n' ' ' < $SCR/probe.txt | cut -c1-200)" | tee -a $R
fi
norm() { grep -a -v -e '^#' -e '^chr_donorA' "$1" | LC_ALL=C sort -S 30G --parallel=8 -T $SCR > "$2"; echo "$(md5sum < "$2" | cut -c1-32) $(wc -l < "$2")"; }
TIMEFORMAT='wall %R user %U sys %S'

# run_arm <sample> <arm label> <dir> <env/perf prefix words...>: one STAR run on the kept input; the logs and the hash of the raw lines are printed
run_arm() {
  local s=$1 arm=$2 od=$3; shift 3
  local -a PRE=("$@") A NA=()
  mkdir -p $od
  mapfile -t A < $KEEP/$s/args.txt
  local i a
  for ((i=0; i<${#A[@]}; i++)); do
    a=${A[i]}
    case $a in
      --readFilesIn) NA+=(--readFilesIn $SCR/$s/chim_dedup.mate1 $SCR/$s/chim_dedup.mate2); i=$((i+2));;
      --outFileNamePrefix) NA+=(--outFileNamePrefix $od/chim_); i=$((i+1));;
      --runThreadN) NA+=(--runThreadN $T); i=$((i+1));;
      *) NA+=("$a");;
    esac
  done
  echo "RUN $s $arm args: ${NA[*]}" | cut -c1-1500 | tee -a $R
  sync
  { time ${PRE[@]+"${PRE[@]}"} "$BIN" star "${NA[@]}" > $od/stdout.txt 2> $od/stderr.txt; echo "rc=$?" > $od/rc.txt; } 2> $od/time.txt
  echo "RUN $s $arm TIME: $(tail -1 $od/time.txt) $(cat $od/rc.txt)" | tee -a $R
  grep -a -h -i -E 'segmentation|fault|killed|abort|terminate' $od/time.txt | cut -c1-200 | sed "s/^/RUN $s $arm SHELL: /" | tee -a $R
  tail -25 $od/stderr.txt | cut -c1-240 > $O/$s.$arm.stderr.txt
  grep -a -h -E 'chimprof|FAULT' $od/stderr.txt | head -12 | sed "s/^/RUN $s $arm STDERR: /" | tee -a $R
  echo "RUN $s $arm mapping: $(grep -a -h -E 'finished mapping|finished Solo' $od/chim_Log.out 2>/dev/null | tr '\n' ' ')" | tee -a $R
  grep -a -h -E 'Number of input reads|Average input read length|Uniquely mapped reads number|Number of reads unmapped: too short|Number of chimeric reads|Mapping speed' $od/chim_Log.final.out $od/chim_Log.progress.out 2>/dev/null | tr -s ' ' | sed "s/^/RUN $s $arm FINAL:/" | tee -a $R
  if [ -s $od/chim_Chimeric.out.junction ]; then
    echo "RUN $s $arm RAWNORM $(norm $od/chim_Chimeric.out.junction $od/sorted.txt)" | tee -a $R
    rm -f $od/sorted.txt
    : > $od/chim_Chimeric.out.junction
  else echo "RUN $s $arm RAWNORM none (no junction file)" | tee -a $R; fi
  cp $od/time.txt $O/$s.$arm.time.txt; cp $od/rc.txt $O/$s.$arm.rc.txt
  [ "$arm" = base ] && { cp $od/chim_Log.out $O/$s.base.Log.out; cp $od/chim_Log.final.out $O/$s.base.Log.final.out; }
  return 0
}

for s in $SAMPLES; do
  [ -s $KEEP/$s/chim_dedup.mate1 ] || { echo "MISSING kept input $s" | tee -a $R; continue; }
  mkdir -p $SCR/$s; t=$(date +%s)
  cp $KEEP/$s/chim_dedup.mate1 $KEEP/$s/chim_dedup.mate2 $SCR/$s/
  echo "RUN $s input copy to node-local disk: $(( $(date +%s) - t )) s; reads $(( $(wc -l < $SCR/$s/chim_dedup.mate1) / 4 ))" | tee -a $R
  for arm in $ARMS; do
    od=$SCR/$s/$arm
    case $arm in
      base) run_arm $s $arm $od;;
      prof)
        run_arm $s $arm $od env SINGLET_CHIM_PROF=$od/p
        if [ -s $od/p.meta ]; then
          cp $od/p.meta $O/$s.prof.meta
          $PY $SRC/t2-chimprof-reads.py $od/p --mate1 $SCR/$s/chim_dedup.mate1 --out $O/$s.reads-summary.txt > /dev/null 2> $O/$s.reads-summary.err
          echo "RUN $s prof analysis rc=$? lines $(wc -l < $O/$s.reads-summary.txt)" | tee -a $R
          cat $od/p.reads.* | gzip -1 > $O/$s.reads.bin.gz
          ls -l $O/$s.reads.bin.gz | awk '{print "RUN '$s' reads records gz bytes " $5}' | tee -a $R
        else echo "RUN $s prof: no meta file" | tee -a $R; fi
        rm -f $od/p.reads.*;;
      samp-*)
        ml=${arm#samp-}; [ "$ml" = auto ] && ml=$MODES
        for m in $ml; do
          od=$SCR/$s/samp-$m
          run_arm $s samp-$m $od env SINGLET_CHIM_SAMPLE=$od/p SINGLET_CHIM_SAMPLE_MODE=$m SINGLET_CHIM_SAMPLE_HZ=${SHZ:-100}
          if [ -s $od/p.meta ] && ls $od/p.samp.* > /dev/null 2>&1; then
            cp $od/p.meta $O/$s.samp-$m.meta; cp $od/p.maps $O/$s.samp-$m.maps
            $PY $SRC/t2-chimprof-samples.py $od/p --exe $BIN --top 25 --out $O/$s.samples-summary-$m.txt > /dev/null 2> $O/$s.samples-summary-$m.err
            echo "RUN $s samp-$m analysis rc=$? lines $(wc -l < $O/$s.samples-summary-$m.txt)" | tee -a $R
            for f in $od/p.samp.*; do gzip -1 -c $f > $O/$s.samp-$m.$(basename $f).gz; done
            rm -f $od/p.samp.*
            [ "${arm#samp-}" = auto ] && break
          else echo "RUN $s samp-$m: no sample files (mode failed)" | tee -a $R; fi
        done;;
      perf)
        if [ "$PERFOK" != 1 ] || ! echo " ${PERF_SAMPLES:-pbmc5p_v1} " | grep -q " $s "; then echo "RUN $s perf arm skipped (perf ok=$PERFOK)" | tee -a $R; continue; fi
        run_arm $s perf $od perf record -o $od/perf.data -F ${PERF_HZ:-49} -g --call-graph dwarf,8192
        timeout 1500 perf report -i $od/perf.data --no-children --stdio --sort symbol -g none --percent-limit 0.4 2> $O/$s.perf-self.err | cut -c1-200 | head -120 > $O/$s.perf-self.txt
        timeout 1500 perf report -i $od/perf.data --children --stdio --sort symbol -g none --percent-limit 0.8 2> $O/$s.perf-incl.err | cut -c1-200 | head -150 > $O/$s.perf-incl.txt
        echo "RUN $s perf report lines self $(wc -l < $O/$s.perf-self.txt) incl $(wc -l < $O/$s.perf-incl.txt)" | tee -a $R;;
      *) echo "unknown arm $arm" | tee -a $R;;
    esac
    rm -rf $SCR/$s/*/chim__STARtmp $SCR/$s/*/perf.data
  done
done
echo "done $(date -Is)" | tee -a $R
