#!/bin/bash
#SBATCH -A debruinz -p short-gpu --exclude=g050,g051,g052 -c 14 --mem=160G --time=05:00:00 -J t2-chimprune-run -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-chimprune-run (chimprune task, throughput track 2): the --chimeric second pass (the bundled STAR, `singlet star <args>`) run on its own, on the pass input kept by
# t2-chimprof-inputs.sbatch ($JR/results/t2-chimprof-inputs/<sample>/{args.txt,chim_dedup.mate1,chim_dedup.mate2}), CPU only, no profiling switches.
# Arms per sample, one after the other on the same node, T threads (A/B/B/A, then verify):
#   pp0a  SINGLET_CHIM_PAIRPRUNE=0   (the old pair loop)       ppa   default (SINGLET_CHIM_PAIRPRUNE unset = pruned loop)
#   ppb   default again               pp0b  SINGLET_CHIM_PAIRPRUNE=0 again      ver   SINGLET_CHIM_PAIRPRUNE=verify (old and new loop per read, compared)
# For each arm: wall, user, sys (bash `time`), exit code, STAR's chimeric counters, and md5 + line count of the sorted data lines of the raw junction file
# (header and `#` lines removed, LC_ALL=C sort). Required (job 819071): pbmc5p_v1 643f187626f6ffaa67d0fe9acf324d42 (2,832,142 lines),
# GSM6250953 (sample name gsm) c2a0f02ef3d48de17e89fab98b17c1ac (6,706,877 lines). Counts, hashes and seconds only.
#   BIN=<singlet-gpu>  SAMPLES="pbmc5p_v1 gsm"  T=8  ARMS="pp0a ppa ppb pp0b ver"  TAG=
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; JR=$W/jobs/gpu-runtime
BIN=${BIN:-$JR/bin/t2-chimprune/singlet-gpu}; T=${T:-8}; SAMPLES=${SAMPLES:-pbmc5p_v1 gsm}; ARMS=${ARMS:-pp0a ppa ppb pp0b ver}; TAG=${TAG:-run}
KEEP=$JR/results/t2-chimprof-inputs; O=$JR/results/t2-chimprune-$SLURM_JOB_ID-$TAG; mkdir -p $O
SCR=/tmp/$USER-chimprune-run-$SLURM_JOB_ID; mkdir -p $SCR; R=$O/results.txt
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM   # own node-local scratch only
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG bin=$BIN samples='$SAMPLES' arms='$ARMS' T=$T $(date -Is)" | tee $R
echo "binary sha256 $(sha256sum $BIN | cut -c1-16)" | tee -a $R
echo "nproc $(nproc) cpus-per-task ${SLURM_CPUS_PER_TASK:-?}" | tee -a $R
echo "node at start: $(awk '/MemFree|^Cached:/{printf "%s %d GB; ", $1, $2/1048576}' /proc/meminfo); squeue on node: $(squeue -h -w $(hostname -s) -o '%i/%j/%u/%m' | paste -sd' ')" | tee -a $R
lscpu | grep -E 'Model name|^CPU\(s\)' | tee -a $R
norm() { grep -a -v -e '^#' -e '^chr_donorA' "$1" | LC_ALL=C sort -S 30G --parallel=8 -T $SCR > "$2"; echo "$(md5sum < "$2" | cut -c1-32) $(wc -l < "$2")"; }
TIMEFORMAT='wall %R user %U sys %S'

# run_arm <sample> <arm> <env words...>
run_arm() {
  local s=$1 arm=$2 od=$SCR/$1/$2; shift 2
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
  echo "RUN $s $arm env: ${PRE[*]:-none} $(date -Is)" | tee -a $R
  sync
  { time ${PRE[@]+"${PRE[@]}"} "$BIN" star "${NA[@]}" > $od/stdout.txt 2> $od/stderr.txt; echo "rc=$?" > $od/rc.txt; } 2> $od/time.txt
  echo "RUN $s $arm TIME: $(tail -1 $od/time.txt) $(cat $od/rc.txt)" | tee -a $R
  tail -25 $od/stderr.txt | cut -c1-240 > $O/$s.$arm.stderr.txt
  grep -a -h -i -E 'MISMATCH|segmentation|fault|killed|abort|terminate' $od/stderr.txt $od/time.txt | cut -c1-300 | sed "s/^/RUN $s $arm MSG: /" | tee -a $R
  echo "RUN $s $arm mapping: $(grep -a -h -E 'finished mapping|finished Solo' $od/chim_Log.out 2>/dev/null | tr '\n' ' ')" | tee -a $R
  grep -a -h -E 'Number of input reads|Uniquely mapped reads number|Number of reads unmapped: too short|Number of chimeric reads' $od/chim_Log.final.out 2>/dev/null | tr -s ' ' | sed "s/^/RUN $s $arm FINAL:/" | tee -a $R
  if [ -s $od/chim_Chimeric.out.junction ]; then
    echo "RUN $s $arm RAWNORM $(norm $od/chim_Chimeric.out.junction $od/sorted.txt)" | tee -a $R
    rm -f $od/sorted.txt
    : > $od/chim_Chimeric.out.junction
  else echo "RUN $s $arm RAWNORM none (no junction file)" | tee -a $R; fi
  cp $od/time.txt $O/$s.$arm.time.txt; cp $od/rc.txt $O/$s.$arm.rc.txt
  return 0
}

for s in $SAMPLES; do
  [ -s $KEEP/$s/chim_dedup.mate1 ] || { echo "MISSING kept input $s" | tee -a $R; continue; }
  mkdir -p $SCR/$s; t=$(date +%s)
  cp $KEEP/$s/chim_dedup.mate1 $KEEP/$s/chim_dedup.mate2 $SCR/$s/
  echo "RUN $s input copy to node-local disk: $(( $(date +%s) - t )) s; reads $(( $(wc -l < $SCR/$s/chim_dedup.mate1) / 4 ))" | tee -a $R
  for arm in $ARMS; do
    case $arm in
      pp0a|pp0b) run_arm $s $arm env SINGLET_CHIM_PAIRPRUNE=0;;
      ppa|ppb)   run_arm $s $arm env;;
      ver)       run_arm $s $arm env SINGLET_CHIM_PAIRPRUNE=verify;;
      *) echo "unknown arm $arm" | tee -a $R;;
    esac
    rm -rf $SCR/$s/*/chim__STARtmp
  done
done
echo "done $(date -Is)" | tee -a $R
