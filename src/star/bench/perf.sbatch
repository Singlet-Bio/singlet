#!/bin/bash
#SBATCH --job-name=sa-perf
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch perf.sbatch <dataset> <reps> <tag>...  (tag = bin/singlet-accel-<tag> or a full bin name,
#                                                  optionally @VAR=VAL,VAR=VAL exported for that run)
# Env SUF = suffix of the output dir.
# Paired timing on one node: every binary runs the t4 config (4 threads, private genome, sorted
# BAM, parallel pileup) round-robin, <reps> rounds. Env MODE=x4 runs 4 concurrent copies sharing
# one /dev/shm genome with disk sort (perf_run's x4_lean). EXTRA = extra singlet args.
# Output: $PB/star-accel/perf/<dataset>[_x4]/<tag>_r<k>/ ; summary line per run.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
DS=$1; REPS=$2; shift 2; MODE=${MODE:-t4}
# The dataset's own reference (species column of bench/pipeline/datasets.tsv). Every dataset used to run on
# GRCh38-2024-A: hgmm1k_v3 (human + mouse) lost its mouse cells and the mouse datasets (neuron1k_v3, nuclei2k_v2)
# aligned to the wrong genome, so their cell calls (hgmm 654 vs 1,026) were not comparable with any other run.
case $DS in
  hgmm*) SPECIES=GRCh38-GRCm39-barnyard ;;
  neuron*|nuclei*) SPECIES=GRCm39-2024-A ;;
  pbmc*) SPECIES=GRCh38-2024-A ;;
  *) SPECIES=${SPECIES:-}; [ -n "$SPECIES" ] || { echo "no reference for $DS: set SPECIES"; exit 2; } ;;
esac
REF=/mnt/projects/debruinz_project/cellarium/reference/$SPECIES; GENOME=$REF/star_2.7.11b
OUT=$PB/star-accel/perf/$DS$([ $MODE = x4 ] && echo _x4)${SUF:-}; mkdir -p $OUT
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR
cp $PB/data/$DS/$DS.1fq $SCR/s.1fq
SHMBIN=""
trap 'rm -rf $SCR; [ -n "$SHMBIN" ] && $SHMBIN genome unload $GENOME >/dev/null 2>&1; true' EXIT
echo "[sa-perf] $DS species=$SPECIES mode=$MODE node=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p') $(date -Is)"
export SINGLET_STALL_TIMEOUT=10800
bin_of() { case $1 in singlet-*) echo $PB/bin/$1 ;; *) echo $PB/bin/singlet-accel-$1 ;; esac; }
run_one() {  # bin outdir extra...
  local B=$1 O=$2; shift 2; mkdir -p $O; cd $SCR
  /usr/bin/time -v -o $O/time.txt $B $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
    --genome-dir $GENOME --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 4 --pipeline \
    --output-format mtx --raw-matrix "$@" ${EXTRA:-} > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
}
ts() { date -d "$(echo "$1" | sed 's/ \.\.\..*//')" +%s 2>/dev/null; }
summ() {  # outdir -> mapping/solo/sort seconds from star_Log.out
  local L=$1/star_Log.out
  local g=$(grep -m1 "^Finished loading the genome" $L | sed 's/^Finished loading the genome: //')
  local m=$(grep -m1 "finished mapping" $L) s=$(grep -m1 "finished Solo counting" $L) b=$(grep -m1 "started sorting BAM" $L)
  local f=$(grep -m1 "Finished on" $1/star_Log.final.out | sed 's/.*|\t//')
  local tg=$(date -d "$g" +%s) tm=$(ts "$m") tsolo=$(ts "$s") tb=$(ts "$b") tf=$(date -d "$f" +%s)
  local wall=$(grep "Elapsed" $1/time.txt | sed 's/.*: //')
  echo "map=$((tm-tg))s solo=$((tsolo-tm))s sort=$((tf-tb))s wall=$wall $(grep -h "SEQCACHE" $L | tail -1)"
}
for r in $(seq 1 $REPS); do
  for tag in "$@"; do
    t0=${tag%%@*}; B=$(bin_of ${t0%%+x*}); O=$OUT/${tag//[@=,+]/_}_r$r; rm -rf $O
    unset SINGLET_STAR_SEQCACHE_MB SINGLET_STAR_SEQCACHE_VERIFY
    if [[ $tag == *@* ]]; then for kv in $(echo ${tag#*@} | tr , " "); do export $kv; done; fi
    if [ $MODE = x4 ]; then
      SHMBIN=$B; $B genome load $GENOME > /dev/null 2>&1
      t0=$(date +%s)
      for i in 1 2 3 4; do run_one $B $O/c$i --genome-shared on --bam-sort-mode disk & done; wait
      echo "[sa-perf] $tag r$r wall_all=$(( $(date +%s)-t0 ))s c1: $(summ $O/c1)"
      $B genome unload $GENOME > /dev/null 2>&1; SHMBIN=""
    else
      if [[ $tag == *+x* ]]; then run_one $B $O --star-extra "$STAR_EXTRA"; else run_one $B $O; fi
      echo "[sa-perf] $tag r$r $(summ $O)"
    fi
  done
done
