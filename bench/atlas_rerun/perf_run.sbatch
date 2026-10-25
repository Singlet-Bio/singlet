#!/bin/bash
#SBATCH --job-name=ar-perf
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch perf_run.sbatch <config> <dataset> <rep>
# One configuration of the Singlet run, timed end to end on 16 reserved cores. Configs:
#   t4 | t8 | t16       present settings (sorted BAM + parallel pileup), N threads
#   t4_shm              genome loaded once into /dev/shm (`singlet genome load`), --genome-shared on
#   t4_stream           --no-parallel-pileup: pileup reads STAR's unsorted BAM as it is written
#   t4_native           binary built with -march=native (bin/singlet-fixed-native)
#   x4_private          4 concurrent copies of the sample, 4 threads each, each STAR loads its own genome
#   x4_shm              4 concurrent copies sharing one genome in /dev/shm
#   t4_nosolo           STAR --soloType CB_samTagOut: barcode correction and tags, no STARsolo counting
#   t4_opt              t4_shm + t4_stream + t4_nosolo together
#   x4_opt              4 concurrent t4_opt copies sharing one genome
#   t4_disk             --bam-sort-mode disk: STAR sort bins on node NVMe instead of RAM
#   x4_lean             4 concurrent samples, one shared genome, disk sort (memory-lean packing)
#   t4_nonhost          campaign default: SINGLET_REF_BASE set, so viral/bacterial/fungal screens run
#   t4_nonhost_shm      same, with the screening indices cached once in /dev/shm (`singlet nonhost load`)
# Env: PERF_BIN (binary name in $PB/bin), PERF_TAG (suffix for the output dir).
# Output: $PB/perf/<dataset>[_<tag>]/<config>_r<rep>/{time.txt,run.log,...}; concurrent configs write c1..c4.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A; GENOME=$REF/star_2.7.11b
CFG=$1; DS=$2; REP=${3:-1}
O=$PB/perf/$DS${PERF_TAG:+_$PERF_TAG}/${CFG}_r$REP; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR; [ -n "${SHM:-}" ] && $BIN genome unload $GENOME >/dev/null 2>&1; [ -n "${NHSHM:-}" ] && $BIN nonhost unload >/dev/null 2>&1; true' EXIT
echo "[perf] $CFG $DS r$REP node=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p') $(date -Is)" | tee $O/node.txt
ONEFQ=$PB/data/$DS/$DS.1fq
[ -f $ONEFQ ] || { echo "missing $ONEFQ (run perf_encode.sbatch)"; exit 2; }
cp $ONEFQ $SCR/s.1fq
BIN=$PB/bin/${PERF_BIN:-singlet-fixed}; T=4; EXTRA=(); N=1; SHM=""
NOSOLO=(--star-extra "--soloType CB_samTagOut")
case $CFG in
  t4) ;;
  t8) T=8 ;;
  t16) T=16 ;;
  t4_shm) SHM=1; EXTRA=(--genome-shared on) ;;
  t4_stream) EXTRA=(--no-parallel-pileup) ;;
  t4_native) BIN=$PB/bin/singlet-fixed-native ;;
  x4_private) N=4 ;;
  x4_shm) N=4; SHM=1; EXTRA=(--genome-shared on) ;;
  t4_nosolo) EXTRA=("${NOSOLO[@]}") ;;
  t4_disk) EXTRA=(--bam-sort-mode disk) ;;
  x4_lean) N=4; SHM=1; EXTRA=(--genome-shared on --bam-sort-mode disk) ;;
  t4_nonhost) export SINGLET_REF_BASE=/mnt/projects/debruinz_project/cellarium/reference ;;
  t4_nonhost_shm) export SINGLET_REF_BASE=/mnt/projects/debruinz_project/cellarium/reference; NHSHM=1 ;;
  t4_opt) SHM=1; EXTRA=(--genome-shared on --no-parallel-pileup "${NOSOLO[@]}") ;;
  x4_opt) N=4; SHM=1; EXTRA=(--genome-shared on --no-parallel-pileup "${NOSOLO[@]}") ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
export SINGLET_STALL_TIMEOUT=10800
if [ -n "${NHSHM:-}" ]; then
  /usr/bin/time -v -o $O/nonhost_load.time.txt $BIN nonhost load > $O/nonhost_load.log 2>&1
fi
if [ -n "$SHM" ]; then
  /usr/bin/time -v -o $O/genome_load.time.txt $BIN genome load $GENOME > $O/genome_load.log 2>&1
fi
run_one() {  # $1 = output dir
  mkdir -p $1; cd $SCR
  /usr/bin/time -v -o $1/time.txt $BIN $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
    --genome-dir $GENOME --exons $REF/genes/genes.gtf --out-prefix $1/ --threads $T --pipeline \
    --output-format mtx --raw-matrix "${EXTRA[@]}" > $1/run.log 2>&1 || echo "exit $?" >> $1/run.log
}
t0=$(date +%s)
if [ $N = 1 ]; then run_one $O; else
  for i in $(seq 1 $N); do run_one $O/c$i & done; wait
fi
t1=$(date +%s)
sacct -j $SLURM_JOB_ID -n -o MaxRSS,MaxVMSize -p 2>/dev/null | head -3 > $O/sacct_mem.txt || true
echo "[perf] $CFG wall_all=$((t1-t0))s" | tee -a $O/node.txt
grep -hE "Elapsed|Maximum resident|User time|System time" $(find $O -name time.txt) | sed 's/^\s*//'
