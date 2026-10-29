#!/bin/bash
#SBATCH --job-name=fqrs-run
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nice=100
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,SET=<GSM>,IN=<label>,BIN=<name>,MODE=<chunk|fifo|file|->,TAG=<run tag>[,T=16][,EXTRA="..."] run.sbatch
# One pipeline run for the fq-read-source evidence table. IN names the input under $W/data/fq-read-source/<SET>/:
# "none" = <SET>/none/*.1fq (runs downloaded with --quality none), "ilmn4" = <SET>/ilmn4/*.1fq (the new default),
# "pooled" = <SET>/pooled.1fq (the orchestrator's decode + re-encode pool). Several runs are passed as --1fq a,b,...
# MODE "-" = do not pass --read-source (a binary that predates it). BIN is under pipeline-benchmark/bin or jobs/.../bin.
# Output: $W/jobs/fq-read-source/runs/<SET>/<TAG>/{run.log,time.txt,monitor.tsv,mem.txt,node.txt,out/} (no BAMs/reads).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
T=${T:-16}; MODE=${MODE:--}
BIN=$W/jobs/fq-read-source/bin/$BIN; [ -x $BIN ] || BIN=$PB/bin/$(basename $BIN)
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
[ "$SET" = GSM5411066 ] && G=$REF/GRCm39-2024-A
D=$W/data/fq-read-source/$SET
case $IN in
  pooled) INPUT=$D/pooled.1fq ;;
  *) INPUT=$(ls $D/$IN/*.1fq | sort | paste -sd,) ;;
esac
O=$W/jobs/fq-read-source/runs/$SET/$TAG; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-fqrs-$SLURM_JOB_ID; mkdir -p $SCR/out
trap 'kill $MON 2>/dev/null; rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF TMPDIR=$SCR
RS=(); [ "$MODE" != "-" ] && RS=(--read-source $MODE)
{
 echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK set=$SET in=$IN mode=$MODE T=$T $(date -Is)"
 echo "input=$INPUT"; sha256sum $BIN; cat ${BIN}.commit 2>/dev/null || true
 echo "loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
 echo "cotenant_cpus=$(squeue -w $(hostname -s) -h -t R -o '%C' | awk '{s+=$1}END{print s-0}')"
 grep -m1 'model name' /proc/cpuinfo
} | tee $O/node.txt
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
( echo -e "t\tmem_mb\tcpu_s"
  while true; do
    printf "%s\t%s\t%s\n" $(date +%s) $(( $(cat $CG/memory.current) / 1048576 )) $(( $(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat) / 1000000 ))
    sleep 5
  done ) > $O/monitor.tsv 2>&1 &
MON=$!
CPU0=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat)
cd $SCR
if [[ $INPUT == *,* ]]; then INARG=(--1fq $INPUT); else INARG=($INPUT); fi
/usr/bin/time -v -o $O/time.txt $BIN "${INARG[@]}" --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
  --out-prefix $SCR/out/ --threads $T --pipeline --output-format mtx --bam-sort-mode disk --genome-shared off \
  "${RS[@]}" ${EXTRA:-} > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
CPU1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat)
kill $MON 2>/dev/null || true
{ echo "cgroup_cpu_s=$(( (CPU1 - CPU0) / 1000000 ))"
  echo "memory_peak_mb=$(( $(cat $CG/memory.peak 2>/dev/null || echo 0) / 1048576 ))"
  grep -E "Elapsed|Maximum resident|User time|System time" $O/time.txt; } > $O/mem.txt
rsync -a --exclude '*.bam' --exclude '*.bai' --exclude '*Unmapped.out.mate*' --exclude '*.fq' --exclude '*.fastq' \
  --exclude '*_STARtmp*' --exclude 'star_tmp*' --exclude private $SCR/out/ $O/out/
# private/ (genotypes, allele counts) is kept apart from out/ so comparisons never print it; it never leaves Clipper
[ -d $SCR/out/private ] && rsync -a $SCR/out/private/ $O/private/
cat $O/mem.txt; tail -3 $O/run.log
