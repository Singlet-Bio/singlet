#!/bin/bash
#SBATCH --job-name=dp-run
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch run_deep.sbatch <binary name in PB/bin> <GSM> <tag> [extra singlet args...]
# One compute-stage run exactly as the orchestrator's compute_stage (s4 class: 4 threads, 80 GB, --bam-sort-mode disk,
# private genome, --run-screen when present), with a 5 s monitor of CPU, RSS, cgroup memory and /tmp disk I/O.
# Input: $W/data/atlas-rerun/perf/<GSM>/sample.1fq (or INPUTS="a.1fq b.1fq" relative to that dir).
# Output: $W/deep-perf/runs/<GSM>/<tag>/ (BAMs and reads excluded; matrices as mtx for perf_equiv when MTX=1).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
BINNAME=$1; GSM=$2; TAG=$3; shift 3
BIN=$PB/bin/$BINNAME
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
D=$W/data/atlas-rerun/perf/$GSM
O=$W/deep-perf/runs/$GSM/$TAG; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-dp-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'kill $MON 2>/dev/null; rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
echo "[dp] $BINNAME $GSM $TAG node=$(hostname) cpus=$SLURM_CPUS_PER_TASK excl=${SLURM_JOB_EXCLUSIVE:-?} $(date -Is)" | tee $O/node.txt
sha256sum $BIN | tee -a $O/node.txt
squeue -w $(hostname -s) -h -o "%u %C %j" | sort | uniq -c | head -20 >> $O/node.txt || true
# Job cgroup (v2) counters every 5 s: memory.current and anon (MB), CPU usage (s, cumulative), I/O bytes read and
# written (MB, cumulative, all devices) -> monitor.tsv. Only this job's processes are counted.
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
echo "cgroup $CG" >> $O/node.txt
( echo -e "t\tmem_mb\tanon_mb\tcpu_s\tread_mb\twrite_mb"
  while true; do
    printf "%s\t%s\t%s\t%s\t%s\n" $(date +%s) \
      $(( $(cat $CG/memory.current) / 1048576 )) $(( $(awk '$1=="anon"{print $2}' $CG/memory.stat) / 1048576 )) \
      $(( $(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat) / 1000000 )) \
      "$(awk '{for(i=2;i<=NF;i++){split($i,a,"=");if(a[1]=="rbytes")r+=a[2];if(a[1]=="wbytes")w+=a[2]}}END{printf "%d\t%d", r/1048576, w/1048576}' $CG/io.stat)"
    sleep 5
  done ) > $O/monitor.tsv 2>&1 &
MON=$!
IN=()
if [ -n "${INPUTS:-}" ]; then for f in $INPUTS; do IN+=($D/$f); done; else IN=($D/sample.1fq); fi
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
FMT=1pz; [ -n "${MTX:-}" ] && FMT=mtx
cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN "${IN[@]}" --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
  --out-prefix $SCR/out/ --threads ${T:-4} --pipeline --output-format $FMT ${MTX:+--raw-matrix} --bam-sort-mode disk \
  --genome-shared off "${SCREEN[@]}" "$@" > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
kill $MON 2>/dev/null || true
rsync -a --exclude '*.bam' --exclude '*.bai' --exclude '*Unmapped.out.mate*' --exclude '*.fq' --exclude '*_STARtmp*' \
  --exclude 'star_tmp*' $SCR/out/ $O/out/
grep -E "Elapsed|Maximum resident" $O/time.txt
grep -E "started|finished|STAR completed|indexed|refresh|Pileup:|\[singlet\] Total|sort" $O/run.log | head -30
