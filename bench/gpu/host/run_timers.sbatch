#!/bin/bash
#SBATCH --job-name=w11-host
#SBATCH --account=debruinz
#SBATCH --nice=10000
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# W11 host-time study, step 4. One real sample per array task with the [w11t]-instrumented DEV binary.
#   sbatch --array=0-5%3 --export=ALL,SETS="GSM4006843 GSM5411066 GSM7901264 GSM7036403 GSM6470208 GSM7656149" run_timers.sbatch
# Same command line as jobs/bam-stream/phase1.sbatch (--threads 4 --pipeline --output-format 1pz --bam-sort-mode disk
# --genome-shared off, run screen when present); cpus-per-task 8 so the 4-thread run is not starved.
# Copies back ONLY run.log, the [w11] lines, monitor.tsv (cgroup CPU every 5 s), node/time/mem files, small json/tsv summaries.
# No BAM, reads, or .1pz leave the job's /tmp.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
[ -n "${SETS:-}" ] && { A=($SETS); SET=${A[$SLURM_ARRAY_TASK_ID]}; }
T=${T:-4}; BIN=${BIN:-$W/jobs/w11/host/bin/singlet-timers}
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
[ "$SET" = GSM5411066 ] && G=$REF/GRCm39-2024-A   # the mouse sample
D=$W/data/bam-stream/$SET; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$SET
O=$W/jobs/w11/host/runs/$SET; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-w11h-$SLURM_JOB_ID; mkdir -p $SCR/out
trap 'kill $MON 2>/dev/null; rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
{
 echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK set=$SET T=$T part=$SLURM_JOB_PARTITION jobid=$SLURM_JOB_ID $(date -Is)"
 sha256sum $BIN; echo "loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
 echo "cotenant_cpus=$(squeue -w $(hostname -s) -h -t R -o '%C' | awk -v j=$SLURM_JOB_ID '{s+=$1}END{print s-0}')"
 grep -m1 'model name' /proc/cpuinfo
} | tee $O/node.txt
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
( echo -e "t\tmem_mb\tanon_mb\tcpu_s\tread_mb\twrite_mb"
  while true; do
    printf "%s\t%s\t%s\t%s\t%s\n" $(date +%s) $(( $(cat $CG/memory.current) / 1048576 )) \
      $(( $(awk '$1=="anon"{print $2}' $CG/memory.stat) / 1048576 )) $(( $(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat) / 1000000 )) \
      "$(awk '{for(i=2;i<=NF;i++){split($i,a,"=");if(a[1]=="rbytes")r+=a[2];if(a[1]=="wbytes")w+=a[2]}}END{printf "%d\t%d", r/1048576, w/1048576}' $CG/io.stat)"
    sleep 5
  done ) > $O/monitor.tsv 2>&1 &
MON=$!
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $D/sample.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
  --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off \
  "${SCREEN[@]}" ${EXTRA:-} > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
kill $MON 2>/dev/null || true
echo "memory_peak_mb=$(( $(cat $CG/memory.peak 2>/dev/null || echo 0) / 1048576 ))" > $O/mem.txt
grep -E '^\[w11[tn]\]' $O/run.log > $O/w11.log || true
# small summaries only
mkdir -p $O/small
for f in summary.json pileup_stats.json metrics_summary.csv provenance.json sex_call.json ancestry_call.json \
         ambient_contamination.json nonhost/nonhost_summary.json read_qc_summary.json rrna_report.json; do
  [ -f $SCR/out/$f ] && { mkdir -p $O/small/$(dirname $f); cp $SCR/out/$f $O/small/$f; }
done
( cd $SCR/out && find . -type f -printf '%s\t%p\n' | sort -k2 ) > $O/small/output_sizes.tsv
python3 $W/bam-stream/parse_run.py $O > $O/stages.json 2>/dev/null || true
grep -E "Elapsed|Maximum resident" $O/time.txt; cat $O/mem.txt; grep -c '^\[w11t\]' $O/run.log
