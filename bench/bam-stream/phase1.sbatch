#!/bin/bash
#SBATCH --job-name=bs-p1
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nice=10000
#SBATCH --mem=80G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   Submit: sbatch --cpus-per-task=T [--array=..%4] --export=ALL,SET=<id>,T=<threads>,REP=<n>[,BIN=name][,EXTRA="..."] phase1.sbatch
#   (SET may also come from an array index through SETS="a b c".)
# Baseline of the CURRENT sorted path, same recipe as deep-perf/run_deep.sbatch (disk sort, private genome, run screen when present),
# plus: cgroup memory.peak, per-device-free cgroup io, peak size of the job's /tmp, node load and co-tenant CPUs.
# Output: $W/bam-stream/runs/<SET>/t<T>_r<REP>/{run.log,time.txt,monitor.tsv,node.txt,mem.txt,out/}  (no BAMs, no reads)
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
[ -n "${SETS:-}" ] && { A=($SETS); SET=${A[$SLURM_ARRAY_TASK_ID]}; }
T=${T:-4}; REP=${REP:-1}; BIN=$PB/bin/${BIN:-singlet-campaign}
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
[ "$SET" = GSM5411066 ] && G=$REF/GRCm39-2024-A   # the mouse sample
# input lookup: bam-stream staging first, then the earlier perf staging
D=$W/data/bam-stream/$SET; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$SET
O=$W/bam-stream/runs${SUF:-}/$SET/t${T}_r${REP}; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-bs-$SLURM_JOB_ID; mkdir -p $SCR/out
trap 'kill $MON $TMON 2>/dev/null; rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
{
 echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK set=$SET T=$T rep=$REP part=$SLURM_JOB_PARTITION $(date -Is)"
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
( pk=0; while true; do u=$(du -sm $SCR 2>/dev/null | cut -f1); [ "${u:-0}" -gt $pk ] && { pk=$u; echo "tmp_peak_mb=$pk" > $O/tmp_peak.txt; }; sleep 10; done ) &
TMON=$!
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
# BCFROM=<runs dir>: use that run's refreshed auto_barcodes.tsv as --barcodes (fixes the barcode set and order for A/B identity runs)
[ -n "${BCFROM:-}" ] && EXTRA="${EXTRA:-} --barcodes $W/bam-stream/$BCFROM/$SET/t4_r1/out/auto_barcodes.tsv"
cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $D/sample.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
  --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off \
  "${SCREEN[@]}" ${EXTRA:-} > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
kill $MON $TMON 2>/dev/null || true
{ echo "memory_peak_mb=$(( $(cat $CG/memory.peak 2>/dev/null || echo 0) / 1048576 ))"; cat $O/tmp_peak.txt 2>/dev/null; } > $O/mem.txt
rsync -a --exclude '*.bam' --exclude '*.bai' --exclude '*Unmapped.out.mate*' --exclude '*.fq' --exclude '*_STARtmp*' --exclude 'star_tmp*' $SCR/out/ $O/out/
python3 $W/bam-stream/parse_run.py $O > $O/stages.json 2>/dev/null || true
grep -E "Elapsed|Maximum resident" $O/time.txt; cat $O/mem.txt
