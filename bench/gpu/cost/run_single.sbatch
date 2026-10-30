#!/bin/bash
#SBATCH --job-name=w11-cost1
#SBATCH --account=debruinz
#SBATCH --nice=10000
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# W11 CPU baseline, step 2: one real sample per task, campaign command line, binary singlet-w3 (PGO 3'/5',
# source-identical to prod/campaign c4a1be38 except bench/star_accel/w3_timing.sbatch).
#   sbatch --array=0-5%3 --export=ALL,SETS="GSM4006843 GSM5411066 GSM7901264 GSM7036403 GSM6470208 GSM7656149" run_single.sbatch
# Reads, BAM and outputs stay in the job /tmp. Copies back only run.log, monitor.tsv (cgroup CPU + run.log size every 2 s), time/node/mem files.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
[ -n "${SETS:-}" ] && { A=($SETS); SET=${A[$SLURM_ARRAY_TASK_ID]}; }
T=${T:-4}; BIN=${BIN:-$PB/bin/singlet-w3}; TAG=${TAG:-single}
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
case $SET in GSM5411066|nuclei*|neuron*) G=$REF/GRCm39-2024-A;; esac
D=$W/data/atlas-rerun/perf/$SET; F=$D/sample.1fq; case $SET in GSM*) ;; *) D=$PB/data/$SET; F=$D/$SET.1fq;; esac; [ -e $F ] || { echo "MISSING INPUT $F"; exit 3; }
O=$W/jobs/w11/cost/runs/$TAG/$SET; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-w11c-$SLURM_JOB_ID; mkdir -p $SCR/out
trap 'kill $MON 2>/dev/null || true; rm -rf $SCR; exit 0' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
{
 echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK set=$SET T=$T part=$SLURM_JOB_PARTITION jobid=$SLURM_JOB_ID $(date -Is)"
 sha256sum $BIN; ls -l $BIN; strings $BIN | grep -m3 -E 'bb51b063|c4a1be38' || true
 echo "loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
 echo "cotenant_cpus=$(squeue -w $(hostname -s) -h -t R -o '%C' | awk -v j=$SLURM_JOB_ID '{s+=$1}END{print s-0}')"
 lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket|Vendor|CPU max)'
 nproc
} | tee $O/node.txt
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
( echo -e "t\tmem_mb\tanon_mb\tcpu_s\tlogbytes"
  while true; do
    printf "%s\t%s\t%s\t%s\t%s\n" $(date +%s.%N) $(( $(cat $CG/memory.current) / 1048576 )) \
      $(( $(awk '$1=="anon"{print $2}' $CG/memory.stat) / 1048576 )) \
      "$(awk '$1=="usage_usec"{printf "%.2f", $2/1e6}' $CG/cpu.stat)" "$(stat -c %s $O/run.log 2>/dev/null || echo 0)"
    sleep 2
  done ) > $O/monitor.tsv 2>/dev/null &
MON=$!
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
cd $SCR
T0=$(date +%s.%N)
/usr/bin/time -v -o $O/time.txt $BIN $F --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
  --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off \
  "${SCREEN[@]}" ${EXTRA:-} > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
T1=$(date +%s.%N); echo "t0=$T0 t1=$T1" > $O/wall.txt
kill $MON 2>/dev/null || true
echo "memory_peak_mb=$(( $(cat $CG/memory.peak 2>/dev/null || echo 0) / 1048576 ))" > $O/mem.txt
grep -b -E 'Loading screening indices|Loading host Bloom filter|EM screening' $O/run.log > $O/nonhost_offsets.txt || true
mkdir -p $O/small; for f in summary.json pileup_stats.json nonhost/nonhost_summary.json; do [ -f $SCR/out/$f ] && cp $SCR/out/$f $O/small/$(basename $f); done
( cd $SCR/out && find . -type f -printf '%s\t%p\n' | sort -k2 ) > $O/output_sizes.tsv
$W/envs/pbench/bin/python $W/bam-stream/parse_run.py $O > $O/stages.json 2>/dev/null || true
grep -E "Elapsed|Maximum resident|exit" $O/time.txt $O/run.log | tail -3; cat $O/mem.txt
exit 0
