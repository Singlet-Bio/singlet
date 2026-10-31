#!/bin/bash
#SBATCH --job-name=fqrs-time
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,SET=<GSM>,BIN=singlet-fqrs-pgo[,IN=none][,REPS=2] timing.sbatch
# Same node, same job, interleaved: fifo, chunk, fifo, chunk, ... (--read-source of one binary on one input), so the
# core-s difference is not node or co-tenant noise. Per run: user+sys of the whole process tree (/usr/bin/time), wall,
# max RSS, cgroup memory peak not separable per run (reported for the job). Outputs are not kept (only the timings).
# Result: $W/jobs/fq-read-source/timing/<SET>_<IN>.tsv
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
BIN=$W/jobs/fq-read-source/bin/$BIN; IN=${IN:-none}; REPS=${REPS:-2}
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
D=$W/data/fq-read-source/$SET; INPUT=$(ls $D/$IN/*.1fq | sort | paste -sd,)
if [[ $INPUT == *,* ]]; then INARG=(--1fq $INPUT); else INARG=($INPUT); fi
O=$W/jobs/fq-read-source/timing; mkdir -p $O; F=$O/${SET}_${IN}.tsv
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
echo -e "rep\tmode\tuser_s\tsys_s\twall_s\tmaxrss_kb\tinput_reads\tpass1_s\tnode" > $F
for r in $(seq 1 $REPS); do
  for m in fifo chunk; do
    SCR=/tmp/$USER-fqt-$SLURM_JOB_ID-$m$r; mkdir -p $SCR/out; export TMPDIR=$SCR; cd $SCR
    /usr/bin/time -f "%U %S %e %M" -o $SCR/t.txt $BIN "${INARG[@]}" --ref-base $REF --genome-dir $G/star_2.7.11b \
      --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads 16 --pipeline --output-format 1pz --bam-sort-mode disk \
      --genome-shared off --read-source $m > $SCR/run.log 2>&1 || echo "exit $?" >> $SCR/run.log
    read U S E M < $SCR/t.txt
    N=$(grep -m1 "Number of input reads" $SCR/out/star_Log.final.out | cut -f2)
    P1=$(grep -oP '\[1fq-pass1\] Scanned \d+ blocks \(\K[\d.]+' $SCR/run.log | head -1)
    echo -e "$r\t$m\t$U\t$S\t$E\t$M\t$N\t$P1\t$(hostname -s)" | tee -a $F
    cd /; rm -rf $SCR
  done
done
echo "cgroup memory peak (job): $(( $(cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')/memory.peak 2>/dev/null || echo 0) / 1048576 )) MB"
