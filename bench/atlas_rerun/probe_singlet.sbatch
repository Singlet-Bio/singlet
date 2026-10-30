#!/bin/bash
#SBATCH --job-name=ar-singlet
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#   sbatch probe_singlet.sbatch <GSM>
# Cost-model check at real depth: Singlet (fixed engine, campaign flags, 4 threads) on every run of
# one deep sample at K = 1 (all reads) and K = 4 (every 4th read pair), timed with GNU time.
set -eu   # no pipefail: zcat | head closes pipes early on purpose
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
GSM=$1; KS=(1 4); K=${KS[$SLURM_ARRAY_TASK_ID]}
D=$W/data/atlas-rerun/depth/$GSM; O=$D/singlet_k$K; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID-$K; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
first=$(ls $D/fq/*_1.fastq.gz | head -1); base=${first%_1.fastq.gz}; CB=""; CDNA=""; best=0
for f in ${base}_*.fastq.gz; do
  n=${f##*_}; n=${n%.fastq.gz}; len=$(zcat $f | head -2 | tail -1 | tr -d '\n' | wc -c)
  if [ $len -ge 26 ] && [ $len -le 30 ]; then CB=$n; fi
  if [ $len -gt $best ]; then best=$len; CDNA=$n; fi
done
for side in $CB $CDNA; do
  ls $D/fq/*_${side}.fastq.gz | while read f; do zcat $f | awk -v k=$K 'int((NR-1)/4) % k == 0'; done \
    | $W/envs/pbench/bin/pigz -p 4 > $SCR/r$side.fastq.gz
done
cd $SCR
/usr/bin/time -v -o $O/encode.time.txt $PB/bin/singlet-fixed encode --reads $SCR/r$CB.fastq.gz $SCR/r$CDNA.fastq.gz \
  -o $SCR/s.1fq > $O/encode.log 2>&1
rm -f $SCR/r*.fastq.gz
export SINGLET_STALL_TIMEOUT=10800
/usr/bin/time -v -o $O/time.txt $PB/bin/singlet-fixed $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
  --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 4 --pipeline \
  --output-format 1pz > $O/run.log 2>&1 || echo "exit $?"
grep -E "Elapsed|Maximum resident|User time" $O/time.txt $O/encode.time.txt
grep -E "total_reads|molecules_" $O/pileup_stats.json
