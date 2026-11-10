#!/bin/bash
#SBATCH --job-name=gsota-snp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:55:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# genotype-sota modality 1: engine end to end from the FASTQs at a chosen .1fq quality mode, then (separately) score.
#   sbatch --array=0-5%2 run_snp_variant.sbatch [BIN_NAME]      task = pool x {binned,full}
# Output: $WORK/bench-public/runs/<rundir>/singlet_<tag>_<quality>/ (private/donor/*.vcf stay on Clipper).
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet
BINNAME=${1:-mc99cpu}; BIN=$W/pipeline-benchmark/bin/$BINNAME; TAG=$BINNAME
POOLS=(kang_b2stim:kang2018_b2_stim kang_b1C:kang2018_b1_c kang_b1A:kang2018_b1_a)
QUALS=(binned full)
T=${SLURM_ARRAY_TASK_ID:-0}; PL=${POOLS[$((T/2))]}; Q=${QUALS[$((T%2))]}
DDIR=$W/bench-public/data/${PL%%:*}; RD=${PL##*:}
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
O=$W/bench-public/runs/$RD/singlet_${TAG}_$Q; rm -rf $O; mkdir -p $O
echo "bin $BINNAME sha256 $(sha256sum $BIN | cut -d' ' -f1) pool $RD quality $Q"
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $DDIR/fastq/R1.fastq.gz $DDIR/fastq/R2.fastq.gz -o $SCR/s.1fq --quality $Q -v > $O/encode.log 2>&1 || { echo encode failed; tail -5 $O/encode.log; exit 1; }
ls -l $SCR/s.1fq
export SINGLET_STALL_TIMEOUT=10800
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
  --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline \
  --n-donors -1 --tagged-bam $O/possorted.bam --bam-sort-mode disk --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "Elapsed|Maximum resident" $O/time.txt; grep -E "auto-K|\[demux\]" $O/run.log | head -10; ls $O/private/donor | head -3
