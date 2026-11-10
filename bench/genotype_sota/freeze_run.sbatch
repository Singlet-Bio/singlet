#!/bin/bash
#SBATCH --job-name=gsota-frz
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --array=0-6%3
#SBATCH --time=03:55:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# genotype-sota FREEZE RUN: every genotype pool end to end on ONE binary (the release freeze commit), so provenance names one commit and SHA.
#   sbatch --export=ALL,BIN=<dir>/singlet-gpu,TAG=<freeze> freeze_run.sbatch   (then freeze_score.sbatch with the same TAG)
# was: pool end to end on the tip binary (prod/campaign f15dad07), auto donor count (K never given), .1fq binned (default).
#   sbatch --array=0-4%2 run_gate_pool.sbatch [cpu|gpu]
# Output tree $WORK/bench-public/runs_$TAG/<rundir>/singlet_auto, with the tool dirs of runs/ symlinked so the
# bench/public/genotype/lib scripts run unchanged on RUNS=runs_$TAG.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet
BIN=${BIN:?set BIN=<freeze binary>}; TAG=${TAG:?set TAG=<short name of the freeze commit>}
echo "gate sha256 $(sha256sum $BIN | cut -d' ' -f1)"
D=$W/bench-public/data
POOLS=("kang2018_b2_stim $D/kang_b2stim/fastq/R1.fastq.gz $D/kang_b2stim/fastq/R2.fastq.gz"
       "kang2018_b1_a $D/kang_b1A/fastq/R1.fastq.gz $D/kang_b1A/fastq/R2.fastq.gz"
       "kang2018_b1_b $D/kang_b1B/fastq/R1.fastq.gz $D/kang_b1B/fastq/R2.fastq.gz"
       "kang2018_b1_c $D/kang_b1C/fastq/R1.fastq.gz $D/kang_b1C/fastq/R2.fastq.gz"
       "maester_k562_bt142_3p $W/data/maester/k562_3p/k562_3p_R1.fastq.gz $W/data/maester/k562_3p/k562_3p_R2.fastq.gz"
       "maester_bpdcn712_3p $W/data/maester/bpdcn_3p/bpdcn_3p_R1.fastq.gz $W/data/maester/bpdcn_3p/bpdcn_3p_R2.fastq.gz"
       "kang2018_b2_ctrl $D/kang_b2ctrl/fastq/R1.fastq.gz $D/kang_b2ctrl/fastq/R2.fastq.gz")
set -- ${POOLS[${SLURM_ARRAY_TASK_ID:-0}]}; RD=$1; R1=$2; R2=$3
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
G=$W/bench-public/runs_$TAG/$RD; mkdir -p $G
for t in vireo souporcell mgatk; do [ -e $W/bench-public/runs/$RD/$t ] && ln -sfn $W/bench-public/runs/$RD/$t $G/$t; done
O=$G/singlet_auto; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $R1 $R2 -o $SCR/s.1fq -v > $O/encode.log 2>&1 || { tail -5 $O/encode.log; exit 1; }
ls -l $SCR/s.1fq
export SINGLET_STALL_TIMEOUT=10800
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
  --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline \
  --n-donors -1 --tagged-bam $O/possorted.bam --bam-sort-mode disk --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "Elapsed|Maximum resident" $O/time.txt; grep -E "auto-K|\[demux\]" $O/run.log | head -12
