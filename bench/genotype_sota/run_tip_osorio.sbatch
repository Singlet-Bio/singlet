#!/bin/bash
#SBATCH --job-name=gsota-otip
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Osorio 2019 GM12878 + GM18502 1:1 mix (10x v2, 150-nt R1 trimmed to 26) on the gate binary, auto-K, then cellsnp-lite 1a + vireo K=2
# (the competitor) on the same tagged BAM and cells. Truth: 1000 Genomes WGS genotypes of the two lines (private, on Clipper).
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
BIN=${TIPBIN:?}/singlet-gpu; echo "gate sha256 $(sha256sum $BIN | cut -d' ' -f1)"
D=$W/bpub/data/osorio2019_lcl_mix/SRR8551678
RD=osorio2019_lcl_mix; G=$W/bench-public/runs_tip/$RD; O=$G/singlet_auto; rm -rf $O; mkdir -p $O
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
zcat $D/SRR8551678_S1_L001_R1_001.fastq.gz | awk 'NR%2==0 {print substr($0,1,26); next} {print}' | pigz -p 8 > $SCR/R1.fq.gz
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $SCR/R1.fq.gz $D/SRR8551678_S1_L001_R2_001.fastq.gz -o $SCR/s.1fq -v > $O/encode.log 2>&1 || { tail -5 $O/encode.log; exit 1; }
ls -l $SCR/s.1fq; rm -f $SCR/R1.fq.gz
export SINGLET_STALL_TIMEOUT=10800
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
  --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline \
  --n-donors -1 --tagged-bam $O/possorted.bam --bam-sort-mode disk --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "Elapsed|Maximum resident" $O/time.txt; grep -E "auto-K|\[demux\]" $O/run.log | head -12
rm -f $SCR/s.1fq
# competitors: reuse the vireo/cellsnp run on the gate-binary BAM (same alignment code)
ln -sfn $W/bench-public/runs_gate/$RD/vireo $G/vireo
