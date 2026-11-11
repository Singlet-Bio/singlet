#!/bin/bash
#SBATCH -J hgmm-bam
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 16
#SBATCH --mem=150G
#SBATCH -t 03:00:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# hgmm1k_v3 with the tagged BAM (CB/UB tags) for the molecule-level soup test. Pin-equivalent binary (92095e29: same data outputs as d9834b8a).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; RG=$W/release-gate; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=$RG/bin/92095e29/singlet-cpu; G=$REF/GRCh38-GRCm39-barnyard; O=$W/expression-sota/hgmm_bam; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; export TMPDIR=$SCR SINGLET_STALL_TIMEOUT=10800
cp $PB/data/hgmm1k_v3/hgmm1k_v3.1fq $SCR/s.1fq; cd $SCR
$BIN $SCR/s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline --tagged-bam $O/possorted.bam --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?"
tail -3 $O/run.log; ls -la $O/possorted.bam
