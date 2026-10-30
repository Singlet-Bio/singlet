#!/bin/bash
#SBATCH --job-name=bs-umigrp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,SET=GSM7901264 umi_group_probe.sbatch     (~0.6 node-h at 8 cores)
# Which UMI grouping should R9 use (per cell x feature, or per cell x gene)? Runs the sorted path keeping STAR's
# sorted BAM in job /tmp with GX/GN added, then umi_group_probe.py on chr1 + the IG/TR loci. Kept: the JSON only.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/${BIN:-singlet-bs-r11}
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A; J=$W/jobs/bam-stream
D=$W/data/bam-stream/$SET; O=$W/bam-stream/umi_group/$SET; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-bsug-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF SINGLET_KEEP_STAR_BAM=1
export SINGLET_EXTRA_SAM_ATTRS="GX GN"
ST=$W/envs/pbench/bin/samtools
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
cd $SCR
$BIN $D/sample.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --threads 8 --pipeline \
  --output-format 1pz --sorted-bam --bam-sort-mode disk --genome-shared off "${SCREEN[@]}" --out-prefix $SCR/out/ > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
BAM=$(ls $SCR/out/*sortedByCoord*.bam | head -1); echo "bam=$BAM"
[ -f $BAM.bai ] || $ST index -@ 8 $BAM
$ST view -H $BAM | grep -m1 "^@PG" | grep -o "outSAMattributes[^-]*" | tee $O/attrs.txt
LOCI="chr1 chr2:88700000-90400000 chr7:38200000-38400000 chr7:142200000-142900000 chr14:21500000-22600000 chr14:105500000-107000000 chr22:22000000-23000000"
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
$ST view $BAM $LOCI > $SCR/reads.sam
in_runtime python -I $J/umi_group_probe.py $G/genes/genes.gtf $REF/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz $O/probe.json < $SCR/reads.sam > $O/probe.log 2>&1
grep -E "Done:|status=" $O/run.log | tail -2
