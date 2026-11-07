#!/bin/bash
#SBATCH --job-name=rg-atac-post
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Diagnose the Multiome 10k ATAC exit 5 (after "cells: 12063 of 65268 candidates"): re-run only the post-alignment step of the pinned
# binary from the kept BAM, with stderr visible. Same whitelists/GEX pairing as run_sample_rg.sbatch.
set -u
source /mnt/projects/debruinz_project/debruinz/singlet/release-gate/jobs/rg_common.sh; rg_check_bin
G=$REF/GRCh38-2024-A; O=$RG/runs/atacsp/pbmc_multiome10k_atac; N=$O/post_diag; rm -rf $N; mkdir -p $N
/usr/bin/time -v -o $N/time_post.txt $BIN atac --bam $O/star_Aligned.sortedByCoord.out.bam --out $N --gtf $G/genes/genes.gtf \
  --whitelist $PB/whitelists/atacsp/737K-cratac-v1.txt --whitelist $PB/whitelists/737K-arc-v1-atac-gex.tsv.gz \
  --paired-gex $RG/runs/atacsp/pbmc_multiome10k_gex > $N/atac.log 2>&1
echo "exit=$?"; tail -n 30 $N/atac.log | cut -c1-300; grep -E "Maximum resident|Elapsed|Exit status" $N/time_post.txt; ls $N
