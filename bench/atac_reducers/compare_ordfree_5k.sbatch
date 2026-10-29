#!/bin/bash
#SBATCH --job-name=ao-cmp5k
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# A1/A2 on the CPU ATAC path: old (singlet-atacsp) vs new (singlet-atacsp-ordfree) binary on the kept sorted BAM of RUN and on the same
# BAM in name-collated (unsorted) order. Expected: new(sorted) == old(sorted); new(collated) == new(sorted); old(collated) over-counts.
#   RUN=<run under runs/atacsp, default pbmc_multiome3k_atac> PAIRED_GEX=<gex run> OLD=singlet-atacsp NEW=singlet-atacsp-ordfree
set -u
# 5k NextGEM ATAC (not Multiome: no --paired-gex). Single-threaded post-alignment stage; no thread option exists in `singlet atac`.
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; ENV=$W/envs/pbench; export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
RUN=${RUN:-atac_pbmc_5k_nextgem}; PAIRED_GEX=${PAIRED_GEX:-}
OLD=$PB/bin/${OLD:-singlet-atacsp-pr48}; NEW=$PB/bin/${NEW:-singlet-atacsp-ordfree}
G=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
O=$PB/runs/atacsp/$RUN; OUT=$PB/runs/atacsp/${RUN}_ordfree; mkdir -p $OUT
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT INT TERM
echo "[cmp] old=$(sha256sum $OLD | cut -c1-12) new=$(sha256sum $NEW | cut -c1-12)"
echo "[cmp] collate $(date +%T)"
samtools collate -@ 8 -T $SCR/col -o $SCR/atac_collated.bam $O/atac_sorted.bam || exit 1
run() {  # name binary bam
  local d=$OUT/$1; rm -rf $d; mkdir -p $d
  /usr/bin/time -v -o $d/time.txt $2 atac --bam $3 --out $d --gtf $G/genes/genes.gtf --whitelist $PB/whitelists/atacsp/737K-cratac-v1.txt \
    --whitelist $PB/whitelists/737K-arc-v1-atac-gex.tsv.gz ${PAIRED_GEX:+--paired-gex $PAIRED_GEX} > $d/atac.log 2>&1
  echo "[cmp] $1 exit=$? wall=$(grep 'Elapsed (wall' $d/time.txt | sed 's/.*: //') rss_kb=$(grep 'Maximum resident' $d/time.txt | sed 's/.*: //')"
  for f in fragments.tsv.gz atac_peak_counts.mtx.gz; do zcat $d/$f | md5sum | cut -c1-32 | sed "s/^/  $1 $f /"; done
  for f in peaks.bed atac_barcode_metrics.tsv atac_cells.tsv; do md5sum $d/$f | cut -c1-32 | sed "s/^/  $1 $f /"; done
  grep -E "fragments:|layout|peaks:|cells:" $d/atac.log | sed "s/^/  $1 | /"
}
run old_sorted $OLD $O/atac_sorted.bam
run new_sorted $NEW $O/atac_sorted.bam
run new_collated $NEW $SCR/atac_collated.bam
run old_collated $OLD $SCR/atac_collated.bam
echo "[cmp] done $(date +%T)"
