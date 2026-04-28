#!/usr/bin/env bash
source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
ulimit -n 10240

SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
FQ=/mnt/projects/debruinz_project/singlify_validation/1fq/SRR32855204.1fq
GENOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
GTF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/genes/genes.gtf
WL=/mnt/home/debruinz/Singlet-AI/singlify/whitelists/gex_737K-arc-v1.txt
BC=/mnt/projects/debruinz_project/singlify_validation/starsolo/SRR32855204/Solo.out/Gene/filtered/barcodes.tsv
OUTDIR=/dev/shm/n11_smoke_$$
mkdir -p "$OUTDIR"
cat "$GENOME/SA" "$GENOME/Genome" "$GENOME/SAindex" > /dev/null 2>&1

/usr/bin/time -f "wall=%e" "$SINGLIFY" "$FQ" \
  --genome-dir "$GENOME" --whitelist "$WL" --barcodes "$BC" \
  --exons "$GTF" --out-prefix "$OUTDIR/" --threads 16 --pipeline \
  --cell-calling 2>&1 | \
  grep -E "ambient|cell_calling|wall=|error|ERROR"

echo "=== output files ==="
ls "$OUTDIR/"ambient* 2>/dev/null || echo "no ambient files"
echo "=== ambient_profile head ==="
head -3 "$OUTDIR/ambient_profile.tsv" 2>/dev/null || echo "missing"
echo "=== contamination head ==="
head -3 "$OUTDIR/ambient_contamination.tsv" 2>/dev/null || echo "missing"
echo "=== stats ==="
awk -F"\t" 'NR>1{sum+=$2;n++} END{print "n_cells="n, "mean_rho="sum/n}' \
  "$OUTDIR/ambient_contamination.tsv" 2>/dev/null
rm -rf "$OUTDIR"
echo DONE
