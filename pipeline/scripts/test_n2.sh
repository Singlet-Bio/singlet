#!/bin/bash
source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
ulimit -n 10240

SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
FQ=/mnt/projects/debruinz_project/singlify_validation/1fq/SRR20020820.1fq
GENOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
GTF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/genes/genes.gtf
OUTDIR=/dev/shm/n2test_${$}
mkdir -p $OUTDIR

echo "=== N2 test: no --whitelist, no --barcodes ==="
$SINGLIFY "$FQ" \
  --genome-dir "$GENOME" \
  --exons "$GTF" \
  --out-prefix "$OUTDIR/" \
  --threads 4 \
  --pipeline 2>&1 | head -25

echo "=== exit $? ==="
rm -rf $OUTDIR /dev/shm/singlify_1fq_*
