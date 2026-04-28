#!/bin/bash
#SBATCH --job-name=bench17

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=45G
#SBATCH --time=60:00
#SBATCH --output=/mnt/projects/debruinz_project/singlify_validation/bench17_%j.log
#SBATCH --error=/mnt/projects/debruinz_project/singlify_validation/bench17_%j.err

set -e
source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
ulimit -n 10240

SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
FQ=/mnt/projects/debruinz_project/singlify_validation/1fq/SRR17873408.1fq
GENOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
GTF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/genes/genes.gtf

echo "=== BENCH17 START: $(date) ==="
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -g | awk '/Mem/{print $2}') GB"

# Clean up leftovers
rm -rf /dev/shm/par17_* /dev/shm/str17_* /dev/shm/singlify_1fq_*

# Unload genome from shared memory if present (avoids limitBAMsortRAM issue with sorted BAM)
$SINGLIFY genome unload "$GENOME" 2>/dev/null || true

# Warm genome cache into page cache
cat "$GENOME/SA" "$GENOME/Genome" "$GENOME/SAindex" > /dev/null 2>&1
echo "[genome warmed: $(date)]"
echo "/dev/shm: $(df -h /dev/shm | awk 'NR==2{print $3" used / "$2" total"}')"

echo ""
echo "=== RUN 1: PARALLEL PILEUP ==="
OUTDIR=/dev/shm/par17_bench
rm -rf $OUTDIR && mkdir -p $OUTDIR
/usr/bin/time -f "par_wall=%e user=%U sys=%S maxrss_kb=%M" $SINGLIFY "$FQ" \
  --genome-dir "$GENOME" --exons "$GTF" \
  --out-prefix "$OUTDIR/" --threads 16
echo "PAR_EXIT=$?"
echo "=== PAR FILES ==="
ls -lh "$OUTDIR/"*.1pz 2>/dev/null || echo "no .1pz"
wc -l "$OUTDIR/exon_counts_barcodes.tsv" 2>/dev/null && cp "$OUTDIR/exon_counts_barcodes.tsv" /mnt/projects/debruinz_project/singlify_validation/par17_barcodes.tsv 2>/dev/null || true
rm -rf $OUTDIR /dev/shm/singlify_1fq_*

echo ""
echo "=== WARM CACHE AGAIN ==="
cat "$GENOME/SA" "$GENOME/Genome" "$GENOME/SAindex" > /dev/null 2>&1

echo ""
echo "=== RUN 2: STREAMING (--no-parallel-pileup) ==="
OUTDIR=/dev/shm/str17_bench
rm -rf $OUTDIR && mkdir -p $OUTDIR
/usr/bin/time -f "str_wall=%e user=%U sys=%S maxrss_kb=%M" $SINGLIFY "$FQ" \
  --genome-dir "$GENOME" --exons "$GTF" \
  --out-prefix "$OUTDIR/" --threads 16 --no-parallel-pileup
echo "STR_EXIT=$?"
echo "=== STR FILES ==="
ls -lh "$OUTDIR/"*.1pz 2>/dev/null || echo "no .1pz"
wc -l "$OUTDIR/exon_counts_barcodes.tsv" 2>/dev/null && cp "$OUTDIR/exon_counts_barcodes.tsv" /mnt/projects/debruinz_project/singlify_validation/str17_barcodes.tsv 2>/dev/null || true
rm -rf $OUTDIR /dev/shm/singlify_1fq_*

echo ""
echo "=== BENCH17 DONE: $(date) ==="
