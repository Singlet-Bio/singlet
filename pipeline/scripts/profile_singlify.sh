#!/bin/bash
# singlify profiling: compare sequential vs pipe mode
# Usage: bash profile_singlify.sh
set -e

# Paths
STAR=/mnt/home/debruinz/Singlet-AI/STAR/source/STAR_stock
PILEUP=/mnt/home/debruinz/Singlet-AI/singlet-pileup/build_v2/singlet-pileup
SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
GENOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
R2=/mnt/home/debruinz/Singlet-AI/STAR/test_data/test_R2.fastq.gz
R1=/mnt/home/debruinz/Singlet-AI/STAR/test_data/test_R1.fastq.gz
WL=/mnt/home/debruinz/Singlet-AI/STAR/test_data/whitelist_v3.txt
BC=/mnt/projects/debruinz_project/cellarium/pipeline/smoke_test/GSM8313394/starsolo/Solo.out/Gene/filtered/barcodes.tsv
GTF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/genes/genes.gtf.gz
VCF=/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.vcf.gz

export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:$LD_LIBRARY_PATH

echo "=== singlify Profiling ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "CPUs: $(nproc)"
echo ""

STAR_THREADS=2

# ── Test 1: Sequential (STAR → file → pileup) ──
echo "--- Test 1: Sequential STAR + pileup ---"
rm -rf /tmp/prof_seq_star /tmp/prof_seq_pileup

T0=$(date +%s%N)

# Step A: STAR alignment
echo "  STAR alignment (${STAR_THREADS} threads)..."
STAR_T0=$(date +%s%N)
$STAR --runMode alignReads \
    --genomeDir $GENOME \
    --readFilesIn $R2 $R1 \
    --readFilesCommand zcat \
    --soloType CB_samTagOut \
    --soloCBwhitelist $WL \
    --soloUMIlen 12 \
    --outSAMtype BAM Unsorted \
    --outBAMcompression 0 \
    --outSAMattributes NH HI nM AS CR UR CB sS sQ sM \
    --clipAdapterType CellRanger4 \
    --outFilterScoreMin 30 \
    --soloCBmatchWLtype 1MM \
    --runThreadN $STAR_THREADS \
    --outFileNamePrefix /tmp/prof_seq_star/ 2>/dev/null
STAR_T1=$(date +%s%N)
STAR_MS=$(( (STAR_T1 - STAR_T0) / 1000000 ))
echo "  STAR: ${STAR_MS}ms"

# Step B: Pileup on BAM
echo "  Pileup (1 thread, file mode)..."
PILEUP_T0=$(date +%s%N)
$PILEUP --barcodes $BC --snps $VCF --exons $GTF \
    --out-prefix /tmp/prof_seq_pileup --threads 1 --workers 1 --output-format mtx \
    /tmp/prof_seq_star/Aligned.out.bam 2>/dev/null
PILEUP_T1=$(date +%s%N)
PILEUP_MS=$(( (PILEUP_T1 - PILEUP_T0) / 1000000 ))
echo "  Pileup: ${PILEUP_MS}ms"

T1=$(date +%s%N)
SEQ_MS=$(( (T1 - T0) / 1000000 ))
echo "  Sequential total: ${SEQ_MS}ms"
echo ""

# ── Test 2: Pipe via singlify ──
echo "--- Test 2: singlify (STAR pipe) ---"
rm -rf /tmp/prof_singlify

T0=$(date +%s%N)
$SINGLIFY \
    --star-binary $STAR \
    --genome-dir $GENOME \
    --reads $R2 $R1 \
    --whitelist $WL \
    --barcodes $BC \
    --exons $GTF \
    --snps $VCF \
    --out-prefix /tmp/prof_singlify \
    --output-format mtx \
    --star-threads $STAR_THREADS \
    --pileup-threads 1 2>/dev/null
T1=$(date +%s%N)
PIPE_MS=$(( (T1 - T0) / 1000000 ))
echo "  singlify total: ${PIPE_MS}ms"
echo ""

# ── Test 3: singlify BAM mode (bypass STAR) ──
echo "--- Test 3: singlify (BAM file, no STAR) ---"
rm -rf /tmp/prof_singlify_bam

T0=$(date +%s%N)
$SINGLIFY \
    --bam-file /tmp/prof_seq_star/Aligned.out.bam \
    --barcodes $BC \
    --exons $GTF \
    --snps $VCF \
    --out-prefix /tmp/prof_singlify_bam \
    --output-format mtx \
    --pileup-threads 1 2>/dev/null
T1=$(date +%s%N)
BAM_MS=$(( (T1 - T0) / 1000000 ))
echo "  singlify BAM: ${BAM_MS}ms"
echo ""

# ── Summary ──
echo "=== Summary ==="
echo "  Sequential (STAR → file → pileup):  ${SEQ_MS}ms (STAR=${STAR_MS}ms + pileup=${PILEUP_MS}ms)"
echo "  singlify pipe (STAR | pileup):       ${PIPE_MS}ms"
echo "  singlify BAM-only (no STAR):         ${BAM_MS}ms"
OVERLAP=$(( SEQ_MS - PIPE_MS ))
echo "  Overlap savings:                     ${OVERLAP}ms"
if [ $SEQ_MS -gt 0 ]; then
    PCT=$(( OVERLAP * 100 / SEQ_MS ))
    echo "  Speedup:                             ${PCT}%"
fi

# ── Correctness check ──
echo ""
echo "=== Correctness (singlify pipe vs sequential) ==="
python3 -c "
import scipy.io
seq = '/tmp/prof_seq_pileup'
pipe = '/tmp/prof_singlify'
for name in ['exon_counts', 'intron_counts', 'snp_ad', 'snp_dp']:
    r = scipy.io.mmread(f'{seq}/{name}.mtx').tocsr()
    t = scipy.io.mmread(f'{pipe}/{name}.mtx').tocsr()
    diff = (r - t).nnz if r.shape == t.shape else -1
    status = 'EXACT' if diff == 0 else f'DIFF {diff}'
    print(f'  {name}: {status}')
"
