source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
ulimit -n 10240

SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
FQ=/mnt/projects/debruinz_project/singlify_validation/1fq/SRR17873408.1fq
GENOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
GTF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/genes/genes.gtf
LOGFILE=/mnt/projects/debruinz_project/singlify_validation/bench17_results.log
rm -f "$LOGFILE"

# Clean up leftover dirs
rm -rf /dev/shm/par17_* /dev/shm/str17_* /dev/shm/singlify_1fq_*

# Warm genome cache
cat "$GENOME/SA" "$GENOME/Genome" "$GENOME/SAindex" > /dev/null 2>&1
echo "[genome cache warmed]" >> "$LOGFILE"

echo "=== RUN 1: PARALLEL PILEUP ===" >> "$LOGFILE"
OUTDIR=/dev/shm/par17_bench
rm -rf $OUTDIR && mkdir -p $OUTDIR
/usr/bin/time -f "par_wall=%e user=%U sys=%S maxrss_kb=%M" $SINGLIFY "$FQ" \
  --genome-dir "$GENOME" --exons "$GTF" \
  --out-prefix "$OUTDIR/" --threads 16 >> "$LOGFILE" 2>&1
echo "PAR_EXIT=$?" >> "$LOGFILE"
echo "=== PAR FILES ===" >> "$LOGFILE"
ls -lh "$OUTDIR/"*.1pz 2>/dev/null >> "$LOGFILE" || echo "no .1pz" >> "$LOGFILE"
wc -l "$OUTDIR/exon_counts_barcodes.tsv" 2>/dev/null >> "$LOGFILE" || true
cp "$OUTDIR/exon_counts_barcodes.tsv" /tmp/par17_barcodes.tsv 2>/dev/null
rm -rf $OUTDIR /dev/shm/singlify_1fq_*

echo "" >> "$LOGFILE"
echo "=== RUN 2: STREAMING ===" >> "$LOGFILE"
OUTDIR=/dev/shm/str17_bench
rm -rf $OUTDIR && mkdir -p $OUTDIR
/usr/bin/time -f "str_wall=%e user=%U sys=%S maxrss_kb=%M" $SINGLIFY "$FQ" \
  --genome-dir "$GENOME" --exons "$GTF" \
  --out-prefix "$OUTDIR/" --threads 16 --no-parallel-pileup >> "$LOGFILE" 2>&1
echo "STR_EXIT=$?" >> "$LOGFILE"
echo "=== STR FILES ===" >> "$LOGFILE"
ls -lh "$OUTDIR/"*.1pz 2>/dev/null >> "$LOGFILE" || echo "no .1pz" >> "$LOGFILE"
wc -l "$OUTDIR/exon_counts_barcodes.tsv" 2>/dev/null >> "$LOGFILE" || true
cp "$OUTDIR/exon_counts_barcodes.tsv" /tmp/str17_barcodes.tsv 2>/dev/null
rm -rf $OUTDIR /dev/shm/singlify_1fq_*

echo "=== BENCHMARKS DONE ===" >> "$LOGFILE"
