#!/usr/bin/env bash
# N11 ambient smoke test on SRR20020820 (known-good --pipeline sample)
source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
ulimit -n 10240

SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
FQ=/mnt/projects/debruinz_project/singlify_validation/1fq/SRR20020820.1fq
GENOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
GTF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/genes/genes.gtf
OUTDIR=/dev/shm/n11_s20_$$
mkdir -p "$OUTDIR"
cat "$GENOME/SA" "$GENOME/Genome" "$GENOME/SAindex" > /dev/null 2>&1

echo "=== N11 smoke: SRR20020820 (--pipeline, auto whitelist + cell calling) ==="
/usr/bin/time -f "wall=%e" "$SINGLIFY" "$FQ" \
  --genome-dir "$GENOME" --exons "$GTF" \
  --out-prefix "$OUTDIR/" --threads 16 --pipeline 2>&1 | \
  grep -E "ambient|cell_calling|wall=|N11|error|ERROR"

echo "=== output files ==="
ls "$OUTDIR/"ambient* 2>/dev/null || echo "no ambient files"
ls "$OUTDIR/cell_calls.tsv" 2>/dev/null && echo "cell_calls.tsv: YES" || echo "cell_calls.tsv: NO"
echo "=== ambient_profile head ==="
head -3 "$OUTDIR/ambient_profile.tsv" 2>/dev/null || echo "missing"
echo "=== contamination head ==="
head -3 "$OUTDIR/ambient_contamination.tsv" 2>/dev/null || echo "missing"
echo "=== contamination stats ==="
awk -F"\t" 'NR>1{sum+=$2;n++;if($2>0.5)high++} END{print "n_cells="n, "mean_rho="sum/n, "high_rho(>0.5)="high+0}' \
  "$OUTDIR/ambient_contamination.tsv" 2>/dev/null
rm -rf "$OUTDIR"
echo DONE
