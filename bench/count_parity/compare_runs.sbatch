#!/bin/bash
#SBATCH --job-name=cp-cmp
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Byte identity of the count matrices of two count-parity runs (CPU pileup vs GPU pileup), header comments aside.
#   sbatch --export=ALL,A=<run dir>,B=<run dir> compare_runs.sbatch      prints one line per matrix and IDENTICAL / DIFFERENT
set -uo pipefail
: "${A:?}" "${B:?}"
bad=0; n=0
for f in exon_counts.mtx.gz intron_counts.mtx.gz ambiguous_counts.mtx.gz spliced_counts.mtx.gz unspliced_counts.mtx.gz \
         exon_reads.mtx.gz intron_reads.mtx.gz raw_feature_bc_matrix/matrix.mtx.gz raw_feature_bc_matrix/barcodes.tsv.gz \
         raw_feature_bc_matrix/features.tsv.gz exon_counts_barcodes.tsv.gz exon_counts_features.tsv.gz; do
  [ -e "$A/$f" ] || { echo "missing in A: $f"; continue; }
  ha=$(zcat "$A/$f" | grep -v '^%' | sha256sum | cut -c1-16); hb=$(zcat "$B/$f" 2>/dev/null | grep -v '^%' | sha256sum | cut -c1-16)
  n=$((n+1)); s=same; [ "$ha" = "$hb" ] || { s=DIFF; bad=$((bad+1)); }
  printf "%-42s %s %s %s\n" "$f" "$ha" "$hb" "$s"
done
grep -h "gpu_pileup\]" "$B/run.log" | head -3
[ $bad = 0 ] && echo "IDENTICAL ($n matrices)" || echo "DIFFERENT ($bad of $n)"
