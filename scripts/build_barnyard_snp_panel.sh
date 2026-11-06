#!/bin/bash
# SPDX-License-Identifier: MIT
# Build the default SNP panel of the human + mouse barnyard reference (GRCh38-GRCm39-barnyard), the one singlet --pipeline
# loads for that reference: <ref_base>/pileup_indices/barnyard_GRCh38_GRCm39.vcf.gz. It is the human panel (1000 Genomes
# phase 3, AF >= 5%, chr-prefixed) and the mouse panel (Ensembl GRCm39 common SNPs) with their contigs renamed to the
# barnyard names (GRCh38___chr1 ..., GRCm39___1 ...). Run it as a batch job, not on a login node.
#   bash scripts/build_barnyard_snp_panel.sh [REF_BASE]
set -euo pipefail
REF=${1:-${SINGLET_REF_BASE:?pass REF_BASE or set SINGLET_REF_BASE}}
P=$REF/pileup_indices
HUMAN=$P/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz
MOUSE=$P/mus_musculus.vcf.gz
OUT=$P/barnyard_GRCh38_GRCm39.vcf.gz
CHR=$REF/GRCh38-GRCm39-barnyard/star_2.7.11b/chrName.txt
for f in "$HUMAN" "$MOUSE" "$CHR"; do [ -s "$f" ] || { echo "missing $f"; exit 2; }; done
TMP=$(mktemp "$OUT.XXXX")
{
  printf '##fileformat=VCFv4.2\n##source=%s + %s, contigs renamed for GRCh38-GRCm39-barnyard\n' "$(basename "$HUMAN")" "$(basename "$MOUSE")"
  printf '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
  zcat "$HUMAN" | awk 'BEGIN{FS=OFS="\t"} /^#/{next} {$1="GRCh38___"$1; print}'
  zcat "$MOUSE" | awk 'BEGIN{FS=OFS="\t"} /^#/{next} {$1="GRCm39___"$1; print}'
} | gzip -c > "$TMP"
mv "$TMP" "$OUT"
# every panel contig must be a reference contig, else its SNPs are silently dropped
zcat "$OUT" | awk -F'\t' '!/^#/{print $1}' | uniq | sort -u > "$OUT.contigs"
MISSING=$(comm -23 "$OUT.contigs" <(sort -u "$CHR") | wc -l)
echo "panel $OUT: $(zcat "$OUT" | grep -vc '^#') SNPs, $(wc -l < "$OUT.contigs") contigs, $MISSING not in the barnyard reference"
comm -23 "$OUT.contigs" <(sort -u "$CHR") | head
rm -f "$OUT.contigs"
sha256sum "$OUT"
