#!/bin/bash
#SBATCH --job-name=download_refs
#SBATCH --partition=cpu
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/projects/debruinz_project/cellarium/pipeline/logs/download_refs_%j.out
set -euo pipefail

module load miniconda3/25.5.1
source /opt/gvsu/clipper/2025.12/spack/apps/linux-cascadelake/miniconda3-25.5.1-xe7kyofwhfxilia75rj5t63zf6wpzzcr/etc/profile.d/conda.sh
set +u; conda activate cellarium; set -u

INDICES_DIR="/mnt/projects/debruinz_project/cellarium/reference/pileup_indices"
mkdir -p "$INDICES_DIR"
cd "$INDICES_DIR"

echo "=== Step 1: Download 1000 Genomes common SNPs ==="
# Use the genome-in-a-bottle / 1000G Phase3 common (MAF>5%) SNP VCF filtered to biallelic SNPs
# This is the standard set used by cellSNP-lite and vireoSNP
SNP_VCF="$INDICES_DIR/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz"
if [[ ! -f "$SNP_VCF" ]]; then
    echo "Downloading 1000G common SNP VCF (hg38, MAF>5%)..."
    wget -q "https://sourceforge.net/projects/cellsnp/files/SNPlist/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz/download" \
        -O "$SNP_VCF"
fi

# Sort and re-index the VCF (the sourceforge file is not position-sorted)
SNP_VCF_SORTED="$INDICES_DIR/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.vcf.gz"
if [[ ! -f "$SNP_VCF_SORTED.tbi" ]]; then
    echo "Sorting and indexing VCF..."
    bcftools sort "$SNP_VCF" -Oz -o "$SNP_VCF_SORTED"
    tabix -p vcf "$SNP_VCF_SORTED"
    SNP_COUNT=$(zcat "$SNP_VCF_SORTED" | grep -v '^#' | wc -l)
    echo "SNP VCF: $SNP_COUNT sites"
else
    echo "Sorted SNP VCF already exists: $SNP_VCF_SORTED"
fi
SNP_VCF="$SNP_VCF_SORTED"

echo "=== Step 2: Extract simple TSV for singlet-pileup ==="
# Convert to simple TSV format: CHROM POS REF ALT
SNP_TSV="$INDICES_DIR/common_snps_hg38.tsv.gz"
if [[ ! -f "$SNP_TSV" ]]; then
    echo "Converting VCF to TSV..."
    bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\n' "$SNP_VCF" | gzip > "$SNP_TSV"
    echo "TSV: $(zcat "$SNP_TSV" | wc -l) sites"
else
    echo "SNP TSV already exists: $SNP_TSV"
fi

echo "=== Step 3: Prepare exon interval BED from GTF ==="
GTF="/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/genes/genes.gtf.gz"
EXON_BED="$INDICES_DIR/exons_hg38.bed.gz"
if [[ ! -f "$EXON_BED" ]]; then
    echo "Extracting exon intervals from GTF..."
    zcat "$GTF" | awk -F'\t' '
    $3 == "exon" {
        # Extract gene_name
        match($9, /gene_name "([^"]+)"/, m);
        gene = m[1];
        # Extract exon_number
        match($9, /exon_number ([0-9]+)/, e);
        enum = e[1];
        # GTF is 1-based inclusive → BED is 0-based half-open
        print $1 "\t" ($4-1) "\t" $5 "\t" gene "_exon" enum "\t0\t" $7
    }' | sort -k1,1 -k2,2n | gzip > "$EXON_BED"
    echo "Exon BED: $(zcat "$EXON_BED" | wc -l) intervals"
else
    echo "Exon BED already exists: $EXON_BED"
fi

echo "=== Step 4: Check/rebuild STAR index ==="
STAR_DIR="/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star"
GENOME_FA="/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/fasta/genome.fa"
STAR_VERSION=$(STAR --version 2>&1)
echo "STAR version: $STAR_VERSION"

# Check existing STAR index version
if [[ -f "$STAR_DIR/genomeParameters.txt" ]]; then
    EXISTING_VERSION=$(grep -o 'STAR_[0-9.]*[a-z]*' "$STAR_DIR/genomeParameters.txt" | head -1 || echo "unknown")
    echo "Existing STAR index version: $EXISTING_VERSION"
fi

# Rebuild if needed (STAR 2.7.11b needs index rebuild from 2.7.1a)
STAR_NEW_DIR="/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b"
if [[ ! -d "$STAR_NEW_DIR" || ! -f "$STAR_NEW_DIR/SA" ]]; then
    echo "Building new STAR index for 2.7.11b..."
    mkdir -p "$STAR_NEW_DIR"

    # Decompress GTF to a temp file (STAR can't read from process substitution)
    GTF_PLAIN="/tmp/genes_$$.gtf"
    zcat "$GTF" > "$GTF_PLAIN"

    STAR --runMode genomeGenerate \
        --genomeDir "$STAR_NEW_DIR" \
        --genomeFastaFiles "$GENOME_FA" \
        --sjdbGTFfile "$GTF_PLAIN" \
        --sjdbOverhang 90 \
        --runThreadN 8 \
        --outTmpDir "/tmp/STARtmp_$$"

    rm -f "$GTF_PLAIN"
    echo "STAR index built at: $STAR_NEW_DIR"
else
    echo "STAR 2.7.11b index already exists: $STAR_NEW_DIR"
fi

echo "=== Step 5: FAI index for genome fasta ==="
if [[ ! -f "$GENOME_FA.fai" ]]; then
    echo "Indexing genome FASTA..."
    samtools faidx "$GENOME_FA"
else
    echo "FASTA index exists"
fi

echo ""
echo "=== Reference preparation complete ==="
ls -lh "$INDICES_DIR"/
