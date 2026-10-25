#!/bin/bash
#SBATCH --job-name=mtl-env
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Reference tools for the mtDNA lineage benchmark: mgatk (bioconda, with the R it needs), MQuad + vireoSNP (PyPI),
# pysam, and the truth tables from the MAESTER-2021 analysis repository (GitHub).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; MM=$W/tools/mm/bin/micromamba; export MAMBA_ROOT_PREFIX=$W/tools/mm/root
E=$W/envs/mtl
if [ ! -x $E/bin/mgatk ]; then
  $MM create -y -p $E -c conda-forge -c bioconda python=3.10 mgatk r-base r-data.table r-matrix \
    bioconductor-summarizedexperiment bioconductor-genomicranges samtools pysam numpy scipy pandas scikit-learn \
    zstandard pyreadr
fi
$E/bin/pip install -q mquad vireoSNP maegatk || $E/bin/pip install -q mquad vireoSNP
$E/bin/mgatk --version || true
$E/bin/python -c "import mquad, vireoSNP, pysam, pyreadr, zstandard; print('py ok', vireoSNP.__version__)"
$E/bin/Rscript -e 'library(SummarizedExperiment); cat("R ok\n")'
# truth tables (author analysis outputs)
T=$W/data/maester/truth; mkdir -p $T; cd $T
G=https://raw.githubusercontent.com/vangalenlab/MAESTER-2021/main
for f in 3_Cell_line_mixes_variants/TenX_CellLineMix_Cells.rds 3_Cell_line_mixes_variants/TenX_CellLineMix_Variants1.txt \
         4_CH_sample/4.2_vois.txt 4_CH_sample/4.3_vois_order.txt 4_CH_sample/4.4_positive_cells.txt; do
  curl -sL --retry 3 -o $(basename $f) $G/$f
done
# gnomAD v3.1 chrM sites (population allele frequencies; used to define the private/public split)
curl -sL --retry 3 -o gnomad.genomes.v3.1.sites.chrM.vcf.bgz \
  https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1/vcf/genomes/gnomad.genomes.v3.1.sites.chrM.vcf.bgz
ls -la $T
