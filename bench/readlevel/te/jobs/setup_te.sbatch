#!/bin/bash
#SBATCH --job-name=te-setup
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# TE comparators: scTE (JiekaiLab/scTE, GitHub) and SoloTE (bvaldebenitom/SoloTE, GitHub) in one env with STAR 2.7.11b, samtools,
# bedtools and pysam; scTE's GRCh38 index built from the same TE GTF and GENCODE annotation the engine uses (10x GRCh38-2024-A).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/tebench; MM=$W/tools/mm/bin/micromamba
export MAMBA_ROOT_PREFIX=$W/tools/mm/root CONDA_PKGS_DIRS=$W/cache/conda_pkgs HOME=$W/cache/condahome
[ -f $ENV/.ok ] || { $MM create -y -p $ENV -c conda-forge -c bioconda --strict-channel-priority python=3.10 star=2.7.11b samtools bedtools pysam pandas numpy scipy anndata git pip minimap2; touch $ENV/.ok; }
[ -x $ENV/bin/minimap2 ] || $MM install -y -p $ENV -c conda-forge -c bioconda --strict-channel-priority minimap2
T=$W/tools/te; mkdir -p $T; cd $T
[ -d scTE ] || git clone -q https://github.com/JiekaiLab/scTE.git
[ -d SoloTE ] || git clone -q https://github.com/bvaldebenitom/SoloTE.git
( cd scTE && git rev-parse HEAD ) > scTE.commit; ( cd SoloTE && git rev-parse HEAD ) > SoloTE.commit
$ENV/bin/pip install -q ./scTE
$ENV/bin/scTE --help | head -5 || true
ls SoloTE
