#!/bin/bash
#SBATCH --job-name=fus-setup
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Comparators for the chimeric-junction benchmark. Arriba 2.5.1 (bioconda env $W/envs/fusion, with STAR 2.7.11b) runs on its own
# STAR alignment against the engine's GRCh38-2024-A index. STAR-Fusion 1.15.1 runs from its Docker image (trinityctat/starfusion,
# pulled to a .sif) with the CTAT genome library GRCh38 GENCODE v44 plug-n-play (36 GB).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/fusion; MM=$W/tools/mm/bin/micromamba
export MAMBA_ROOT_PREFIX=$W/tools/mm/root CONDA_PKGS_DIRS=$W/cache/conda_pkgs HOME=$W/cache/condahome
[ -f $ENV/.ok ] || { $MM create -y -p $ENV -c conda-forge -c bioconda --strict-channel-priority arriba star samtools python=3.11 pandas; touch $ENV/.ok; }
SIF=$W/images/starfusion-1.15.1.sif
[ -s $SIF ] || { export APPTAINER_CACHEDIR=$W/cache/apptainer APPTAINER_TMPDIR=/tmp; apptainer pull $SIF docker://trinityctat/starfusion:1.15.1; }
apptainer exec $SIF STAR-Fusion --version || true
L=$W/data/readlevel/ctat; mkdir -p $L; cd $L
N=GRCh38_gencode_v44_CTAT_lib_Oct292023.plug-n-play
if [ ! -d $N/ctat_genome_lib_build_dir ]; then
  curl -sSfL -C - -o $N.tar.gz https://data.broadinstitute.org/Trinity/CTAT_RESOURCE_LIB/$N.tar.gz
  tar -xzf $N.tar.gz && rm -f $N.tar.gz; fi
ls $N/ctat_genome_lib_build_dir | head -40
