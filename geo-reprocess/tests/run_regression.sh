#!/bin/bash
#SBATCH --partition=all
#SBATCH --nodelist=g002
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=proto-regression
#SBATCH --output=/mnt/projects/debruinz_project/cellarium/pipeline/logs/regression_%j.log

source /opt/gvsu/clipper/2025.12/spack/apps/linux-cascadelake/miniconda3-25.5.1-xe7kyofwhfxilia75rj5t63zf6wpzzcr/etc/profile.d/conda.sh
conda activate cellarium
export ALEVIN_FRY_HOME=/mnt/projects/debruinz_project/cellarium/af_home

cd /mnt/projects/debruinz_project/cellarium/workspace/geo-reprocess
python tests/test_pipeline_matrix.py --layer 5 --species none --timeout 600 2>&1
