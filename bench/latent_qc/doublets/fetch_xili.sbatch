#!/bin/bash
# Xi & Li 2021 doublet benchmark, 16 real datasets with experimental doublet labels (Zenodo 4062232, real_datasets.zip),
# converted from .rds to Matrix Market + labels under $LQC/dbl/data/<name>/.
#SBATCH --job-name=lqc-xili
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/study/latent-qc:refs/heads/study/latent-qc
git --git-dir=$W/repos/singlet-py.git show refs/heads/study/latent-qc:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh
source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
D=$LQC/dbl/raw; mkdir -p $D $LQC/dbl/data
cd $D
[ -f real_datasets.zip ] || curl -sSL --retry 5 -o real_datasets.zip "https://zenodo.org/api/records/4062232/files/real_datasets.zip/content"
ls -la real_datasets.zip; [ -d real_datasets ] || unzip -q -o real_datasets.zip
find . -name "*.rds" | sort
R $SRC/bench/latent_qc/doublets/xili_convert.R $D $LQC/dbl/data
ls $LQC/dbl/data
