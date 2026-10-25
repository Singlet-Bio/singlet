#!/bin/bash
#SBATCH --job-name=bpg-mgatk-env
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# mgatk 0.9.0 in envs/mtl imports `dedup` (PyPI package `bam-dedup`) but does not declare it. Install it into a separate
# directory that run_mgatk.sbatch puts on PYTHONPATH, so the shared mtl environment is not modified.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet
$W/envs/mtl/bin/pip install --no-deps --ignore-requires-python --target $W/envs/mgatk_extra "bam-dedup>=0.3.0" 2>&1 | tail -3
PYTHONPATH=$W/envs/mgatk_extra $W/envs/mtl/bin/python -c "from dedup import picardlike; print('dedup ok')"
