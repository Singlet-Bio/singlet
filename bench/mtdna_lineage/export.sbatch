#!/bin/bash
#SBATCH --job-name=mtl-export
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch export.sbatch <name>    Singlet's private counts + public heteroplasmy -> out/mtdna_lineage/<name>/singlet_mt.npz
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet; J=$WORK/jobs/mtl; O=$WORK/out/mtdna_lineage/$1
source $WORK/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh   # main has no synced worktree here
PYSRC=$WORK/src/singlet-py/claude__called-v2-migration/python   # a worktree with the built singlet._pz extension
export PYTHONPATH=$PYSRC
in_runtime python $J/singlet_export.py $O $O/singlet_mt.npz
ls -la $O/singlet_mt.npz
