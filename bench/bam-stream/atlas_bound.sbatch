#!/bin/bash
#SBATCH --job-name=bs-bound
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# ~0.05 node-h. Usage: sbatch --export=ALL,T=4 atlas_bound.sbatch
W=/mnt/projects/debruinz_project/debruinz/singlet
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
in_runtime python $W/bam-stream/atlas_bound.py ${T:-4}
