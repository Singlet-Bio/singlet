#!/bin/bash
#SBATCH --job-name=bs-collect
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# ~0.2 node-h. Atlas bounds for each configuration, then results.json.
W=/mnt/projects/debruinz_project/debruinz/singlet
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
for cfg in "4 1.47 " "16 1.47 " "4 1.47 _fa" "4 1.47 _memsort"; do set -- $cfg; echo "== T=$1 suffix=${3:-none}"; in_runtime python $W/bam-stream/atlas_bound.py $1 $2 ${3:-}; done
in_runtime python $W/bam-stream/collect_results.py
