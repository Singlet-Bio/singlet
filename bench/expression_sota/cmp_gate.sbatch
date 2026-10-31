#!/bin/bash
#SBATCH -J ec-cmp
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet
for d in pbmc1k_v3 hgmm1k_v3 nuclei2k_v2; do echo "== $d"; python3 $W/expression-sota/code/cmp_gate.py $W/release-gate/runs/ec/${SHA8:-92095e29}-cpu/$d $W/pipeline-benchmark/runs/$d/singlet-public; done
