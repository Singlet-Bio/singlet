#!/bin/bash
#SBATCH -J nh-comp
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota; PY="$W/envs/pbench/bin/python3 -I"; mkdir -p $X/nuclei
$PY $X/code/nuclei_composition.py $X/nucleus_hash/singlet $X/nuclei/nucleus_hash.json $X/nucleus_hash/hashtags.counts.tsv
for ds in nuclei2k_v2 pbmc1k_v3 hgmm1k_v3; do echo == $ds; $PY $X/code/nuclei_composition.py $W/release-gate/runs/ec/92095e29-cpu/$ds $X/nuclei/$ds.json | head -14; done
