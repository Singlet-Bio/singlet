#!/bin/bash
#SBATCH -J intr-med
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; P=$W/pipeline-benchmark/runs; X=$W/expression-sota
$W/envs/pbench/bin/python3 -I $X/code/intronic_median.py cells:pbmc1k_v3=$P/pbmc1k_v3/singlet-public cells:pbmc1k_v2=$P/pbmc1k_v2/singlet-public cells:pbmc1k_cite_v3=$P/pbmc1k_cite_v3/singlet-public cells:pbmc5p_v1=$P/pbmc5p_v1/singlet-public cells:hgmm1k_v3=$P/hgmm1k_v3/singlet-public cells:kang_ctrl=$P/kang2018_ctrl/singlet-public cells:neuron1k_v3=$P/neuron1k_v3/singlet-public nuclei:nuclei2k_v2=$P/nuclei2k_v2/singlet-public nuclei:gaublomme=$X/nucleus_hash/singlet
