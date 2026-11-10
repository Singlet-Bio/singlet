#!/bin/bash
#SBATCH -J soup-feat
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota; P=$W/pipeline-benchmark/runs
$W/envs/pbench/bin/python3 -I $X/code/soup_features.py $X/nuclei/soup_features.json hgmm1k_v3=$P/hgmm1k_v3/singlet-public kang_ctrl=$P/kang2018_ctrl/singlet-public pbmc1k_v3=$P/pbmc1k_v3/singlet-public neuron1k_v3=$P/neuron1k_v3/singlet-public nuclei2k_v2=$P/nuclei2k_v2/singlet-public gaublomme=$X/nucleus_hash/singlet 2>&1 | cut -c1-900
