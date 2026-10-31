#!/bin/bash
#SBATCH -J nh-xist
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota
$W/envs/pbench/bin/python3 -I $X/code/nh_xist.py $X/nucleus_hash/singlet $X/nucleus_hash/hashtags.counts.tsv $X/nuclei/xist_ambient.json 2>&1 | grep -v -i -E "warn|spmatrix|sparse|^ +(That|Unless|on 2D|See|https)|^$"
