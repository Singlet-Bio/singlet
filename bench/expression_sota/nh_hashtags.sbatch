#!/bin/bash
#SBATCH -J nh-tags
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/expression-sota/data/nucleus_hash; O=$W/expression-sota/nucleus_hash; mkdir -p $O
# barcode set: the 10x v2 whitelist
WL=$W/pipeline-benchmark/whitelists/737K-august-2016.txt
$W/envs/pbench/bin/python3 -I $W/expression-sota/code/nh_hashtags.py $D/SRR8703774_2.fastq.gz $D/SRR8703774_3.fastq.gz $WL $O/hashtags
