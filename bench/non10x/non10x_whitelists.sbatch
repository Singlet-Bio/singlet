#!/bin/bash
#SBATCH --account=debruinz
#SBATCH --job-name=ve-wl
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Whitelist directory for the validation-engine runs (D-152): every file of pipeline-benchmark/whitelists linked, except the files the
# commit ships itself (indrop_bc1.txt: the pipeline-benchmark copy is the 8-nt prefix of the published table, not the 8-11 nt list in
# python/singlet/data/whitelists). The commit's file is expected beside this script as indrop_bc1.txt.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/validation-engine/non10x/whitelists
mkdir -p $D
for f in $W/pipeline-benchmark/whitelists/*; do b=$(basename $f); [ "$b" = indrop_bc1.txt ] || ln -sfn $f $D/$b; done
cp $W/jobs/validation-engine/non10x/indrop_bc1.txt $D/indrop_bc1.txt
ls -la $D | head -40; wc -l $D/indrop_bc1.txt
