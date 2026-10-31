#!/bin/bash
#SBATCH --job-name=nht-compare
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Reference sgRNA junction table from the reference STARsolo viral alignments, then the full comparison. Run after nht-refsolo and nht-proto.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/nonhost
export PATH=$W/envs/pbench/bin:$PATH
python -I $X/code/sgrna_junctions.py $X/ref_starsolo/dpi2/viral_aligned.sam $X/ref/viral/viral.trs.json $X/ref_starsolo/dpi2/sgrna
for d in kept all_unmapped; do python -I $X/code/sgrna_junctions.py $X/proto/dpi2/$d/aligned.sam $X/ref/viral/viral.trs.json $X/proto/dpi2/$d/sgrna; done
python -I -W ignore $X/code/compare.py $X/compare_dpi2.json
