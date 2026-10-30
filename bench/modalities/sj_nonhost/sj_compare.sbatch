#!/bin/bash
#SBATCH --job-name=sjnh-cmp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch sj_compare.sbatch [STARSOLO_TAG]    (dir starsolo[_TAG] under pbmc1k_v3/; output sj_compare[_TAG].json)
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; B=$W/modality-bench/sj_nonhost; T=${1:+_$1}
$W/envs/pbench/bin/python -I -W ignore $B/scripts/sj_compare.py $W/pipeline-benchmark/perf/pbmc1k_v3_det/t4_r1 $B/pbmc1k_v3/starsolo$T /mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/genes/genes.gtf $B/pbmc1k_v3/sj_compare$T.json
