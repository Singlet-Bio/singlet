#!/bin/bash
#SBATCH --job-name=cc-dev
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
T=$W/state/cellcall_truth/truth
O=$W/bench/calls-sota/cell_calling; mkdir -p $O
PY=$W/envs/gpool/venv/bin/python
SRC=${SRC:-$W/bench/calls-sota/cell_calling}
$PY -P $SRC/score_tables.py $O/dev_scores.json \
  kang_b1A=$T/kang_b1A.truth.tsv.gz:G kang_b1B=$T/kang_b1B.truth.tsv.gz:G kang_b1C=$T/kang_b1C.truth.tsv.gz:G kang_b2ctrl=$T/kang_b2ctrl.truth.tsv.gz:G \
  hgmm1k_v2=$T/hgmm1k_v2.truth.tsv.gz:S hgmm1k_v3=$T/hgmm1k_v3.truth.tsv.gz:S hgmm5k_v3=$T/hgmm5k_v3.truth.tsv.gz:S
