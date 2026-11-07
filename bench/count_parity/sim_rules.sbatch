#!/bin/bash
#SBATCH --job-name=cp-sim
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Replay multi-gene UMI rules on a run's molecule dump (sim_rules.py).  sbatch --export=ALL,DS=hgmm1k_v3,TAG=<run>,CODE=<dir> sim_rules.sbatch
set -euo pipefail
: "${DS:?}" "${TAG:?}" "${CODE:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; R=$W/count-parity/runs/$DS/$TAG
H5=$(ls $PB/data/$DS/*filtered_feature_bc_matrix.h5 | head -1)
$W/envs/pbench/bin/python -I $CODE/bench/count_parity/sim_rules.py --dump $R/molecules.tsv.gz --cr $H5 --out $R/sim_rules.json
