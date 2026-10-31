#!/bin/bash
#SBATCH --job-name=hto-diag
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/adt-bench/bin:$PATH
cd $W/expression-sota/hto/code/bench/expression_sota/hto
python -P diag_bff.py $W/data/adt_bench/zenodo_8304003 $W/bpub/artifacts/hto
