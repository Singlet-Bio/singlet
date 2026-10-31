#!/bin/bash
#SBATCH --job-name=hto-lodo
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# PREREG section 6: BFF-style caller, leave-one-dataset-out on Howitt 2023.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/adt-bench/bin:$PATH
O=$W/expression-sota/hto/results; mkdir -p $O; cd $W/expression-sota/hto/code/bench/expression_sota/hto
export OMP_NUM_THREADS=4
python -P lodo_bff.py --zenodo $W/data/adt_bench/zenodo_8304003 --artifacts $W/bpub/artifacts/hto --out $O/lodo_bff.json --boot 200
