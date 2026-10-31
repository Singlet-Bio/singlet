#!/bin/bash
#SBATCH --job-name=hto-fid
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/adt-bench/bin:$PATH
O=$W/expression-sota/hto/results; mkdir -p $O; cd $W/expression-sota/hto/code/bench/expression_sota/hto
python -W ignore -P fidelity.py $W/data/adt_bench/zenodo_8304003 $W/bpub/artifacts/hto $O/fidelity.json
