#!/bin/bash
#SBATCH --job-name=hto-i1
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# I1 (Stoeckius 2018 PBMC, 8 donors): every caller once, truth = vireo genotype donors on Singlet's called cells.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/adt-bench/bin:$PATH
D=$W/expression-sota/hto/data/stoeckius2018
O=$W/expression-sota/hto/results; mkdir -p $O; cd $W/expression-sota/hto/code/bench/expression_sota/hto
export OMP_NUM_THREADS=4
python -W ignore -P eval_independent.py --data $D --i1-truth $D/vireo/vireo/donor_ids.tsv --i1-cells $D/singlet/cell_calls.tsv \
  --frozen frozen.json --only I1 --out $O/independent_I1.json
