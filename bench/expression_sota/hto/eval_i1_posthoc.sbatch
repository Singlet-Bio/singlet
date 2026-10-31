#!/bin/bash
#SBATCH --job-name=hto-i1posthoc
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# POST-HOC (not pre-registered), after the pre-registered I1 run degenerated: callers see only cells with >= 1 tag count.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/adt-bench/bin:$PATH
D=$W/expression-sota/hto/data/stoeckius2018
O=$W/expression-sota/hto/results; cd $W/expression-sota/hto/code/bench/expression_sota/hto
export OMP_NUM_THREADS=4
python -W ignore -P eval_independent.py --data $D --i1-truth $D/vireo_full/vireo/donor_ids.tsv --i1-cells $D/singlet_full/cell_calls.tsv \
  --frozen frozen.json --only I1 --drop-zero --out $O/independent_I1_posthoc_dropzero.json
