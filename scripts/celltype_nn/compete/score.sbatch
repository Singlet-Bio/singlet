#!/bin/bash
#SBATCH --job-name=ctnn-score
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
O=$D/compete
# Singlet's methods are written under calls_<model>/: link them in under calls/ with public names
for m in nn nn_t0.3 nn_argmax guide; do [ -d $O/calls_m1_notissue/$m ] && ln -sfn $O/calls_m1_notissue/$m $O/calls/singlet_$m; done
in_runtime python -u $SRC/scripts/celltype_nn/compete/score.py --compete $O --data $D --out $O/results.json
