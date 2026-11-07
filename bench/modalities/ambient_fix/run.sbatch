#!/bin/bash
#SBATCH --job-name=amb-fix
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compile the header-only estimator against the barnyard inputs, run variants, score against truth.
# Usage: sbatch run.sbatch "TAG:kappa:z:K[:cell_iters[:phi_pseudocount[:equal_weight[:outlier_round]]]] ..."   (default one variant: base:20:3:0)
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/ambient; S=$W/jobs/modfix-ambient; A=$W/modality-bench/ambient
export PATH=$E/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl
VARIANTS=${1:-base:20:3:0}
mkdir -p $A/fix_out
[ -f $A/fix_in/dims.txt ] || python -I $S/dump.py
$W/envs/pbench/bin/x86_64-conda-linux-gnu-g++ -std=c++20 -O2 -pthread -I$S/include $S/harness.cpp -o $S/harness
TAGS=""
for v in $VARIANTS; do
  IFS=: read -r tag kappa z K iters eps ew ol off s mf <<< "$v"
  mkdir -p $A/fix_out/$tag
  /usr/bin/time -v $S/harness $A/fix_in $A/fix_out/$tag 16 $kappa $z $K ${iters:-30} ${eps:-1e-4} ${ew:-0} ${ol:-1} ${off:-1} ${s:-5} ${mf:-1.25} 2> $A/fix_out/$tag/harness.log || { cat $A/fix_out/$tag/harness.log; exit 1; }
  grep -E "harness|Maximum resident|Elapsed" $A/fix_out/$tag/harness.log
  TAGS="$TAGS $tag"
done
python -I $S/evaluate_fix.py $TAGS | tee $A/fix_out/evaluate_fix.out
