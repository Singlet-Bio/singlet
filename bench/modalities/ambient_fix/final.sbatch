#!/bin/bash
#SBATCH --job-name=amb-final
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Default configuration end to end: 16-thread run, 1-thread run (must be byte-identical), scoring, per-cluster diagnostics, timing.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/ambient; S=$W/jobs/modfix-ambient; A=$W/modality-bench/ambient
export PATH=$E/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl
CXX=$W/envs/pbench/bin/x86_64-conda-linux-gnu-g++
$CXX -std=c++20 -O2 -pthread -I$S/include $S/harness.cpp -o $S/harness
$CXX -std=c++20 -O2 -pthread -I$S/include $S/perf.cpp -o $S/perf
mkdir -p $A/fix_out/final $A/fix_out/final_t1
$S/harness $A/fix_in $A/fix_out/final 16 2>&1 | tee $A/fix_out/final/harness.log
$S/harness $A/fix_in $A/fix_out/final_t1 1 2>&1 | tee $A/fix_out/final_t1/harness.log
for f in rho.f64 cluster.u32 removed.u32 corr.indptr corr.indices corr.data; do
  cmp $A/fix_out/final/$f $A/fix_out/final_t1/$f && echo "identical: $f"
done
python -I $S/evaluate_fix.py final | tee $A/fix_out/evaluate_final.out
python -I $S/diag.py final
echo "== timing, synthetic 15k cells x 38k genes (+100k empty droplets)"
/usr/bin/time -v $S/perf 16 2>&1 | grep -E "perf\]|Maximum resident|Elapsed"
$S/perf 1 2>&1 | grep "perf\] threads"
