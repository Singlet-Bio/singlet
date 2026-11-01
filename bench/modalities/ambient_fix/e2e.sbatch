#!/bin/bash
#SBATCH --job-name=amb-e2e
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Harness fed from a pipeline run directory (cell set = cell_calls.tsv called_v2), scored on the 879 pure benchmark cells.
# Usage: sbatch e2e.sbatch "TAG:ex|ei[:harness args as kappa:z:K:iters:eps:ew:ol:off:s:mf ...]" ...
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/ambient; S=$W/jobs/modfix-ambient; A=$W/modality-bench/ambient
RUN=${RUN:-$W/pipeline-benchmark/runs/hgmm1k_v3/singlet-modfix-t8}
export PATH=$E/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl
for m in ex ei sp; do [ -f $A/e2e_in_$m/dims.txt ] || python -I $S/dump_e2e.py $RUN $A/e2e_in_$m $m; done
$W/envs/pbench/bin/x86_64-conda-linux-gnu-g++ -std=c++20 -O2 -pthread -I$S/include $S/harness.cpp -o $S/harness
for v in "$@"; do
  IFS=: read -r tag mode rest <<< "$v"
  mkdir -p $A/e2e_out/$tag
  args=${rest//:/ }
  echo "=== $tag mode=$mode args=[$args]"
  $S/harness $A/e2e_in_$mode $A/e2e_out/$tag 16 $args 2>&1 | grep -E "harness\] (n_outl|load)"
  python -I $S/score_e2e.py $A/e2e_in_$mode $A/e2e_out/$tag
done
