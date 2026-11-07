#!/bin/bash
# Submit every (dataset, tool) run. Login-node safe: only sbatch.
#   bash submit_runs.sh [dataset ...]       default: all datasets in datasets.tsv
HERE=$(cd $(dirname $0) && pwd)
DS=("$@"); [ ${#DS[@]} -gt 0 ] || DS=($(tail -n +2 $HERE/datasets.tsv | cut -f1))
for d in "${DS[@]}"; do
  for t in ${TOOLS:-singlet-baseline singlet-fixed singlet-baseline-t4 singlet-fixed-t4 starsolo alevinfry}; do
    j=$(sbatch -A debruinz --parsable --job-name=pb-$t-$d $HERE/run_tool.sbatch $d $t)
    echo "$d $t $j"
  done
done
