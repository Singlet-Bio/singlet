#!/bin/bash
# Submit the public benchmark: fetch every dataset, then every (dataset, tool) run, then collect the results.
# Login-node safe: only reads the cohort table and calls sbatch.
#   bash bench/public/harness/submit.sh [--tier core|all] [--tools "singlet starsolo"] [dataset ...]
# Concurrency: fetches %4, runs %12 (the cohort budget assumes 12 concurrent 16-thread tasks; check `clipper.py free`).
# Tasks whose output exists (.done) exit at once, so the same command resumes a partial campaign.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__bench-public-cohort-485cf7/bench/public/harness/common.sh
H=$SRC/bench/public/harness; PART=${PART:-scavenger}
TIER=core; ONLY_TOOLS=""; DS=()
while [ $# -gt 0 ]; do
  case $1 in --tier) TIER=$2; shift 2 ;; --tools) ONLY_TOOLS=$2; shift 2 ;; *) DS+=("$1"); shift ;; esac
done
# datasets handled by their own pipelines (per-sample studies, bulk ancestry) are not in the generic arrays
SPECIAL="cxg16 geo_sex21 mage150"
if [ ${#DS[@]} -eq 0 ]; then
  mapfile -t DS < <(grep -v '^#' $COHORT | awk -F'\t' -v t=$TIER 'NR>1 && (t=="all" || $3==t) {print $1}')
fi
: > $BP/work/datasets.txt; : > $BP/work/tasks.tsv
for d in "${DS[@]}"; do
  [[ " $SPECIAL " == *" $d "* ]] && { echo "skip $d (own pipeline: see cohort.md)"; continue; }
  echo $d >> $BP/work/datasets.txt
  cohort_row $d
  for t in ${C[tools]//,/ }; do
    [ -n "$ONLY_TOOLS" ] && [[ " $ONLY_TOOLS " != *" $t "* ]] && continue
    printf "%s\t%s\n" $d $t >> $BP/work/tasks.tsv
  done
done
ND=$(wc -l < $BP/work/datasets.txt); NT=$(wc -l < $BP/work/tasks.tsv)
echo "$ND datasets, $NT runs"
F=$(sbatch --parsable --partition=$PART --requeue --array=1-$ND%4 $H/fetch.sbatch)
R=$(sbatch --parsable --partition=$PART --requeue --dependency=afterany:$F --array=1-$NT%12 $H/run.sbatch)
K=$(sbatch --parsable --partition=$PART --requeue --dependency=afterany:$R $H/collect.sbatch)
echo "fetch $F  runs $R  collect $K"
echo "$(date -Is) fetch=$F runs=$R collect=$K tier=$TIER n_runs=$NT" >> $BP/work/submissions.log
