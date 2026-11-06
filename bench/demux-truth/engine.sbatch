#!/bin/bash
#SBATCH --job-name=dmx2-engine
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --array=0-10%6
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Our engine (October header) on one mixture: K given and auto, 3 seeds x 2 thread counts. On the m2 mixtures the K-given
# runs also run the diagnostic: ELBO of VB started from the truth labels against the ELBO at the optimum the engine finds.
# The mixture is rebuilt with the same mix seed; truth.tsv must be identical to the one the build job wrote.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/demux-truth2
CAP=${CAP:-2000}
row=$(grep -v '^#' $J/repo/bench/demux-truth/mixtures.tsv | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
IFS=$'\t' read -r id donors rate mseed <<< "$row"
K=$(echo $donors | tr ',' '\n' | wc -l)
OUT=$J/results/$id; cp $OUT/truth.tsv $OUT/truth.build.tsv
rm -f $OUT/runs.jsonl $OUT/diag.jsonl
RUNS=""
for s in 42 7 123; do
  for t in 4 16; do
    D=""; [[ $id == m2_* && $t == 16 ]] && D=":diag"
    RUNS="$RUNS singlet_k_t${t}_s${s}:$K:$t:$s$D singlet_auto_t${t}_s${s}:-1:$t:$s"
  done
done
echo "node=$(hostname) mix=$id donors=$donors K=$K $(date -Is)"
$J/bin/demux_mix --samples $J/samples.tsv --donors $donors --rate $rate --cap $CAP --mix-seed $mseed \
  --keep-lo 0.6 --keep-hi 0.8 --out $OUT --runs "$RUNS" 2>&1 | grep -E '^\[mix\]|^\[diag\]|^\[demux\] (Best|Covered)|rror' || true
cmp $OUT/truth.tsv $OUT/truth.build.tsv && echo "truth identical to the build job" || { echo "ERROR: truth differs from the build job"; exit 1; }
rm -f $OUT/truth.build.tsv
echo "done $(date -Is)"
