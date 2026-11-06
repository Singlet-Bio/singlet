#!/bin/bash
#SBATCH --job-name=dmx-mix
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# One array task = one mixture (row of mixtures.tsv). Env: CAP (cells per donor), RESULTS (output root).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/demux-truth
CAP=${CAP:-2000}; RESULTS=${RESULTS:-$J/results/overnight}
PY=$W/envs/gpool/venv/bin/python
row=$(grep -v '^#' $J/repo/bench/demux-truth/mixtures.tsv | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
IFS=$'\t' read -r id donors rate mseed <<< "$row"
K=$(echo $donors | tr ',' '\n' | wc -l)
OUT=$RESULTS/$id; rm -rf $OUT; mkdir -p $OUT
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK mix=$id donors=$donors rate=$rate cap=$CAP $(date -Is)"
RUNS="k_t4_r1:$K:4:42 k_t4_r2:$K:4:42 k_t8_r1:$K:8:42 k_t16_r1:$K:16:42 k_t16_r2:$K:16:42 k_t16_s7:$K:16:7 auto_t16_r1:-1:16:42 auto_t16_r2:-1:16:42"
$J/bin/demux_mix --samples $J/samples.tsv --donors $donors --rate $rate --cap $CAP --mix-seed $mseed --out $OUT --runs "$RUNS" 2>&1 \
  | grep -E '^\[mix\]|^\[demux\] (Best|Covered)' || true
$PY $J/repo/bench/demux-truth/score.py $OUT
echo "done $(date -Is)"
