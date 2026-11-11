#!/bin/bash
# Summarize the per-matrix ablate results into generated/tab_ablation.tsv (medians and quartiles over matrices) and print it:
#   sbatch --export=ALL,S1_SHA=<sha> ablate_report.sbatch
# Uses the private clone made by ablate.sbatch when S1_OVERLAY was set (src/<sha>-ablate), else the shared checkout.
#SBATCH --job-name=s1-ablate-report
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/logs/%x-%j.out
set -euo pipefail
SRC0=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/src/${S1_SHA:?set S1_SHA}
source "$SRC0/papers/manuscripts/s1-format/bench/slurm/clipper/common.sh"
[ -d "$S1/src/$SHA-ablate" ] && { SRC=$S1/src/$SHA-ablate; BENCH=$SRC/papers/manuscripts/s1-format/bench; }
s1_activate
cd "$BENCH"
python - <<PY
from pathlib import Path
from s1bench import ablate, util
cfg = util.load_config(None, ["results_dir=$S1/results", "generated_dir=$S1/generated"])
Path("$S1/generated").mkdir(exist_ok=True)
ablate.aggregate(cfg, "$S1/generated")
PY
column -t -s"$(printf '\t')" "$S1/generated/tab_ablation.tsv" | cut -c1-200
