#!/bin/bash
# Singlet 1.0.0 public benchmark: gex-counts, spliced-unspliced, junctions-psi, cell-calling on Clipper (login node: only sbatch).
#   bash run_all.sh [runs|extract|all]      default: all
# 1. runs     STARsolo (cr + default), Singlet, CellBender on the eight datasets (competitor outputs already on disk are not re-run:
#             alevin-fry comes from bench/pipeline run_tool.sbatch `alevinfry` / `alevinfry-fw`, run on the same FASTQs and reference).
# 2. extract  per-cell artifacts under $WORK/bench-public/artifacts/<capability>/ (what the notebook downloads).
# Then, anywhere with numpy and pandas:
#   python bench/public/capabilities/<module>.py $WORK/bench-public/artifacts/<capability> --out bench/public/results/<capability>.json
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STEP=${1:-all}
if [ "$STEP" = runs ] || [ "$STEP" = all ]; then
  S1=$(sbatch --parsable --array=0-7 --export=ALL,MODE=cr "$HERE/starsolo_public.sbatch")
  sbatch --parsable --array=0-7 --export=ALL,MODE=default "$HERE/starsolo_public.sbatch" > /dev/null
  sbatch --parsable --array=0-7 "$HERE/singlet_public.sbatch" > /dev/null
  sbatch --parsable --array=0-7 --dependency=afterok:$S1 "$HERE/cellbender_public.sbatch" > /dev/null
fi
if [ "$STEP" = extract ] || [ "$STEP" = all ]; then
  sbatch --parsable --array=0-7 "$HERE/extract_public.sbatch"
fi
