#!/bin/bash
#SBATCH --job-name=pub-extract
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Artifacts of the public benchmark capabilities, one array task per dataset (order of starsolo_public.sbatch):
#   sbatch --array=0-7 --export=ALL,CAPS="gex-counts spliced-unspliced junctions-psi cell-calling meta" extract_public.sbatch
# Reads the tool outputs under $PB/runs/<dataset>/ and writes $WORK/bench-public/artifacts/<capability>/<dataset>.* (+ resources.json,
# provenance.json). Code: SRC, a checkout of branch bench/public-counts (default: the folder this script was put with).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
SRC=${SRC:-$W/jobs/pub/src}
ART=${ART:-$W/bench-public/artifacts}; mkdir -p $ART
PY=$W/envs/pbench/bin/python
DSS=(pbmc1k_v3 pbmc1k_v2 neuron1k_v3 hgmm1k_v3 pbmc1k_cite_v3 nuclei2k_v2 pbmc5p_v1 kang2018_ctrl)
ID=${DSS[$SLURM_ARRAY_TASK_ID]}
CAPS=${CAPS:-"gex-counts spliced-unspliced junctions-psi cell-calling meta"}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for CAP in $CAPS; do
  if [ $CAP = meta ]; then
    for C2 in gex-counts spliced-unspliced junctions-psi cell-calling; do
      mkdir -p $ART/$C2
      flock $ART/.meta.lock $PY -P $SRC/bench/public/capabilities/extract/run_extract.py meta $ID --pb $PB --out $ART/$C2
    done
  else
    echo "[extract] $CAP $ID"
    mkdir -p $ART/$CAP
    $PY -P $SRC/bench/public/capabilities/extract/run_extract.py $CAP $ID --pb $PB --out $ART/$CAP
  fi
done
