#!/bin/bash
# Fixed called_ed (cellcall 2.1) against the study's recommended counts, and the golden vectors (tests/data/called_ed_golden).
# Code comes from branch $BRANCH of singlet-py (git archive of the fetched branch; nothing uploaded from a laptop).
#
#   sbatch --export=ALL,BRANCH=fix/called-v2-knee-bounds verify.sbatch   (array 0-6)
#   tasks 0-5: verify_cellcall.py per dataset (+ a golden vector for hgmm1k_v3, neuron1k_v3, pbmc1k_v3); task 6: synthetic vectors
#SBATCH --job-name=cv2fix
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --array=0-6
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
sdir() { [ -d "$1/state/$2" ] && echo "$1/state/$2" || echo "$1/state/${2/called_ed/called_v2}"; }   # state/called_ed* was renamed called_v2* on Clipper
BRANCH=${BRANCH:-fix/called-v2-knee-bounds}
OUT=$(sdir "$W" called_ed_fix)
PB=$W/pipeline-benchmark
MB=$W/modality-bench
DS=$(sdir "$W" called_ed_truth)/datasets
PY=$W/envs/pbench/bin/python
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} NUMBA_CACHE_DIR=/tmp/${USER:-x}-numba-${SLURM_JOB_ID:-x} OMP_NUM_THREADS=4
TASK=${SLURM_ARRAY_TASK_ID:-0}
SCRATCH=/tmp/${USER:-x}-${SLURM_JOB_ID:-x}-$TASK; mkdir -p "$SCRATCH" "$OUT/verify" "$OUT/golden"
trap 'rm -rf "$SCRATCH" "$NUMBA_CACHE_DIR"' EXIT
REPO=$W/repos/singlet-py.git
( flock 9; git --git-dir=$REPO fetch -q origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" ) 9>"$OUT/.fetch.lock"  # array tasks share the repo
SHA=$(git --git-dir=$REPO rev-parse "origin/$BRANCH")
mkdir -p $SCRATCH/src && git --git-dir=$REPO archive "$SHA" python bench/called_ed_truth tests/data/called_ed_golden/generate.py | tar -x -C $SCRATCH/src
S=$SCRATCH/src
echo "task=$TASK branch=$BRANCH sha=$SHA host=$(hostname)"

run() {  # NAME RUN_DIR [golden RATE GENES]
  $PY -P $S/bench/called_ed_truth/verify_cellcall.py "$1" "$2" $DS/$1.json $OUT/verify/$1.json
  if [ "${3:-}" = golden ]; then
    $PY -P $S/tests/data/called_ed_golden/generate.py real "real_$1" "$2" $OUT/golden --rate "$4" --genes "$5"
  fi
}
case $TASK in
  0) run kang $MB/genotype_pool/singlet_modfix ;;
  1) run hgmm1k_v3 $PB/runs/hgmm1k_v3/singlet-modfix-t8 golden 0.05 100 ;;
  2) run pbmc1k_v3 $PB/runs/pbmc1k_v3/singlet-modfix-t8 golden 0.1 150 ;;
  3) run neuron1k_v3 $PB/runs/neuron1k_v3/singlet-modfix-t8 golden 0.1 150 ;;
  4) run nuclei2k_v2 $PB/runs/nuclei2k_v2/singlet-modfix-t8 ;;
  5) run pbmc5p_v1 $PB/runs/pbmc5p_v1/singlet-modfix-t8 ;;
  6) $PY -P $S/tests/data/called_ed_golden/generate.py synthetic $OUT/golden ;;
esac
echo "$SHA" > $OUT/sha-$TASK.txt
