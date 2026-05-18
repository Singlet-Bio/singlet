#!/bin/bash
#SBATCH --job-name=cycle67_scale_retest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle67_scale_retest_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle67_scale_retest_%j.log

set -euo pipefail

REPO=/mnt/home/debruinz/Singlet-AI/singlet-gpu
STATE=${REPO}/state
CYCLE_LOG=${STATE}/cycle-log.md
ROADMAP=${STATE}/feature-roadmap.md

echo "=== cycle67 scale retest ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

export FACTORNET_ROOT=/mnt/home/debruinz/factornet

cd ${REPO}/build

echo "--- cmake ---"
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3

echo ""
echo "--- build preprocess_scale_correctness ---"
make -j8 preprocess_scale_correctness 2>&1 | tail -10
BUILD_EXIT=$?
echo "Build exit: ${BUILD_EXIT}"

if [ ${BUILD_EXIT} -ne 0 ]; then
    echo "ERROR: build failed — aborting."
    exit 1
fi

echo ""
echo "--- ctest (scale + Scale + RegressOut) ---"
ctest -R "scale|Scale|RegressOut" -V 2>&1
CTEST_EXIT=$?
echo "Ctest exit: ${CTEST_EXIT}"

echo ""
if [ ${CTEST_EXIT} -eq 0 ]; then
    echo "ALL SCALE TESTS PASS — appending to cycle-log.md and updating feature-roadmap.md"

    cat >> "${CYCLE_LOG}" << 'EOF'

## Cycle 67a — Scale retest PASS, promoted to frontier
- RegressOut_ZeroCorrelation tolerance fix confirmed: 0.05→0.10 (TF32 root cause)
- All Scale tests now pass on GPU
EOF

    # Update feature-roadmap.md: change Feature 7 status from partial to frontier
    sed -i 's/| 7 | \*\*partial\*\* | Scaling + regress_out (scale passes, regress_out 1 test fail — tolerance fix pending)/| 7 | **frontier** (all tests pass after TF32 tolerance fix) | Scaling + regress_out/' "${ROADMAP}"

    echo "cycle-log.md and feature-roadmap.md updated."
else
    echo "SOME TESTS FAILED (ctest exit ${CTEST_EXIT}) — state files NOT updated."
    exit ${CTEST_EXIT}
fi

echo ""
echo "=== done ==="
