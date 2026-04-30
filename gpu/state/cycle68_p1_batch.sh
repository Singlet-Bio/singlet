#!/bin/bash
#SBATCH --job-name=cycle68_p1_batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle68_p1_batch_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle68_p1_batch_%j.err

set -euo pipefail

export FACTORNET_ROOT=/mnt/home/debruinz/factornet
BUILD=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build
LOGDIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs
STATE=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state

echo "=== cycle68_p1_batch: Features 12-14 GPU compile + test ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unavailable')"
echo ""

cd "$BUILD"

echo "=== CMake configure ==="
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3
echo ""

# P1 targets: Feature 12 (GSEA), Feature 13 (Annotation), Feature 14 (Integration)
P1_TARGETS=(
    gsea_fgsea_correctness
    gsea_aucell_correctness
    anno_marker_score_correctness
    anno_reference_map_correctness
    integrate_harmony_correctness
    integrate_bbknn_correctness
)

declare -A BUILD_RESULTS
declare -A TEST_RESULTS

echo "=== Building P1 targets ==="
for TARGET in "${P1_TARGETS[@]}"; do
    echo "--- Building $TARGET ---"
    if make -j8 "$TARGET" 2>&1 | tail -10; then
        BUILD_RESULTS[$TARGET]="OK"
    else
        BUILD_RESULTS[$TARGET]="FAIL"
    fi
    echo "$TARGET build: ${BUILD_RESULTS[$TARGET]}"
    echo ""
done

echo "=== P1 ctest ==="
ctest -R "gsea_fgsea|gsea_aucell|anno_marker|anno_reference|integrate_harmony|integrate_bbknn" -V 2>&1
CTEST_EXIT=$?
echo "P1 ctest exit: $CTEST_EXIT"
echo ""

echo "=== Build Summary ==="
for TARGET in "${P1_TARGETS[@]}"; do
    echo "  $TARGET: ${BUILD_RESULTS[$TARGET]}"
done
echo "  ctest exit: $CTEST_EXIT"
echo "=== Done: $(date) ==="

# Append to cycle-log.md
{
echo ""
echo "## cycle68_p1_batch (Job $SLURM_JOB_ID, $(date +%Y-%m-%d))"
echo "Features: 12 (GSEA), 13 (Annotation), 14 (Integration)"
echo ""
echo "| Target | Build |"
echo "|--------|-------|"
for TARGET in "${P1_TARGETS[@]}"; do
    echo "| $TARGET | ${BUILD_RESULTS[$TARGET]} |"
done
echo ""
echo "ctest exit: $CTEST_EXIT"
echo "Log: \`bench/logs/cycle68_p1_batch_${SLURM_JOB_ID}.log\`"
} >> "$STATE/cycle-log.md"
