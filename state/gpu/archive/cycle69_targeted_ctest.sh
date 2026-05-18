#!/bin/bash
#SBATCH --job-name=sg69_targeted_ctest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle69_targeted_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle69_targeted_%j.log

set -euo pipefail

export FACTORNET_ROOT=/mnt/home/debruinz/factornet
BUILD=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build
TESTS=$BUILD/tests

echo "=== cycle69 targeted ctest — $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

# Quick cmake reconfigure if needed (no-op if up-to-date)
if [ ! -f "$TESTS/de_wilcoxon_correctness" ]; then
    echo "=== Rebuilding (binaries missing) ==="
    cd "$BUILD"
    cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3
fi

cd "$TESTS"

# ──────────────────────────────────────────────
# DE TESTS
# ──────────────────────────────────────────────

echo ""
echo "=== DE Wilcoxon ==="
[ ! -f ./de_wilcoxon_correctness ] && \
    make -C "$BUILD" -j8 de_wilcoxon_correctness 2>&1 | tail -5
ctest --test-dir "$BUILD" -R "WilcoxonTest" -V --timeout 120 2>&1
echo "Wilcoxon exit: $?"

echo ""
echo "=== DE t-test ==="
[ ! -f ./de_ttest_correctness ] && \
    make -C "$BUILD" -j8 de_ttest_correctness 2>&1 | tail -5
ctest --test-dir "$BUILD" -R "TtestDE" -V --timeout 120 2>&1
echo "Ttest exit: $?"

echo ""
echo "=== DE Donor Pseudobulk ==="
[ ! -f ./de_donor_pseudobulk_correctness ] && \
    make -C "$BUILD" -j8 de_donor_pseudobulk_correctness 2>&1 | tail -5
ctest --test-dir "$BUILD" -R "DonorPseudobulk" -V --timeout 120 2>&1
echo "Donor exit: $?"

# ──────────────────────────────────────────────
# P1 TESTS
# ──────────────────────────────────────────────

echo ""
echo "=== GSEA fgsea ==="
[ ! -f ./gsea_fgsea_correctness ] && \
    make -C "$BUILD" -j8 gsea_fgsea_correctness 2>&1 | tail -5
ctest --test-dir "$BUILD" -R "^Fgsea\." -V --timeout 120 2>&1
echo "Fgsea exit: $?"

echo ""
echo "=== GSEA AUCell ==="
[ ! -f ./gsea_aucell_correctness ] && \
    make -C "$BUILD" -j8 gsea_aucell_correctness 2>&1 | tail -5
ctest --test-dir "$BUILD" -R "^AUCell\." -V --timeout 120 2>&1
echo "Aucell exit: $?"

echo ""
echo "=== Anno marker ==="
[ ! -f ./anno_marker_score_correctness ] && \
    make -C "$BUILD" -j8 anno_marker_score_correctness 2>&1 | tail -5
ctest --test-dir "$BUILD" -R "MarkerScoreTest" -V --timeout 120 2>&1
echo "Marker exit: $?"

echo ""
echo "=== Anno reference ==="
[ ! -f ./anno_reference_map_correctness ] && \
    make -C "$BUILD" -j8 anno_reference_map_correctness 2>&1 | tail -5
ctest --test-dir "$BUILD" -R "ReferenceMapTest" -V --timeout 120 2>&1
echo "RefMap exit: $?"

echo ""
echo "=== Integration Harmony ==="
[ ! -f ./integrate_harmony_correctness ] && \
    make -C "$BUILD" -j8 integrate_harmony_correctness 2>&1 | tail -5
ctest --test-dir "$BUILD" -R "HarmonyCorrectness" -V --timeout 120 2>&1
echo "Harmony exit: $?"

echo ""
echo "=== Integration BBKNN ==="
make -C "$BUILD" -j8 integrate_bbknn_correctness 2>&1 | tail -5
ctest --test-dir "$BUILD" -R "BbknnTest" -V --timeout 120 2>&1
echo "Bbknn exit: $?"

echo ""
echo "=== cycle69 targeted ctest complete — $(date) ==="
