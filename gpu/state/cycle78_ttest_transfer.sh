#!/bin/bash
#SBATCH --job-name=sg_cycle78_ttest
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle78_ttest_transfer_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle78_verify
SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu

echo "=== Cycle 78 T-test Transfer (m/n swap + expm1 + negate + target_count + aligned Spearman) ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
date

rm -rf "$BUILD_DIR"

echo "--- cmake configure ---"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DEIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    2>&1 | tail -5
CMAKE_EXIT=$?
echo "CMAKE_EXIT=$CMAKE_EXIT"
[ "$CMAKE_EXIT" -ne 0 ] && { echo "CMAKE CONFIGURE FAILED"; exit $CMAKE_EXIT; }

echo "--- cmake build (target de_ttest_correctness) ---"
cmake --build "$BUILD_DIR" --target de_ttest_correctness -j8 2>&1 | tail -20
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED"; exit $BUILD_EXIT; }

# ── T-test TinyPlanted tests (try all casings) ─────────────────────────────────
echo ""
echo "--- ctest: TinyPlanted (try all casings) ---"
for REGEX in "Ttest.*TinyPlanted" "TTest.*TinyPlanted" "ttest.*TinyPlanted"; do
    COUNT=$(ctest --test-dir "$BUILD_DIR" -R "$REGEX" -N 2>/dev/null | grep -c "Test #" || true)
    if [ "$COUNT" -gt 0 ]; then
        echo "Using regex: $REGEX  (matched $COUNT tests)"
        TTEST_TINY_REGEX="$REGEX"
        break
    fi
done
TTEST_TINY_REGEX="${TTEST_TINY_REGEX:-Ttest.*TinyPlanted}"

ctest --test-dir "$BUILD_DIR" -R "$TTEST_TINY_REGEX" \
      --output-on-failure -V 2>&1 | tee /tmp/ttest_tiny_${SLURM_JOB_ID}.txt
TINY_EXIT=$?
echo "TINY_EXIT=$TINY_EXIT"

# ── T-test RealData test ────────────────────────────────────────────────────────
echo ""
echo "--- ctest: Ttest.*RealData | GSM4037629 ---"
ctest --test-dir "$BUILD_DIR" -R "Ttest.*GSM4037629|Ttest.*RealData" \
      --output-on-failure -V 2>&1 | tee /tmp/ttest_real_${SLURM_JOB_ID}.txt
REAL_EXIT=$?
echo "REAL_EXIT=$REAL_EXIT"

# ── Wilcoxon no-regression ──────────────────────────────────────────────────────
echo ""
echo "--- Build wilcoxon (no-regression) ---"
cmake --build "$BUILD_DIR" --target de_wilcoxon_correctness -j8 2>&1 | tail -5
WILCOX_BUILD_EXIT=$?
echo "WILCOX_BUILD_EXIT=$WILCOX_BUILD_EXIT"

if [ "$WILCOX_BUILD_EXIT" -eq 0 ]; then
    echo "--- ctest: Wilcoxon.*TinyPlanted (no-regression) ---"
    ctest --test-dir "$BUILD_DIR" -R "Wilcoxon.*TinyPlanted" \
          --output-on-failure -V 2>&1 | tee /tmp/wilcox_tiny_${SLURM_JOB_ID}.txt
    TINY_WILCOX_EXIT=$?
    echo "TINY_WILCOX_EXIT=$TINY_WILCOX_EXIT"

    echo "--- ctest: Test83_RealDataSized_PostNormalize_NoCrash (no-regression) ---"
    ctest --test-dir "$BUILD_DIR" -R "Test83_RealDataSized_PostNormalize_NoCrash" \
          --output-on-failure -V 2>&1 | tee /tmp/test83_${SLURM_JOB_ID}.txt
    TEST83_EXIT=$?
    echo "TEST83_EXIT=$TEST83_EXIT"

    echo "--- ctest: Wilcoxon_GSM4037629_RealDataPlanted (no-regression) ---"
    ctest --test-dir "$BUILD_DIR" -R "Wilcoxon_GSM4037629_RealDataPlanted" \
          --output-on-failure -V 2>&1 | tee /tmp/wilcox_real_${SLURM_JOB_ID}.txt
    WILCOX_REAL_EXIT=$?
    echo "WILCOX_REAL_EXIT=$WILCOX_REAL_EXIT"
else
    echo "Wilcoxon build failed — skipping no-regression tests"
    TINY_WILCOX_EXIT=99
    TEST83_EXIT=99
    WILCOX_REAL_EXIT=99
fi

# ── Key metrics extraction ──────────────────────────────────────────────────────
echo ""
echo "=== KEY METRICS: T-test TinyPlanted ==="
grep -E "Jaccard|Spearman|rho=|registry|PASSED|FAILED|intersection|cluster [0-9]" \
    /tmp/ttest_tiny_${SLURM_JOB_ID}.txt | head -40 || true

echo ""
echo "=== KEY METRICS: T-test RealData ==="
grep -E "Jaccard|Spearman|rho=|registry|PASSED|FAILED|scanpy|cluster [0-9]" \
    /tmp/ttest_real_${SLURM_JOB_ID}.txt | head -30 || true

echo ""
echo "=== Wilcoxon TinyPlanted VERDICT ==="
grep -E "PASSED|FAILED|tests passed|tests failed" \
    /tmp/wilcox_tiny_${SLURM_JOB_ID}.txt | tail -5 || true

echo ""
echo "=== Test83 VERDICT ==="
grep -E "PASSED|FAILED|PASS|FAIL|CUDA error" \
    /tmp/test83_${SLURM_JOB_ID}.txt | tail -5 || true

echo ""
echo "=== Wilcoxon RealDataPlanted VERDICT ==="
grep -E "PASSED|FAILED|Jaccard|Spearman|rho=" \
    /tmp/wilcox_real_${SLURM_JOB_ID}.txt | tail -10 || true

echo ""
echo "=== SUMMARY ==="
echo "BUILD_EXIT=$BUILD_EXIT  TINY_EXIT=$TINY_EXIT  REAL_EXIT=$REAL_EXIT"
echo "WILCOX_BUILD=$WILCOX_BUILD_EXIT  WILCOX_TINY=$TINY_WILCOX_EXIT  TEST83=$TEST83_EXIT  WILCOX_REAL=$WILCOX_REAL_EXIT"
date
