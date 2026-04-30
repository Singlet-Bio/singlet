#!/bin/bash
#SBATCH --job-name=cycle68_de_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle68_de_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle68_de_%j.err

set -euo pipefail

LOG=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle68_de_${SLURM_JOB_ID}.log
CYCLE_LOG=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle-log.md
BENCH_REG=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/benchmark-registry.md
BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build

export FACTORNET_ROOT=/mnt/home/debruinz/factornet

echo "======================================================"
echo "cycle68 DE bench — job ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "======================================================"

# ── Build ──────────────────────────────────────────────────────────────────
echo ""
echo "=== cmake configure ==="
cd "$BUILD_DIR"
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3

BUILD_RESULTS=()
for target in de_wilcoxon_correctness de_ttest_correctness de_donor_pseudobulk_correctness; do
    echo ""
    echo "=== Building $target ==="
    make -j8 $target 2>&1 | tail -15
    BUILD_EXIT=$?
    BUILD_RESULTS+=("$target:$BUILD_EXIT")
    echo "Build exit: $BUILD_EXIT"
done

# ── Correctness tests ──────────────────────────────────────────────────────
echo ""
echo "=== DE ctest ==="
set +e
ctest -R "de_|wilcoxon|ttest|donor|De" -V 2>&1
CTEST_EXIT=$?
set -e
echo "ctest exit: $CTEST_EXIT"

# ── Bench driver ───────────────────────────────────────────────────────────
echo ""
echo "=== Building bench_de_wilcoxon_perf ==="
set +e
make -j8 bench_de_wilcoxon_perf 2>&1 | tail -10
BENCH_BUILD_EXIT=$?
set -e
echo "Bench build exit: $BENCH_BUILD_EXIT"

BENCH_THROUGHPUT="N/A"
if [ $BENCH_BUILD_EXIT -eq 0 ] && [ -f "$BUILD_DIR/bench/bench_de_wilcoxon_perf" ]; then
    echo ""
    echo "=== Running bench_de_wilcoxon_perf ==="
    set +e
    BENCH_OUT=$("$BUILD_DIR/bench/bench_de_wilcoxon_perf" 2>&1)
    BENCH_EXIT=$?
    set -e
    echo "$BENCH_OUT"
    echo "Bench exit: $BENCH_EXIT"
    # Extract throughput line (cells/s or genes/s or similar)
    BENCH_THROUGHPUT=$(echo "$BENCH_OUT" | grep -Ei "cells/s|genes/s|throughput|ms|μs|us" | head -3 | tr '\n' '|' || echo "N/A")
fi

echo ""
echo "======================================================"
echo "End: $(date)"
echo "======================================================"

# ── Append to cycle-log.md ─────────────────────────────────────────────────
BUILD_SUMMARY=""
for r in "${BUILD_RESULTS[@]}"; do
    name="${r%%:*}"; exit_code="${r##*:}"
    if [ "$exit_code" = "0" ]; then
        BUILD_SUMMARY+="  - $name: OK\n"
    else
        BUILD_SUMMARY+="  - $name: FAIL (exit $exit_code)\n"
    fi
done

cat >> "$CYCLE_LOG" << EOF

## Cycle 68 — DE compile + correctness ($(date +%Y-%m-%d))
- **Job**: ${SLURM_JOB_ID} on $(hostname)
- **Phase**: E — GPU compile + correctness test (Wilcoxon, t-test, pseudobulk)
- **Build results**:
$(printf "$BUILD_SUMMARY")- **ctest exit**: ${CTEST_EXIT}
- **Bench build exit**: ${BENCH_BUILD_EXIT}
- **Bench throughput**: ${BENCH_THROUGHPUT}
EOF

# ── Append to benchmark-registry.md ───────────────────────────────────────
cat >> "$BENCH_REG" << EOF

## Feature 11 — Differential Expression (Cycle 68, $(date +%Y-%m-%d))
| Target | Build | Notes |
|--------|-------|-------|
| de_wilcoxon_correctness | $(echo "${BUILD_RESULTS[0]}" | grep -q ':0' && echo OK || echo FAIL) | GPU Wilcoxon rank-sum |
| de_ttest_correctness | $(echo "${BUILD_RESULTS[1]}" | grep -q ':0' && echo OK || echo FAIL) | GPU Welch t-test |
| de_donor_pseudobulk_correctness | $(echo "${BUILD_RESULTS[2]}" | grep -q ':0' && echo OK || echo FAIL) | GPU pseudobulk NB GLM |
| bench_de_wilcoxon_perf | $([ $BENCH_BUILD_EXIT -eq 0 ] && echo OK || echo FAIL) | Bench driver |

- **ctest result**: $([ $CTEST_EXIT -eq 0 ] && echo "PASS" || echo "FAIL (exit $CTEST_EXIT)")
- **Wilcoxon bench throughput**: ${BENCH_THROUGHPUT}
- **Job ID**: ${SLURM_JOB_ID}
EOF

echo "State files updated."
