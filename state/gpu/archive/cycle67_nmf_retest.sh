#!/bin/bash
#SBATCH --job-name=cycle67_nmf_retest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle67_nmf_retest_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle67_nmf_retest_%j.err

set -euo pipefail

LOG=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle67_nmf_retest_${SLURM_JOB_ID}.log
BENCH_REG=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/benchmark-registry.md
CYCLE_LOG=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle-log.md
BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build

echo "=== Cycle 67b NMF solver routing fix re-benchmark ===" | tee -a "$LOG"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG"
echo "Node: $(hostname)" | tee -a "$LOG"
echo "Date: $(date)" | tee -a "$LOG"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Build: pick up fit.h change ---
echo "=== Rebuild bench_reduce_nmf_phaseE ===" | tee -a "$LOG"
export FACTORNET_ROOT=/mnt/home/debruinz/factornet
cd "$BUILD_DIR"
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3 | tee -a "$LOG"

# Force rebuild of NMF bench object to pick up fit.h change
rm -f bench/CMakeFiles/bench_reduce_nmf_phaseE.dir/*.o 2>/dev/null || true
echo "Removed old object files (if any)" | tee -a "$LOG"

make -j8 bench_reduce_nmf_phaseE 2>&1 | tail -10 | tee -a "$LOG"
echo "bench_reduce_nmf_phaseE build exit: $?" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Run NMF benchmark ---
echo "=== NMF Benchmark (k=10, k=20, k=50 with fix) ===" | tee -a "$LOG"
BENCH_OUTPUT=$(./bench/bench_reduce_nmf_phaseE 2>&1)
echo "$BENCH_OUTPUT" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Extract k=50 wall time from output (look for lines containing k=50)
K50_LINE=$(echo "$BENCH_OUTPUT" | grep -i "k=50\|k50\| 50 " | head -5 || echo "k50 line not found")
K10_LINE=$(echo "$BENCH_OUTPUT" | grep -i "k=10\|k10\| 10 " | head -5 || echo "k10 line not found")
K20_LINE=$(echo "$BENCH_OUTPUT" | grep -i "k=20\|k20\| 20 " | head -5 || echo "k20 line not found")

echo "=== Extracted k lines ===" | tee -a "$LOG"
echo "k=10: $K10_LINE" | tee -a "$LOG"
echo "k=20: $K20_LINE" | tee -a "$LOG"
echo "k=50: $K50_LINE" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Build and run correctness tests ---
echo "=== Build correctness tests ===" | tee -a "$LOG"
make -j8 reduce_nmf_correctness 2>&1 | tail -5 | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "=== ctest NMF correctness ===" | tee -a "$LOG"
CTEST_OUTPUT=$(ctest -R "nmf|NMF" -V 2>&1)
echo "$CTEST_OUTPUT" | tee -a "$LOG"
CTEST_PASS=$(echo "$CTEST_OUTPUT" | grep -E "tests passed|passed" | tail -1 || echo "see log")
echo "ctest summary: $CTEST_PASS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Append to benchmark-registry.md ---
echo "=== Appending to benchmark-registry.md ===" | tee -a "$LOG"
cat >> "$BENCH_REG" << EOF

## Cycle 67b — NMF solver routing fix re-benchmark (Job $SLURM_JOB_ID, $(date +%Y-%m-%d))
- k=50 wall BEFORE: 15,980ms (Cholesky O(k^3))
- k=50 wall AFTER: see log — $K50_LINE
- k=10 wall: $K10_LINE
- k=20 wall: $K20_LINE
- Speedup (50 vs before): expected ~16x (k^3 → k^2 at k=50)
- ctest: $CTEST_PASS
- Log: bench/logs/cycle67_nmf_retest_${SLURM_JOB_ID}.log
EOF

# --- Append to cycle-log.md ---
echo "=== Appending to cycle-log.md ===" | tee -a "$LOG"
cat >> "$CYCLE_LOG" << EOF

## Cycle 67b — NMF solver routing fix re-benchmark ($(date +%Y-%m-%d))
- **Job ID**: $SLURM_JOB_ID on $(hostname)
- **Fix**: k>=30 now routes to Coordinate Descent (O(k^2)) instead of Cholesky (O(k^3))
- **k=50 wall BEFORE**: 15,980ms
- **k=50 wall AFTER**: $K50_LINE
- **k=10 wall**: $K10_LINE
- **k=20 wall**: $K20_LINE
- **ctest**: $CTEST_PASS
- **Verdict**: $(echo "$BENCH_OUTPUT" | grep -i "ms\|time" | wc -l) timing lines captured; see benchmark-registry.md for details
EOF

echo "=== Done ===" | tee -a "$LOG"
echo "Completed at: $(date)" | tee -a "$LOG"
