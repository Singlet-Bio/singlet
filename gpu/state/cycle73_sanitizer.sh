#!/bin/bash
#SBATCH --job-name=cycle73_sanitizer
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle73_sanitizer_%j.log
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -uo pipefail
cd /mnt/home/debruinz/Singlet-AI/singlet-gpu

# Reuse the Cycle 72 clean build tree (already has the 4 sync fixes and Test83 compiled in).
# If it doesn't exist or doesn't have the test binary, rebuild.
BUILD=build_cycle72_verify
if [ ! -x "$BUILD/tests/de_wilcoxon_correctness" ]; then
  echo "Rebuild required"
  rm -rf "$BUILD"
  cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -10
  cmake --build "$BUILD" -j8 --target de_wilcoxon_correctness 2>&1 | tail -15
fi

BIN="$BUILD/tests/de_wilcoxon_correctness"
echo "=== NODE: $(hostname) ==="
echo "=== BIN: $BIN ==="
ls -la "$BIN"
nvidia-smi | head -15

echo "=== COMPUTE-SANITIZER (memcheck) ==="
# Locate compute-sanitizer — CUDA 12.x ships it; older toolkits had cuda-memcheck.
SANITIZER=$(command -v compute-sanitizer || command -v cuda-memcheck || echo "")
if [ -z "$SANITIZER" ]; then
  echo "SANITIZER_NOT_FOUND"
  # Try CUDA toolkit path heuristics
  for p in /usr/local/cuda/bin/compute-sanitizer /opt/cuda/bin/compute-sanitizer; do
    if [ -x "$p" ]; then SANITIZER="$p"; break; fi
  done
fi
echo "Using sanitizer: $SANITIZER"
"$SANITIZER" --version 2>&1 | head -5 || true

# Run memcheck only on Test83 to keep the log short and focused.
# --print-limit 10 caps output; --show-backtrace device adds device backtraces.
"$SANITIZER" \
    --tool memcheck \
    --print-limit 10 \
    --show-backtrace device \
    --error-exitcode 100 \
    "$BIN" --gtest_filter="*Test83_RealDataSized_PostNormalize_NoCrash*" 2>&1 \
    | tee /mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle73_sanitizer_memcheck.txt
echo "SANITIZER_EXIT=$?"

echo "=== SHORT TAIL (first 100 lines of memcheck block only) ==="
grep -A 40 "Invalid \|========= ERROR" /mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle73_sanitizer_memcheck.txt | head -150

echo "=== DONE ==="
