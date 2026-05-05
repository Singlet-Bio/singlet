#!/usr/bin/env bash
# ── Benchmark Suite Orchestration ────────────────────────────────────────────
# Submits all benchmark jobs to SLURM with correct dependency ordering.
# Re-encodes all .1pz files with the CURRENT codec before measuring.
#
# Usage (from login node):
#     bash code/run_benchmarks.sh          # submit all
#     bash code/run_benchmarks.sh --dry    # print commands without submitting
#
# After all jobs finish:
#     1. python3 code/benchmarks/extract_write_benchmarks.py
#     2. cd code/data && Rscript ../figures/figures_v4.R
#     3. cd code/figures && Rscript figures_frontier.R
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ROOT/code/benchmarks"
DATA="$ROOT/code/data"
VENV="source /mnt/home/debruinz/venv/bin/activate"
LOGDIR="$ROOT/code/logs"
DRY=false

[[ "${1:-}" == "--dry" ]] && DRY=true

mkdir -p "$DATA" "$LOGDIR"

submit() {
    local name="$1"; shift
    local args=("$@")
    if $DRY; then
        echo "[DRY] sbatch ${args[*]}" >&2
        echo "0"  # fake job ID
    else
        local out
        out=$(sbatch "${args[@]}")
        local jid
        jid=$(echo "$out" | grep -oP '\d+')
        echo "$jid"
        echo "  Submitted $name → job $jid" >&2
    fi
}

echo "═══════════════════════════════════════════════════════════"
echo "  SinglePress Benchmark Suite"
echo "  Root: $ROOT"
echo "  Data: $DATA"
echo "  Dry run: $DRY"
echo "═══════════════════════════════════════════════════════════"

# ── 1. Survey (longest: ~3-4 hours, scans all ~3000 .1pz files) ─────────
echo ""
echo "─── Step 1: Dataset Survey ───"
JID_SURVEY=$(submit "survey" \
    --job-name=sp-survey \
    --time=360 \
    --mem=128G \
    --cpus-per-task=8 \
    --output="$LOGDIR/survey_%j.out" \
    --error="$LOGDIR/survey_%j.err" \
    --wrap="$VENV && cd /tmp && python3 -u $BENCH/survey_all_datasets.py")

# ── 2. CPU benchmarks (independent of survey, except gpu_flops) ──────────
echo ""
echo "─── Step 2: CPU Benchmarks ───"

# Threading benchmark (~1-2 hours, needs 32 CPUs)
JID_THREAD=$(submit "threading" \
    --job-name=sp-thread \
    --time=180 \
    --mem=64G \
    --cpus-per-task=32 \
    --output="$LOGDIR/threading_%j.out" \
    --error="$LOGDIR/threading_%j.err" \
    --wrap="$VENV && cd /tmp && python3 -u $BENCH/benchmark_threading_v2.py")

# Operations benchmark (~1-2 hours)
JID_OPS=$(submit "operations" \
    --job-name=sp-ops \
    --time=180 \
    --mem=64G \
    --cpus-per-task=8 \
    --output="$LOGDIR/ops_%j.out" \
    --error="$LOGDIR/ops_%j.err" \
    --wrap="$VENV && cd /tmp && python3 -u $BENCH/benchmark_operations.py")

# Compression frontier (~1-2 hours)
JID_FRONTIER=$(submit "frontier" \
    --job-name=sp-frontier \
    --time=180 \
    --mem=64G \
    --cpus-per-task=8 \
    --output="$LOGDIR/frontier_%j.out" \
    --error="$LOGDIR/frontier_%j.err" \
    --wrap="$VENV && cd /tmp && python3 -u $BENCH/benchmark_compression_frontier.py")

# Write benchmarks (v3: .1pz, H5AD, 10x H5, npz, RDS — ~2 hours)
JID_WRITE=$(submit "write-v3" \
    --job-name=sp-write \
    --time=180 \
    --mem=64G \
    --cpus-per-task=8 \
    --output="$LOGDIR/write_%j.out" \
    --error="$LOGDIR/write_%j.err" \
    --wrap="$VENV && cd /tmp && python3 -u $BENCH/benchmarks_v3.py")

# ── 3. GPU benchmark (depends on survey for dataset selection) ────────────
echo ""
echo "─── Step 3: GPU Benchmark ───"
if $DRY; then
    JID_GPU=$(submit "gpu-flops" \
        --job-name=sp-gpu \
        --dependency=afterok:0 \
        --partition=gpu \
        --gres=gpu:1 \
        --constraint=nvidia_h100_nvl \
        --time=120 \
        --mem=64G \
        --cpus-per-task=8 \
        --output="$LOGDIR/gpu_%j.out" \
        --error="$LOGDIR/gpu_%j.err" \
        --wrap="$VENV && cd /tmp && python3 -u $BENCH/benchmark_gpu_flops.py")
else
    JID_GPU=$(submit "gpu-flops" \
        --job-name=sp-gpu \
        --dependency=afterok:$JID_SURVEY \
        --partition=gpu \
        --gres=gpu:1 \
        --constraint=nvidia_h100_nvl \
        --time=120 \
        --mem=64G \
        --cpus-per-task=8 \
        --output="$LOGDIR/gpu_%j.out" \
        --error="$LOGDIR/gpu_%j.err" \
        --wrap="$VENV && cd /tmp && python3 -u $BENCH/benchmark_gpu_flops.py")
fi

# ── 4. R benchmarks (depend on survey for dataset list) ──────────────────
echo ""
echo "─── Step 4: R Benchmarks ───"
if $DRY; then
    R_DEP="--dependency=afterok:0"
else
    R_DEP="--dependency=afterok:$JID_SURVEY"
fi

JID_RFMT=$(submit "r-formats" \
    --job-name=sp-rfmt \
    $R_DEP \
    --time=120 \
    --mem=64G \
    --cpus-per-task=8 \
    --output="$LOGDIR/rfmt_%j.out" \
    --error="$LOGDIR/rfmt_%j.err" \
    --wrap="module load r/4.5.2 && $VENV && cd /tmp && Rscript $BENCH/benchmark_r_formats_v2.R")

JID_BPCELLS=$(submit "bpcells" \
    --job-name=sp-bpcells \
    $R_DEP \
    --time=120 \
    --mem=64G \
    --cpus-per-task=8 \
    --output="$LOGDIR/bpcells_%j.out" \
    --error="$LOGDIR/bpcells_%j.err" \
    --wrap="module load r/4.5.2 && $VENV && cd /tmp && Rscript $BENCH/benchmark_bpcells_v3.R")

# ── 5. Post-processing (depends on write-v3 finishing) ───────────────────
echo ""
echo "─── Step 5: Extract & R Write Benchmarks ───"
if $DRY; then
    WRITE_DEP="--dependency=afterok:0"
else
    WRITE_DEP="--dependency=afterok:$JID_WRITE"
fi

JID_EXTRACT=$(submit "extract-csv" \
    --job-name=sp-extract \
    $WRITE_DEP \
    --time=10 \
    --mem=4G \
    --cpus-per-task=1 \
    --output="$LOGDIR/extract_%j.out" \
    --error="$LOGDIR/extract_%j.err" \
    --wrap="$VENV && python3 $BENCH/extract_write_benchmarks.py")

if $DRY; then
    RW_DEP="--dependency=afterok:0"
else
    RW_DEP="--dependency=afterok:$JID_EXTRACT"
fi

JID_RW=$(submit "r-write" \
    --job-name=sp-rwrite \
    $RW_DEP \
    --time=60 \
    --mem=64G \
    --cpus-per-task=8 \
    --output="$LOGDIR/rwrite_%j.out" \
    --error="$LOGDIR/rwrite_%j.err" \
    --wrap="module load r/4.5.2 && $VENV && cd /tmp && python3 -u $BENCH/benchmark_write_rds.py")

# ── 6. Figure generation (depends on ALL data CSVs) ──────────────────────
echo ""
echo "─── Step 6: Generate Figures ───"
if $DRY; then
    FIG_DEP="--dependency=afterok:0"
else
    FIG_DEP="--dependency=afterok:$JID_SURVEY:$JID_THREAD:$JID_OPS:$JID_GPU:$JID_RFMT:$JID_BPCELLS:$JID_EXTRACT:$JID_RW:$JID_FRONTIER"
fi

JID_FIG=$(submit "figures" \
    --job-name=sp-figures \
    $FIG_DEP \
    --time=30 \
    --mem=16G \
    --cpus-per-task=4 \
    --output="$LOGDIR/figures_%j.out" \
    --error="$LOGDIR/figures_%j.err" \
    --wrap="module load r/4.5.2 && cd $DATA && Rscript $ROOT/code/figures/figures_v4.R && cd $ROOT/code/figures && Rscript figures_frontier.R")

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All jobs submitted. Monitor with:"
echo "    squeue -u \$USER --name=sp-survey,sp-thread,sp-ops,sp-frontier,sp-write,sp-gpu,sp-rfmt,sp-bpcells,sp-extract,sp-rwrite,sp-figures"
echo ""
echo "  Job chain:"
echo "    survey ($JID_SURVEY) ─┬─> gpu ($JID_GPU)"
echo "                          ├─> r-formats ($JID_RFMT)"
echo "                          └─> bpcells ($JID_BPCELLS)"
echo "    threading ($JID_THREAD)  ──> [independent]"
echo "    operations ($JID_OPS)    ──> [independent]"
echo "    frontier ($JID_FRONTIER) ──> [independent]"
echo "    write ($JID_WRITE) ──> extract ($JID_EXTRACT) ──> r-write ($JID_RW)"
echo "    ALL ──> figures ($JID_FIG)"
echo "═══════════════════════════════════════════════════════════"
