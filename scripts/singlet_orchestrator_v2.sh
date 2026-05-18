#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# singlet_orchestrator_v2.sh — Submit canonical v2 pipeline jobs over a batch
# of accessions and validate outputs after each job completes.
#
# Usage:
#   singlet_orchestrator_v2.sh BATCH_FILE OUTPUT_ROOT [extra args...]
#
# BATCH_FILE
#     One accession or URL per line (blank lines and # comments allowed).
# OUTPUT_ROOT
#     Directory under which one subdirectory per sample is created.
# extra args
#     Forwarded verbatim to `python -m singlet.pipeline run ... -- <extra>`.
#
# Environment:
#   SINGLET_REF_BASE    Reference bundles root (passed to the pipeline).
#   SINGLET_THREADS     Override pipeline threads (default: nproc).
#   SLURM_PARTITION     If set, submit each sample as an sbatch job.
#                       Otherwise run sequentially in the foreground.
#
# Exit codes: 0 on full success, non-zero on any per-sample failure
# or validation error.

set -euo pipefail

usage() {
    sed -n '3,24p' "$0" >&2
    exit 2
}

if [[ $# -lt 2 ]]; then usage; fi

BATCH_FILE=$1
OUTPUT_ROOT=$2
shift 2
EXTRA_ARGS=("$@")

if [[ ! -f "$BATCH_FILE" ]]; then
    echo "BATCH_FILE not found: $BATCH_FILE" >&2
    exit 1
fi
mkdir -p "$OUTPUT_ROOT"

THREADS=${SINGLET_THREADS:-$(nproc)}
PARTITION=${SLURM_PARTITION:-}

FAILED=0
TOTAL=0

run_one() {
    local accession=$1
    local sample_dir="$OUTPUT_ROOT/$accession"
    mkdir -p "$sample_dir"
    echo ">>> singlet pipeline run $accession → $sample_dir"
    python -m singlet.pipeline run \
        "$accession" \
        "$sample_dir" \
        --threads "$THREADS" \
        ${SINGLET_REF_BASE:+--ref-base "$SINGLET_REF_BASE"} \
        -- "${EXTRA_ARGS[@]}"
    echo ">>> validating $sample_dir"
    python -m singlet.manifest "$sample_dir"
}

submit_sbatch() {
    local accession=$1
    local sample_dir="$OUTPUT_ROOT/$accession"
    mkdir -p "$sample_dir"
    local script
    script=$(mktemp -p "$sample_dir" _sbatch.XXXXXX.sh)
    cat >"$script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=singlet_${accession}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${THREADS}
#SBATCH --output=${sample_dir}/sbatch.log
set -euo pipefail
python -m singlet.pipeline run \\
    "${accession}" \\
    "${sample_dir}" \\
    --threads ${THREADS} \\
    ${SINGLET_REF_BASE:+--ref-base "${SINGLET_REF_BASE}"} \\
    -- ${EXTRA_ARGS[*]}
python -m singlet.manifest "${sample_dir}"
EOF
    chmod +x "$script"
    sbatch "$script"
}

while IFS= read -r line; do
    line=${line%%#*}
    line=$(echo "$line" | xargs)
    [[ -z "$line" ]] && continue
    TOTAL=$((TOTAL + 1))
    if [[ -n "$PARTITION" ]]; then
        submit_sbatch "$line" || FAILED=$((FAILED + 1))
    else
        if ! run_one "$line"; then
            FAILED=$((FAILED + 1))
        fi
    fi
done <"$BATCH_FILE"

echo "==================================================="
echo "submitted/ran ${TOTAL} samples; failures so far: ${FAILED}"
echo "(SLURM jobs: check 'squeue -u $USER' for live status)"
exit "$FAILED"
