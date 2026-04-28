#!/bin/bash
# job_dispatch.sh — Submit any command as an async SLURM batch job.
#
# Wraps a command in a SLURM script that:
#   1. Runs the command on the target node
#   2. Writes structured JSON result to a known path
#   3. Returns immediately so the agent can continue working
#
# The orchestrator polls for results via job_poll.py at Phase 0.
#
# Usage:
#   bash job_dispatch.sh submit \
#     --tag "bench-c01-cycle67" \
#     --node c001 \
#     --threads 20 \
#     --timeout 3600 \
#     --cmd 'source /opt/rh/gcc-toolset-13/enable && ...'
#
#   bash job_dispatch.sh list          # show all pending/completed jobs
#   bash job_dispatch.sh cancel TAG    # cancel a running job
#
# Result files land in: singlify/state/jobs/<tag>/
#   result.json  — structured output (exit code, wall time, stdout tail)
#   stdout.log   — full stdout+stderr
#   job.meta     — submission metadata (tag, node, cmd, submit time)
#
# Design principles:
#   - Every job is identified by a unique TAG (human-readable, cycle-scoped)
#   - Jobs are fire-and-forget: submit returns instantly with SLURM job ID
#   - Results are self-contained JSON: any agent can parse them without logs
#   - Short jobs (<30s expected) should use direct SSH — don't batch those
#   - Medium jobs (30s-120s): submit async but poll within cycle before closing
#   - Long jobs (>120s): submit and harvest next cycle
#
# job_dispatch.sh wait TAG [--timeout 120]  — poll until complete or timeout

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_DIR="$(cd "${SCRIPT_DIR}/../state/jobs" 2>/dev/null && pwd || echo "${SCRIPT_DIR}/../state/jobs")"

# ─── Helper ──────────────────────────────────────────────────────────────────

preamble() {
    cat << 'ENV'
source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
export PKG_CONFIG_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib/pkgconfig
ulimit -n 10240
ENV
}

# ─── Submit ──────────────────────────────────────────────────────────────────

cmd_submit() {
    local tag="" node="" threads=20 timeout=3600 cmd="" expected_duration=0 dag_task="" cycle=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --tag)               tag="$2"; shift 2 ;;
            --node)              node="$2"; shift 2 ;;
            --threads)           threads="$2"; shift 2 ;;
            --timeout)           timeout="$2"; shift 2 ;;
            --cmd)               cmd="$2"; shift 2 ;;
            --expected-duration) expected_duration="$2"; shift 2 ;;
            --dag-task)          dag_task="$2"; shift 2 ;;
            --cycle)             cycle="$2"; shift 2 ;;
            *) echo "Unknown: $1" >&2; exit 1 ;;
        esac
    done

    if [[ -z "$tag" || -z "$cmd" ]]; then
        echo "ERROR: --tag and --cmd are required" >&2
        exit 1
    fi

    # Default node: c001
    node="${node:-c001}"

    # Create job directory
    local job_dir="${JOBS_DIR}/${tag}"
    mkdir -p "$job_dir"

    # Write metadata (use Python for safe JSON encoding)
    python3 -c "
import json, pathlib
pathlib.Path('${job_dir}/job.meta').write_text(json.dumps({
    'tag': '${tag}',
    'node': '${node}',
    'threads': ${threads},
    'timeout': ${timeout},
    'expected_duration': ${expected_duration},
    'dag_task': '${dag_task}',
    'cycle': ${cycle},
    'submitted': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
    'status': 'pending',
    'cmd_preview': $(python3 -c "import sys,json; print(json.dumps(sys.argv[1][:120]))" "$cmd")
}, indent=2))
"

    # Build SLURM script
    local slurm_script="${job_dir}/job.slurm"
    cat > "$slurm_script" << SLURM
#!/bin/bash
#SBATCH --job-name=sf_${tag}
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --nodelist=${node}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${threads}
#SBATCH --time=$(( timeout / 60 + 1 ))
#SBATCH --output=${job_dir}/stdout.log
#SBATCH --error=${job_dir}/stdout.log

# ── Environment ──
$(preamble)

# ── Timing wrapper ──
JOB_DIR="${job_dir}"
T_START=\$(date +%s.%N)

# Run the user command, capture exit code
set +e
(
${cmd}
)
CMD_EXIT=\$?
set -e

T_END=\$(date +%s.%N)
WALL=\$(python3 -c "print(round(\$T_END - \$T_START, 3))")

# ── Write structured result ──
STDOUT_TAIL=\$(tail -20 "\${JOB_DIR}/stdout.log" 2>/dev/null | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' 2>/dev/null || echo '""')

cat > "\${JOB_DIR}/result.json" << RESULT
{
  "tag": "${tag}",
  "exit_code": \${CMD_EXIT},
  "wall_seconds": \${WALL},
  "completed": "\$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "node": "${node}",
  "stdout_tail": \${STDOUT_TAIL}
}
RESULT

# Update meta status
python3 -c "
import json, pathlib
p = pathlib.Path('\${JOB_DIR}/job.meta')
d = json.loads(p.read_text())
d['status'] = 'done' if \${CMD_EXIT} == 0 else 'failed'
d['slurm_job_id'] = '\${SLURM_JOB_ID:-unknown}'
p.write_text(json.dumps(d, indent=2))
"

echo "[job_dispatch] ${tag}: exit=\${CMD_EXIT} wall=\${WALL}s"
SLURM

    # Submit
    local jid
    jid=$(sbatch --parsable "$slurm_script" 2>&1)

    if [[ "$jid" =~ ^[0-9]+$ ]]; then
        # Record SLURM job ID in meta
        python3 -c "
import json, pathlib
p = pathlib.Path('${job_dir}/job.meta')
d = json.loads(p.read_text())
d['slurm_job_id'] = '$jid'
p.write_text(json.dumps(d, indent=2))
"
        echo "SUBMITTED: tag=${tag} job_id=${jid} node=${node}"
        echo "${jid}"
    else
        echo "SUBMIT FAILED: ${jid}" >&2
        echo "failed" > "${job_dir}/result.json"
        exit 1
    fi
}

# ─── List ────────────────────────────────────────────────────────────────────

cmd_list() {
    if [[ ! -d "$JOBS_DIR" ]]; then
        echo "(no jobs directory)"
        return
    fi

    printf "%-30s %-10s %-8s %-20s %s\n" "TAG" "STATUS" "EXIT" "SUBMITTED" "WALL"
    printf "%-30s %-10s %-8s %-20s %s\n" "---" "------" "----" "---------" "----"

    for d in "${JOBS_DIR}"/*/; do
        [[ -d "$d" ]] || continue
        local tag=$(basename "$d")
        local meta="${d}job.meta"
        local result="${d}result.json"

        if [[ -f "$result" ]] && python3 -c "import json; json.load(open('$result'))" 2>/dev/null; then
            local exit_code wall status submitted
            read -r exit_code wall status submitted < <(python3 -c "
import json
r = json.load(open('$result'))
m = json.load(open('$meta')) if __import__('pathlib').Path('$meta').exists() else {}
print(r.get('exit_code','?'), r.get('wall_seconds','?'), m.get('status','?'), m.get('submitted','?')[:19])
" 2>/dev/null || echo "? ? ? ?")
            printf "%-30s %-10s %-8s %-20s %ss\n" "$tag" "$status" "$exit_code" "$submitted" "$wall"
        elif [[ -f "$meta" ]]; then
            local status submitted
            read -r status submitted < <(python3 -c "
import json
m = json.load(open('$meta'))
print(m.get('status','?'), m.get('submitted','?')[:19])
" 2>/dev/null || echo "? ?")
            printf "%-30s %-10s %-8s %-20s %s\n" "$tag" "$status" "-" "$submitted" "-"
        fi
    done
}

# ─── Cancel ──────────────────────────────────────────────────────────────────

cmd_cancel() {
    local tag="$1"
    local meta="${JOBS_DIR}/${tag}/job.meta"

    if [[ ! -f "$meta" ]]; then
        echo "No job found with tag: $tag" >&2
        exit 1
    fi

    local jid
    jid=$(python3 -c "import json; print(json.load(open('$meta')).get('slurm_job_id',''))")
    if [[ -n "$jid" && "$jid" != "None" ]]; then
        scancel "$jid" 2>/dev/null && echo "Cancelled SLURM job $jid (tag=$tag)"
    fi

    python3 -c "
import json, pathlib
p = pathlib.Path('$meta')
d = json.loads(p.read_text())
d['status'] = 'cancelled'
p.write_text(json.dumps(d, indent=2))
"
}

# ─── Wait ────────────────────────────────────────────────────────────────────

cmd_wait() {
    local tag="${1:?TAG required}"
    shift
    local poll_timeout=120

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --timeout) poll_timeout="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    local job_dir="${JOBS_DIR}/${tag}"
    local result="${job_dir}/result.json"
    local meta="${job_dir}/job.meta"

    if [[ ! -f "$meta" ]]; then
        echo "No job found: $tag" >&2
        return 1
    fi

    local elapsed=0
    local interval=5
    echo "Waiting for ${tag} (timeout=${poll_timeout}s, poll=${interval}s)..."

    while [[ $elapsed -lt $poll_timeout ]]; do
        # Check if result.json exists (job completed)
        if [[ -f "$result" ]] && python3 -c "import json; json.load(open('$result'))" 2>/dev/null; then
            local exit_code wall
            read -r exit_code wall < <(python3 -c "
import json
r = json.load(open('$result'))
print(r.get('exit_code','?'), r.get('wall_seconds','?'))
" 2>/dev/null || echo "? ?")
            echo "COMPLETED: ${tag} exit=${exit_code} wall=${wall}s (waited ${elapsed}s)"
            return 0
        fi

        sleep "$interval"
        elapsed=$((elapsed + interval))

        # Adaptive interval: start at 5s, grow to 15s after 30s
        if [[ $elapsed -gt 30 ]]; then interval=15; fi
    done

    echo "TIMEOUT: ${tag} still running after ${poll_timeout}s"
    return 2
}

# ─── Main ────────────────────────────────────────────────────────────────────

case "${1:-help}" in
    submit)  shift; cmd_submit "$@" ;;
    list)    cmd_list ;;
    cancel)  cmd_cancel "${2:?TAG required}" ;;
    wait)    shift; cmd_wait "$@" ;;
    *)
        echo "Usage: job_dispatch.sh {submit|list|cancel|wait} [options]"
        echo ""
        echo "  submit --tag TAG --cmd 'COMMAND' [--node NODE] [--threads N] [--timeout S]"
        echo "         [--expected-duration SECS] [--dag-task TASK] [--cycle N]"
        echo "  list                     Show all jobs and their status"
        echo "  cancel TAG               Cancel a running job"
        echo "  wait TAG [--timeout 120] Poll until job completes or timeout"
        ;;
esac
