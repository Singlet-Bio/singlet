#!/usr/bin/env bash
# dag_lock.sh — Atomic DAG lock for parallel singlify agents
#
# Uses mkdir(2) atomicity on POSIX filesystems (Lustre, NFS, ext4).
# Two operations: acquire / release
#
# Usage:
#   dag_lock.sh acquire <agent-id> [timeout_seconds]
#   dag_lock.sh release <agent-id>
#   dag_lock.sh status
#
# The lock prevents two parallel orchestrator agents from editing
# singlify/state/dag.md simultaneously. Each agent must acquire the
# lock before modifying state files and release after.
#
# If the lock is stale (>300 seconds old), it is automatically broken.

set -euo pipefail

LOCK_DIR="/mnt/home/debruinz/Singlet-AI/singlify/state/.dag_lock"
STALE_THRESHOLD=300  # seconds before auto-break

acquire() {
    local agent_id="${1:?agent-id required}"
    local timeout="${2:-30}"
    local elapsed=0

    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        # Check for stale lock
        if [ -f "$LOCK_DIR/owner" ]; then
            local lock_ts
            lock_ts=$(stat -c %Y "$LOCK_DIR/owner" 2>/dev/null || echo 0)
            local now_ts
            now_ts=$(date +%s)
            local age=$(( now_ts - lock_ts ))
            if [ "$age" -gt "$STALE_THRESHOLD" ]; then
                local old_owner
                old_owner=$(cat "$LOCK_DIR/owner" 2>/dev/null || echo "unknown")
                echo "[dag_lock] Breaking stale lock (age=${age}s, owner=${old_owner})" >&2
                rm -rf "$LOCK_DIR"
                continue
            fi
            local holder
            holder=$(cat "$LOCK_DIR/owner" 2>/dev/null || echo "unknown")
            echo "[dag_lock] Waiting for lock (held by ${holder}, age=${age}s)..." >&2
        fi
        sleep 1
        elapsed=$((elapsed + 1))
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "[dag_lock] TIMEOUT after ${timeout}s — lock not acquired" >&2
            return 1
        fi
    done

    echo "${agent_id} $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_DIR/owner"
    echo "[dag_lock] Acquired by ${agent_id}" >&2
    return 0
}

release() {
    local agent_id="${1:?agent-id required}"
    if [ ! -d "$LOCK_DIR" ]; then
        echo "[dag_lock] No lock to release" >&2
        return 0
    fi
    local owner
    owner=$(awk '{print $1}' "$LOCK_DIR/owner" 2>/dev/null || echo "")
    if [ "$owner" != "$agent_id" ] && [ -n "$owner" ]; then
        echo "[dag_lock] WARNING: Lock owned by '${owner}', not '${agent_id}' — releasing anyway" >&2
    fi
    rm -rf "$LOCK_DIR"
    echo "[dag_lock] Released by ${agent_id}" >&2
    return 0
}

status() {
    if [ -d "$LOCK_DIR" ] && [ -f "$LOCK_DIR/owner" ]; then
        local owner ts
        owner=$(awk '{print $1}' "$LOCK_DIR/owner" 2>/dev/null || echo "unknown")
        ts=$(awk '{print $2}' "$LOCK_DIR/owner" 2>/dev/null || echo "unknown")
        local lock_ts now_ts age
        lock_ts=$(stat -c %Y "$LOCK_DIR/owner" 2>/dev/null || echo 0)
        now_ts=$(date +%s)
        age=$(( now_ts - lock_ts ))
        echo "LOCKED by=${owner} since=${ts} age=${age}s"
    else
        echo "UNLOCKED"
    fi
}

case "${1:-}" in
    acquire) acquire "${2:-}" "${3:-30}" ;;
    release) release "${2:-}" ;;
    status)  status ;;
    *)
        echo "Usage: $0 {acquire|release|status} [agent-id] [timeout]" >&2
        exit 1
        ;;
esac
