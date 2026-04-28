#!/bin/bash
# genome_load.sh — Pre-load/unload STAR genome indexes into POSIX shared memory.
#
# Usage:
#   ./genome_load.sh load    [human] [mouse]
#   ./genome_load.sh unload  [human] [mouse]
#   ./genome_load.sh status
#
# When genomes are loaded, all singlify jobs on this node share one copy of the
# 32GB genome SA array, cutting per-concurrent-job memory from ~80GB to ~48GB.
#
# Workflow:
#   1. ssh c001 'bash genome_load.sh load human mouse'
#   2. sbatch --array=2-178%20 val1_process.sh   # now runs 20 at a time safely
#   3. (after all jobs done) ssh c001 'bash genome_load.sh unload human mouse'

set -euo pipefail

source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}

SINGLIFY="/mnt/home/debruinz/Singlet-AI/singlify/build/singlify"
REFBASE="/mnt/projects/debruinz_project/cellarium/reference"

H38="${REFBASE}/GRCh38-2024-A/star_2.7.11b"
M39="${REFBASE}/GRCm39-2024-A/star_2.7.11b"

OP="${1:-status}"
shift 2>/dev/null || true
TARGETS="${*:-human mouse}"

load_genome() {
    local name="$1" dir="$2"
    echo "=== Loading $name genome ==="
    "$SINGLIFY" genome load "$dir" 2>&1
}

unload_genome() {
    local name="$1" dir="$2"
    echo "=== Unloading $name genome ==="
    "$SINGLIFY" genome unload "$dir" 2>&1
}

status_genome() {
    local name="$1" dir="$2"
    echo -n "  $name: "
    "$SINGLIFY" genome status "$dir" 2>&1 | grep -E "status =" || echo "NOT loaded"
}

case "$OP" in
    load)
        [[ "$TARGETS" == *human* ]] && load_genome "GRCh38" "$H38"
        [[ "$TARGETS" == *mouse* ]] && load_genome "GRCm39" "$M39"
        echo "=== Shared memory after load ==="
        ipcs -m | tail -n +4 | head -10
        ;;
    unload)
        [[ "$TARGETS" == *human* ]] && unload_genome "GRCh38" "$H38"
        [[ "$TARGETS" == *mouse* ]] && unload_genome "GRCm39" "$M39"
        echo "=== Shared memory after unload ==="
        ipcs -m | tail -n +4 | head -10
        ;;
    status)
        echo "=== Genome shared memory status ==="
        status_genome "GRCh38 (human)" "$H38"
        status_genome "GRCm39 (mouse)" "$M39"
        echo ""
        echo "=== All shared memory segments ==="
        ipcs -m
        ;;
    *)
        echo "Usage: $0 load|unload|status [human] [mouse]"
        exit 1
        ;;
esac
