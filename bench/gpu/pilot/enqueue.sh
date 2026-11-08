#!/bin/bash
# Copy item scripts into the GPU pilot's queue (bench/gpu/pilot/README.md). Usage: enqueue.sh ITEM.g1.sh|ITEM.g2.sh ...
set -euo pipefail
Q=${QUEUE:-/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-campaign/queue}
mkdir -p $Q/new
for f in "$@"; do
  case $f in *.g1.sh|*.g2.sh) ;; *) echo "!! $f: name must end in .g1.sh or .g2.sh" >&2; exit 1 ;; esac
  bash -n "$f"
  install -m 0755 "$f" "$Q/new/$(basename "$f")"
  echo "queued $(basename "$f")"
done
squeue -h -u "$USER" -n gpu-pilot -o "%i %T %N %L" | sed 's/^/pilot: /' || true
