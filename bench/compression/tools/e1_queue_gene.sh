#!/bin/bash
# gene-resolution pass: tools/e1_queue_gene.sh <log> sample:member ...   (cap 2000, secondary measurement)
cd "$(dirname "$0")/.." || exit 1
log=$1; shift
bash tools/e1_run_all.sh gene 2000 "$log" "$@"
