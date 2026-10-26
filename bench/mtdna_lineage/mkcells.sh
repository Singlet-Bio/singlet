#!/bin/bash
# mkcells.sh <singlet out dir> <truth cells> <out>: Singlet called cells, then truth cells it did not call (barcodes, no -1)
set -euo pipefail
awk -F'\t' 'NR>1 && $3==1 {sub(/-.*/,"",$1); print $1}' $1/cell_calls.tsv > $3
grep -vxF -f $3 $2 >> $3 || true
wc -l < $3
