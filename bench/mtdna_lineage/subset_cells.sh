#!/bin/bash
# subset_cells.sh <cells.txt> <truth cells> <n> <out>: all truth cells present in cells.txt plus random others, n in total
set -euo pipefail
grep -xF -f $2 $1 > $4 || true
grep -vxF -f $2 $1 | shuf --random-source=<(yes) | head -n $(( $3 - $(wc -l < $4) )) >> $4 || true
wc -l < $4
