#!/bin/bash
# mkcells.sh <singlet out dir> <truth cells> <out> [column]: Singlet's called cells, then truth cells it did not call
# (barcodes, no -1). column: cell_calls.tsv column of the call to use, 4 = called_v2 (default, the final call),
# 3 = the knee call (K562/BT142 mix: called_v2 adds 123,744 ambient barcodes, median 312 UMIs, there).
set -euo pipefail
COL=${4:-4}
awk -F'\t' -v c=$COL 'NR>1 && $c==1 {sub(/-.*/,"",$1); print $1}' $1/cell_calls.tsv > $3
grep -vxF -f $3 $2 >> $3 || true
wc -l < $3
