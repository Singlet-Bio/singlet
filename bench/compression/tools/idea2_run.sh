#!/usr/bin/env bash
# usage: tools/idea2_run.sh "ENV=val ENV2=val" sample member [codec]   -> prints one summary line (encode only, reps 0)
cd "$(dirname "$0")/.."
envs="$1"; sample="$2"; member="$3"; codec="${4:-idea2_env}"
out=$(env $envs ./bin/idea2_bench.exe --no-file-check --reps 0 --codec $codec --out results/idea2_sweep_tmp.tsv corpus/$sample/${member}_counts.1pz 2>&1)
c=$(echo "$out" | grep "cells:" | sed 's/.*TOTAL/TOTAL/')
top=$(echo "$out" | grep "cells:" | grep -o "cell_top [0-9]* B ([0-9.]* b/nnz)" )
tail=$(echo "$out" | grep "cells:" | grep -o "cell_tail [0-9]* B ([0-9.]* b/nnz)" )
val=$(echo "$out" | grep "cells:" | grep -o "cell_val [0-9]* B ([0-9.]* b/nnz)" )
e=$(echo "$out" | grep "empties:" | sed 's/.*TOTAL/TOTAL/')
echo "[$envs] $sample/$member | $top | $tail | $val | cells $c | empt $e"
