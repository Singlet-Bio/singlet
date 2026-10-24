#!/usr/bin/env bash
cd "$(dirname "$0")/.."
F="corpus/pbmc_v2_typical/exon_counts.1pz corpus/pbmc_v3_lowgenes/exon_counts.1pz corpus/tumor_3p/exon_counts.1pz"
OUT=results/e2/sweep2.txt
run() { echo "## $*" >> $OUT; ./bin/e2_cost.exe "$@" $F >> $OUT 2>&1; }
run --n0 16 --rounds 1 --mix 0
run --n0 16 --rounds 1 --mix 0 --theta 0.5 --zlim 0
run --n0 16 --rounds 1 --mix 0 --theta 1.0 --zlim 0
run --n0 16 --rounds 1 --mix 0 --theta 0.5 --zlim 2
run --n0 16 --rounds 1 --mix 0 --theta 1.0 --zlim 2
run --n0 16 --rounds 1 --mix 0 --theta 1.0 --zlim 8
run --n0 8 --rho 1.5 --rounds 1 --mix 0
echo DONE >> $OUT
