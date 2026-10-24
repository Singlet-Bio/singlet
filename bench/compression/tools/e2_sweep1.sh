#!/usr/bin/env bash
# E2 phase-1 sweeps: ideal code length vs model/schedule parameters (single thread, sequential)
cd "$(dirname "$0")/.."
F="corpus/pbmc_v2_typical/exon_counts.1pz corpus/pbmc_v3_lowgenes/exon_counts.1pz corpus/tumor_3p/exon_counts.1pz"
OUT=results/e2/sweep1.txt
run() { echo "## $*" >> $OUT; ./bin/e2_cost.exe "$@" $F >> $OUT 2>&1; }
run --B 1024
run --B 16 --n0 16 --rho 1.1
run --B 64
run --B 256
run --B 4096
run --whole 1
run --B 1024 --n0 16
run --B 1024 --n0 32
run --B 1024 --n0 128
run --B 1024 --rho 1.5
run --B 1024 --rho 3
run --B 1024 --bpo 2
run --B 1024 --bpo 8
run --B 1024 --gt 0 --apc 0.01
run --B 1024 --gt 0 --apc 0.05
run --B 1024 --gt 0 --apc 0.2
run --B 1024 --mix 0
run --B 1024 --rounds 1
run --B 1024 --rounds 3
run --B 1024 --ctrust 5
run --B 1024 --ctrust 80
run --B 1024 --flat 0.4
run --B 1024 --flat 1.0
run --B 1024 --perm 0
run --B 1024 --ctx 1
run --B 1024 --kappa 5
run --B 1024 --kappa 20
echo DONE >> $OUT
