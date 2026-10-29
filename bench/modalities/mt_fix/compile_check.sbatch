#!/bin/bash
#SBATCH --job-name=mtfix-compile
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Syntax/type check of the edited engine and export headers (export.h, pileup_engine.h) with htslib, and the unit tests.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/modfix-mt; E=$W/envs/pbench; CXX=$E/bin/x86_64-conda-linux-gnu-g++
cd $J && rm -rf inc && mkdir inc && tar xzf inc.tgz -C inc
INC="-I$J/inc/include -I$J/inc/include/singlet/pileup -I$E/include -I$E/include/htslib"
cat > chk_export.cpp <<'EOF'
#include "singlet/pileup/pileup_engine.h"
#include "singlet/pileup/export.h"
int main() { return 0; }
EOF
echo "== syntax check export.h + pileup_engine.h"
$CXX -std=c++20 -fsyntax-only -fopenmp -w $INC chk_export.cpp 2>&1 | head -40
echo "syntax-check exit=${PIPESTATUS[0]}"
echo "== unit tests"
LIBS="-L$E/lib -lzstd -lz -Wl,-rpath,$E/lib -lpthread"
$CXX -std=c++20 -O2 -w $INC test_mt_caller.cpp -o bin_test_mt_caller $LIBS && ./bin_test_mt_caller | tail -3
$CXX -std=c++20 -O2 -w $INC test_mt_events.cpp -o bin_test_mt_events $LIBS && ./bin_test_mt_events | tail -3
