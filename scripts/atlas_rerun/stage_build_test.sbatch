#!/bin/bash
#SBATCH --job-name=s9-stagebuild
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9: build singlet-stage from the branch, then stage + --verify real runs (full .sra from S3 -> .1fq with binned qualities -> .1fg2) and report core-s/M, B/pair, unique fraction.
# Env: BRANCH (default claude/stage-1fg2), RUNS (space separated SRR list). Nothing is kept: all data in node /tmp.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; SRC=$W/jobs/s9/src
BRANCH=${BRANCH:-claude/stage-1fg2}
RUNS=${RUNS:-"SRR19421372 SRR19573735 SRR30477995"}
T=/tmp/$USER-stage-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
BIN=$W/release-gate/bin/d9834b8a/singlet-cpu
cd $T && git clone -q --depth 1 -b $BRANCH git@github.com:Singlet-Bio/singlet.git repo 2>&1 | tail -2
cd $T/repo && git log --oneline -1
ZI=$W/tools/dl/include; ZL=$W/tools/dl/lib
echo "== build"; g++ -O3 -march=x86-64-v3 -msse4.2 -std=c++17 -pthread -Iinclude -Isrc/stage -I$ZI src/stage/singlet_stage.cpp -o $T/singlet-stage -L$ZL -Wl,-rpath,$ZL -lzstd -lz 2>&1 | head -30
ls -l $T/singlet-stage || exit 1
for R in $RUNS; do
  echo "== $R"
  python3 - <<PY
import sys, os
sys.path.insert(0, "$SRC")
import fetch_worker as F
p, info = F.fetch_run("$R", "$T", 16, print)
print("fetched", info["bytes"])
PY
  $BIN download $T/$R.sra --quality binned --vdb-threads 2 --no-profile -o $T/$R.1fq 2>&1 | tail -2
  rm -f $T/$R.sra
  ls -l $T/$R.1fq
  /usr/bin/time -v $T/singlet-stage --whitelist-dir $SINGLET_WHITELIST_DIR --accessions $R --scratch $T -o $T/$R.1fg2 --verify $T/$R.1fq 2> $T/time.txt | tee $T/$R.json
  grep -E "verify:|Maximum resident|Elapsed" $T/time.txt
  ls -l $T/$R.1fq $T/$R.1fg2 | awk '{print $5, $9}'
  rm -f $T/$R.1fg2
done
echo "== done"
