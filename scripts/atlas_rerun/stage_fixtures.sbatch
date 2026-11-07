#!/bin/bash
#SBATCH --job-name=s9-fixtures
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9: test fixtures for S8's readers: three real public runs as .1fq (ILLUMINA4 qualities from the full .sra) and .1fg2, kept in $WORK/jobs/s9/fixtures
# (temporary: delete when S8's identity gate is done).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; SRC=$W/jobs/s9/src; F=$W/jobs/s9/fixtures; mkdir -p $F
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
BIN=$W/release-gate/bin/d9834b8a/singlet-cpu
T=/tmp/$USER-fx-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
cd $T && git clone -q --depth 1 -b claude/stage-1fg2 git@github.com:Singlet-Bio/singlet.git repo 2>&1 | tail -2
cd $T/repo && git log --oneline -1
ZI=$W/tools/dl/include; ZL=$W/tools/dl/lib
g++ -O3 -march=x86-64-v3 -msse4.2 -std=c++17 -pthread -Iinclude -Isrc/stage -I$ZI src/stage/singlet_stage.cpp -o $F/singlet-stage -L$ZL -Wl,-rpath,$ZL -lzstd -lz || exit 1
for R in SRR19421372 SRR19573735 SRR30477995; do
  python3 - <<PY
import sys
sys.path.insert(0, "$SRC")
import fetch_worker as F
F.fetch_run("$R", "$T", 16, print)
PY
  $BIN download $T/$R.sra --quality binned --vdb-threads 2 --no-profile -o $F/$R.1fq 2>&1 | tail -1
  rm -f $T/$R.sra
  $F/singlet-stage --whitelist-dir $SINGLET_WHITELIST_DIR --accessions $R --scratch $T -o $F/$R.1fg2 --verify --json $F/$R.stage.json $F/$R.1fq | tail -c 200
done
sha256sum $BIN > $F/BINARY.txt; ls -l $F
