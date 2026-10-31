#!/bin/bash
#SBATCH --job-name=s9-combined
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9: per run, the three fetch-tier paths from the same local .sra (stage_combined_one.py): A baseline (SRA -> sorted .1fq), B two-step (+ singlet-stage),
# C combined (singlet download --stage-out, no .1fq, no sort). CPU of each, bytes B vs C (barcode-sorted vs SRA-order source), digest equality of B and C.
# Env: RUNS (SRR list), BRANCH. singlet-cpu from build_singlet_stage.sbatch ($WORK/jobs/s9/bin); singlet-stage is rebuilt here from the branch (has --digest).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; SRC=$W/jobs/s9/src; BIN=$W/jobs/s9/bin
RUNS=${RUNS:-"SRR19421372 SRR19573735 SRR30477995"}
T=$W/jobs/s9/combined_$SLURM_JOB_ID; mkdir -p $T; B=/tmp/$USER-cmb-$SLURM_JOB_ID; mkdir -p $B; trap 'rm -rf $B' EXIT
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
cd $B && git clone -q --depth 1 -b ${BRANCH:-claude/stage-1fg2} git@github.com:Singlet-Bio/singlet.git repo 2>&1 | tail -1
cd $B/repo && git log --oneline -1
ZI=$W/tools/dl/include; ZL=$W/tools/dl/lib
g++ -O3 -march=x86-64-v3 -msse4.2 -std=c++17 -pthread -Iinclude -Isrc/stage -I$ZI src/stage/singlet_stage.cpp -o $B/singlet-stage -L$ZL -Wl,-rpath,$ZL -lzstd -lz || exit 1
for R in $RUNS; do
  python3 -c "
import sys
sys.path.insert(0, '$SRC')
import fetch_worker as F
F.fetch_run('$R', '$B', 16, print)
"
  python3 $SRC/stage_combined_one.py $R $B $BIN $SINGLET_WHITELIST_DIR $B/singlet-stage | tee -a $T/results.jsonl
  rm -f $B/$R.sra
done
echo "== done"
