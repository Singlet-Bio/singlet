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
# the engine must be an immutable, identified build: <dir>-<commit>/bin/<binary> (or <dir>/<binary>) beside a `commit` file starting with that commit (never a shared mutable directory)
pin() { local r d c; r=$(readlink -f "$1"); d=$(dirname "$r"); [ "$(basename $d)" = bin ] && d=$(dirname $d); c=$(basename $d); c=${c##*[-_]}; [[ $c =~ ^[0-9a-f]{7,40}$ ]] && [ -x "$r" ] && case "$(cat $d/commit 2>/dev/null)" in $c*) return 0;; esac; echo "ABORT: $1 is not a pinned build (<dir>-<commit>/bin/<binary> with a commit file starting with the commit)"; exit 3; }
W=/mnt/projects/debruinz_project/debruinz/singlet; SRC=$W/jobs/s9/src; BIN=${BIN:?set BIN=$W/builds/<commit8>/bin/<binary> (an immutable build from scripts/engine_gpu/build_clipper.sbatch)}; pin "$BIN/singlet-cpu"
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
