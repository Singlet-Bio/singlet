#!/bin/bash
#SBATCH --job-name=s9-stagescale
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9: singlet-stage on ~45 real runs (the todo list of the earlier fetch-scale run 793624): per run fetch (global rate cap 600 MB/s) -> .1fq -> .1fg2,
# CPU of each step, B/read of .1fq vs .1fg2, unique-sequence fraction; --verify on the first 6. Results: $WORK/jobs/s9/stagescale_<job>/results.jsonl
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; SRC=$W/jobs/s9/src
BRANCH=${BRANCH:-claude/stage-1fg2}; N=${N:-45}; P=${P:-8}
T=$W/jobs/s9/stagescale_$SLURM_JOB_ID; mkdir -p $T/cap; B=/tmp/$USER-ss-$SLURM_JOB_ID; mkdir -p $B; trap 'rm -rf $B' EXIT
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
cd $B && git clone -q --depth 1 -b $BRANCH git@github.com:Singlet-Bio/singlet.git repo 2>&1 | tail -2
cd $B/repo && git log --oneline -1
ZI=$W/tools/dl/include; ZL=$W/tools/dl/lib
g++ -O3 -march=x86-64-v3 -msse4.2 -std=c++17 -pthread -Iinclude -Isrc/stage -I$ZI src/stage/singlet_stage.cpp -o $B/singlet-stage -L$ZL -Wl,-rpath,$ZL -lzstd -lz || exit 1
python3 -c "import sys; sys.path.insert(0,'$SRC'); import ratecap; ratecap.write_cap_mbps('$T/cap', 600)"
python3 - <<PY
import json
runs = [json.loads(l)["runs"][0] for l in open("$W/jobs/s9/scale_793624/todo.jsonl")][:$N]
open("$T/runs.txt", "w").write("\n".join(runs) + "\n")
print(len(runs), "runs")
PY
i=0
cat $T/runs.txt | while read R; do i=$((i+1)); V=""; [ $i -le 6 ] && V="--verify"; echo "$R${V:+ $V}"; done | \
  xargs -P $P -L1 sh -c 'python3 '$SRC'/stage_one.py $0 --stage-bin '$B'/singlet-stage --singlet '$W'/release-gate/bin/d9834b8a/singlet-cpu --whitelist-dir '$SINGLET_WHITELIST_DIR' --scratch '$B' --cap-stage '$T'/cap $1' > $T/results.jsonl 2> $T/errors.log
echo "== done $(date -Is) $(wc -l < $T/results.jsonl) results"
