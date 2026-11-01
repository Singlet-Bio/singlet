#!/bin/bash
#SBATCH --job-name=s9-multirun
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9: multi-run samples. ENA: human 10x-like experiments with 2-3 runs of 8-35 M reads each (N_SAMPLES of them); per sample the two-step path (one pooled
# singlet-stage over the .1fq of every run) vs per-run parts + singlet-stage --merge-parts (stage_multirun_one.py). Needs singlet-cpu/-stage from build_singlet_stage.sbatch.
set -uo pipefail
# the engine must be an immutable, identified build: <dir>-<commit>/bin/<binary> (or <dir>/<binary>) beside a `commit` file starting with that commit (never a shared mutable directory)
pin() { local r d c; r=$(readlink -f "$1"); d=$(dirname "$r"); [ "$(basename $d)" = bin ] && d=$(dirname $d); c=$(basename $d); c=${c##*[-_]}; [[ $c =~ ^[0-9a-f]{7,40}$ ]] && [ -x "$r" ] && case "$(cat $d/commit 2>/dev/null)" in $c*) return 0;; esac; echo "ABORT: $1 is not a pinned build (<dir>-<commit>/bin/<binary> with a commit file starting with the commit)"; exit 3; }
W=/mnt/projects/debruinz_project/debruinz/singlet; SRC=$W/jobs/s9/src; BIN=${BIN:?set BIN=$W/builds/<commit8>/bin/<binary> (an immutable build from scripts/engine_gpu/build_clipper.sbatch)}; pin "$BIN/singlet-cpu"
N=${N_SAMPLES:-5}
T=$W/jobs/s9/multirun_$SLURM_JOB_ID; mkdir -p $T; B=/tmp/$USER-mr-$SLURM_JOB_ID; mkdir -p $B; trap 'rm -rf $B' EXIT
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
cd $B && git clone -q --depth 1 -b ${BRANCH:-claude/stage-1fg2} git@github.com:Singlet-Bio/singlet.git repo 2>&1 | tail -1
cd $B/repo && git log --oneline -1
ZI=$W/tools/dl/include; ZL=$W/tools/dl/lib
g++ -O3 -march=x86-64-v3 -msse4.2 -std=c++17 -pthread -Iinclude -Isrc/stage -I$ZI src/stage/singlet_stage.cpp -o $B/singlet-stage -L$ZL -Wl,-rpath,$ZL -lzstd -lz || exit 1
python3 - <<PY > $T/samples.txt
import json, sys, urllib.parse, urllib.request, collections
q = urllib.parse.quote('library_source="TRANSCRIPTOMIC SINGLE CELL" AND instrument_platform="ILLUMINA" AND library_strategy="RNA-Seq" AND tax_id=9606')
rows = json.load(urllib.request.urlopen("https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&query=%s&fields=run_accession,experiment_accession,read_count&format=json&limit=6000" % q, timeout=120))
by = collections.defaultdict(list)
for r in rows:
    if r.get("read_count") and 8e6 <= int(r["read_count"]) <= 35e6:
        by[r["experiment_accession"]].append(r["run_accession"])
# prefer experiments whose runs are all in the size window and that have 2-3 runs
exp = sorted((e for e, rs in by.items() if 2 <= len(rs) <= 3), key=lambda e: by[e][0])
step = max(1, len(exp) // ($N * 3))
for e in exp[::step][:$N * 3]:
    print(e, ",".join(sorted(by[e])))
PY
head -$((N*3)) $T/samples.txt > $T/cand.txt; echo "candidates: $(wc -l < $T/cand.txt)"
DONE=0
while read E RUNS; do
  [ $DONE -ge $N ] && break
  OK=1
  for R in $(echo $RUNS | tr ',' ' '); do
    python3 -c "
import sys
sys.path.insert(0, '$SRC')
import fetch_worker as F
F.fetch_run('$R', '$B', 16, lambda m: None)
" || OK=0
  done
  if [ $OK = 1 ]; then
    out=$(python3 $SRC/stage_multirun_one.py $E $RUNS $B $BIN $SINGLET_WHITELIST_DIR $B/singlet-stage)
    echo "$out" | tee -a $T/results.jsonl
    echo "$out" | grep -q '"skipped"' || DONE=$((DONE+1))
  fi
  rm -f $B/*.sra
done < $T/cand.txt
echo "== done $DONE samples"
