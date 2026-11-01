#!/bin/bash
#SBATCH --job-name=s14-votegate
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Real-data gate of the per-sample protocol vote: the production fetch_worker (head screen + decode) on GSM4987144 (4 runs; SRR13296299 is called 10x-3p-v2, the other three
# 10x-5p-v2; GEO: 5p v1). Passes when the manifest has protocol_vote {status voted, winner 10x-5p-v2}, every kept run decodes, and all four .1fq headers carry the same protocol.
# Env: BRANCH (default fix/protocol-vote), GSM / RUNS / READS for another sample, NOHEAD=1 runs the fetch worker without --head-screen (the vote then probes 5000 spots of each run), OLD=1 uses the pre-vote worker (prod/campaign).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
BIN=${BIN:-$W/jobs/s14/bin_r1new/singlet-cpu}; WL=$W/pipeline-benchmark/whitelists
T=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
BR=${BRANCH:-fix/protocol-vote}; [ -n "${OLD:-}" ] && BR=prod/campaign
git clone -q --depth 1 -b $BR git@github.com:Singlet-Bio/singlet.git $T/repo 2>&1 | tail -1; (cd $T/repo && git log --oneline -1)
SRC=$T/repo/scripts/atlas_rerun; ST=$T/stage
GSM=${GSM:-GSM4987144}; RUNS=${RUNS:-SRR13296299,SRR13296300,SRR13296301,SRR13296302}; READS=${READS:-77297667,74867509,79590901,80111048}
python3 - $SRC $ST <<PY
import json, os, sys
sys.path.insert(0, sys.argv[1]); import stagequeue as Q
Q.init(sys.argv[2])
json.dump({"gsm": "$GSM", "gse": "GSE163781", "organism": "Homo sapiens; Mus musculus", "runs": "$RUNS".split(","), "run_reads": [int(x) for x in "$READS".split(",")]},
          open(sys.argv[2] + "/todo/$GSM.json", "w"))
PY
python3 -u $SRC/fetch_worker.py --stage $ST --binary $BIN --whitelist-dir $WL ${NOHEAD:+} $([ -z "${NOHEAD:-}" ] && echo --head-screen) --scratch $T --cap-tb 1 --room-need-gb 1 --streams 16 --decode-threads 4 \
    --idle-exit-min 0.2 --max-samples 1 2>&1 | grep -a -v "^\[fetch\].*MB/s" | cut -c1-250 | tail -30
echo "== manifest"
python3 - $ST <<'PY'
import glob, json, sys
for m in glob.glob(sys.argv[1] + "/ready/*/manifest.json"):
    d = json.load(open(m)); print(m.split("/")[-2], "runs", d["runs"], "pairs", d["pairs"], "format", d["format"]); print(json.dumps(d.get("protocol_vote"), indent=1)[:1500])
PY
echo "== run headers"
for f in $ST/ready/*/*.1fq; do echo "$(basename $f): $($BIN decode $f --header 2>&1 | grep -a -o 'Protocol: [^ ,]*' | head -1)  $($BIN decode $f --header 2>&1 | grep -a -o 'Reads: [0-9]*' | head -1)"; done
