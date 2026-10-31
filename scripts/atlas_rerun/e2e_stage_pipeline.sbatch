#!/bin/bash
#SBATCH --job-name=s9-e2e
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9 Phase 2 end-to-end test of the decoupled pipeline on N real small samples, as ONE Slurm job standing in for the three tiers:
#   fetch_worker (background) -> bounded stage (backpressure) -> drain_worker A (killed with -9 mid-sample) -> drain_worker B (reaps A's
#   claim, drains the rest) -> dry-run publish -> staged reads and /tmp deleted. launcher.py --once --dry-run prints its decisions.
# Env: N (samples, default 12), BIN (singlet binary), CAP_GB (stage cap, default 3), QUALITY (default binned).
# Nothing is written to Cloudflare (publish = dry-run); reads and outputs are deleted; test root is $W/jobs/s9/e2e_$SLURM_JOB_ID.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
SRC=${SRC:-$W/jobs/s9/src}          # scripts/atlas_rerun copied here by the submitter (git sync is the normal route)
N=${N:-10}; CAP_GB=${CAP_GB:-3}; QUALITY=${QUALITY:-binned}
BIN=${BIN:-$W/pipeline-benchmark/bin/mc99cpu}
T=$W/jobs/s9/e2e_$SLURM_JOB_ID; STAGE=$T/stage; mkdir -p $T/root $STAGE
export SINGLET_BIN=$BIN S9_HB_STALE_S=45
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
echo "== node $(hostname) $(date -Is) N=$N cap=${CAP_GB}GB quality=$QUALITY bin=$BIN"

# config: the campaign config pointed at the test root
python3 - <<PY
import json
c = json.load(open("$SRC/config.json"))
k = c["clusters"]["clipper"]
k.update(root="$T/root", binary="$BIN", whitelist_dir="$W/pipeline-benchmark/whitelists", scratch="/tmp")
json.dump(c, open("$T/config.json", "w"))
PY

# todo items: N small human 10x-like runs (ENA search; the staged decode drops non-10x runs, which the test then reports)
python3 -c "import sys; sys.path.insert(0,'$SRC'); import stagequeue as Q; Q.init('$STAGE')"
python3 - <<PY
import json, os, urllib.parse, urllib.request
q = urllib.parse.quote('library_source="TRANSCRIPTOMIC SINGLE CELL" AND instrument_platform="ILLUMINA" AND library_strategy="RNA-Seq" AND tax_id=9606')
rows = json.load(urllib.request.urlopen("https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&query=%s&fields=run_accession,read_count,study_accession&format=json&limit=3000" % q, timeout=90))
rows = [r for r in rows if r.get("read_count") and 8e6 < int(r["read_count"]) < 22e6]
rows.sort(key=lambda r: r["run_accession"])
step = max(1, len(rows) // $N)
for i, r in enumerate(rows[::step][:$N]):
    gsm = "RUN%02d_%s" % (i, r["run_accession"])
    json.dump({"gsm": gsm, "gse": r["study_accession"], "organism": "Homo sapiens", "runs": [r["run_accession"]], "reads": int(r["read_count"])},
              open("$STAGE/todo/%s.json" % gsm, "w"))
PY
echo "todo: $(ls $STAGE/todo | wc -l)"

echo "== launcher decision on an empty stage (dry-run)"
python3 $SRC/launcher.py --stage $STAGE --config $T/config.json --once --dry-run --no-successor

echo "== start fetch worker"
python3 -u $SRC/fetch_worker.py --stage $STAGE --cap-tb $(python3 -c "print($CAP_GB/1000)") --room-need-gb 1 --streams 32 --quality $QUALITY \
    --config $T/config.json --idle-exit-min 2 > $T/fetch.log 2>&1 &
FPID=$!
# wait for the first staged sample, then show the launcher's decision with something in the stage
for i in $(seq 1 120); do [ -n "$(ls $STAGE/ready 2>/dev/null)" ] && break; sleep 5; done
python3 $SRC/launcher.py --stage $STAGE --config $T/config.json --once --dry-run --no-successor
du -sh /tmp/s9-fetch-* 2>/dev/null | head -2

echo "== drain worker A (will be killed with -9 while it holds a claim)"
python3 -u $SRC/drain_worker.py --stage $STAGE --engine cpu --config $T/config.json --threads 8 --idle-exit-min 1 > $T/drainA.log 2>&1 &
APID=$!
for i in $(seq 1 120); do [ -n "$(ls $STAGE/claimed 2>/dev/null)" ] && break; sleep 2; done
sleep 8; kill -9 $APID; pkill -9 -P $APID 2>/dev/null; pkill -9 -f "mc99cpu .*--out-prefix $T" 2>/dev/null; pkill -9 -f "mc99cpu .*--out-prefix /tmp/s9-drain" 2>/dev/null
echo "killed A holding: $(ls $STAGE/claimed)"
sleep 50   # past the shortened heartbeat timeout
echo "== drain worker B (reaps A's claim and drains the rest)"
python3 -u $SRC/drain_worker.py --stage $STAGE --engine cpu --config $T/config.json --threads 8 --idle-exit-min 3 > $T/drainB.log 2>&1 &
BPID=$!
wait $FPID; echo "fetch worker exit $?"
wait $BPID; echo "drain B exit $?"

echo "== results"
echo "--- fetch log"; grep -E "\[fetch\]|\[stage\]" $T/fetch.log | cut -c1-200
echo "--- drain A"; grep -E "\[drain\]" $T/drainA.log | cut -c1-200
echo "--- drain B"; grep -E "\[drain\]|reaped" $T/drainB.log | cut -c1-200
echo "--- stage dirs"; for d in todo fetching incoming ready claimed failed deleting; do echo "$d: $(ls $STAGE/$d | wc -l)"; done; ls $STAGE/failed
echo "--- events: done per sample (must be exactly 1 each), failed"
cat $T/root/inbox/*.json 2>/dev/null | python3 -c "
import sys, json, collections
dec = json.JSONDecoder(); s = sys.stdin.read(); i = 0; ev = []
while i < len(s):
    while i < len(s) and s[i].isspace(): i += 1
    if i >= len(s): break
    o, i = dec.raw_decode(s, i); ev.append(o)
c = collections.Counter((e['gsm'], e['event']) for e in ev if 'gsm' in e)
done = {g: n for (g, e), n in c.items() if e == 'done'}
print('done samples', len(done), 'double-done', [g for g, n in done.items() if n > 1], 'failed events', sum(n for (g, e), n in c.items() if e == 'failed'))
for e in ev:
    if e['event'] == 'done': print(e['gsm'], 'pairs', e.get('pairs'), 'compute_s', e.get('compute_s'), 'out_public_bytes', e.get('out_public_bytes'), e.get('publish'))
"
echo "--- manifests of staged samples"
cat $T/fetch.log | grep "\[stage\]" | cut -c1-200
echo "--- leftover node scratch"; ls /tmp | grep -E "^s9-" | head; df -h /tmp | tail -1
echo "== done $(date -Is)"
