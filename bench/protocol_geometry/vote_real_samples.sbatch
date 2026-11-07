#!/bin/bash
#SBATCH --job-name=s14-voteReal
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# The merged fetch_worker (prod/campaign, #160) on real multi-run samples whose vote excludes a run or hands one to the companion classifier: fetch -> vote -> stage -> GPU drain.
# Env: GSMS (comma list), BRANCH (default prod/campaign), BIN (singlet-gpu). Nothing is published (dry-run); reads and outputs are deleted at the end except the manifests.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/bpub/bin:$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
BIN=${BIN:-$W/jobs/s14/bin_bbf6497d/singlet-gpu}; GSMS=${GSMS:-GSM6634344,GSM9015940}
T=$W/jobs/s14/voteReal_$SLURM_JOB_ID; STAGE=$T/stage; mkdir -p $T/root $STAGE
git clone -q --depth 1 -b ${BRANCH:-prod/campaign} git@github.com:Singlet-Bio/singlet.git $T/repo 2>&1 | tail -1; (cd $T/repo && git log --oneline -1 | cut -c1-90)
SRC=$T/repo/scripts/atlas_rerun
export SINGLET_BIN=$BIN SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
nvidia-smi --query-gpu=name --format=csv,noheader; ls -la $(dirname $BIN)/../whitelists | head -2
python3 - <<PY
import json
c = json.load(open("$SRC/config.json")); k = c["clusters"]["clipper"]
k.update(root="$T/root", binary="$BIN", whitelist_dir="$W/pipeline-benchmark/whitelists", scratch="/tmp"); c["binary"]["path"] = "$BIN"; c["binary"].pop("sha256", None)
json.dump(c, open("$T/config.json", "w"))
import glob, pandas as pd
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("$W/data/atlas-rerun/runs*.parquet"))], ignore_index=True).drop_duplicates("run_accession")
d["gsm"] = d["gsm"].fillna(d["sample_alias"].where(d["sample_alias"].astype(str).str.startswith("GSM"), d["sample_accession"]))
s = pd.read_parquet("$W/data/atlas-rerun/samples.parquet").set_index("gsm_id")
with open("$T/plan.jsonl", "w") as fh:
    for g in "$GSMS".split(","):
        x = d[(d.gsm == g) & (d.read_count > 0)].sort_values("run_accession")
        fh.write(json.dumps(dict(gsm=g, gse=s.gse_id[g], organism=s.organism[g], runs=list(x.run_accession), run_reads=[int(n) for n in x.read_count], protocol_inferred=s.protocol_inferred[g])) + "\n")
print(open("$T/plan.jsonl").read())
PY
python3 $SRC/stagequeue.py enqueue $STAGE $T/plan.jsonl
echo "== fetch worker"
python3 -u $SRC/fetch_worker.py --stage $STAGE --cap-tb 2 --room-need-gb 1 --streams 32 --quality binned --config $T/config.json --idle-exit-min 3 > $T/fetch.log 2>&1 &
FPID=$!
echo "== drain worker (gpu)"
python3 -u $SRC/drain_worker.py --stage $STAGE --engine gpu --config $T/config.json --threads 12 --idle-exit-min 6 --max-samples 2 --mem-gb 190 > $T/drain.log 2>&1 &
DPID=$!
wait $FPID; echo "fetch exit $?"; wait $DPID; echo "drain exit $?"
echo "--- fetch log"; grep -a -E "\[vote\]|\[fetch\]|\[stage\]|excluded|companion" $T/fetch.log | cut -c1-420
echo "--- drain log"; grep -a -E "\[drain\]|species|pileup|Error|error" $T/drain.log | cut -c1-300 | head -40
for d in todo fetching incoming ready claimed failed; do echo "$d: $(ls $STAGE/$d | tr '\n' ' ')"; done
echo "--- manifests"
for f in $STAGE/ready/*/manifest.json $STAGE/failed/*/manifest.json; do [ -f "$f" ] && { echo "## $f"; python3 -c "
import json,sys
m=json.load(open('$f'))
for k in ('gsm','runs','protocol','protocol_vote','excluded_runs','companion_candidates','protocol_check','pairs','files','attempts'):
    if k in m: print(k, json.dumps(m[k])[:900])
print('keys', sorted(m)[:40])"; ls -la $(dirname $f) | head -12; }; done
echo "--- events"; cat $T/root/inbox/*.json 2>/dev/null | cut -c1-700
echo "== done"
