#!/bin/bash
#SBATCH --job-name=s9-headscreen
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9: cost and verdicts of the head screen (1 M pairs by VDB streaming, then screen-runs) on the runs whose real outcome is known
# ($WORK/jobs/s9/prefilter/labels.json: staged / parked_non10x / parked_no_gex). 8 runs in parallel, 2 VDB threads each.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; R=$W/jobs/s9; SRC=$R/src; O=$R/prefilter
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
BIN=$W/pipeline-benchmark/bin/mc99cpu
cat > $O/head_one.py <<PY
import json, os, sys, time
sys.path.insert(0, "$SRC")
import fetch_worker as F
run = sys.argv[1]
scr = "/tmp/s9-head-%s-%d" % (run, os.getpid()); os.makedirs(scr)
v, secs = F.head_screen("$BIN", run, scr, dict(os.environ))
print(json.dumps({"run": run, "verdict": v, "secs": round(secs, 1)}), flush=True)
import shutil; shutil.rmtree(scr, ignore_errors=True)
PY
cut -d'"' -f2 $O/labels.json >/dev/null 2>&1
python3 -c "import json;print('\n'.join(sorted(json.load(open('$O/labels.json')))))" > $O/runs_all.txt
cat $O/runs_all.txt | xargs -P 8 -I{} python3 $O/head_one.py {} > $O/head_results.jsonl 2> $O/head_err.log
python3 - <<PY
import json
lab = json.load(open("$O/labels.json"))
res = [json.loads(l) for l in open("$O/head_results.jsonl")]
import collections
t = collections.defaultdict(list)
for r in res: t[lab[r["run"]]].append(r)
for k, v in t.items():
    print(k, len(v), "verdicts", dict(collections.Counter(x["verdict"] for x in v)), "median s %.0f max %.0f" % (sorted(x["secs"] for x in v)[len(v)//2], max(x["secs"] for x in v)))
PY
echo "== done"
