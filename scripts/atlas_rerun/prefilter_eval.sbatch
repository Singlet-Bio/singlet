#!/bin/bash
#SBATCH --job-name=s9-prefilter
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9: validate prefilter.py against the real outcomes of the fetch tests (staged vs parked at the decode gate / GEX screen), then measure
# the share of a larger random sample of "single-cell" runs it rejects. Metadata only: no read bytes are downloaded.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; R=$W/jobs/s9; SRC=$R/src; O=$R/prefilter; mkdir -p $O
python3 - <<PY
import glob, json, os, re
labels = {}
for d in ("$R/scale_793624", "$R/e2e_793579"):
    todo = {}
    if os.path.exists(d + "/todo.jsonl"):
        for l in open(d + "/todo.jsonl"):
            j = json.loads(l); todo[j["gsm"]] = j["runs"][0]
    logs = "".join(open(f).read() for f in glob.glob(d + "/fetch*.log"))
    parked = dict(re.findall(r"\] (\S+): (parked_\w+)", logs))
    staged = set(re.findall(r"\] (\S+): ready", logs))
    for g, lab in list(parked.items()) + [(g, "staged") for g in staged]:
        run = todo.get(g) or g.split("_", 1)[-1]
        labels[run] = lab
json.dump(labels, open("$O/labels.json", "w"))
open("$O/runs.txt", "w").write("\n".join(sorted(labels)) + "\n")
print(len(labels), "labelled runs:", {k: sum(1 for v in labels.values() if v == k) for k in set(labels.values())})
PY
python3 $SRC/prefilter.py annotate $O/runs.txt $O/annotated.jsonl
python3 $SRC/prefilter.py eval $O/annotated.jsonl $O/labels.json
echo "== random sample of single-cell RNA-Seq runs (unlabelled): share rejected, by reason"
python3 - <<PY
import json, random, sys, urllib.parse, collections
sys.path.insert(0, "$SRC")
import prefilter as P
q = urllib.parse.quote('library_source="TRANSCRIPTOMIC SINGLE CELL" AND instrument_platform="ILLUMINA" AND library_strategy="RNA-Seq"')
rows = json.loads(P.get("https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&query=%s&fields=run_accession,read_count&format=json&limit=20000" % q))
random.seed(1)
runs = [r["run_accession"] for r in random.sample(rows, 400)]
recs = P.annotate(runs)
c = collections.Counter(); n_acc = 0
for m in recs:
    ok, why = P.accept(m); n_acc += ok; c[why or "accepted"] += 1
print(len(recs), "runs; accepted", n_acc, "(%.0f%%)" % (100 * n_acc / len(recs)))
for k, v in c.most_common(12): print("  ", v, k)
PY
echo "== done"
