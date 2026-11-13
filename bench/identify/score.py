#!/usr/bin/env python3
"""Score `singlet identify` on a panel: confusion matrix by family, 10x protocol accuracy, action check, every miss with its evidence.

usage: score.py PANEL.tsv RESULT_DIR [OUT.md]     (RESULT_DIR holds summary.tsv files from the batch parts, plus <id>.json per sample)
"""
import csv, glob, json, os, sys
from collections import Counter, defaultdict

panel = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
res = {}
for f in glob.glob(os.path.join(sys.argv[2], "**", "summary.tsv"), recursive=True):
    for r in csv.DictReader(open(f), delimiter="\t"):
        res[r["id"]] = (r, os.path.join(os.path.dirname(f), r["id"] + ".json"))
out = sys.stdout if len(sys.argv) < 4 else open(sys.argv[3], "w")
P = lambda *a: print(*a, file=out)

fams = sorted({r["family"] for r in panel} | {v[0]["verdict"] for v in res.values()})
cm = defaultdict(Counter)
misses, no_result, bad_action = [], [], []
tenx_n = tenx_ok = fam_ok = n = 0
for r in panel:
    if r["id"] not in res:
        no_result.append(r["id"]); continue
    s, jpath = res[r["id"]]
    n += 1
    cm[r["family"]][s["verdict"]] += 1
    ok = s["verdict"] == r["family"]
    fam_ok += ok
    proto_ok = True
    if r["protocol"].startswith("10x-") and r["family"] in ("droplet_3p", "droplet_5p", "multiome_gex"):
        tenx_n += 1
        want = r["protocol"]
        got = s["protocol"]
        proto_ok = got == want or (want in ("10x-3p", "10x-5p") and got.startswith(want))
        tenx_ok += ok and proto_ok
    if not (ok and proto_ok):
        misses.append((r, s, jpath))
    if r["family"] in ("atac", "multiome_atac", "feature_barcode", "bulk_rna", "genomic_dna") and s["action"] == "process":
        bad_action.append(r["id"])

P("# Identify development panel: results\n")
P(f"Scored {n} of {len(panel)} panel rows ({len(no_result)} without a result).\n")
P("## Confusion matrix (rows: label family, columns: verdict)\n")
P("| label \ verdict | " + " | ".join(fams) + " | total | correct |")
P("|---|" + "---|" * (len(fams) + 2))
for f in fams:
    if f not in cm: continue
    tot = sum(cm[f].values())
    P(f"| {f} | " + " | ".join(str(cm[f][g]) if cm[f][g] else "" for g in fams) + f" | {tot} | {cm[f][f]} |")
P(f"\nFamily correct: {fam_ok} of {n}. 10x family and protocol correct: {tenx_ok} of {tenx_n}.")
P(f"Runs labeled ATAC, multiome ATAC, feature barcode, bulk or genomic that received `process`: {len(bad_action)}" + (" (" + ", ".join(bad_action) + ")" if bad_action else "") + ".\n")
act = Counter((r["family"], res[r["id"]][0]["action"]) for r in panel if r["id"] in res)
P("## Actions by label family\n")
P("| label family | " + " | ".join(["process", "classify_only", "defer", "refuse"]) + " |\n|---|---|---|---|---|")
for f in sorted({k[0] for k in act}):
    P(f"| {f} | " + " | ".join(str(act[(f, a)]) for a in ["process", "classify_only", "defer", "refuse"]) + " |")
P("\n## Misses\n")
for r, s, jpath in misses:
    why = ""
    try:
        j = json.load(open(jpath))
        top = j["interpretations"][:2]
        why = "; ".join(f"{t['family']}/{t['protocol']} support {t['support']}: " + " / ".join(t["reasons"][:2]) for t in top)
        lists = ", ".join(f"{l['family']} {l['one_mismatch']:.2f}" for l in j["lists"] if l["one_mismatch"] > 0.05)
        why += f" | lists>5%: {lists or 'none'} | layout " + ",".join(str(l["len_mode"]) for l in j["layout"])
        why += f" | spots {j['source']['spots']} route {j['source']['route']} err {j['source']['error']}"
    except Exception as e:
        why = f"(no json: {e}) error: {s.get('error','')}"
    P(f"- `{r['id']}` label {r['family']}/{r['protocol']}; verdict {s['verdict']}/{s['protocol']} ({s['action']}). {why}")
if no_result:
    P("\n## Rows without a result\n\n" + ", ".join(no_result))
