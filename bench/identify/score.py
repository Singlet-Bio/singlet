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
def ptag(row):
    """registry id of a verdict; summary files written before record version 2 only have the internal tag"""
    return row["protocol_tag"] if "protocol_tag" in row else row["protocol"]


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
        got = ptag(s)
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
def has_bc_read(jpath):
    # a deposit has a barcode read when some non-index segment is 40 bases or shorter (10x barcode reads are 24 to 28)
    try:
        j = json.load(open(jpath))
    except Exception:
        return True
    return any(l["len_mode"] <= 40 and not l["index_like"] for l in j["layout"])
nb_n = nb_ok = 0
for r in panel:
    if r["id"] in res and r["protocol"].startswith("10x-") and r["family"] in ("droplet_3p", "droplet_5p", "multiome_gex"):
        sres, jp = res[r["id"]]
        if has_bc_read(jp):
            nb_n += 1
            got = ptag(sres); want = r["protocol"]
            nb_ok += sres["verdict"] == r["family"] and (got == want or (want in ("10x-3p", "10x-5p") and got.startswith(want)))
P("\nFamily correct: %d of %d. 10x family and protocol correct: %d of %d; among the %d 10x rows whose deposit holds a barcode read (a non-index segment of 40 bases or fewer): %d." % (fam_ok, n, tenx_ok, tenx_n, nb_n, nb_ok))
pm = Counter()
for r in panel:
    if r["id"] in res and res[r["id"]][0]["verdict"] == r["family"] and not r["protocol"].startswith("10x-"):
        pm[(r["family"], ptag(res[r["id"]][0]) == r["protocol"])] += 1
P("Protocol tag matches inside correct non-10x families: " + ", ".join("%s %s %d" % (k[0], "tag right" if k[1] else "tag differs", v) for k, v in sorted(pm.items())) + ".")
P(f"Runs labeled ATAC, multiome ATAC, feature barcode, bulk or genomic that received `process`: {len(bad_action)}" + (" (" + ", ".join(bad_action) + ")" if bad_action else "") + ".\n")
# 10x GEX family and version over the whole panel: every non-10x row is a potential false positive
TENX = ("droplet_3p", "droplet_5p")
def vcall(v): return v["verdict"] in TENX
def vlabel(r): return r["family"] in TENX
tp = sum(1 for r in panel if r["id"] in res and vlabel(r) and vcall(res[r["id"]][0]))
fp = [r["id"] for r in panel if r["id"] in res and not vlabel(r) and vcall(res[r["id"]][0])]
fn = [r["id"] for r in panel if r["id"] in res and vlabel(r) and not vcall(res[r["id"]][0])]
ver_ok = sum(1 for r in panel if r["id"] in res and vlabel(r) and vcall(res[r["id"]][0]) and r["protocol"].startswith("10x-3p-v" if r["family"] == "droplet_3p" else "10x-5p-v") and ptag(res[r["id"]][0]) == r["protocol"])
ver_n = sum(1 for r in panel if r["id"] in res and vlabel(r) and r["protocol"][-2:-1] == "v")
ver_pred = sum(1 for r in panel if r["id"] in res and vcall(res[r["id"]][0]) and r["protocol"][-2:-1] == "v" and r["family"] in TENX)
P("## 10x 3' and 5' (v1 to v4), whole panel\n")
P(f"Family: {tp} of {tp + len(fn)} labeled 10x 3'/5' rows called 10x 3'/5' (recall {tp / max(1, tp + len(fn)):.2f}); {tp} of {tp + len(fp)} calls are labeled 10x 3'/5' (precision {tp / max(1, tp + len(fp)):.2f}); false positives from the other {sum(1 for r in panel if r['id'] in res and not vlabel(r))} rows: {', '.join(fp) or 'none'}.")
P(f"Version (rows with a stated version): {ver_ok} of {ver_n} exact (recall {ver_ok / max(1, ver_n):.2f}); of {ver_pred} calls on those rows, {ver_ok} carry the labeled version (precision {ver_ok / max(1, ver_pred):.2f}).")
P("Not called: " + (", ".join("%s (%s)" % (i, res[i][0]["verdict"]) for i in fn) or "none") + ".")
# the three bars
nonproc = [r["id"] for r in panel if r["id"] in res and r["family"] in ("atac", "multiome_atac", "feature_barcode", "bulk_rna", "genomic_dna") and res[r["id"]][0]["action"] == "process"]
cell_fams = {"droplet_3p", "droplet_5p", "droplet_other", "multiome_gex", "plate_barcoded", "smartseq", "visium"}
refused = [(r["id"], res[r["id"]][0]["verdict"]) for r in panel if r["id"] in res and r["family"] in cell_fams and res[r["id"]][0]["action"] == "refuse"]
nonten_proc = [r["id"] for r in panel if r["id"] in res and r["family"] not in TENX + ("multiome_gex", "visium") and (vcall(res[r["id"]][0]) or res[r["id"]][0]["action"] == "process")]
P("\nBars: ATAC, feature-barcode, bulk or genomic rows given `process`: %d. Single-cell rows given `refuse`: %d (%s). Rows of other protocols given `process` or called 10x: %d (%s).\n" % (len(nonproc), len(refused), ", ".join("%s as %s" % x for x in refused) or "none", len(nonten_proc), ", ".join(nonten_proc) or "none"))
brm = [i for i, v in res.items() if v[0]["verdict"] == "barcode_read_missing"]
if brm:
    P("## barcode_read_missing\n")
    P("%d rows. Endpoint that carries a barcode read (from the endpoints record):\n" % len(brm))
    for i in sorted(brm):
        try:
            j = json.load(open(res[i][1]))
            eps = [e["endpoint"] for e in j.get("endpoints", []) if e.get("barcode_read_present")]
            seen = [e["endpoint"] for e in j.get("endpoints", [])]
            P("- `%s`: %s" % (i, ", ".join(eps) if eps else ("none of " + ", ".join(seen) if seen else "endpoints not surveyed")))
        except Exception as e:
            P("- `%s`: no json (%s)" % (i, e))
    P("")
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
    P(f"- `{r['id']}` label {r['family']}/{r['protocol']}; verdict {s['verdict']}/{ptag(s)} ({s['action']}). {why}")
if no_result:
    P("\n## Rows without a result\n\n" + ", ".join(no_result))
