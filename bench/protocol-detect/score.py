"""Score protocol_detect output against the truth panel, by the rules in REGISTRATION.md.

    python score.py PANEL.tsv OUTDIR [--trial TRIAL.tsv] [--md OUT.md]

OUTDIR holds <id>.json from run_panel.sbatch.  TRIAL.tsv (id, chosen_tag, ...) holds the trial-mapping referee's pick for rows
whose decision was "trial"; without it those rows count as ambiguous.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

TENX_LISTS = ("v1", "v2", "v3", "v4_3p", "v4_5p", "arc_gex", "visium")
FAM_OF_TAG = {"10x-3p-v1": "10x_3p", "10x-3p-v2": "10x_3p", "10x-3p-v3": "10x_3p", "10x-3p-v4": "10x_3p",
              "10x-5p-v2": "10x_5p", "10x-5p-v3": "10x_5p", "10x-5p-v4": "10x_5p", "10x-arc-gex": "10x_arc_gex",
              "10x-visium": "visium"}
VERSIONED = {"10x_3p", "10x_5p", "10x_arc_gex", "visium"}


def best10(res: dict) -> float:
    return max((s["lists"].get(n, {}).get("rate", 0.0) for s in res.get("slots", []) for n in TENX_LISTS), default=0.0)


def predict(res: dict, trial: dict | None):
    """(family, version, protocol, how)."""
    d = res.get("decision") or {}
    top = d.get("top") or {}
    if d.get("status") in ("clear", "unsupported"):
        fam = top.get("family", "unknown")
        # the version is the protocol tag (an engine tag for the versioned families), whether or not the layout lets the engine run it
        return fam, (top.get("protocol") or "") if fam in VERSIONED else "", top.get("protocol", ""), "detector"
    if d.get("status") == "trial":
        if trial and trial.get("chosen_tag"):
            tag = trial["chosen_tag"]
            return FAM_OF_TAG.get(tag, "unknown"), tag, tag, "trial"
        return "ambiguous", "", "", "trial pending"
    return "unknown", "", "", d.get("status", "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("outdir")
    ap.add_argument("--trial", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--rerank", action="store_true", help="rank again from the saved slot evidence with the current protocol_detect.py")
    a = ap.parse_args()
    if a.rerank:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import protocol_detect as pdet
    rows = list(csv.DictReader(open(a.panel), delimiter="\t"))
    trial = {}
    if a.trial and os.path.isfile(a.trial):
        trial = {r["id"]: r for r in csv.DictReader(open(a.trial), delimiter="\t")}
    lines, recs = [], []
    for r in rows:
        p = os.path.join(a.outdir, r["id"] + ".json")
        res = json.load(open(p)) if os.path.isfile(p) else {"error": "no output"}
        if a.rerank and "slots" in res:
            slots = [pdet.Slot(**sd) for sd in res["slots"]]
            cands = pdet.rank(slots)
            res["candidates"] = [c.to_dict() for c in cands]
            res["decision"] = pdet.decide(cands, "")
        fam_t, ver_t = r["family"], r["version"]
        relabeled = False
        if "error" not in res and r.get("deposit_check") == "barcode_read" and best10(res) < 0.05:
            fam_t, ver_t, relabeled = "no_cell_barcode", "", True
        if "error" in res:
            fam_p, ver_p, proto_p, how = "error", "", "", res["error"]
        else:
            fam_p, ver_p, proto_p, how = predict(res, trial.get(r["id"]))
        recs.append(dict(id=r["id"], fam_t=fam_t, ver_t=ver_t, fam_p=fam_p, ver_p=ver_p, proto_t=r["protocol"], proto_p=proto_p,
                         how=how, relabeled=relabeled, res=res))
    n = len(recs)
    fam_ok = sum(x["fam_t"] == x["fam_p"] for x in recs)
    calls = [x for x in recs if x["fam_p"] not in ("ambiguous", "unknown", "error")]
    prec_ok = sum(x["fam_t"] == x["fam_p"] for x in calls)
    vrows = [x for x in recs if x["ver_t"]]
    ver_ok = sum(x["ver_t"] == x["ver_p"] for x in vrows)
    out = [f"# Protocol detection score: {os.path.basename(a.panel)}", "",
           f"Rows {n}. Family recall {fam_ok}/{n} = {fam_ok / n:.3f}. Family precision {prec_ok}/{len(calls)} = "
           f"{(prec_ok / len(calls)) if calls else 0:.3f}. Version recall {ver_ok}/{len(vrows)} = {(ver_ok / len(vrows)) if vrows else 0:.3f}.",
           f"Relabeled to no_cell_barcode by the deposit rule: {sum(x['relabeled'] for x in recs)}. "
           f"Decided by trial mapping: {sum(x['how'] == 'trial' for x in recs)}; trial pending: {sum(x['how'] == 'trial pending' for x in recs)}.", ""]
    fams = sorted({x["fam_t"] for x in recs} | {x["fam_p"] for x in recs})
    out += ["| family | rows | recall | calls | precision |", "|---|---|---|---|---|"]
    for f in fams:
        t = [x for x in recs if x["fam_t"] == f]
        c = [x for x in recs if x["fam_p"] == f]
        out.append(f"| {f} | {len(t)} | {sum(x['fam_p'] == f for x in t)} | {len(c)} | {sum(x['fam_t'] == f for x in c)} |")
    nontenx = [x for x in recs if x["fam_t"] == x["fam_p"] == "other_barcoded"]
    out += ["", f"Protocol tag right within other_barcoded: {sum(x['proto_t'].split('-')[0] in x['proto_p'] for x in nontenx)}/{len(nontenx)}", "",
            "## Misses", ""]
    for x in recs:
        if x["fam_t"] != x["fam_p"] or (x["ver_t"] and x["ver_t"] != x["ver_p"]):
            res = x["res"]
            cands = "; ".join(f"{c['family']}/{c['protocol']} {c['score']}" for c in res.get("candidates", [])[:3])
            ev = " | ".join((res.get("decision", {}).get("top") or {}).get("evidence", []))[:300]
            out.append(f"- `{x['id']}` truth {x['fam_t']}/{x['ver_t'] or x['proto_t']}; called {x['fam_p']}/{x['ver_p'] or x['proto_p']} "
                       f"({x['how']}). candidates: {cands}. evidence: {ev} {res.get('error', '')} {res.get('log', '')[:200]}")
    txt = "\n".join(out)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")
    with open(os.path.join(os.path.dirname(os.path.abspath(a.md or a.panel)), os.path.basename(a.panel).replace(".tsv", "_calls.tsv")), "w") as f:
        f.write("id\tfamily_truth\tversion_truth\tfamily_call\tversion_call\thow\tstatus\ttrial_tags\n")
        for x in recs:
            d = x["res"].get("decision", {})
            f.write(f"{x['id']}\t{x['fam_t']}\t{x['ver_t']}\t{x['fam_p']}\t{x['ver_p']}\t{x['how']}\t{d.get('status', '')}\t"
                    f"{','.join(d.get('trial_tags', []))}\n")


if __name__ == "__main__":
    main()
