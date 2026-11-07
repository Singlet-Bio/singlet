#!/usr/bin/env python3
"""Beta release gate: turn the capability results into the release policy and a pass/fail table.

    python bench/public/release_gate.py bench/public/results [--out DIR] [--waivers waivers.json] [--binary SHA]

Reads every ``<capability>.json`` in the results directory (the contract's own ``capability_status`` recomputes pass and
fail from values and rules, nothing is hand-written) and writes ``release-gate.json`` and ``release-gate.md``:

  * one row per capability of ``contract.CAPABILITIES`` (a missing file is ``missing``),
  * for every failing or incomplete capability the failing metrics with the gap to the reference and the fix owner,
  * the release policy: the Beta is releasable when every capability is ``pass``, or ``fail``/``incomplete``/``missing``
    only with a waiver (``waivers.json``: {capability: {"reason": ..., "approved_by": ...}}). A waiver never hides the
    row; the table prints it as ``waived``.
  * the exit code is 0 when the policy is met and 1 otherwise, so CI and the campaign launcher can gate on it.

Real-data rule: the contract rejects a capability with no gating metric on real data, so ``pass`` here always rests on
real truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract as C  # noqa: E402

# who fixes a failing capability (session names of the Beta accuracy work); falls back to the contract's owner
FIX_OWNER = {
    "spliced-unspliced": "intron rule (fix/intron-rule-default)",
    "doublets": "doublet combiner (study/latent-qc, #43)",
    "hto": "HTO demux (adt-modality)",
    "donor-snps": "base qualities in .1fq (fq-read-source) + SRA Lite verdict",
    "mtdna": "base qualities in .1fq (fq-read-source) + SRA Lite verdict",
    "alignment-performance": "engine-blockers / GPU engine",
    "download": "engine-blockers",
}


def gap(m: dict) -> dict:
    """The distance to the reference for one failing metric (positive = Singlet is worse)."""
    s = (m.get("singlet") or {}).get("value")
    rule = m.get("rule") or {}
    direction = m.get("direction")
    comps = {c["tool"]: c.get("value") for c in m.get("competitors", []) if c.get("value") is not None}
    ref = tool = None
    if rule.get("type") == "match_or_beat":
        against = rule.get("against", "best")
        names = list(comps) if against == "best" else [t for t in against if t in comps]
        if names:
            sign = 1 if direction == "higher" else -1
            tool = max(names, key=lambda t: sign * comps[t])
            ref = comps[tool]
    elif rule.get("type") == "threshold":
        ref, tool = float(rule["value"]), "threshold"
    elif rule.get("type") == "equivalence":
        t = rule["target"]
        ref, tool = (comps.get(t), t) if isinstance(t, str) else (t, "target")
    d = None
    if s is not None and ref is not None:
        d = (ref - s) if direction == "higher" else (s - ref)
    return {"metric": m["id"], "dataset": m.get("dataset"), "singlet": s, "reference": ref, "reference_tool": tool,
            "gap": d, "rule": rule.get("type")}


def build(results_dir: Path, waivers: dict, binary: str | None) -> dict:
    rows = []
    for cap, owner in C.CAPABILITIES.items():
        p = results_dir / f"{cap}.json"
        if not p.exists():
            rows.append({"capability": cap, "status": "missing", "owner": owner, "fix_owner": FIX_OWNER.get(cap, owner),
                         "failing": []})
            continue
        doc = json.loads(p.read_text(encoding="utf8"))
        st = C.capability_status(doc)
        failing = [gap(m) for m in doc.get("metrics", [])
                   if (m.get("rule") or {}).get("type") != "informational" and C.evaluate_metric(m) == "fail"]
        failing.sort(key=lambda g: -(g["gap"] if g["gap"] is not None else 0))
        rows.append({"capability": cap, "status": st["status"], "n_gating": st["n_metrics"], "n_pass": st["n_pass"],
                     "n_fail": st["n_fail"], "n_na": st["n_na"], "doc_status": doc.get("status"), "owner": owner,
                     "fix_owner": FIX_OWNER.get(cap, owner), "singlet_commit": (doc.get("singlet") or {}).get("commit"),
                     "failing": failing})
    for r in rows:
        r["waived"] = bool(r["status"] != "pass" and r["capability"] in waivers)
        if r["waived"]:
            r["waiver"] = waivers[r["capability"]]
    blocking = [r["capability"] for r in rows if r["status"] != "pass" and not r["waived"]]
    commits = sorted({r.get("singlet_commit") for r in rows if r.get("singlet_commit")})
    return {"schema": "release-gate/1", "binary_commit": binary, "result_commits": commits,
            "policy": "releasable only when every capability passes on real-data gating metrics; a waiver is explicit and listed",
            "releasable": not blocking, "blocking": blocking,
            "n_pass": sum(r["status"] == "pass" for r in rows), "n_total": len(rows), "capabilities": rows}


def markdown(g: dict) -> str:
    out = [f"# Beta release gate: {'RELEASABLE' if g['releasable'] else 'NOT releasable'}",
           f"{g['n_pass']} of {g['n_total']} capabilities pass. Policy: {g['policy']}.",
           f"Binary: {g.get('binary_commit') or 'not given'}; results computed with: {', '.join(g['result_commits']) or 'n/a'}.", "",
           "| capability | status | gating pass/fail/na | worst gap | fix owner |", "|---|---|---|---|---|"]
    for r in g["capabilities"]:
        w = next((f for f in r["failing"] if f["gap"] is not None), None)
        gp = f"{w['metric']} on {w['dataset']}: {w['singlet']:.4g} vs {w['reference']:.4g} ({w['reference_tool']})" if w else ""
        st = r["status"] + (" (waived)" if r["waived"] else "")
        cnt = f"{r['n_pass']}/{r['n_fail']}/{r['n_na']}" if "n_pass" in r else ""
        out.append(f"| {r['capability']} | {st} | {cnt} | {gp} | {r['fix_owner'] if r['status'] != 'pass' else ''} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: the results directory")
    ap.add_argument("--waivers", type=Path, default=None)
    ap.add_argument("--binary", default=None, help="full SHA of the binary under test")
    a = ap.parse_args(argv)
    waivers = json.loads(a.waivers.read_text(encoding="utf8")) if a.waivers else {}
    g = build(a.results, waivers, a.binary)
    out = a.out or a.results
    out.mkdir(parents=True, exist_ok=True)
    (out / "release-gate.json").write_text(json.dumps(g, indent=1) + "\n", encoding="utf8")
    (out / "release-gate.md").write_text(markdown(g), encoding="utf8")
    print(markdown(g))
    return 0 if g["releasable"] else 1


if __name__ == "__main__":
    sys.exit(main())
