#!/usr/bin/env python3
"""Assemble the Beta release-gate table for one pinned binary.

    python make_gate.py NEW_DIR CARRY_DIR OUT.md [--pin SHA]

NEW_DIR   result documents measured on the pinned binary (bench/release_gate jobs on Clipper)
CARRY_DIR the newest published document of every other capability (bench/public/collect_results.py); used only where NEW_DIR has none,
          and always shown as carried forward with the commit it was computed at.
Per capability: status and gating pass/fail from the contract (contract.capability_status), the worst failing metric against the best
tool, where the number was measured, and the proposed class of a failing capability:

  blocker    wrong output: the run errors, a deterministic-output or identity check fails, or a gating metric is far from the best tool
             (a higher-is-better gap > 0.05, or an accuracy/precision/F-score/recall below 0.90 and more than 0.02 behind the best
             tool, or a distance metric more than 5x the best tool's);
  should-fix behind the best tool on a gating metric, but valid output and within those bounds.

The class is a proposal from these rules and the per-capability notes in NOTES; the rules are printed in the document.
"""
import json, sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "public"))
import contract as C  # noqa: E402
import release_gate as G  # noqa: E402

NOTES = json.loads((Path(__file__).with_name("gate_notes.json")).read_text(encoding="utf8")) if Path(__file__).with_name("gate_notes.json").exists() else {}


def classify(row, doc):
    """blocker / should-fix / '' from the failing metrics (rules in the module docstring)."""
    if row["status"] == "pass":
        return ""
    if row["status"] in ("missing", "incomplete"):
        return "missing"
    for f in row["failing"]:
        s, r, mid = f["singlet"], f["reference"], f["metric"]
        if s is None or r is None:
            continue
        higher = next((m for m in doc["metrics"] if m["id"] == mid and m.get("dataset") == f["dataset"]), {}).get("direction") == "higher"
        if higher:
            lowlevel = ("acc" in mid or "precision" in mid or "f_score" in mid or "recall" in mid) and s < 0.90
            if r - s > 0.05 or (lowlevel and r - s > 0.02):
                return "blocker"
        else:
            if r > 0 and s / r > 5 and s > 0.02:
                return "blocker"
    return "should-fix"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new"); ap.add_argument("carry"); ap.add_argument("out"); ap.add_argument("--pin", default="")
    a = ap.parse_args()
    rows = []
    for cap, owner in C.CAPABILITIES.items():
        src = "re-run"
        p = Path(a.new) / f"{cap}.json"
        if not p.exists():
            p = Path(a.carry) / f"{cap}.json"; src = "carried"
        if not p.exists():
            rows.append({"capability": cap, "status": "missing", "src": "none", "failing": [], "owner": owner, "cls": "missing"})
            continue
        doc = json.loads(p.read_text(encoding="utf8"))
        st = C.capability_status(doc)
        failing = [G.gap(m) for m in doc["metrics"] if (m.get("rule") or {}).get("type") != "informational" and C.evaluate_metric(m) == "fail"]
        failing.sort(key=lambda g: -(g["gap"] if g["gap"] is not None else 0))
        commit = (doc.get("singlet") or {}).get("commit", "")
        if src == "re-run" and a.pin and commit and not commit.startswith(a.pin[:8]):
            src = "carried"   # a "new" file that was not computed at the pin
        r = {"capability": cap, "status": st["status"], "n_pass": st["n_pass"], "n_fail": st["n_fail"], "n_na": st["n_na"],
             "n": st["n_metrics"], "src": src, "commit": commit[:8], "failing": failing, "owner": G.FIX_OWNER.get(cap, owner)}
        r["cls"] = classify(r, doc)
        r.update(NOTES.get(cap, {}))
        if "cls_override" in r:
            r["cls"] = r["cls_override"]
        rows.append(r)
    out = ["| capability | status | pass/fail | measured | worst gap vs best tool | class | note |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        w = next((f for f in r["failing"] if f["gap"] is not None), None)
        gp = f"`{w['metric']}` {w['dataset']}: {w['singlet']:.4g} vs {w['reference']:.4g} ({w['reference_tool']})" if w else ""
        meas = "-" if r["src"] == "none" else (f"pin {a.pin[:8]}" if r["src"] == "re-run" else f"carried, {r['commit']}")
        pf = f"{r['n_pass']}/{r['n_fail']}" if "n_pass" in r else ""
        out.append(f"| {r['capability']} | {r['status']} | {pf} | {meas} | {gp} | {r['cls']} | {r.get('note', '')} |")
    n_re = sum(r["src"] == "re-run" for r in rows)
    summary = {"n_total": len(rows), "n_pass": sum(r["status"] == "pass" for r in rows), "n_fail": sum(r["status"] == "fail" for r in rows),
               "n_missing": sum(r["status"] in ("missing", "incomplete") for r in rows), "n_measured_at_pin": n_re,
               "blockers": [r["capability"] for r in rows if r["cls"] == "blocker"],
               "should_fix": [r["capability"] for r in rows if r["cls"] == "should-fix"]}
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf8")
    Path(a.out).with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf8")
    print(json.dumps(summary, indent=1)); print("\n".join(out))


if __name__ == "__main__":
    main()
