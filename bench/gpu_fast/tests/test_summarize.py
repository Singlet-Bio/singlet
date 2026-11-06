import json
import os

import summarize
from helpers import leg, result
from test_demand import dleg


def mkrun(root, name, res, cot=None):
    run = root / name
    (run / "res").mkdir(parents=True)
    (run / "res" / "result.json").write_text(json.dumps(res))
    for l in res["legs"]:
        d = run / "res" / l["dir"]
        d.mkdir()
        (d / "runs.tsv").write_text("0\t0\ts\t1000.0\t1100.0\t0\n")
        if cot:
            (d / "cotenants.log").write_text("## start 2026-11-05T14:00:00-05:00\n" + cot)
    return str(run)


def t2res(extra_cot=False):
    legs = []
    for i, arm in enumerate("ABBA"):
        l = leg(i + 1, arm, core={"A": 9.8, "B": 8.7}[arm], held=40.0, span=1.7)
        l["dir"] = f"leg{i + 1}{arm}"
        l["metrics"]["progress_rate"] = {"A": 1.9, "B": 2.3}[arm]
        if extra_cot and i == 0:
            l["metrics"]["other_jobs_max"] = 1
        legs.append(l)
    return result(legs, 2)


def test_summary_has_verdicts_both_ways_cotenants_limits_and_refit(tmp_path):
    t1 = result([dict(dleg(i + 1, a, c, h), dir=f"leg{i + 1}{a}") for i, (a, c, h) in enumerate([("A", 8.94, 39.1), ("B", 7.79, 27.6), ("B", 7.79, 27.6), ("A", 8.94, 39.1)])], 1)
    r1 = mkrun(tmp_path, "t1", t1)
    r2 = mkrun(tmp_path, "t2", t2res(True), cot="800365 debruinz mine 28 x\n800370 debruinz gate 18 x\n")
    man = {"camp": "c-1", "steps": [{"id": "s", "tier": 2, "job": "800365", "run": r2, "rule_params": {}, "intended": False, "notes": []},
                                    {"id": "gone", "tier": 2, "job": "1", "run": str(tmp_path / "nothing"), "rule_params": {}}],
           "summary": {"limits_for": [r1], "alone": [r1], "loaded": [], "loaded_from_steps_arm": "A"}}
    mp = tmp_path / "campaign.json"
    mp.write_text(json.dumps(man))
    out = tmp_path / "out"
    assert summarize.main(str(mp), str(out)) == 0
    text = open(out / "summary.md").read()
    assert "### s (job 800365)" in text and "All legs: **PROMOTE**" in text and "Without the co-tenant legs (leg 1 (A))" in text
    assert "No result. INCONCLUSIVE (J1)" in text
    assert "CO-TENANT job 800370 gate" in text and "Demand vector and node rate limits" in text and "node rate (smallest)" in text
    assert "Load factors and efficiency" in text
    assert json.load(open(out / "steps" / "s" / "verdict.json"))["without_contaminated"]["rule"] == "T2-4"
