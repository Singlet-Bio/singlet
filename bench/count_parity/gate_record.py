"""Collect the per-dataset parity.json files of one gate.sbatch run into the gate record <G>.json.

  python gate_record.py G COMMIT SOURCE_HASH "DS1 DS2 ..." ENGINE
"""
import datetime
import json
import os
import sys

g, commit, h, dss, engine = sys.argv[1:6]
out = {"source_hash": h, "commit": commit, "engine": engine, "date": datetime.date.today().isoformat(), "datasets": {}}
for ds in dss.split():
    with open(os.path.join(g, ds, "parity.json")) as f:
        p = json.load(f)
    out["datasets"][ds] = {"axes": {k: {f: v[f] for f in ("singlet", "starsolo", "diff", "diff_ci", "verdict")} for k, v in p["axes"].items()}}
fails = [a["verdict"] == "FAIL" for d in out["datasets"].values() for a in d["axes"].values()]
out["gate"] = "FAIL" if any(fails) or not fails else "PASS"
with open(g + ".json", "w") as f:
    json.dump(out, f, indent=1)
print("GATE", out["gate"], g + ".json")
