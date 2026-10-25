# SPDX-License-Identifier: MIT
"""``provenance.json`` of the crispr-guides artifacts: commit, datasets and the resources of each run.

    python -P make_provenance.py RUNS_DIR ARTIFACT_DIR --commit SHA
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vdj"))
from make_artifacts import gnu_time  # noqa: E402

RUNS = {"a549": "tenx_crispr_a549_5k", "replogle_k562_gg41": "replogle2022_k562_gg41"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("out")
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    prov = {"commit": a.commit, "datasets": [], "resources": []}
    for run, ds in RUNS.items():
        rj = os.path.join(a.runs, run, "singlet", "run.json")
        if not os.path.exists(rj) or not os.path.exists(os.path.join(a.out, f"{ds}.meta.json")):
            continue
        prov["datasets"].append(ds)
        run_ = json.load(open(rj))
        t = run_["timings_s"]
        g = gnu_time(os.path.join(a.runs, run, "singlet.time"))
        prov["resources"].append({"tool": "singlet", "dataset": ds, "wall_s": round(g.get("wall_s", run_["wall_s"]), 1),
                                  "max_rss_gb": round(g.get("max_rss_gb", run_["peak_rss_gb"]), 2), "threads": 1,
                                  "notes": f"counting {t['count_s']:.0f} s + assignment {t['assign_singlet_s']:.1f} s "
                                           f"({run_['stats'].get('reads', 0):,} read pairs)"})
        for c, name in (("cellranger_gmm", "gmm-cellranger-style"), ("poisson_gaussian", "poisson-gaussian-mixture")):
            if f"assign_{c}_s" in t:
                prov["resources"].append({"tool": name, "dataset": ds, "wall_s": round(t[f"assign_{c}_s"], 1),
                                          "max_rss_gb": prov["resources"][-1 if name.startswith("gmm") else -2]["max_rss_gb"],
                                          "threads": 1, "notes": "assignment only, on Singlet's counts, in the same process "
                                                                 "(memory is the process peak)"})
    with open(os.path.join(a.out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
