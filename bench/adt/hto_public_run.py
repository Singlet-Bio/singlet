# SPDX-License-Identifier: MIT
"""Singlet's calls for the public ``hto`` capability: run include/singlet/pileup/hto_demux.h on the Howitt et al. 2023 captures.

    python -P bench/adt/hto_public_run.py --zenodo DIR --exe hto_demux_cli --artifacts ART --selection selection.json

``hto_demux_cli`` is bench/adt/hto_demux_cli.cpp compiled against the header. ``selection.json`` is the output of
bench/adt/hto_lodo.py: each dataset is called with the setting chosen on the *other* datasets. ``ART`` must already hold the
truth, map and competitor files of the capability (from ``hto_public_run.py`` of the adt-modality branch); this script
writes ``hto_<dataset>_singlet.parquet`` and ``resources_hto_<dataset>_singlet.json`` there, and checks that the calls do not
change when the cells are fed in reverse order.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import pandas as pd
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hto_lodo as L  # noqa: E402

PUBLIC = {"howitt2023_bal1": "BAL1", "howitt2023_bal2": "BAL2", "howitt2023_bal3": "BAL3", "howitt2023_cl": "CL"}


def count_file(z, ds, cap):
    return (os.path.join(z, "BAL_data", f"batch{ds[-1]}_{cap}_hto_counts.csv") if ds != "CL"
            else os.path.join(z, "cell_line_data", f"lmo_counts_capture{cap[-1]}.csv"))


def run(exe, src, out, params, extra=()):
    cmd = [exe, src, out, "--confident-min", str(params["confident_min"]), "--doublet-nmads", str(params["doublet_nmads"]),
           "--doublet-min", str(params["doublet_min"]), "--veto", str(params["veto"] if params["veto"] is not None else 2.0),
           *extra]
    t0 = time.perf_counter()
    p = psutil.Popen(cmd)
    peak = 0
    while p.poll() is None:
        try:
            peak = max(peak, p.memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(0.02)
    if p.returncode:
        raise SystemExit(f"{cmd} failed ({p.returncode})")
    return time.perf_counter() - t0, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--commit", default=os.environ.get("SINGLET_COMMIT", ""))
    a = ap.parse_args()
    sel = json.load(open(a.selection))
    held = {h["held_out"]: h for h in sel["held_out"]}
    data = L.load_howitt(a.zenodo)
    tmp = tempfile.mkdtemp(prefix="hto_run_")
    for pub, ds in PUBLIC.items():
        caps, hmap = data[ds]
        params = held[ds]["params"]
        parts, wall, rss = [], 0.0, 0
        for cap, h, _ in caps:
            src = count_file(a.zenodo, ds, cap)
            out, out_r = os.path.join(tmp, "o.tsv"), os.path.join(tmp, "r.tsv")
            w, m = run(a.exe, src, out, params)
            run(a.exe, src, out_r, params, ("--reverse",))
            c1 = pd.read_csv(out, sep="\t", dtype=str).set_index("barcode")["call"]
            c2 = pd.read_csv(out_r, sep="\t", dtype=str).set_index("barcode")["call"]
            if not (c1 == c2.reindex(c1.index)).all():
                raise SystemExit(f"{ds} {cap}: calls depend on the order of the cells")
            wall += w
            rss = max(rss, m)
            parts.append(pd.DataFrame({"barcode": c1.index, "call": c1.values}))
        pd.concat(parts).to_parquet(os.path.join(a.artifacts, f"hto_{pub}_singlet.parquet"), index=False)
        rec = {"tool": "singlet", "dataset": pub, "wall_s": round(wall, 2), "max_rss_gb": round(rss / 1024 ** 3, 3),
               "threads": 1, "cpu_h": round(wall / 3600, 5), "returncode": 0, "singlet_commit": a.commit,
               "command": f"{os.path.basename(a.exe)} COUNTS OUT --confident-min {params['confident_min']} --doublet-nmads "
                          f"{params['doublet_nmads']} --doublet-min {params['doublet_min']} --veto {params['veto']}"}
        json.dump(rec, open(os.path.join(a.artifacts, f"resources_hto_{pub}_singlet.json"), "w"), indent=1)
        print(pub, params, round(wall, 2), "s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
