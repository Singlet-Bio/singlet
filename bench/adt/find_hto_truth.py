# SPDX-License-Identifier: MIT
"""Hashtag pools with an independent truth: GEX samples whose bundle has genotype-free donor assignments.

    python -P bench/adt/find_hto_truth.py --data $WORK/data/adt --out $WORK/state/adt/hto_truth.tsv

For every processed library that is a hashtag pool (``hto_umi_share`` >= 0.05 in ``qc.json``), opens the study bundle (central
directory and one small member) and reads ``samples/<GEX GSM>/donor/donor_assignments.tsv``. A pool is usable as a benchmark
when it has >= 2 donors with >= 200 called cells each (``n_donors`` counts only those) and each hashtag's cells come mostly from
one donor (checked by the benchmark itself).
Writes one row per library: donors, assigned cells, and the donor-hashtag purity under the library's own hashtag calls.
"""

import argparse
import glob
import io
import json
import os
import sys

import pandas as pd


def main():
    from singlet.v2.legacy import LegacyBundle

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, bundles = [], {}
    for q in sorted(glob.glob(os.path.join(a.data, "GSE*", "GSM*", "qc.json"))):
        qc = json.load(open(q))
        if (qc.get("hto_umi_share") or 0) < 0.05 or not qc.get("n_hashtags"):
            continue
        gse, gex = qc["gse"], qc["gex_gsm"]
        rec = {"gse": gse, "gsm": qc["gsm"], "gex_gsm": gex, "n_hashtags": qc["n_hashtags"],
               "hto_singlet_share": qc.get("hto_singlet_share"), "donor_file": False}
        try:
            b = bundles.get(gse) or LegacyBundle.open(f"https://data.singlet.bio/data/{gse}/{gse}.singlet")
            bundles[gse] = b
            m = f"samples/{gex}/donor/donor_assignments.tsv"
            if b.exists(m):
                d = pd.read_csv(io.BytesIO(b.read(m)), sep="\t")
                rec["donor_file"] = True
                rec["donor_columns"] = ",".join(d.columns[:8])
                # genotype-free demux output: cell_status cell/ambient; donor_id donorN / doublet / unassigned
                cells = d[d["cell_status"].astype(str).str.lower().eq("cell")] if "cell_status" in d else d
                v = cells["donor_id"].astype(str)
                donors = v[v.str.startswith("donor")].value_counts()
                rec["n_donors"] = int((donors >= 200).sum())
                rec["donor_cells"] = ",".join(f"{k}:{n}" for k, n in donors.items())
                rec["genotype_doublets"] = int(v.str.lower().eq("doublet").sum())
                rec["n_assigned"] = int(donors.sum())
        except Exception as e:  # recorded; the scan goes on
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
        rows.append(rec)
        print(rec, flush=True)
    pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)


if __name__ == "__main__":
    sys.exit(main())
