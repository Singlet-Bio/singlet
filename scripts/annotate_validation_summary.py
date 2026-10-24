#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarise an annotation run (the ``<GSE>/<GSM>.summary.json`` and ``.labels.parquet`` files written by annotate_sample.py).

Reports: samples and cells annotated; seconds per million cells by stage (wall clock of one task, ``--threads`` threads);
fraction of cells with a cell type call, by depth reached; sample-level sex calls against the sex in the GEO metadata
(``--sex-labels``: JSON {GSM: 'female' | 'male'}); fraction of cells whose label disagrees with their sample's call; ancestry
calls per donor. Writes one JSON with the aggregates (no per-cell data).

  python scripts/annotate_validation_summary.py --run out/run1 --sex-labels sex_labels.json --threads 8 --out summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--sex-labels", type=Path)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    import pandas as pd

    truth = json.loads(a.sex_labels.read_text()) if a.sex_labels else {}
    rows, sex_rows, anc_rows = [], [], []
    for sp in sorted(a.run.glob("GSE*/GSM*.summary.json")):
        s = json.loads(sp.read_text())
        lab = sp.with_name(sp.name.replace(".summary.json", ".labels.parquet"))
        if not lab.exists() or s.get("skipped"):
            continue
        df = pd.read_parquet(lab)
        t = s.get("timings", {})
        n = len(df)
        typed = df["cell_type_cl"].notna()
        rows.append(
            dict(
                gse=s["gse"],
                gsm=s["gsm"],
                species=s["species"],
                n_cells=n,
                typed=float(typed.mean()) if n else 0.0,
                method=s.get("cell_type", {}).get("method_used"),
                read_s=t.get("read_s"),
                cell_type_s=t.get("cell_type_s"),
                sex_s=t.get("sex_s"),
                ancestry_s=t.get("ancestry_s"),
                total_s=(t.get("total_s") or 0) + (t.get("read_s") or 0),
            )
        )
        groups = s.get("sex", {}).get("groups", {})
        if s["gsm"] in truth and len(groups) == 1:
            g = next(iter(groups.values()))
            sex_rows.append(
                dict(
                    gse=s["gse"],
                    gsm=s["gsm"],
                    species=s["species"],
                    truth=truth[s["gsm"]],
                    call=g["call"],
                    pi_male=g["pi_male"],
                    n_informative=g["n_informative"],
                    n_cells=n,
                    cells_agree=float((df["sex_chrom"] == truth[s["gsm"]]).mean()),
                    cells_contrary=float(
                        (
                            df["sex_chrom"]
                            == ({"male": "female", "female": "male"}[truth[s["gsm"]]])
                        ).mean()
                    ),
                )
            )
        for d, c in (s.get("ancestry") or {}).items():
            anc_rows.append(
                dict(
                    gse=s["gse"],
                    gsm=s["gsm"],
                    donor=d,
                    group=c.get("group"),
                    prob=c.get("prob"),
                    n_sites=c.get("n_sites"),
                    stability=c.get("stability"),
                )
            )
    R = pd.DataFrame(rows)
    out: dict = {"n_samples": len(R), "n_cells": int(R["n_cells"].sum()) if len(R) else 0}
    if len(R):
        tot = R["n_cells"].sum()
        out["seconds_per_million_cells"] = {
            k: float(R[k].fillna(0).sum() / tot * 1e6)
            for k in ("read_s", "cell_type_s", "sex_s", "ancestry_s", "total_s")
        }
        out["cpu_hours_per_million_cells"] = (
            out["seconds_per_million_cells"]["total_s"] * a.threads / 3600
        )
        out["typed_fraction_cells"] = float((R["typed"] * R["n_cells"]).sum() / tot)
        out["typed_fraction_by_species"] = {
            sp: float((g["typed"] * g["n_cells"]).sum() / g["n_cells"].sum())
            for sp, g in R.groupby("species")
        }
        out["samples"] = R.to_dict("records")
    if sex_rows:
        S = pd.DataFrame(sex_rows)
        decided = S[S["call"] != "ambiguous"]
        out["sex"] = dict(
            n_samples=len(S),
            n_studies=int(S["gse"].nunique()),
            called=int(len(decided)),
            ambiguous=int((S["call"] == "ambiguous").sum()),
            correct=int((decided["call"] == decided["truth"]).sum()),
            accuracy_of_calls=float((decided["call"] == decided["truth"]).mean())
            if len(decided)
            else None,
            by_species={
                sp: dict(
                    n=len(g),
                    correct=int((g["call"] == g["truth"]).sum()),
                    ambiguous=int((g["call"] == "ambiguous").sum()),
                )
                for sp, g in S.groupby("species")
            },
            median_cells_agree=float(S["cells_agree"].median()),
            mean_cells_contrary=float(S["cells_contrary"].mean()),
            wrong=S[(S["call"] != S["truth"]) & (S["call"] != "ambiguous")].to_dict("records"),
            ambiguous_samples=S[S["call"] == "ambiguous"].to_dict("records"),
        )
    if anc_rows:
        A = pd.DataFrame(anc_rows)
        out["ancestry"] = dict(
            n_donors=len(A),
            groups=A["group"].fillna("none").value_counts().to_dict(),
            median_sites=float(np.nanmedian(A["n_sites"].astype(float))) if len(A) else None,
        )
    a.out.write_text(json.dumps(out, indent=1, default=str))
    print(
        json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=1, default=str)[:6000]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
