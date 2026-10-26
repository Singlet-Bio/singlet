# SPDX-License-Identifier: MIT
"""Per-cell doublet artifact for one genotype-multiplexed Kang 2018 lane (no genotypes, no allele counts: labels and scores only).

    python -I kang_artifacts.py ID RUN_DIR WORK_DIR ARTIFACT_ROOT TRUTH_TSV LANE_FILTER
      LANE_FILTER  stim=ctrl (batch 2 control lane)   |   batch=C (batch 1 lane C)

Writes ARTIFACT_ROOT/doublets/<id>_doublets.tsv.gz with one row per cell Singlet called:
  barcode, n_umi, truth (demuxlet label: singlet / doublet / ambs / none), singlet_score / singlet_call (expression),
  singlet_gt_pdbl / singlet_gt_call (genotype doublet probability and `doublet` label of donor_assignments.tsv),
  scrublet_*, scdblfinder_*, vireo_pdbl / vireo_call (vireo's prob_doublet and `doublet` donor_id)
and ARTIFACT_ROOT/doublets/resources_<id>.json.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barnyard_artifacts import parse_time  # noqa: E402


def main(ds, run, work, art, truth_tsv, lane):
    cells = pd.read_csv(f"{work}/cells.csv")
    bc = cells.barcode.astype(str).values
    out = cells[["barcode", "n_umi"]].copy()

    t = pd.read_csv(truth_tsv, sep="\t", index_col=0)
    col, val = lane.split("=")
    t = t[t[col].astype(str) == val].copy()
    t.index = [i.rsplit("-", 1)[0] for i in t.index]
    assert not t.index.duplicated().any(), "barcode collisions inside the lane"
    mcol = "multiplets"
    out["truth"] = t[mcol].reindex(bc).fillna("none").values

    sd = pd.read_csv(f"{run}/doublet_scores.tsv", sep="\t")
    sd["barcode"] = sd.barcode.astype(str).str.replace(r"-1$", "", regex=True)
    sd = sd.set_index("barcode").reindex(bc)
    out["singlet_score"] = sd.doublet_score.values
    out["singlet_call"] = sd.is_doublet.astype(str).str.lower().eq("true").values.astype(int)

    da = pd.read_csv(f"{run}/donor/donor_assignments.tsv", sep="\t")
    da["cell"] = da.cell.astype(str).str.replace(r"-1$", "", regex=True)
    da = da.set_index("cell").reindex(bc)
    out["singlet_gt_pdbl"] = da.prob_doublet.values
    out["singlet_gt_call"] = (da.donor_id == "doublet").astype(int).values
    out["singlet_n_donors"] = da.donor_id[da.donor_id.str.startswith("donor", na=False)].nunique()

    for tname in ("scrublet", "scdblfinder"):
        d = pd.read_csv(f"{work}/tools/{tname}.csv").set_index("barcode").reindex(bc)
        out[f"{tname}_score"] = d.score.values
        out[f"{tname}_call"] = d.call.astype(str).str.lower().eq("true").values.astype(int)

    v = pd.read_csv(f"{work}/vireo/out/donor_ids.tsv", sep="\t")
    v["cell"] = v.cell.astype(str).str.replace(r"-1$", "", regex=True)
    v = v.drop_duplicates("cell").set_index("cell").reindex(bc)
    out["vireo_pdbl"] = v.prob_doublet.values
    out["vireo_call"] = (v.donor_id == "doublet").astype(int).values

    os.makedirs(f"{art}/doublets", exist_ok=True)
    out.to_csv(f"{art}/doublets/{ds}_doublets.tsv.gz", sep="\t", index=False, float_format="%.6g")

    et = parse_time(f"{run}/encode.time.txt") if os.path.exists(f"{run}/encode.time.txt") else (0, 0, 0)
    pt = parse_time(f"{run}/time.txt")
    node = open(f"{run}/node.txt").read().strip()
    res = [dict(tool="singlet", dataset=ds, wall_s=round(et[0] + pt[0], 1), max_rss_gb=round(max(et[1], pt[1]), 2), threads=16,
                cpu_h=round(et[2] + pt[2], 3), node=node,
                notes="whole pipeline including alignment, counting, donor demultiplexing with the donor count estimated (encode %.0f s + pipeline %.0f s)" % (et[0], pt[0]))]
    nodes = open(f"{work}/tools/node.txt").read().split("\n")
    for t_, thr in (("scrublet", 1), ("scdblfinder", 1)):
        w, r, c = parse_time(f"{work}/tools/{t_}.time")
        res.append(dict(tool=t_, dataset=ds, wall_s=round(w, 1), max_rss_gb=round(r, 2), threads=thr, cpu_h=round(c, 3), node=nodes[0]))
    w1, r1, c1 = parse_time(f"{work}/tools/cellsnp.time")
    w2, r2, c2 = parse_time(f"{work}/tools/vireo.time")
    res.append(dict(tool="vireo", dataset=ds, wall_s=round(w1 + w2, 1), max_rss_gb=round(max(r1, r2), 2), threads=16, cpu_h=round(c1 + c2, 3),
                    node=nodes[0], notes="cellsnp-lite pileup %.0f s + vireo %.0f s; excludes the BAM, which Singlet wrote" % (w1, w2)))
    json.dump(res, open(f"{art}/doublets/resources_{ds}.json", "w"), indent=1)
    print(ds, "rows", len(out), out.truth.value_counts().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:7])
