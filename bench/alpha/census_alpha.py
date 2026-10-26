"""Research Alpha QC census: what the published atlas holds after the Alpha exclusions.

    python -I census_alpha.py gsms CENSUS_DIR POLICY_JSON OUT_TSV           # the Alpha public sample list (for the reads job)
    python -I census_alpha.py json CENSUS_DIR POLICY_JSON READS_TSV_GZ OUT_JSON

CENSUS_DIR holds read-only D1 dumps (d1/*.jsonl: sample_qc, gsm, bundle_index, gse) and studies_action.tsv;
POLICY_JSON is alpha-policy.json (withheld_studies, excluded_samples). READS_TSV_GZ comes from census_reads.py.

Levels:
  physical = every sample in a published bundle whose study is not withheld, minus excluded samples
  public   = physical, restricted to studies downloadable now and samples usable with QC tier 0 (the site's basis)
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

mode, CEN, POL = sys.argv[1], sys.argv[2], sys.argv[3]
L = lambda f: [json.loads(l) for l in open(f"{CEN}/d1/{f}.jsonl", encoding="utf-8")]  # noqa: E731
qc = {r["gsm_id"]: r for r in L("sample_qc")}
gsm = {r["gsm_id"]: r for r in L("gsm")}
pol = json.load(open(POL, encoding="utf-8"))
wh, ex = set(pol["withheld_studies"]), set(pol["excluded_samples"])
act = {r["gse"]: r for r in csv.DictReader(open(f"{CEN}/studies_action.tsv", encoding="utf-8"), delimiter="\t")}
excl_reason = {}
for r in csv.DictReader(open(f"{CEN}/samples_excluded.tsv", encoding="utf-8"), delimiter="\t"):
    excl_reason[r["gsm"]] = r["reasons"]
import os  # noqa: E402

before, phys = [], []
HAVE_INDEX = os.path.exists(f"{CEN}/d1/bundle_index.jsonl")
if HAVE_INDEX:
    for line in open(f"{CEN}/d1/bundle_index.jsonl", encoding="utf-8"):
        r = json.loads(line)
        g = r["gse_id"]
        for s in sorted({e["p"].split("/")[1] for e in json.loads(r["entries"]) if e["p"].startswith("samples/")}):
            before.append((g, s))
            if g not in wh and s not in ex:
                phys.append((g, s))


def tier0(s):
    q = qc.get(s)
    return bool(q) and q["usable"] == 1 and (q["qc_tier"] or 0) == 0


if HAVE_INDEX:
    pub = [(g, s) for g, s in phys if act.get(g, {}).get("downloadable_now") == "1" and tier0(s)]
else:  # on Clipper: the public list and the bundle-level counts come from the census files (alpha_public_gsms.tsv, stats.json)
    pub = [(r["gse_id"], r["gsm_id"]) for r in csv.DictReader(open(f"{CEN}/alpha_public_gsms.tsv", encoding="utf-8"), delimiter="\t")]
    STATS = json.load(open(f"{CEN}/stats.json", encoding="utf-8"))
SPECIES = {"GRCh38-2024-A": "human", "GRCm39-2024-A": "mouse", "GRCh38-GRCm39-barnyard": "human+mouse (barnyard)",
           "GRCz11": "zebrafish", "mRatBN7.2": "rat", "Mmul_10": "macaque"}
CHEM = {"10xv2": "10x 3' v2", "10xv3": "10x 3' v3", "10xv1": "10x 3' v1", "10xv3-5p": "10x 5'", "10x-arc-gex": "10x Multiome GEX",
        "10x-visium": "10x Visium", "dropseq": "Drop-seq", "indrop": "inDrop", "seqwell": "Seq-Well", "bd-rhapsody": "BD Rhapsody",
        "splitseq": "SPLiT-seq / Parse", "celseq2": "CEL-seq2", "marsseq2": "MARS-seq2", "quartzseq2": "Quartz-seq2",
        "dnbelab-c4": "DNBelab C4", "microwell-seq": "Microwell-seq", "surecell": "SureCell", "ddseq": "ddSEQ"}


def sp_of(s):
    return SPECIES.get((qc.get(s) or {}).get("reference_build"), (qc.get(s) or {}).get("reference_build") or "unknown")


def chem_of(s):
    q, m = qc.get(s) or {}, gsm.get(s) or {}
    if m.get("assay_family") == "10x 5'" and str(q.get("protocol", "")).startswith("10x"):
        return "10x 5' (GEO-curated; counted with a 3' label)" if q.get("protocol") != "10xv3-5p" else "10x 5'"
    return CHEM.get(q.get("protocol"), q.get("protocol") or "unknown")


if mode == "gsms":
    with open(sys.argv[4], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["gsm_id", "gse_id", "species", "protocol"])
        for g, s in pub:
            w.writerow([s, g, sp_of(s), (qc.get(s) or {}).get("protocol")])
    print(len(pub))
    sys.exit(0)

READS, OUTJ = sys.argv[4], sys.argv[5]
cells = lambda s: (qc.get(s) or {}).get("n_cells_called") or 0  # noqa: E731


def level(rows):
    return {"studies": len({g for g, _ in rows}), "samples": len(rows), "cells": int(sum(cells(s) for _, s in rows))}


def q(x, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return None
    return {"n": int(len(x)), **{f"p{int(p * 100):02d}": float(np.quantile(x, p)) for p in ps}, "mean": float(x.mean())}


def table(rows, key):
    d = defaultdict(lambda: [set(), 0, 0])
    for g, s in rows:
        k = key(s)
        d[k][0].add(g)
        d[k][1] += 1
        d[k][2] += cells(s)
    return [{"name": k, "studies": len(v[0]), "samples": v[1], "cells": int(v[2])} for k, v in sorted(d.items(), key=lambda kv: -kv[1][1])]


out = {
    "schema": "singlet.alpha.census", "schema_version": "1.0.0", "engine": "alpha (published atlas)",
    "dataset_scope": "every sample published in Research Alpha",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    "policy_version": pol.get("version"), "policy_source": pol.get("source"),
    "sources": ["singlet-catalog D1 (read-only SELECT dumps): gse, gsm, sample_qc, bundle_index",
                "alpha-policy.json (withheld studies, excluded samples)",
                "published bundles: samples/<GSM>/summary.json (reads used, engine build); Clipper: pipeline-benchmark/results/campaign/campaign_samples.csv (strand), "
                "cellarium processing_catalog.parquet (SRA read_count)"],
    "levels": ({"bundles_before_exclusions": level(before), "physical_after_exclusions": level(phys), "public": level(pub)} if HAVE_INDEX else
               {"bundles_before_exclusions": STATS["physical_before"], "physical_after_exclusions": STATS["physical_after"], "public": level(pub),
                "public_before_exclusions": STATS["public_before"]}),
    "withheld_studies": len(wh),
    "excluded_samples": {"total": len(ex),
                         "by_reason_in_bundles": (dict(Counter(r for g, s in before if s in ex for r in excl_reason.get(s, "unknown").split(",")))
                                                  if HAVE_INDEX else STATS["reason_counts_in_bundle"]),
                         "public_samples_and_cells_removed_by_reason": None if HAVE_INDEX else STATS["excluded_public_by_reason"]},
    "definitions": {
        "physical_after_exclusions": "samples in published bundles of studies not withheld, minus excluded samples (strand, non-RNA library, empty matrix)",
        "public": "physical, in studies downloadable now, samples usable with QC tier 0 (singlet.bio's basis)",
        "cells": "cells called by Singlet (sample_qc.n_cells_called; STARsolo CellRanger2.2 knee filter on the capped reads)"},
}
out["public_by_species"] = table(pub, sp_of)
out["public_by_chemistry"] = table(pub, chem_of)
out["public_by_assay_family"] = table(pub, lambda s: (gsm.get(s) or {}).get("assay_family") or "unknown")
out["public_cells_per_sample"] = q([cells(s) for _, s in pub])
out["public_median_genes_per_cell"] = q([(qc.get(s) or {}).get("median_genes") for _, s in pub])
out["public_mapping_rate"] = q([(qc.get(s) or {}).get("mapping_rate") for _, s in pub])
out["public_exonic_fraction"] = q([(qc.get(s) or {}).get("exonic_fraction") for _, s in pub])
d = pd.read_csv(READS, sep="\t")
d = d[d.gsm_id.isin({s for _, s in pub})].drop_duplicates("gsm_id")
SUMM = sys.argv[6:]  # summaries_<k>.tsv from census_summaries.py (summary.json of each published sample)
sm = pd.concat([pd.read_csv(p, sep="\t") for p in SUMM]).drop_duplicates("gsm").rename(columns={"gsm": "gsm_id"})
d = d.merge(sm[["gsm_id", "n_input_reads", "git_sha", "singlet_version", "wall_seconds", "peak_rss_gb", "error"]], on="gsm_id", how="left")
has = d.n_input_reads.notna()
fr = (d.n_input_reads / d.read_count).where(d.read_count.gt(0)).dropna().clip(upper=1.0)
H = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0001]
out["reads"] = {
    "coverage": {"public_samples": int(len(d)), "with_reads_used": int(has.sum()), "with_reads_available": int(d.read_count.gt(0).sum()),
                 "with_both": int(len(fr)), "summary_read_errors": {str(k): int(v) for k, v in d.error.dropna().str[:40].value_counts().head(5).items()}},
    "reads_used_source": "n_input_reads in each published sample's summary.json (read from the bundles)",
    "reads_available_source": "read_count (SRA spots) in Singlet's processing catalog; checked against NCBI runinfo on the benchmark samples",
    "reads_used": q(d.loc[has, "n_input_reads"].values),
    "share_at_30M_cap": float((d.loc[has, "n_input_reads"] >= 29_999_000).mean()) if has.any() else None,
    "share_below_10M": float((d.loc[has, "n_input_reads"] < 10_000_000).mean()) if has.any() else None,
    "reads_available": q(d.loc[d.read_count.gt(0), "read_count"].values),
    "frac_reads_used": q(fr.values),
    "share_frac_below_0.5": float((fr < 0.5).mean()) if len(fr) else None,
    "share_frac_below_0.25": float((fr < 0.25).mean()) if len(fr) else None,
    "share_frac_above_0.9": float((fr > 0.9).mean()) if len(fr) else None,
    "frac_reads_used_histogram": {"bins": H, "counts": np.histogram(fr.values, bins=H)[0].tolist()},
    "samples_with_multiple_runs": int((d.n_srr > 1).sum()), "share_multiple_runs": float((d.n_srr > 1).mean()),
    "note": "the campaign kept only the first SRA run and its first 30M reads; samples with several runs lost the other runs entirely",
}
out["engine_builds"] = {"git_sha": {str(k): int(v) for k, v in d.git_sha.value_counts().items()},
                        "singlet_version": {str(k): int(v) for k, v in d.singlet_version.value_counts().items()}}
out["wall_s_per_sample_4_threads"] = q(d.wall_seconds.values)
out["peak_rss_gb_per_sample"] = q(d.peak_rss_gb.values)
ws = d.wrong_strand / (d.wrong_strand + d.exon_hits)
out["strand"] = {"wrong_strand_fraction": q(ws.values), "share_above_0.5": float((ws > 0.5).mean()) if ws.notna().any() else None,
                 "n": int(ws.notna().sum()),
                 "note": "human samples with >50% wrong-strand reads were excluded from Alpha; mouse strand is not classified yet"}
out["caveats"] = [
    "Mouse samples were not screened for wrong-strand counting (policy: 'mouse strand not yet classified'); some mouse 5' libraries in Alpha may be counted antisense.",
    "Cell counts are Singlet's knee-filter calls on reads capped at the first 30M of the first SRA run; they undercount relative to Cell Ranger / EmptyDrops (see the cell-recovery benchmark).",
    "Every count matrix in Alpha comes from the atlas engine (v0.3.0, git 8e1803ba), which double-counts some molecules (+2% to +19% per cell vs the fixed engine).",
]
json.dump(out, open(OUTJ, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("levels", "reads", "strand")}, indent=1))
