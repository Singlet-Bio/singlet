"""Classify the campaign's wrong-strand samples per protocol; write the re-run list.

    python strand_classify.py <campaign_samples.csv> <results_dir> <gsm_metadata.parquet> <out_dir>
                              [<bam_probe.tsv> ...]

wf = wrong_strand / (wrong_strand + exon_hits) from each sample's pileup_stats.json (the atlas
engine's counters): the share of strand-informative exonic reads on the strand the atlas did
NOT count. It understates a wrong orientation (reads touching exons of both strands, most chrM
reads, count as hits). The atlas JSON's reverse_strand is the caller's flag, not what the engine
counted: the old parallel probe flipped only its own copy. So wf > 0.5 means "counted on the
minority strand", whether a 5' library was not flipped or a 3' library was flipped wrongly.

The library's own strandedness does not depend on which way it was counted: |1 - 2 wf| (0 for
an unstranded library, 1 for a perfectly stranded one). Classes (flagged = wf > 0.5):

  misoriented         wf >= 0.7: a stranded library counted on its minority strand. The fixed
                      engine counts it the other way. Re-run (count_gain ~ wf / (1 - wf)).
  likely_misoriented  0.6 <= wf < 0.7: partly stranded, counted on the minority strand. Re-run.
  antisense_heavy     0.5 < wf < 0.6: about as many reads on each strand (unstranded protocol,
                      or antisense/noise-dominated). Not a strand error; re-counting gains < 1.5x.
                      The fixed probe leaves these as configured (no-call band 0.4-0.6); its
                      safety net counts the majority strand. Optional re-run.
  borderline          0.35 < wf <= 0.5 with chrM > 15% of mapped reads: may be misoriented
                      under the chrM dilution. Listed for checking, not re-run.

GEO text (overall design, data processing, extract protocol, description, library/kit/chemistry
fields) is searched for 5' / V(D)J wording: weak, independent evidence for 10x samples.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

csv, res, meta, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
probe_tsvs = [Path(p) for p in sys.argv[5:]]
out.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(csv, low_memory=False)
d = d[d.exon_hits.notna()].copy()


def stats(g):
    try:
        s = json.loads((res / g / "out" / "pileup_stats.json").read_text())
    except Exception:  # noqa: BLE001
        return {}
    return {"chrm_reads": s.get("chrm_reads"), "mapped": s.get("mapped_reads"),
            "multigene": s.get("multigene_reads")}


extra = pd.DataFrame([{"gsm": g, **stats(g)} for g in d.gsm])
d = d.merge(extra, on="gsm", how="left")
d["wf"] = d.wrong_strand / (d.wrong_strand + d.exon_hits)
d["chrm_share"] = d.chrm_reads / d.mapped
d["reverse_strand"] = d.reverse_strand.astype(str).str.lower() == "true"


d["strandedness"] = (1 - 2 * d.wf).abs()
d["count_gain"] = d.wf / (1 - d.wf)


def klass(r):
    if r.wf >= 0.7:
        return "misoriented"
    if r.wf >= 0.6:
        return "likely_misoriented"
    if r.wf > 0.5:
        return "antisense_heavy"
    if r.wf > 0.35 and (r.chrm_share or 0) > 0.15:
        return "borderline"
    return "ok"


d["class"] = d.apply(klass, axis=1)

# ── GEO text: 5' wording ─────────────────────────────────────────────────────
# "5'" alone also matches oligo sequences written 5'-...-3'; require a kit-like word after it
FIVE = re.compile(r"5\s*['’′`]\s*(?:gene|gex|kit|v\d|librar|chem|sequenc|end|capture|immune)|5\s*-?\s*prime|"
                  r"five[\s-]prime|\bv\s*\(\s*d\s*\)\s*j\b|\bvdj\b", re.I)
THREE = re.compile(r"3\s*['’′`]\s*(?:gene|gex|kit|v\d|librar|chem|sequenc|end|capture)|3\s*-?\s*prime|"
                   r"three[\s-]prime", re.I)
want = set(d.gsm[d["class"] != "ok"])
ctrl = d[(d["class"] == "ok") & d.protocol.isin(["10xv2", "10xv3", "10xv1", "10x-visium"])]
want |= set(ctrl.sample(n=min(3000, len(ctrl)), random_state=0).gsm)
import pyarrow.parquet as pq  # noqa: E402

pf = pq.ParquetFile(meta)
names = pf.schema_arrow.names
text_cols = [c for c in names if c in ("overall_design", "data_processing", "extract_protocol", "description",
                                       "characteristics", "library_construction", "library_construction_kit",
                                       "library_kit", "kit", "chemistry", "assay")
             or c.startswith("10x_") or "chemistry" in c or c.startswith("library_strategy")]
rows = []
for i in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(i, columns=["gsm_id", "gse_id"] + text_cols).to_pandas()
    t = t[t.gsm_id.isin(want)]
    if len(t):
        txt = t[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        rows.append(pd.DataFrame({"gsm": t.gsm_id, "gse": t.gse_id,
                                  "geo_5p": txt.str.contains(FIVE), "geo_3p": txt.str.contains(THREE)}))
g = pd.concat(rows).drop_duplicates("gsm") if rows else pd.DataFrame(columns=["gsm", "gse", "geo_5p", "geo_3p"])
d = d.merge(g, on="gsm", how="left")

# ── BAM probe (orientation-free) for the samples whose BAM was kept ─────────
if probe_tsvs:
    pr = pd.concat([pd.read_csv(p, sep="\t") for p in probe_tsvs if p.is_file()])
    pr = pr.rename(columns={"name": "gsm", "antisense_frac": "bam_antisense_frac"})
    d = d.merge(pr[["gsm", "bam_antisense_frac", "informative", "top50_share_sense", "top50_share_anti"]],
                on="gsm", how="left")

# ── tables ──────────────────────────────────────────────────────────────────
def per_protocol(x):
    n = len(x)
    f = x[x.wf > 0.5]
    m = x[x.geo_5p.notna()]
    return pd.Series({
        "samples": n, "wf_median": x.wf.median(), "flagged_gt50": len(f), "flagged_pct": 100 * len(f) / n,
        "misoriented_ge70": int((x["class"] == "misoriented").sum()),
        "likely_60_70": int((x["class"] == "likely_misoriented").sum()), "gt90": int((x.wf > 0.9).sum()),
        "antisense_heavy_50_60": int((x["class"] == "antisense_heavy").sum()),
        "borderline": int((x["class"] == "borderline").sum()),
        "reverse_strand_applied_in_flagged": int(f.reverse_strand.sum()),
        "flagged_wf_median": f.wf.median() if len(f) else float("nan"),
        "flagged_strandedness_median": f.strandedness.median() if len(f) else float("nan"),
        "ok_strandedness_median": x[x.wf <= 0.5].strandedness.median(),
        "flagged_studies": f.gse.nunique() if "gse" in f else float("nan"),
        "geo_5p_pct_flagged": 100 * m[m.wf > 0.5].geo_5p.mean() if (m.wf > 0.5).any() else float("nan"),
        "geo_5p_pct_ok": 100 * m[m.wf <= 0.5].geo_5p.mean() if (m.wf <= 0.5).any() else float("nan"),
    })


tab = d.groupby("protocol")[list(d.columns)].apply(per_protocol).sort_values("flagged_gt50", ascending=False)
tab.loc["ALL"] = per_protocol(d)
tab.to_csv(out / "per_protocol.csv")

# wf histogram per protocol (0.05 bins) for the figure / eyeballing bimodality
bins = [i / 20 for i in range(21)]
h = d.groupby("protocol").wf.apply(lambda s: pd.cut(s, bins, include_lowest=True).value_counts().sort_index())
h.unstack().to_csv(out / "wf_hist_per_protocol.csv")

# per-study view of flagged samples: a study is usually all one chemistry
flag = d[d.wf > 0.5]
st = flag.groupby(["protocol", "gse"]).agg(samples=("gsm", "size"), wf_median=("wf", "median"),
                                           geo_5p=("geo_5p", "mean")).reset_index()
tot = d.groupby("gse").gsm.size().rename("study_samples")
st = st.merge(tot, left_on="gse", right_index=True, how="left").sort_values("samples", ascending=False)
st.to_csv(out / "flagged_studies.csv", index=False)

cols = ["gsm", "gse", "protocol", "class", "wf", "strandedness", "count_gain", "exon_hits", "wrong_strand",
        "chrm_share", "reverse_strand",
        "geo_5p", "geo_3p"] + [c for c in ("bam_antisense_frac", "top50_share_anti") if c in d]
rerun = d[d["class"].isin(["misoriented", "likely_misoriented"])].sort_values(["protocol", "wf"], ascending=[True, False])
rerun[cols].to_csv(out / "rerun_samples.csv", index=False)
d[d["class"] == "antisense_heavy"][cols].to_csv(out / "antisense_heavy_samples.csv", index=False)
d[d["class"] == "borderline"][cols].to_csv(out / "borderline_samples.csv", index=False)
d[cols].to_csv(out / "all_samples_strand.csv", index=False)

with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 80):
    print(tab.round(3))
    print("\nflagged studies (top 40):\n", st.head(40).round(3).to_string(index=False))
    print("\nclass counts:", d["class"].value_counts().to_dict())
    if "bam_antisense_frac" in d:
        print(d[d.bam_antisense_frac.notna()][cols].round(3).to_string(index=False))
