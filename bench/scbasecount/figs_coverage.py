"""Coverage figures and tables for docs/scbasecount-comparison.md (runs anywhere; inputs are the small join tables).

Usage: python figs_coverage.py COVERAGE_DIR CAMPAIGN_SAMPLES_CSV OUT_DIR
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COV, CAMP, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
SINGLET, SBC, AUTHOR, GRAY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#8f8e88", "#d9d8d3"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LIGHT, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

a = pd.read_parquet(os.path.join(COV, "singlet_atlas_joined.parquet"))
s = pd.read_parquet(os.path.join(COV, "sbc_joined.parquet"))
A = a[a.status.isin(["DONE", "DONE_QC_WARN"])].copy()
A["in_sbc"] = A.srx.isin(set(s.srx_accession))
s["in_singlet_atlas"] = s.srx_accession.isin(set(A.srx.dropna()))
camp = pd.read_csv(CAMP, low_memory=False)
camp["ws"] = camp.wrong_strand / (camp.wrong_strand + camp.exon_hits)
tables = {}

# ---- headline numbers
n_both = int(s.in_singlet_atlas.sum())
head = {
    "singlet_samples": len(A), "singlet_studies": int(A.gse_id.nunique()),
    "sbc_samples": len(s), "sbc_geo_samples": int(s.is_geo.sum()),
    "sbc_geo_studies": int(s[s.is_geo].gse.nunique() + s[s.is_geo & s.gse.isna()].ncbi_study.nunique()),
    "both_samples": n_both, "both_studies": int(A[A.in_sbc].gse_id.nunique()),
    "singlet_only": int((~A.in_sbc).sum()), "sbc_only": int((~s.in_singlet_atlas).sum()),
    "sbc_only_geo": int((s.is_geo & ~s.in_singlet_atlas).sum()), "sbc_non_geo": int((~s.is_geo).sum()),
    "singlet_non10x": int((~A.assay_family.fillna("").str.startswith("10x")).sum()),
    "sbc_studies_total": int(s.ncbi_study.where(s.ncbi_study.astype(str) != "nan").nunique()),
}
A10 = A[A.assay_family.fillna("").str.startswith("10x")]
head["singlet_10x"] = len(A10)
head["singlet_10x_in_sbc"] = int(A10.in_sbc.sum())
json.dump(head, open(os.path.join(OUT, "coverage_headline.json"), "w"), indent=1)

# ---- Fig 1: the two atlases as aligned bars (sample-level overlap)
fig, ax = plt.subplots(figsize=(8, 2.6))
rows = [("Singlet", [("in both", n_both, SINGLET), ("only here", head["singlet_only"], "#9ec3ef")]),
        ("scBaseCount", [("in both", n_both, SBC), ("only here, GEO", head["sbc_only_geo"], "#f5b496"),
                         ("only here, not GEO", head["sbc_non_geo"], LIGHT)])]
for y, (name, segs) in enumerate(rows[::-1]):
    x = 0
    for lab, v, col in segs:
        ax.barh(y, v, left=x, color=col, height=0.55, edgecolor="#fcfcfb", linewidth=2)
        if v > 6000:
            ax.text(x + v / 2, y, f"{lab}\n{v:,}", ha="center", va="center", fontsize=8.5,
                    color="white" if col in (SINGLET, SBC) else INK)
        x += v
    ax.text(x + 800, y, f"{x:,}", va="center", color=INK2, fontsize=9)
ax.set_yticks([0, 1], ["scBaseCount", "Singlet"])
ax.set_xlim(0, 84000)
ax.set_xlabel("samples (one GEO GSM / SRA experiment each)")
ax.set_title("Samples in each atlas, matched by SRA experiment", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_coverage_overlap.png"), dpi=200)
plt.close(fig)

# ---- Fig 2: Singlet atlas by assay family, in scBaseCount or not
fam = A.groupby(A.assay_family.fillna("Unknown")).in_sbc.agg(["size", "sum"]).sort_values("size")
fam.columns = ["samples", "in_sbc"]
tables["singlet_by_assay"] = fam.sort_values("samples", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.6))
y = range(len(fam))
ax.barh(y, fam.in_sbc, color=SINGLET, height=0.6, edgecolor="#fcfcfb", linewidth=2, label="also in scBaseCount")
ax.barh(y, fam.samples - fam.in_sbc, left=fam.in_sbc, color="#9ec3ef", height=0.6, edgecolor="#fcfcfb", linewidth=2,
        label="only in Singlet")
for i, (n, k) in enumerate(zip(fam.samples, fam.in_sbc)):
    ax.text(n + 400, i, f"{n:,}  ({k / n:.0%} in scBaseCount)", va="center", fontsize=8, color=INK2)
ax.set_yticks(list(y), fam.index)
ax.set_xlim(0, fam.samples.max() * 1.35)
ax.set_xlabel("Singlet atlas samples")
ax.legend(frameon=False, loc="lower right")
ax.set_title("Singlet's atlas by assay: what scBaseCount also has", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_singlet_by_assay.png"), dpi=200)
plt.close(fig)

# ---- Fig 3: GEO samples scBaseCount has and Singlet's atlas lacks, by Singlet catalog status
lab = {"skip_plate_based": "Singlet classed it plate-based (skipped)", "pending": "queued, not yet processed",
       "done": "processed, study not in the atlas release", "not in Singlet GEO catalog": "not in Singlet's GEO catalog",
       "skip_atac": "Singlet classed it ATAC (skipped)", "skip_vdj": "Singlet classed it V(D)J (skipped)",
       "skip_non_rna": "classed non-RNA (skipped)", "in_progress": "in progress", "skip_spatial": "classed spatial (skipped)",
       "rejected": "rejected at curation"}
# rebuild the "scBaseCount GEO sample, not in the atlas" list on the same atlas definition as Fig 1
cat = pd.read_parquet(os.path.join(COV, "catalog_map.parquet")).drop_duplicates("gsm_id").set_index("gsm_id")
w = s[s.is_geo & ~s.in_singlet_atlas].copy()
w["cat_processing_status"] = w.geo_gsm.map(cat.processing_status)
api_status = a.set_index("gsm_id").status
w.loc[w.geo_gsm.map(api_status).isin(["FAIL", "HARD_FAIL"]), "cat_processing_status"] = "failed"
lab["failed"] = "processed, failed Singlet QC"
st = w.cat_processing_status.fillna("not in Singlet GEO catalog")
st = st.where(st.isin(lab.keys()), "other (failures, bulk, ...)")
cnt = st.map(lambda x: lab.get(x, x)).value_counts().sort_values()
tables["sbc_geo_missing_reason"] = pd.crosstab(st.map(lambda x: lab.get(x, x)), w.tech_10x)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh(range(len(cnt)), cnt.values, color=SBC, height=0.6)
for i, v in enumerate(cnt.values):
    ax.text(v + 60, i, f"{v:,}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(cnt)), cnt.index)
ax.set_xlabel("GEO samples in scBaseCount but not in Singlet's atlas")
ax.set_title(f"{len(w):,} GEO samples in scBaseCount, not in Singlet: why", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_sbc_geo_missing_reason.png"), dpi=200)
plt.close(fig)

# ---- Fig 4: organism
s["org"] = s.organism.astype(str)
A["org"] = A.organism.astype(str).str.split(";").str[0].str.strip()
top = list(pd.concat([A.org, s.org]).value_counts().head(9).index)
oc = pd.DataFrame({"Singlet": A.org.where(A.org.isin(top), "other").value_counts(),
                   "scBaseCount": s.org.where(s.org.isin(top), "other").value_counts()}).fillna(0).astype(int)
oc = oc.loc[[t for t in top if t in oc.index] + (["other"] if "other" in oc.index else [])]
tables["by_organism"] = oc
fig, ax = plt.subplots(figsize=(8, 3.6))
import numpy as np  # noqa: E402

x = np.arange(len(oc))
ax.bar(x - 0.2, oc.Singlet, 0.38, color=SINGLET, label="Singlet")
ax.bar(x + 0.2, oc.scBaseCount, 0.38, color=SBC, label="scBaseCount")
ax.set_yscale("log")
ax.set_xticks(x, [t.replace(" ", "\n", 1) for t in oc.index], fontsize=8)
ax.set_ylabel("samples (log)")
ax.legend(frameon=False)
ax.set_title("Samples by organism", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_by_organism.png"), dpi=200)
plt.close(fig)

# ---- label agreement on shared samples, with Singlet's wrong-strand fraction
both = s[s.in_singlet_atlas].merge(A[["srx", "gsm_id", "assay_family", "protocol", "modality"]], left_on="srx_accession",
                                   right_on="srx").merge(camp[["gsm", "ws"]], left_on="gsm_id", right_on="gsm", how="left")
tables["label_agreement"] = pd.crosstab(both.assay_family.fillna("Unknown"), both.tech_10x)
both["strand"] = pd.cut(both.ws, [-1, 0.2, 0.5, 0.9, 1.01], labels=["<20%", "20-50%", "50-90%", ">90%"])
tables["strand_by_sbc_label"] = pd.crosstab(both.tech_10x, both.strand)
tables["prep"] = pd.crosstab(both.modality.fillna("unknown"), both.cell_prep)
# Fig 5: share of shared samples Singlet counted mostly on the wrong strand, by scBaseCount's library label
lbl = {"3_prime_gex": "3' GEX", "5_prime_gex": "5' GEX", "multiome": "Multiome GEX", "vdj": "V(D)J",
       "feature_barcoding": "feature barcode", "other": "other"}
bb = both[both.ws.notna() & both.tech_10x.isin(lbl)]
rate = bb.groupby("tech_10x").ws.agg(n="size", bad=lambda v: (v > 0.5).sum())
rate["share"] = rate.bad / rate.n
rate = rate.loc[[k for k in lbl if k in rate.index]]
tables["wrong_strand_rate_by_sbc_label"] = rate
fig, ax = plt.subplots(figsize=(8, 3.0))
ax.bar(range(len(rate)), rate.share * 100, color=[SBC if k == "5_prime_gex" else SINGLET for k in rate.index], width=0.6)
for i, (k, r) in enumerate(rate.iterrows()):
    ax.text(i, r.share * 100 + 0.6, f"{r.share:.0%}\n{int(r.bad):,} of {int(r.n):,}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(len(rate)), [lbl[k] for k in rate.index])
ax.set_ylabel("% with >50% wrong-strand reads")
ax.set_ylim(0, rate.share.max() * 100 * 1.45)
ax.set_title("Shared samples Singlet counted mostly antisense, by scBaseCount's library type", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_strand_vs_sbc_label.png"), dpi=200)
plt.close(fig)

with open(os.path.join(OUT, "coverage_tables.md"), "w", encoding="utf-8") as fh:
    fh.write("# Coverage tables (generated by bench/scbasecount/figs_coverage.py)\n\n```json\n" + json.dumps(head, indent=1) + "\n```\n")
    for k, t in tables.items():
        fh.write(f"\n## {k}\n\n" + t.to_markdown() + "\n")
print(json.dumps(head, indent=1))
for k, t in tables.items():
    print("\n==", k)
    print(t.to_string())
