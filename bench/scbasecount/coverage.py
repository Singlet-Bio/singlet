"""Coverage of Singlet vs scBaseCount (2026-01-12 release), joined by SRA experiment (SRX).

Inputs (from fetch_meta.py): META/sbc_samples.parquet, META/singlet_gsm.parquet, META/catalog_map.parquet,
and the frozen per-sample study list PLAN (plan.tsv, 8,306 studies).

Steps
  1. Singlet atlas samples = /api/gsm rows of plan studies; GSM -> SRX from Singlet's processing catalog.
  2. scBaseCount samples = unique SRX of the GeneFull_Ex50pAS tables (one row per SRX/ERX/NRX).
  3. For scBaseCount SRX that Singlet's GEO catalog does not know, ask NCBI esummary (db=sra, by entrez_id) whether
     the sample alias is a GSM (i.e. GEO-deposited) and which GSE it belongs to.
  4. Write per-sample join tables and summary counts to OUT.
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

META, PLAN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "singlet-bench-scbasecount/1.0"}


def fetch(url, data=None, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print("retry", e, flush=True)
            time.sleep(3 + 5 * i)


plan = pd.read_csv(PLAN, sep="\t")
plan_gse = set(plan.gse)
sg = pd.read_parquet(os.path.join(META, "singlet_gsm.parquet"))
cat = pd.read_parquet(os.path.join(META, "catalog_map.parquet"))
sbc = pd.read_parquet(os.path.join(META, "sbc_samples.parquet"))
sbc = sbc[sbc.feature == "GeneFull_Ex50pAS"].drop_duplicates("srx_accession").copy()
print("singlet rows", len(sg), "status", sg.status.value_counts().to_dict())
print("sbc srx", len(sbc))

# --- Singlet atlas ------------------------------------------------------------------------------------------------
atlas = sg[sg.gse_id.isin(plan_gse)].copy()
print("atlas rows (plan studies)", len(atlas), "studies", atlas.gse_id.nunique(), "status",
      atlas.status.value_counts().to_dict())
cat1 = cat.drop_duplicates("gsm_id").set_index("gsm_id")
atlas["srx"] = atlas.gsm_id.map(cat1.srx_accession)
atlas["submission_date"] = atlas.gsm_id.map(cat1.submission_date)
atlas["catalog_protocol"] = atlas.gsm_id.map(cat1.protocol_inferred)
# fall back on SRR -> SRX through the catalog's srr lists
srr2srx = {}
for srx, srrs in zip(cat.srx_accession, cat.srr_accessions):
    if isinstance(srx, str) and isinstance(srrs, str):
        for r in re.split(r"[;,\s]+", srrs):
            if r:
                srr2srx[r] = srx
miss = atlas.srx.isna() | (atlas.srx.astype(str).str.len() < 4)
atlas.loc[miss, "srx"] = atlas.loc[miss, "srr_ids"].map(
    lambda s: next((srr2srx[r] for r in str(s).split(",") if r in srr2srx), None))
print("atlas samples without SRX", int(atlas.srx.isna().sum()))

# --- scBaseCount -> GEO -------------------------------------------------------------------------------------------
srx2gsm = cat.dropna(subset=["srx_accession"]).drop_duplicates("srx_accession").set_index("srx_accession")
sbc["gsm"] = sbc.srx_accession.map(srx2gsm.gsm_id)
sbc["gse"] = sbc.srx_accession.map(srx2gsm.gse_id)
sbc["in_singlet_catalog"] = sbc.gsm.notna()
unk = sbc[~sbc.in_singlet_catalog & sbc.srx_accession.str.match(r"^[SED]RX")]
print("sbc srx unknown to the Singlet catalog", len(unk))

cache = os.path.join(OUT, "ncbi_esummary_cache.jsonl")
seen = {}
if os.path.exists(cache):
    for line in open(cache):
        r = json.loads(line)
        seen[r["uid"]] = r
todo = [str(int(u)) for u in unk.entrez_id if str(int(u)) not in seen]
print("esummary to fetch", len(todo), flush=True)
with open(cache, "a") as fh:
    for i in range(0, len(todo), 200):
        ids = ",".join(todo[i:i + 200])
        x = fetch("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                  data=urllib.parse.urlencode({"db": "sra", "id": ids}).encode())
        root = ET.fromstring(x)
        for ds in root.findall("DocSum"):
            uid = ds.findtext("Id")
            expxml = next((it.text for it in ds.findall("Item") if it.get("Name") == "ExpXml"), "") or ""
            m_s = re.search(r'<Sample acc="([^"]+)" name="([^"]*)"', expxml)
            m_t = re.search(r"<Title>(.*?)</Title>", expxml)
            m_st = re.search(r'<Study acc="([^"]+)" name="([^"]*)"', expxml)
            m_bp = re.search(r"<Bioproject>(.*?)</Bioproject>", expxml)
            m_center = re.search(r'center_name="([^"]*)"', expxml)
            title = m_t.group(1) if m_t else ""
            gsm = re.search(r"GSM\d+", (m_s.group(2) if m_s else "") + " " + title)
            rec = {"uid": uid, "sample_acc": m_s.group(1) if m_s else "", "sample_name": m_s.group(2) if m_s else "",
                   "title": title, "study": m_st.group(1) if m_st else "", "bioproject": m_bp.group(1) if m_bp else "",
                   "center": m_center.group(1) if m_center else "", "gsm": gsm.group(0) if gsm else ""}
            seen[uid] = rec
            fh.write(json.dumps(rec) + "\n")
        time.sleep(0.4)
        if i % 4000 == 0:
            print("esummary", i, flush=True)
es = pd.DataFrame(seen.values())
es["uid"] = es.uid.astype("int64")
sbc = sbc.merge(es.add_prefix("ncbi_"), left_on="entrez_id", right_on="ncbi_uid", how="left")
sbc["geo_gsm"] = sbc.gsm.fillna(sbc.ncbi_gsm.where(sbc.ncbi_gsm.astype(str).str.startswith("GSM")))
sbc["is_geo"] = sbc.geo_gsm.notna() & (sbc.geo_gsm.astype(str) != "")
sbc["in_singlet_atlas"] = sbc.srx_accession.isin(set(atlas.srx.dropna()))
atlas["in_sbc"] = atlas.srx.isin(set(sbc.srx_accession))
# GSE-level: a Singlet study is "in scBaseCount" when any of its SRX is
atlas_gse_in_sbc = atlas.groupby("gse_id").in_sbc.any()
cat_srx = set(cat.srx_accession.dropna())
sg_all_srx = set(sg.gsm_id.map(cat1.srx_accession).dropna())

sbc.to_parquet(os.path.join(OUT, "sbc_joined.parquet"))
atlas.to_parquet(os.path.join(OUT, "singlet_atlas_joined.parquet"))
# catalog row for every GEO sample scBaseCount has that the atlas lacks: why is it missing from Singlet?
geo_only = sbc[sbc.is_geo & ~sbc.in_singlet_atlas]
cg = cat.drop_duplicates("gsm_id").set_index("gsm_id")
why = geo_only[["srx_accession", "geo_gsm", "gse", "organism", "tech_10x", "cell_prep", "obs_count"]].copy()
for c in ["processing_status", "exclusion_reason", "protocol_inferred", "gse_id", "submission_date", "target_assay"]:
    why["cat_" + c] = why.geo_gsm.map(cg[c]) if c in cg else None
why["gse_in_plan"] = why.cat_gse_id.isin(plan_gse)
why["in_singlet_api"] = why.geo_gsm.isin(set(sg.gsm_id))
why.to_parquet(os.path.join(OUT, "sbc_geo_not_in_singlet.parquet"))

summary = {
    "singlet_atlas_samples": int(len(atlas)), "singlet_atlas_studies": int(atlas.gse_id.nunique()),
    "singlet_atlas_samples_with_srx": int(atlas.srx.notna().sum()),
    "sbc_srx": int(len(sbc)), "sbc_geo_srx": int(sbc.is_geo.sum()),
    "sbc_geo_gse": int(pd.concat([sbc.gse.dropna()]).nunique()),
    "overlap_samples": int(sbc.in_singlet_atlas.sum()),
    "overlap_studies_singlet_side": int(atlas_gse_in_sbc.sum()),
    "singlet_only_samples": int((~atlas.in_sbc).sum()),
    "sbc_only_samples": int((~sbc.in_singlet_atlas).sum()),
    "sbc_only_geo_samples": int((sbc.is_geo & ~sbc.in_singlet_atlas).sum()),
    "sbc_geo_not_in_singlet_by_reason": why.cat_processing_status.fillna("not in Singlet GEO catalog").value_counts().head(25).to_dict(),
    "sbc_geo_not_in_singlet_gse_in_plan": int(why.gse_in_plan.sum()),
}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
