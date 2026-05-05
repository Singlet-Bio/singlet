#!/usr/bin/env python3
"""Gap analysis: USA-eligible samples not yet quantified.

Uses `find` for fast filesystem enumeration, then joins against catalog.
"""
import json, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("/mnt/projects/debruinz_project/cellarium")
CATALOG_DIR = PROJECT / "catalog"
QUANT_DIR = PROJECT / "pipeline" / "quant"

# ── Protocol classifications ─────────────────────────────────────────
DROPLET = {
    "10xv3", "10xv2", "10xv3_5prime", "10xv4",
    "dropseq", "indrop", "parse", "bd_rhapsody",
    "seqwell", "dnbelab", "ddseq", "surecell", "splitseq",
    "scirna", "scopeseq",
}
MULTIOME = {"10x_multiome", "multiome_atac"}
CITE = {"citeseq"}
PLATE = {
    "smartseq2", "smartseq3", "smart-seq", "smartseq",
    "marsseq", "mars-seq", "mars_seq",
    "icell8", "cel-seq", "celseq", "celseq2",
    "strt-seq", "strt_seq", "strtseq",
    "quartzseq", "quartz-seq",
    "plate_based", "fluidigm_c1", "microwell",
}
NON_RNA = {
    "scatac", "10x_atac", "chipseq", "chip-seq", "methylation",
    "hi_c", "hi-c", "hic", "visium", "slideseq",
    "dnase_seq", "mnase_seq", "mirna_seq", "rip_seq",
}
SUSPECT = {"10x_suspect", "unknown", "unknown_sc", "snrna_unknown"}

def classify(proto):
    if not proto: return "unknown"
    p = proto.lower().strip()
    if p in DROPLET:  return "usa_droplet"
    if p in MULTIOME: return "usa_multiome"
    if p in CITE:     return "cite_rna_eligible"
    if p in PLATE:    return "plate_based"
    if p in NON_RNA:  return "non_rna"
    if p in SUSPECT:  return "suspect_triage"
    return "other"

print("=" * 80)
print("USA-MODE GAP ANALYSIS")
print("=" * 80)

# ── Step 1: Fast filesystem enumeration via find ─────────────────────
print("\n[1/5] Enumerating pipeline via find...")

# count matrix GSMs (.1pz or legacy .spz)
r = subprocess.run(
    ["find", str(QUANT_DIR), "-maxdepth", "3", "-name", "counts.1pz", "-o", "-name", "counts.spz"],
    capture_output=True, text=True, timeout=120
)
spz_gsms = set()
for line in r.stdout.strip().split("\n"):
    if line:
        parts = line.split("/")
        # .../GSE.../GSM.../counts.1pz or counts.spz
        for i, p in enumerate(parts):
            if p.startswith("GSM"):
                spz_gsms.add(p)
                break

# sample_manifest.json paths
r2 = subprocess.run(
    ["find", str(QUANT_DIR), "-maxdepth", "3", "-name", "sample_manifest.json"],
    capture_output=True, text=True, timeout=120
)
manifest_paths = []  # (gsm, full_path)
manifest_gsms = set()
for line in r2.stdout.strip().split("\n"):
    if line:
        parts = line.split("/")
        for i, p in enumerate(parts):
            if p.startswith("GSM"):
                manifest_paths.append((p, line.strip()))
                manifest_gsms.add(p)
                break

print(f"  count matrix GSMs:         {len(spz_gsms):>9,}")
print(f"  sample_manifest.json GSMs: {len(manifest_gsms):>9,}")

# ── Step 2: Read manifests for status ────────────────────────────────
print(f"\n[2/5] Reading {len(manifest_paths):,} manifests...")

completed = set()
failed = set()
fail_errors = {}
cell_counts = {}

for gsm, path in manifest_paths:
    try:
        with open(path) as f:
            mf = json.load(f)
        nc = mf.get("n_cells", 0) or 0
        cell_counts[gsm] = nc
        qc = mf.get("qc_status", "")
        err = mf.get("error", "")

        if nc > 0:
            completed.add(gsm)
        elif err or mf.get("status") == "failed":
            failed.add(gsm)
            fail_errors[gsm] = err
    except:
        pass

print(f"  Completed (cells > 0):     {len(completed):>9,}")
print(f"  Failed:                     {len(failed):>9,}")
print(f"  Other:                      {len(manifest_gsms) - len(completed) - len(failed):>9,}")

# ── Step 3: Load catalog ─────────────────────────────────────────────
print(f"\n[3/5] Loading master catalog...")
import pyarrow.parquet as pq

cat = pq.read_table(
    CATALOG_DIR / "geo_single_cell_catalog.parquet",
    columns=["gsm_id", "organism", "protocol_inferred", "protocol_confidence",
             "library_source", "library_layout", "read_count", "srx_accession",
             "instrument_platform"]
)
print(f"  Catalog samples: {cat.num_rows:,}")

gsm_col = cat.column("gsm_id").to_pylist()
proto_col = cat.column("protocol_inferred").to_pylist()
org_col = cat.column("organism").to_pylist()
conf_col = cat.column("protocol_confidence").to_pylist()
src_col = cat.column("library_source").to_pylist()
lay_col = cat.column("library_layout").to_pylist()
rc_col = cat.column("read_count").to_pylist()
srx_col = cat.column("srx_accession").to_pylist()
plat_col = cat.column("instrument_platform").to_pylist()

catalog = {}
elig_all = Counter()
for i in range(cat.num_rows):
    gsm = gsm_col[i]
    proto = str(proto_col[i]) if proto_col[i] else ""
    elig = classify(proto)
    elig_all[elig] += 1
    catalog[gsm] = {
        "protocol": proto, "eligibility": elig,
        "organism": str(org_col[i]).split(";")[0].strip() if org_col[i] else "Unknown",
        "confidence": str(conf_col[i]) if conf_col[i] else "",
        "lib_source": str(src_col[i]).upper() if src_col[i] else "",
        "lib_layout": str(lay_col[i]).upper() if lay_col[i] else "",
        "read_count": rc_col[i],
        "srx": str(srx_col[i]) if srx_col[i] else "",
        "platform": str(plat_col[i]).upper() if plat_col[i] else "",
    }

print(f"\n  Eligibility breakdown:")
for e, cnt in sorted(elig_all.items(), key=lambda x: -x[1]):
    print(f"    {e:25s}: {cnt:>9,}")

usa_eligible = {g for g, d in catalog.items()
                if d["eligibility"] in ("usa_droplet", "usa_multiome", "cite_rna_eligible")}
print(f"\n  Total USA-eligible:        {len(usa_eligible):>9,}")

# ── Step 4: Gap computation ──────────────────────────────────────────
print(f"\n[4/5] Computing gap...")

eligible_completed = usa_eligible & completed
eligible_failed = usa_eligible & failed
eligible_in_pipeline = usa_eligible & manifest_gsms
eligible_not_attempted = usa_eligible - manifest_gsms
gap = usa_eligible - eligible_completed

print(f"\n  USA-eligible:              {len(usa_eligible):>9,}")
print(f"  |-- Completed:             {len(eligible_completed):>9,}")
print(f"  |-- Failed:                 {len(eligible_failed):>9,}")
print(f"  |-- In pipeline (other):   {len(eligible_in_pipeline - eligible_completed - eligible_failed):>9,}")
print(f"  +-- Not yet attempted:     {len(eligible_not_attempted):>9,}")
print(f"\n  Total gap:                 {len(gap):>9,}")
print(f"  Completion rate:           {len(eligible_completed)/max(len(usa_eligible),1)*100:.1f}%")

# ── Step 5: Detailed breakdown ───────────────────────────────────────
print(f"\n[5/5] Breakdown of {len(gap):,} gap samples...")

gap_proto = Counter()
gap_org = Counter()
gap_conf = Counter()
risk = defaultdict(set)

for gsm in gap:
    cd = catalog.get(gsm, {})
    gap_proto[cd.get("protocol", "?")] += 1
    gap_org[cd.get("organism", "?")] += 1
    gap_conf[cd.get("confidence", "?")] += 1

    if not cd.get("srx"):
        risk["no_srx"].add(gsm)
    if "SINGLE" in cd.get("lib_layout", "") and "PAIRED" not in cd.get("lib_layout", ""):
        risk["single_end"].add(gsm)
    src = cd.get("lib_source", "")
    if src and src == "TRANSCRIPTOMIC":
        risk["not_sc_tagged"].add(gsm)
    plat = cd.get("platform", "")
    if plat and not any(x in plat for x in ("ILLUMINA", "DNBSEQ", "BGISEQ")):
        risk["non_illumina"].add(gsm)
    if cd.get("confidence") == "low":
        risk["low_confidence"].add(gsm)
    rc = cd.get("read_count")
    if rc:
        try:
            if int(rc) < 100000:
                risk["very_low_reads"].add(gsm)
        except: pass

print(f"\n  By protocol (top 20):")
for proto, cnt in gap_proto.most_common(20):
    print(f"    {proto:25s}: {cnt:>8,} ({cnt/len(gap)*100:5.1f}%)")

print(f"\n  By organism (top 15):")
for org, cnt in gap_org.most_common(15):
    print(f"    {org:35s}: {cnt:>8,} ({cnt/len(gap)*100:5.1f}%)")

print(f"\n  By confidence:")
for conf, cnt in gap_conf.most_common():
    print(f"    {conf:15s}: {cnt:>8,} ({cnt/len(gap)*100:5.1f}%)")

print(f"\n  Risk flags:")
for r_name in sorted(risk, key=lambda r: -len(risk[r])):
    print(f"    {r_name:25s}: {len(risk[r_name]):>8,}")
multi = sum(1 for gsm in gap if sum(1 for r in risk if gsm in risk[r]) >= 2)
print(f"    {'MULTI-RISK (>=2)':25s}: {multi:>8,}")

# Failure breakdown
if eligible_failed:
    print(f"\n  Failure categories ({len(eligible_failed):,} eligible failed):")
    fail_cats = Counter()
    for gsm in eligible_failed:
        err = fail_errors.get(gsm, "").lower()
        if "download" in err or "curl" in err:       cat_name = "download"
        elif "no r2" in err or "no_r2" in err:       cat_name = "no_r2"
        elif "permit" in err or "25856" in err:       cat_name = "permit_list"
        elif "timeout" in err or "timed out" in err:  cat_name = "timeout"
        elif "piscem" in err or "mapping" in err:     cat_name = "mapping"
        elif "protocol" in err:                       cat_name = "protocol_detect"
        elif "simpleaf" in err or "alevin" in err:    cat_name = "simpleaf_other"
        elif "qc" in err or "few cells" in err:       cat_name = "qc_failure"
        elif err:                                     cat_name = "other"
        else:                                         cat_name = "no_error_msg"
        fail_cats[cat_name] += 1
    for c, n in fail_cats.most_common():
        print(f"    {c:25s}: {n:>6,}")

# ── Summary ──────────────────────────────────────────────────────────
high_risk = risk.get("single_end", set()) | risk.get("non_illumina", set()) | risk.get("very_low_reads", set())
low_risk_unattempted = eligible_not_attempted - high_risk

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"  Master catalog:         {cat.num_rows:>9,}")
print(f"  USA-eligible:           {len(usa_eligible):>9,}")
print(f"  DONE:                   {len(eligible_completed):>9,}  ({len(eligible_completed)/max(len(usa_eligible),1)*100:.1f}%)")
print(f"  FAILED:                  {len(eligible_failed):>9,}")
print(f"  NOT ATTEMPTED:           {len(eligible_not_attempted):>9,}")
print(f"    Low-risk (clean):      {len(low_risk_unattempted):>9,}")
print(f"    High-risk (flagged):   {len(eligible_not_attempted & high_risk):>9,}")
