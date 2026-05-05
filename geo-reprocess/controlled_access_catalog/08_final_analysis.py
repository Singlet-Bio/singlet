#!/usr/bin/env python3
"""
Final controlled-access catalog analysis and summary generation.

Produces:
1. Unified JSON catalog with deduplication
2. Per-repository analysis
3. Protocol and disease breakdown
4. Overlap analysis with GEO
5. Actionable access path recommendations
"""

import json
import os
import sys
from collections import defaultdict

import pandas as pd

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all_data():
    """Load all discovery results."""
    data = {}

    with open(os.path.join(OUTPUT_DIR, "controlled_access_catalog.json")) as f:
        data["dbgap"] = json.load(f)

    with open(os.path.join(OUTPUT_DIR, "ega_controlled_sc.json")) as f:
        data["ega"] = json.load(f)

    with open(os.path.join(OUTPUT_DIR, "portal_controlled_sc.json")) as f:
        data["portals"] = json.load(f)

    with open(os.path.join(OUTPUT_DIR, "additional_portals.json")) as f:
        data["additional"] = json.load(f)

    # Load GEO catalog for comparison
    geo_path = "/mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet"
    if os.path.exists(geo_path):
        data["geo"] = pd.read_parquet(geo_path)

    return data


def analyze_dbgap(data):
    """Analyze dbGaP controlled-access SC landscape."""
    projects = data["dbgap"]["dbgap_projects"]

    # Deduplicate
    by_sig = defaultdict(list)
    for uid, p in projects.items():
        sig = (p.get("name", ""), p.get("sra_run_count", 0))
        by_sig[sig].append(uid)
    dup_uids = set()
    for sig, uids in by_sig.items():
        for uid in uids[1:]:
            dup_uids.add(uid)
    unique = {uid: p for uid, p in projects.items() if uid not in dup_uids}

    # Classify
    sc_protos = {"scRNA", "snRNA", "10x_chromium", "atac", "multiome",
                 "cite-seq", "spatial", "merfish", "smart-seq", "drop-seq"}
    high_conf = {uid: p for uid, p in unique.items()
                 if set(p.get("protocols_inferred", [])) & sc_protos}

    # Protocol breakdown
    proto_counts = defaultdict(int)
    proto_runs = defaultdict(int)
    for p in high_conf.values():
        for pr in p.get("protocols_inferred", []):
            proto_counts[pr] += 1
            proto_runs[pr] += p.get("sra_run_count", 0)

    # Disease breakdown
    disease_counts = defaultdict(int)
    for p in high_conf.values():
        for d in p.get("disease_categories", []):
            disease_counts[d] += 1

    # phs accessions
    phs_list = [p["phs_accession"] for p in unique.values()
                if p.get("phs_accession")]

    return {
        "total_bioprojects": len(unique),
        "high_confidence_sc": len(high_conf),
        "total_sra_runs": sum(p.get("sra_run_count", 0) for p in unique.values()),
        "high_conf_sra_runs": sum(p.get("sra_run_count", 0) for p in high_conf.values()),
        "phs_accessions_found": len(phs_list),
        "phs_list": sorted(set(phs_list)),
        "protocol_by_projects": dict(sorted(proto_counts.items(), key=lambda x: -x[1])),
        "protocol_by_runs": dict(sorted(proto_runs.items(), key=lambda x: -x[1])),
        "disease_breakdown": dict(sorted(disease_counts.items(), key=lambda x: -x[1])),
        "projects_with_runs": sum(1 for p in high_conf.values() if p.get("sra_run_count", 0) > 0),
        "projects_dbgap_only": sum(1 for p in high_conf.values() if p.get("sra_run_count", 0) == 0),
    }


def analyze_ega(data):
    """Analyze EGA controlled-access SC landscape."""
    entries = data["ega"]["ega_entries"]
    studies = {k: v for k, v in entries.items() if k.startswith("EGAS")}
    datasets = {k: v for k, v in entries.items() if k.startswith("EGAD")}
    dacs = {k: v for k, v in entries.items() if k.startswith("EGAC")}

    # Classify protocols from study names
    proto_counts = defaultdict(int)
    disease_counts = defaultdict(int)

    for eid, entry in studies.items():
        fields = entry.get("fields", {})
        name_raw = fields.get("name", "")
        if isinstance(name_raw, list):
            name = name_raw[0].lower() if name_raw else ""
        else:
            name = str(name_raw).lower()

        desc_raw = fields.get("description", "")
        if isinstance(desc_raw, list):
            desc = desc_raw[0].lower() if desc_raw else ""
        else:
            desc = str(desc_raw).lower()

        text = name + " " + desc

        # Protocol
        if "10x" in text or "chromium" in text:
            proto_counts["10x_chromium"] += 1
        if "smart-seq" in text or "smartseq" in text:
            proto_counts["smart-seq"] += 1
        if "drop-seq" in text or "dropseq" in text:
            proto_counts["drop-seq"] += 1
        if "cite-seq" in text or "citeseq" in text:
            proto_counts["cite-seq"] += 1
        if "atac" in text:
            proto_counts["atac"] += 1
        if "multiome" in text:
            proto_counts["multiome"] += 1
        if "spatial" in text or "visium" in text:
            proto_counts["spatial"] += 1
        if "snrna" in text or "single nucleus" in text or "single-nucleus" in text:
            proto_counts["snRNA"] += 1
        if "scrna" in text or "single cell rna" in text or "single-cell rna" in text:
            proto_counts["scRNA"] += 1

        # Disease
        disease_map = {
            "cancer": ["cancer", "tumor", "carcinoma", "melanoma", "leukemia",
                       "lymphoma", "glioma", "sarcoma", "myeloma"],
            "neurological": ["brain", "neural", "neurodegen", "alzheimer"],
            "autoimmune": ["autoimmune", "inflammatory", "lupus", "rheumatoid",
                          "crohn", "colitis", "psoriasis"],
            "infectious": ["hiv", "covid", "sars", "infection", "viral", "sepsis",
                          "dengue", "malaria"],
            "pulmonary": ["lung", "pulmonary", "fibrosis", "copd"],
            "normal": ["healthy", "normal", "atlas", "reference"],
        }
        for cat, keywords in disease_map.items():
            if any(kw in text for kw in keywords):
                disease_counts[cat] += 1

    return {
        "total_studies": len(studies),
        "total_datasets": len(datasets),
        "total_dacs": len(dacs),
        "protocol_breakdown": dict(sorted(proto_counts.items(), key=lambda x: -x[1])),
        "disease_breakdown": dict(sorted(disease_counts.items(), key=lambda x: -x[1])),
        "geo_overlap": "~0.7% (6 entries mention GEO)",
        "dbgap_overlap": "~0.6% (5 entries mention dbGaP)",
    }


def analyze_portals(data):
    """Consolidate portal data."""
    portals = data["portals"]

    return {
        "gdc": {
            "sc_files": portals.get("gdc", {}).get("total_files", 0),
            "controlled_files": portals.get("gdc", {}).get("access_controlled", 46),
            "strategies": ["scRNA-Seq"],
            "projects": [p["project"] for p in portals.get("gdc", {}).get("projects", [])],
            "access": "Controlled via dbGaP; open for processed count matrices",
        },
        "encode": {
            "sc_experiments": portals.get("encode", {}).get("total_sc_experiments", 963),
            "assay_breakdown": portals.get("encode", {}).get("assay_breakdown", []),
            "access": "Open access (ENCODE is public)",
            "note": "ENCODE SC data is reprocessable; mostly snATAC and snRNA",
        },
        "hubmap": {
            "sc_datasets": portals.get("hubmap", {}).get("sc_datasets", 574),
            "total_datasets": portals.get("hubmap", {}).get("total_datasets", 6790),
            "access": "Mix - protected data requires DUA; processed data open",
        },
        "htan": {
            "dbgap_study": "phs002371",
            "assays": ["scRNA-seq", "snRNA-seq", "scATAC-seq", "snATAC-seq",
                       "10x Multiome", "10x Visium", "MERFISH", "CODEX", "CITE-seq"],
            "cases": "~3,000",
            "access": "Controlled via dbGaP (phs002371)",
        },
        "brain_nemo": {
            "biccn_cells": "50M+",
            "bican_cells": "200M+ (ongoing)",
            "assays": ["snRNA-seq", "scRNA-seq", "snATAC-seq", "10x Multiome",
                       "MERFISH", "Patch-seq"],
            "dbgap_studies": ["phs002673"],
            "access": "Mix of open + controlled via dbGaP",
        },
        "gtex": {
            "snrna_nuclei": "~200K across 8 tissues",
            "dbgap_study": "phs000424",
            "protocol": "10x Chromium 3' v3 (snRNA-seq)",
            "access": "Controlled via dbGaP; processed counts on GTEx portal",
        },
        "cellxgene": {
            "collections": portals.get("cellxgene", {}).get("total_collections", 359),
            "datasets": portals.get("cellxgene", {}).get("total_datasets", 2083),
            "access": "Open (processed h5ad files)",
            "note": "Overlap with GEO is high; these are processed data",
        },
    }


def geo_comparison(data):
    """Compare with GEO catalog."""
    geo = data.get("geo")
    if geo is None:
        return {}

    return {
        "geo_total_runs": len(geo),
        "geo_unique_gsms": geo["gsm_id"].nunique(),
        "geo_unique_gses": geo["gse_id"].nunique(),
        "geo_run_prefixes": {
            "SRR": int(geo["run_accession"].str.startswith("SRR", na=False).sum()),
            "ERR": int(geo["run_accession"].str.startswith("ERR", na=False).sum()),
            "DRR": int(geo["run_accession"].str.startswith("DRR", na=False).sum()),
        },
        "geo_organisms": dict(geo["organism"].value_counts().head(5)),
        "geo_protocols": dict(geo["protocol_inferred"].value_counts()),
        "overlap_assessment": {
            "ega_to_geo": "Near zero - GEO catalog has 0 ERR runs; EGA is entirely separate",
            "dbgap_to_geo": "Low - dbGaP controlled metadata hidden from SRA public; some studies may have GEO counterparts for open-access processed data",
            "gdc_to_geo": "Some overlap - GDC open data may be in GEO; controlled BAMs are not",
            "brain_to_geo": "Partial - BICCN has open GEO deposits for mouse; human controlled data is separate",
        },
    }


def main():
    data = load_all_data()

    print("Building final analysis...")

    dbgap_analysis = analyze_dbgap(data)
    ega_analysis = analyze_ega(data)
    portal_analysis = analyze_portals(data)
    geo_comp = geo_comparison(data)

    # Build final catalog
    final = {
        "metadata": {
            "generated": "2026-03-31",
            "description": "Controlled-access single-cell data catalog - comprehensive inventory",
            "geo_catalog_version": "processing_catalog.parquet",
        },
        "dbgap": dbgap_analysis,
        "ega": ega_analysis,
        "portals": portal_analysis,
        "geo_comparison": geo_comp,
    }

    # Print summary report
    print("\n" + "=" * 80)
    print("CONTROLLED-ACCESS SINGLE-CELL DATA INVENTORY")
    print("=" * 80)

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  REPOSITORY            STUDIES    EST. SAMPLES   ACCESS MODEL              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  dbGaP (SC projects)   {dbgap_analysis['high_confidence_sc']:>6}     50K-200K        dbGaP DAR + IRB             ║
║  EGA (SC studies)      {ega_analysis['total_studies']:>6}     30K-100K        EGA DAC per study           ║
║  GDC (SC files)           268     <10K            dbGaP DAR                 ║
║  ENCODE (SC+sn exps)      963     ~10K            Open access               ║
║  HuBMAP (SC datasets)     574     ~10K            DUA (most open)           ║
║  HTAN (sc atlas)            1     ~50K            dbGaP (phs002371)         ║
║  Brain/NeMO (BICCN)       50+     250M+ cells     dbGaP + open              ║
║  GTEx (snRNA)               1     ~200K nuclei    dbGaP (phs000424)         ║
║  GSA-Human (est.)       ~1000     unknown         Chinese DAC               ║
║  JGAS (est.)             ~100     unknown         Japanese NBDC             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  GEO (our catalog)     {geo_comp.get('geo_unique_gses', '?'):>6}     {geo_comp.get('geo_unique_gsms', '?'):>6} GSMs    Open (already cataloged)    ║
╚══════════════════════════════════════════════════════════════════════════════╝""")

    print(f"""
PROTOCOL BREAKDOWN (dbGaP projects):
  {json.dumps(dbgap_analysis['protocol_by_projects'], indent=4)}

PROTOCOL BREAKDOWN (EGA studies):
  {json.dumps(ega_analysis['protocol_breakdown'], indent=4)}

DISEASE BREAKDOWN (dbGaP):
  {json.dumps(dbgap_analysis['disease_breakdown'], indent=4)}

GEO OVERLAP ANALYSIS:
  - EGA overlap with GEO: {ega_analysis['geo_overlap']}
  - GEO has ZERO ERR-prefix runs → EGA data is entirely non-overlapping
  - dbGaP controlled metadata is hidden from SRA public → low overlap
  - Most controlled-access SC data is unique and not in our GEO catalog

KEY dbGaP phs ACCESSIONS FOUND ({len(dbgap_analysis['phs_list'])}):
  {', '.join(dbgap_analysis['phs_list'][:20])}
  {'...' if len(dbgap_analysis['phs_list']) > 20 else ''}
""")

    # Save
    outfile = os.path.join(OUTPUT_DIR, "final_catalog_analysis.json")
    with open(outfile, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"Full analysis saved to {outfile}")


if __name__ == "__main__":
    main()
