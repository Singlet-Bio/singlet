#!/usr/bin/env python3
"""Verify superseries subseries presence in catalog and analyze multiome/gap GSEs.

For each superseries: resolve ALL subseries via GEO SOFT, check catalog presence.
For each multiome GSE: identify RNA-containing samples.
For each sc-RNA-seq gap: confirm modality and identify what needs processing.
"""

import json
import re
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

CATALOG_DIR = Path("/mnt/projects/debruinz_project/cellarium/catalog")

# From the analysis output
SUPERSERIES = {
    "GSE196830": "GSE196829", "GSE120716": "GSE117403", "GSE171892": "GSE171891",
    "GSE161267": "GSE164157", "GSE204684": "GSE204683", "GSE244594": "GSE244593",
    "GSE137400": "GSE137863", "GSE103275": "GSE103272", "GSE107957": "GSE107956",
    "GSE114727": "GSE114725", "GSE120508": "GSE120507", "GSE130775": "GSE130774",
    "GSE131258": "GSE131256", "GSE146737": "GSE146736", "GSE145929": "GSE145865",
    "GSE145928": "GSE145843", "GSE144175": "GSE161290", "GSE141428": "GSE141427",
    "GSE150681": "GSE150660", "GSE161383": "GSE161382", "GSE157997": "GSE157996",
    "GSE165555": "GSE165554", "GSE163532": "GSE163495", "GSE165839": "GSE165838",
    "GSE151202": "GSE151192", "GSE173096": "GSE172515", "GSE172357": "GSE172316",
    "GSE166179": "GSE193240", "GSE178362": "GSE178519", "GSE178454": "GSE178453",
    "GSE224959": "GSE225671", "GSE227191": "GSE227190", "GSE254090": "GSE254089",
    "GSE92280": "GSE92279",
}

MULTIOME = [
    "GSE196830",  # also a superseries — skip, already tracked above
    "GSE161267",  # also a superseries
    "GSE204684",  # also a superseries
    "GSE244594",  # also a superseries
    "GSE161383",  # also a superseries
    "GSE151202",  # also a superseries
    "GSE173096",  # also a superseries
    "GSE178362",  # also a superseries
    "GSE178454",  # also a superseries
    "GSE224959",  # also a superseries
    "GSE254090",  # also a superseries
]

# Pure sc-RNA-seq gaps + multiome + subseries gaps from the analysis
SC_RNASEQ_GAPS = [
    "GSE139324", "GSE152042", "GSE83139", "GSE153723", "GSE188528",
    "GSE234713", "GSE234933", "GSE75330", "GSE76381", "GSE135355",
    "GSE135354", "GSE156702", "GSE156339", "GSE156704", "GSE203273",
    "GSE213688", "GSE178451", "GSE252772", "GSE295862", "GSE77944",
    "GSE109979", "GSE130772", "GSE152197", "GSE240429",
]

SUBSERIES_GAPS = [
    "GSE135355", "GSE135354", "GSE156339", "GSE156702",  # subseries sc-RNA-seq
]


def fetch_superseries_subseries(gse_id):
    """Get all subseries of a superseries via GEO SOFT."""
    url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    r = requests.get(url, params={"acc": gse_id, "targ": "self", "form": "text", "view": "brief"}, timeout=20)
    subseries = []
    title = ""
    stype = ""
    if r.status_code == 200:
        for line in r.text.split("\n"):
            line = line.strip()
            if line.startswith("!Series_title"):
                title = line.split("=", 1)[-1].strip()
            elif line.startswith("!Series_type"):
                stype += line.split("=", 1)[-1].strip() + "; "
            elif line.startswith("!Series_relation"):
                val = line.split("=", 1)[-1].strip()
                # SuperSeries of: GSE123456
                m = re.search(r'SuperSeries of:?\s*(GSE\d+)', val)
                if m:
                    subseries.append(m.group(1))
                # Also check SubSeries of: ...
                m2 = re.search(r'SubSeries of:?\s*(GSE\d+)', val)
                if m2:
                    subseries.append(("parent", m2.group(1)))
    return {"title": title, "type": stype, "subseries": subseries}


def fetch_gse_metadata_brief(gse_id):
    """Get brief metadata for a GSE."""
    url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    try:
        r = requests.get(url, params={"acc": gse_id, "targ": "self", "form": "text", "view": "brief"}, timeout=20)
        if r.status_code != 200:
            return {"title": "FETCH_FAILED", "type": "", "n_samples": 0, "platforms": []}
        
        info = {"title": "", "type": "", "summary": "", "overall_design": "", 
                "n_samples": 0, "platforms": [], "organisms": [], "relations": []}
        sample_ids = []
        for line in r.text.split("\n"):
            line = line.strip()
            if line.startswith("!Series_title"):
                info["title"] = line.split("=", 1)[-1].strip()
            elif line.startswith("!Series_type"):
                info["type"] += line.split("=", 1)[-1].strip() + "; "
            elif line.startswith("!Series_summary"):
                info["summary"] += line.split("=", 1)[-1].strip() + " "
            elif line.startswith("!Series_overall_design"):
                info["overall_design"] += line.split("=", 1)[-1].strip() + " "
            elif line.startswith("!Series_platform_id"):
                info["platforms"].append(line.split("=", 1)[-1].strip())
            elif line.startswith("!Series_sample_id"):
                sample_ids.append(line.split("=", 1)[-1].strip())
            elif line.startswith("!Series_sample_taxid"):
                info["organisms"].append(line.split("=", 1)[-1].strip())
            elif line.startswith("!Series_relation"):
                info["relations"].append(line.split("=", 1)[-1].strip())
        info["n_samples"] = len(sample_ids)
        info["gsm_ids"] = sample_ids[:10]
        info["summary"] = info["summary"][:500]
        info["overall_design"] = info["overall_design"][:500]
        return info
    except Exception as e:
        return {"title": "ERROR: " + str(e)[:80], "type": "", "n_samples": 0, "platforms": []}


def main():
    # Load catalog GSE set
    print("Loading catalog...")
    cat = pd.read_parquet(str(CATALOG_DIR / "geo_single_cell_catalog.parquet"), columns=["gse_id", "gsm_id"])
    catalog_gses = set(cat["gse_id"].unique())
    catalog_gsms = set(cat["gsm_id"].unique())
    print(f"  Catalog: {len(catalog_gses):,} GSEs, {len(catalog_gsms):,} GSMs")

    # Also load descriptions catalog if available
    desc_path = CATALOG_DIR / "all_gse_descriptions.parquet"
    desc_gses = set()
    if desc_path.exists():
        desc_df = pd.read_parquet(str(desc_path), columns=["gse_id"])
        desc_gses = set(desc_df["gse_id"].unique())
        print(f"  Descriptions catalog: {len(desc_gses):,} GSEs")

    # Check quant pipeline for processed status
    quant_dir = Path("/mnt/projects/debruinz_project/cellarium/pipeline/quant")

    # =====================================================================
    # PART 1: SUPERSERIES VERIFICATION
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 1: SUPERSERIES VERIFICATION")
    print("  Checking if subseries of 34 SuperSeries are in our catalog")
    print("=" * 80)

    ss_all_covered = 0
    ss_partial = 0
    ss_none = 0
    ss_missing_subseries = []  # subseries that should be in catalog but aren't

    for super_gse, noted_sub in sorted(SUPERSERIES.items()):
        info = fetch_superseries_subseries(super_gse)
        subseries = [s for s in info["subseries"] if isinstance(s, str)]
        time.sleep(0.35)

        in_cat = [s for s in subseries if s in catalog_gses]
        not_in_cat = [s for s in subseries if s not in catalog_gses]
        in_desc = [s for s in not_in_cat if s in desc_gses]

        if len(not_in_cat) == 0 and len(subseries) > 0:
            status = "ALL_IN_CATALOG"
            ss_all_covered += 1
        elif len(in_cat) > 0:
            status = "PARTIAL"
            ss_partial += 1
        else:
            status = "NONE_IN_CATALOG"
            ss_none += 1

        print(f"\n  {super_gse} → {len(subseries)} subseries [{status}]")
        print(f"    Title: {info['title'][:80]}")
        print(f"    Type:  {info['type'][:80]}")
        if subseries:
            print(f"    In catalog:     {in_cat}")
            if not_in_cat:
                print(f"    NOT in catalog: {not_in_cat}")
                if in_desc:
                    print(f"    In desc (discovered but filtered): {in_desc}")
                # For missing subseries, fetch their metadata to understand why
                for sub_gse in not_in_cat:
                    sub_info = fetch_gse_metadata_brief(sub_gse)
                    time.sleep(0.35)
                    combined = (sub_info.get("title", "") + " " + sub_info.get("summary", "")).lower()
                    is_rna = any(x in combined for x in ["rna-seq", "rnaseq", "transcriptom", "gene expression", "scrna", "snrna"])
                    is_sc = any(x in combined for x in ["single-cell", "single cell", "scrna", "10x", "drop-seq", "smart-seq", "snrna", "single-nucleus", "droplet"])
                    is_atac = any(x in combined for x in ["atac", "chromatin", "chip-seq", "cut&tag"])
                    modality = []
                    if is_sc: modality.append("SC")
                    if is_rna: modality.append("RNA")
                    if is_atac: modality.append("ATAC")
                    if not modality: modality.append("?")
                    
                    print(f"      {sub_gse}: n={sub_info['n_samples']}, [{'/'.join(modality)}], {sub_info['title'][:60]}")
                    print(f"        type: {sub_info['type'][:60]}")
                    
                    if is_sc and is_rna:
                        ss_missing_subseries.append({
                            "gse": sub_gse,
                            "parent": super_gse,
                            "title": sub_info["title"],
                            "n_samples": sub_info["n_samples"],
                            "type": sub_info["type"],
                            "modality": "/".join(modality),
                        })
        else:
            # The noted_sub is what we saw in the SOFT — check that
            if noted_sub in catalog_gses:
                print(f"    Noted subseries {noted_sub}: IN CATALOG")
            else:
                print(f"    Noted subseries {noted_sub}: NOT IN CATALOG")

    print(f"\n--- SUPERSERIES SUMMARY ---")
    print(f"  All subseries in catalog:  {ss_all_covered}")
    print(f"  Partial coverage:          {ss_partial}")
    print(f"  No subseries in catalog:   {ss_none}")
    print(f"  Missing SC-RNA subseries:  {len(ss_missing_subseries)}")
    
    if ss_missing_subseries:
        print(f"\n  Missing SC-RNA subseries that should be added:")
        for item in ss_missing_subseries:
            print(f"    {item['gse']} (parent={item['parent']}, n={item['n_samples']}) [{item['modality']}] {item['title'][:60]}")

    # =====================================================================
    # PART 2: MULTIOME GSEs — RNA COMPONENT
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 2: MULTIOME GSEs — identifying RNA component")
    print("  Note: All 11 multiome GSEs are also superseries (handled above).")
    print("  Checking if any have RNA subseries NOT yet covered.")
    print("=" * 80)

    # The multiome GSEs are all superseries — their RNA subseries should have been
    # identified in Part 1. Print a cross-reference.
    multiome_rna_missing = [m for m in ss_missing_subseries if "RNA" in m["modality"]]
    print(f"\n  Multiome-derived RNA subseries missing from catalog: {len(multiome_rna_missing)}")
    for item in multiome_rna_missing:
        print(f"    {item['gse']} [{item['modality']}] n={item['n_samples']} — {item['title'][:60]}")

    # =====================================================================
    # PART 3: SC-RNA-SEQ GAPS — DETAILED INVESTIGATION
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 3: SC-RNA-SEQ GAP INVESTIGATION")
    print(f"  Investigating {len(SC_RNASEQ_GAPS)} GSEs classified as genuine gaps")
    print("=" * 80)

    gap_details = []
    for gse in SC_RNASEQ_GAPS:
        meta = fetch_gse_metadata_brief(gse)
        time.sleep(0.35)
        
        combined = (meta.get("title", "") + " " + meta.get("summary", "") + " " + meta.get("overall_design", "")).lower()
        
        # Determine modality more precisely
        is_10x = any(x in combined for x in ["10x chromium", "10x genomics", "chromium"])
        is_dropseq = any(x in combined for x in ["drop-seq", "dropseq"])
        is_smartseq = any(x in combined for x in ["smart-seq", "smartseq", "plate-based"])
        is_spatial = any(x in combined for x in ["visium", "spatial", "merfish", "slide-seq", "cosmx", "nanostring", "xenium", "stereo-seq"])
        is_rna = any(x in combined for x in ["rna-seq", "rnaseq", "transcriptom", "gene expression"])
        is_sc = any(x in combined for x in ["single-cell", "single cell", "scrna", "snrna", "single-nucleus", "droplet", "microfluidic"])
        is_atac = any(x in combined for x in ["atac", "chromatin", "chip-seq", "cut&tag"])
        
        # Check organism
        organisms = meta.get("organisms", [])
        is_human = any(t in ["9606"] for t in organisms) or "human" in combined or "homo sapiens" in combined
        is_mouse = any(t in ["10090"] for t in organisms) or "mouse" in combined or "mus musculus" in combined
        
        platform_str = ", ".join(meta.get("platforms", []))
        
        # Determine processability
        processable = True
        notes = []
        if is_spatial and not is_sc:
            processable = False
            notes.append("Spatial-only (no droplet-based SC)")
        if not is_rna and not is_sc:
            processable = False
            notes.append("No clear RNA/SC signal")
        if is_smartseq:
            notes.append("SmartSeq/plate-based")
        if is_10x:
            notes.append("10x Chromium")
        if is_dropseq:
            notes.append("Drop-seq")
        if not is_human and not is_mouse:
            orgs_list = [str(o) for o in organisms[:3]]
            notes.append(f"Non-human/mouse (taxids: {','.join(orgs_list)})")
        
        # Check if any GSMs are already processed
        gsm_ids = meta.get("gsm_ids", [])
        gsms_in_cat = [g for g in gsm_ids if g in catalog_gsms]
        
        # Check quant directory
        quant_exists = (quant_dir / gse).exists() if quant_dir.exists() else False
        
        gap_details.append({
            "gse": gse,
            "title": meta["title"],
            "n_samples": meta["n_samples"],
            "platforms": platform_str,
            "type": meta["type"],
            "processable": processable,
            "notes": notes,
            "is_sc": is_sc,
            "is_rna": is_rna,
            "is_spatial": is_spatial,
            "is_10x": is_10x,
            "is_smartseq": is_smartseq,
            "gsms_in_cat": len(gsms_in_cat),
            "quant_exists": quant_exists,
            "organisms": organisms[:3],
            "summary": meta["summary"][:200],
        })
        
        status = "PROCESSABLE" if processable else "SKIP"
        note_str = "; ".join(notes) if notes else "standard SC-RNA"
        print(f"\n  {gse} [{status}] n={meta['n_samples']}")
        print(f"    Title: {meta['title'][:80]}")
        print(f"    Type:  {meta['type'][:60]}")
        print(f"    GPL:   {platform_str}")
        print(f"    Notes: {note_str}")
        if gsms_in_cat:
            print(f"    GSMs already in catalog: {gsms_in_cat}/{len(gsm_ids)} checked")
        if quant_exists:
            print(f"    Quant dir exists: YES")

    # Tally
    processable = [d for d in gap_details if d["processable"]]
    not_processable = [d for d in gap_details if not d["processable"]]
    
    print(f"\n--- SC-RNA GAP SUMMARY ---")
    print(f"  Processable: {len(processable)}")
    print(f"  Skip:        {len(not_processable)}")
    
    if processable:
        total_samples = sum(d["n_samples"] for d in processable)
        print(f"\n  Processable GSEs ({total_samples} total samples):")
        for d in sorted(processable, key=lambda x: -x["n_samples"]):
            note_str = "; ".join(d["notes"]) if d["notes"] else "standard"
            print(f"    {d['gse']} n={d['n_samples']:>5} [{note_str}] {d['title'][:55]}")

    # =====================================================================
    # GRAND SUMMARY
    # =====================================================================
    print("\n" + "=" * 80)
    print("GRAND SUMMARY — ALL GAPS")
    print("=" * 80)
    
    all_missing_gses = set()
    
    # From superseries check
    for item in ss_missing_subseries:
        all_missing_gses.add(item["gse"])
    
    # From sc-rnaseq gaps that are processable
    for d in processable:
        all_missing_gses.add(d["gse"])
    
    print(f"\n  Total unique GSEs to add to catalog: {len(all_missing_gses)}")
    print(f"    From superseries (uncovered subseries): {len(ss_missing_subseries)}")
    print(f"    From sc-RNA-seq gaps (processable):     {len(processable)}")
    
    if all_missing_gses:
        print(f"\n  Complete list of GSEs to add:")
        for gse in sorted(all_missing_gses):
            print(f"    {gse}")
    
    # Save actionable list
    action_file = CATALOG_DIR / "gap_gses_to_add.json"
    action_data = {
        "superseries_missing_subseries": ss_missing_subseries,
        "sc_rnaseq_gaps": [d for d in gap_details if d["processable"]],
        "all_gses_to_add": sorted(all_missing_gses),
        "skip_spatial_only": [d["gse"] for d in gap_details if not d["processable"]],
    }
    # Clean for JSON serialization
    for item in action_data.get("sc_rnaseq_gaps", []):
        item.pop("gsms_in_cat", None)
    
    with open(str(action_file), "w") as f:
        json.dump(action_data, f, indent=2, default=str)
    print(f"\n  Saved actionable list to: {action_file}")


if __name__ == "__main__":
    main()
