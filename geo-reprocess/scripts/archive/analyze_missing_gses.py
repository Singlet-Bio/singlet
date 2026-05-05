#!/usr/bin/env python3
"""Detailed analysis of GSEs present in external repos but missing from our catalog.

Uses NCBI GEO SOFT format to get actual series metadata (more reliable than GDS).
"""

import json
import re
import time
import sys
import requests
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

CATALOG_DIR = Path("/mnt/projects/debruinz_project/cellarium/catalog")


def fetch_geo_metadata(gse_ids):
    """Fetch GEO series metadata via SOFT format."""
    results = {}
    for i, gse in enumerate(gse_ids):
        try:
            geo_url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
            r = requests.get(geo_url, params={
                "acc": gse,
                "targ": "self",
                "form": "text",
                "view": "brief",
            }, timeout=20)
            
            if r.status_code == 200 and "!Series_title" in r.text:
                info = {"source": "geo_soft"}
                sample_ids = []
                for line in r.text.split("\n"):
                    line = line.strip()
                    if line.startswith("!Series_title"):
                        info["title"] = line.split("=", 1)[-1].strip()
                    elif line.startswith("!Series_summary"):
                        info["summary"] = (info.get("summary", "") + " " + line.split("=", 1)[-1].strip()).strip()
                    elif line.startswith("!Series_overall_design"):
                        info["overall_design"] = (info.get("overall_design", "") + " " + line.split("=", 1)[-1].strip()).strip()
                    elif line.startswith("!Series_type"):
                        info["gdstype"] = (info.get("gdstype", "") + line.split("=", 1)[-1].strip() + "; ").strip()
                    elif line.startswith("!Series_platform_id"):
                        info["gpl"] = (info.get("gpl", "") + " " + line.split("=", 1)[-1].strip()).strip()
                    elif line.startswith("!Series_sample_id"):
                        sample_ids.append(line.split("=", 1)[-1].strip())
                    elif "Series_sample_taxid" in line:
                        info["taxon_id"] = line.split("=", 1)[-1].strip()
                    elif line.startswith("!Series_submission_date"):
                        info["pdat"] = line.split("=", 1)[-1].strip()
                    elif line.startswith("!Series_relation"):
                        rel = line.split("=", 1)[-1].strip()
                        if "SuperSeries" in rel:
                            info["is_superseries"] = True
                            info["superseries_ref"] = rel
                        elif "SubSeries" in rel:
                            info["is_subseries"] = True
                            info["subseries_of"] = rel
                        elif "BioProject" in rel:
                            info["bioproject"] = rel
                    elif line.startswith("!Series_pubmed_id"):
                        info["pubmed"] = line.split("=", 1)[-1].strip()
                    elif line.startswith("!Series_geo_accession"):
                        info["accession"] = line.split("=", 1)[-1].strip()
                
                info["n_samples"] = len(sample_ids)
                info["sample_ids"] = sample_ids[:5]  # keep a few for reference
                if info.get("summary"):
                    info["summary"] = info["summary"][:600]
                results[gse] = info
            else:
                results[gse] = {"title": "FETCH_FAILED", "source": "none", 
                                "http_status": r.status_code,
                                "snippet": r.text[:200] if r.text else ""}
            time.sleep(0.4)
        except Exception as e:
            results[gse] = {"title": "ERROR", "source": "error", "error": str(e)[:100]}
            time.sleep(0.4)
        
        if (i + 1) % 20 == 0:
            print("  Fetched {}/{}".format(i + 1, len(gse_ids)))
    
    return results


def classify_gse(gse, meta):
    """Classify why a GSE is missing from our catalog."""
    title = (meta.get("title", "") or "").lower()
    summary = (meta.get("summary", "") or "").lower()
    gdstype = (meta.get("gdstype", "") or "").lower()
    overall_design = (meta.get("overall_design", "") or "").lower()
    n_samples = meta.get("n_samples", 0)
    gpl = (meta.get("gpl", "") or "").lower()
    combined = title + " " + summary + " " + overall_design
    
    # Superseries
    if meta.get("is_superseries"):
        return "superseries", "SuperSeries umbrella — subseries likely already in catalog"
    
    # Controlled access
    if any(x in combined for x in ["dbgap", "controlled access", "restricted", "phs0"]):
        return "controlled_access", "Controlled access — requires dbGaP/EGA authorization"
    
    # Pure epigenomic (ATAC/ChIP/CUT&Tag) without RNA
    epigenomic_kw = ["atac-seq", "atacseq", "atac seq", "chromatin accessibility",
                      "cut&tag", "cutandtag", "cut-and-tag", "chip-seq", "chipseq",
                      "dnase-seq", "hi-c", "hic"]
    rna_kw = ["rna-seq", "rnaseq", "rna seq", "transcriptom", "gene expression",
              "scrna", "snrna", "single-cell rna", "single cell rna"]
    
    is_epigenomic = any(x in combined for x in epigenomic_kw)
    is_rna = any(x in combined for x in rna_kw)
    
    sc_kw = ["single-cell", "single cell", "scrna", "sc-rna", "10x chromium",
             "10x genomics", "drop-seq", "dropseq", "indrop", "cel-seq",
             "smart-seq", "smartseq", "snrna", "sn-rna", "single-nucleus",
             "single nucleus", "droplet", "microfluidic"]
    is_sc = any(x in combined for x in sc_kw)
    
    if is_epigenomic and not is_rna:
        return "epigenomic_only", "Epigenomic only (ATAC/ChIP/CUT&Tag) — no RNA-seq"
    
    # Spatial only
    spatial_kw = ["spatial transcriptom", "visium", "slide-seq", "slideseq",
                  "merfish", "seqfish", "10x xenium", "stereo-seq", "seq-scope"]
    if any(x in combined for x in spatial_kw) and not is_sc:
        return "spatial_only", "Spatial transcriptomics without single-cell component"

    # Multiome
    if is_sc and is_epigenomic and is_rna:
        return "multiome", "Multiome (RNA+ATAC) — has RNA, should be in catalog"
    if is_sc and is_epigenomic:
        return "sc_epigenomic", "Single-cell epigenomic without clear RNA component"
    
    # Clear single-cell RNA-seq
    if is_sc and is_rna:
        # Check for subseries (component might be in catalog)
        if meta.get("is_subseries"):
            return "subseries_sc_rnaseq", "SubSeries sc-RNA-seq — parent or sibling may be in catalog"
        return "sc_rnaseq_gap", "Single-cell RNA-seq — genuine gap, should be in catalog"
    
    if is_sc and not is_rna and not is_epigenomic:
        # SC mentioned but no specific modality
        return "sc_unclear_modality", "Single-cell mentioned but modality unclear"
    
    # Bulk RNA-seq
    if is_rna and not is_sc:
        return "bulk_rnaseq", "Bulk RNA-seq — not single-cell"
    
    # Check for withdrawn/unavailable
    if meta.get("source") in ("none", "error"):
        return "unavailable", "Could not fetch metadata — may be withdrawn/embargoed"
    
    if not title and not summary:
        return "no_metadata", "No metadata available"
    
    return "other", "Unclassified: type='{}', title='{}'".format(
        gdstype[:40], (meta.get("title", "") or "")[:60])


def main():
    print("=" * 80)
    print("DETAILED ANALYSIS: GSEs in external repos but NOT in our catalog")
    print("=" * 80)
    
    # Load crossref
    xref = pd.read_parquet(str(CATALOG_DIR / "repository_gse_crossref.parquet"))
    missing = xref[~xref["in_our_catalog"] & (xref["n_repos"] > 0)].sort_values("n_repos", ascending=False)
    missing_gses = missing["gse_id"].tolist()
    print("\nTotal missing GSEs: {}".format(len(missing_gses)))

    # Check if any are in descriptions catalog (discovered but filtered)
    desc_path = CATALOG_DIR / "all_gse_descriptions.parquet"
    desc_gses = set()
    if desc_path.exists():
        desc_df = pd.read_parquet(str(desc_path), columns=["gse_id"])
        desc_gses = set(desc_df["gse_id"].unique())
        in_desc = [g for g in missing_gses if g in desc_gses]
        not_in_desc = [g for g in missing_gses if g not in desc_gses]
        print("  In descriptions catalog (discovered, filtered out): {}".format(len(in_desc)))
        print("  NOT in descriptions catalog (never discovered):     {}".format(len(not_in_desc)))

    # Fetch metadata
    print("\nFetching NCBI GEO metadata for {} GSEs...".format(len(missing_gses)))
    metadata = fetch_geo_metadata(missing_gses)
    
    sources = Counter(m.get("source", "?") for m in metadata.values())
    print("Metadata sources: {}".format(dict(sources)))

    # Classify
    print("\n" + "=" * 80)
    print("CLASSIFICATION")
    print("=" * 80)
    
    categories = Counter()
    cat_items = defaultdict(list)
    
    for gse in missing_gses:
        meta = metadata.get(gse, {})
        cat, reason = classify_gse(gse, meta)
        
        row = missing[missing["gse_id"] == gse].iloc[0]
        repos = [c.replace("in_", "") for c in ["in_cellxgene", "in_scea", "in_hca", "in_scp", "in_ucsc_cb"]
                 if row.get(c, False)]
        
        categories[cat] += 1
        cat_items[cat].append({
            "gse": gse,
            "title": (meta.get("title", "") or "?")[:90],
            "n_samples": meta.get("n_samples", "?"),
            "gpl": meta.get("gpl", ""),
            "pdat": meta.get("pdat", ""),
            "repos": repos,
            "n_repos": row.get("n_repos", 0),
            "reason": reason,
            "summary": (meta.get("summary", "") or "")[:250],
            "gdstype": meta.get("gdstype", ""),
            "superseries_ref": meta.get("superseries_ref", ""),
            "subseries_of": meta.get("subseries_of", ""),
            "in_desc": gse in desc_gses,
        })

    # Summary table
    print("\n{:30s} {:>5s}  {}".format("Category", "Count", "Description"))
    print("-" * 100)
    for cat, n in categories.most_common():
        sample_reason = cat_items[cat][0]["reason"]
        print("{:30s} {:5d}  {}".format(cat, n, sample_reason[:60]))
    print("{:30s} {:5d}".format("TOTAL", sum(categories.values())))

    # Detailed per-category
    for cat in [c for c, _ in categories.most_common()]:
        items = cat_items[cat]
        print("\n" + "=" * 80)
        print("{} ({} GSEs)".format(cat.upper(), len(items)))
        print("=" * 80)
        for item in sorted(items, key=lambda x: -x["n_repos"]):
            repos_str = ",".join(item["repos"])
            desc_flag = " [KNOWN]" if item["in_desc"] else " [NEVER_SEEN]"
            print("\n  {} (n={}, submitted={}{})".format(
                item["gse"], item["n_samples"], item["pdat"], desc_flag))
            print("    Title: {}".format(item["title"]))
            print("    Repos: [{}] (n_repos={})".format(repos_str, item["n_repos"]))
            if item["gdstype"]:
                print("    Type:  {}".format(item["gdstype"][:80]))
            if item["gpl"]:
                print("    GPL:   {}".format(item["gpl"]))
            if item["superseries_ref"]:
                print("    Super: {}".format(item["superseries_ref"][:100]))
            if item["subseries_of"]:
                print("    SubOf: {}".format(item["subseries_of"][:100]))
            print("    Why:   {}".format(item["reason"]))
            if item["summary"]:
                summary_short = item["summary"][:180].replace("\n", " ")
                print("    Sum:   {}...".format(summary_short))

    # Grand summary
    print("\n" + "=" * 80)
    print("ACTIONABLE SUMMARY")
    print("=" * 80)
    
    gap_cats = ["sc_rnaseq_gap", "multiome", "subseries_sc_rnaseq"]
    review_cats = ["sc_unclear_modality", "sc_epigenomic", "other", "unavailable", "no_metadata"]
    excluded_cats = ["epigenomic_only", "spatial_only", "bulk_rnaseq"]
    structural_cats = ["superseries", "controlled_access"]
    
    gaps = []
    for c in gap_cats:
        gaps.extend(cat_items.get(c, []))
    reviews = []
    for c in review_cats:
        reviews.extend(cat_items.get(c, []))
    excluded = []
    for c in excluded_cats:
        excluded.extend(cat_items.get(c, []))
    structural = []
    for c in structural_cats:
        structural.extend(cat_items.get(c, []))
    
    print("\n  A. GENUINE GAPS — should add to catalog:   {:3d} GSEs".format(len(gaps)))
    print("  B. NEEDS MANUAL REVIEW:                    {:3d} GSEs".format(len(reviews)))
    print("  C. CORRECTLY EXCLUDED (wrong assay/scope):  {:3d} GSEs".format(len(excluded)))
    print("  D. STRUCTURAL (superseries/controlled):     {:3d} GSEs".format(len(structural)))
    
    if gaps:
        print("\n--- A. GENUINE GAPS ---")
        for item in sorted(gaps, key=lambda x: -x["n_repos"]):
            print("  {} n={:>4s} {} [{}]".format(
                item["gse"], str(item["n_samples"]), item["title"][:55], ",".join(item["repos"])))
    
    if reviews:
        print("\n--- B. NEEDS REVIEW ---")
        for item in sorted(reviews, key=lambda x: -x["n_repos"]):
            print("  {} n={:>4s} {} [{}]".format(
                item["gse"], str(item["n_samples"]), item["title"][:55], ",".join(item["repos"])))

    if excluded:
        print("\n--- C. CORRECTLY EXCLUDED ---")
        for item in excluded:
            print("  {} — {}".format(item["gse"], item["reason"][:60]))

    if structural:
        print("\n--- D. STRUCTURAL ---")
        for item in structural:
            print("  {} — {}".format(item["gse"], item["reason"][:60]))

    # Discovery gap analysis
    print("\n" + "=" * 80)
    print("DISCOVERY GAP ANALYSIS")
    print("=" * 80)
    never_seen = [g for g in missing_gses if g not in desc_gses]
    known_but_filtered = [g for g in missing_gses if g in desc_gses]
    
    print("\n  Never discovered (not in any catalog):  {} GSEs".format(len(never_seen)))
    print("  Discovered but filtered out:            {} GSEs".format(len(known_but_filtered)))
    
    if never_seen:
        print("\n  Never-seen GSEs and their external repos:")
        for gse in never_seen[:30]:
            row = missing[missing["gse_id"] == gse].iloc[0]
            repos = [c.replace("in_", "") for c in ["in_cellxgene", "in_scea", "in_hca", "in_scp", "in_ucsc_cb"]
                     if row.get(c, False)]
            meta = metadata.get(gse, {})
            print("    {} [{}] — {}".format(gse, ",".join(repos), 
                  (meta.get("title", "") or "?")[:60]))


if __name__ == "__main__":
    main()
