#!/usr/bin/env python3
"""
Phase 2: Get detailed metadata for discovered dbGaP single-cell studies.

The gap database IDs from Phase 1 are dbGaP study records.
This script fetches their detailed metadata including:
- phs accession numbers
- Study descriptions
- Number of participants/samples
- Associated SRA BioProject accessions
- Disease/phenotype information

Also queries SRA directly for library_source=TRANSCRIPTOMIC SINGLE CELL
studies and cross-references against our GEO catalog.
"""

import requests
import xml.etree.ElementTree as ET
import json
import time
import os
import re
from collections import defaultdict

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DELAY = 0.35


def esearch(db, term, retmax=100000):
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {"db": db, "term": term, "retmax": retmax, "retmode": "json", "usehistory": "y"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    result = r.json()["esearchresult"]
    return {
        "count": int(result["count"]),
        "ids": result["idlist"],
        "webenv": result.get("webenv"),
        "query_key": result.get("querykey"),
    }


def efetch_xml(db, webenv, query_key, retstart, retmax):
    url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {
        "db": db, "query_key": query_key, "WebEnv": webenv,
        "retstart": retstart, "retmax": retmax, "retmode": "xml",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.text


def fetch_dbgap_study_details():
    """
    Fetch details for all dbGaP single-cell related studies.
    """
    # Combined query
    query = (
        '("single cell"[All Fields] OR "single-cell"[All Fields] OR '
        '"scRNA"[All Fields] OR "snRNA"[All Fields] OR '
        '"10x genomics"[All Fields] OR "10x chromium"[All Fields] OR '
        '"Drop-seq"[All Fields] OR "single nucleus"[All Fields] OR '
        '"scATAC"[All Fields] OR "CITE-seq"[All Fields])'
    )
    
    print("Searching dbGaP gap database...")
    time.sleep(DELAY)
    search = esearch("gap", query, retmax=100000)
    total = search["count"]
    print(f"  Found {total} studies")
    
    if not search["webenv"]:
        print("  No webenv, aborting")
        return []
    
    studies = []
    batch_size = 100
    fetched = 0
    
    while fetched < total:
        time.sleep(DELAY)
        try:
            xml_text = efetch_xml("gap", search["webenv"], search["query_key"],
                                  fetched, batch_size)
            
            # Parse the XML response
            # dbGaP returns DocumentSummary records
            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError:
                # Sometimes the XML is malformed, try wrapping
                try:
                    root = ET.fromstring(f"<root>{xml_text}</root>")
                except:
                    fetched += batch_size
                    continue
            
            # Look for study records
            for doc in root.iter("DocumentSummary"):
                study = {}
                
                # Extract key fields
                for elem_name in ["d_study_id", "study_accession", "study_name", 
                                   "d_disease_name", "d_participants", "d_platform",
                                   "d_description", "study_description"]:
                    elem = doc.find(elem_name)
                    if elem is not None and elem.text:
                        study[elem_name] = elem.text.strip()
                
                # Also check for phs accession pattern
                for elem in doc.iter():
                    if elem.text and re.match(r'phs\d+', str(elem.text)):
                        study["phs_accession"] = elem.text.strip()
                
                if study:
                    studies.append(study)
            
        except Exception as e:
            print(f"  Error at batch {fetched}: {e}")
        
        fetched += batch_size
        if fetched % 500 == 0:
            print(f"  Processed {fetched}/{total}, {len(studies)} studies parsed")
    
    print(f"  Total studies parsed: {len(studies)}")
    return studies


def search_sra_sc_by_biosample():
    """
    Query SRA for all single-cell studies (both open and controlled)
    using the library_source field, then identify which have
    BioProject accessions that map to dbGaP.
    """
    print("\nSearching SRA for ALL single-cell transcriptomic studies...")
    
    # First, get the total count of SC transcriptomic studies
    query = '"TRANSCRIPTOMIC SINGLE CELL"[Source]'
    time.sleep(DELAY)
    search = esearch("sra", query, retmax=0)
    total_sc = search["count"]
    print(f"  Total SC transcriptomic SRA experiments: {total_sc:,}")
    
    # Now search specifically for studies (not individual runs)
    # Group by study accession
    time.sleep(DELAY)
    query2 = '"TRANSCRIPTOMIC SINGLE CELL"[Source]'
    search2 = esearch("sra", query2, retmax=50000)
    
    print(f"  Fetching up to 50,000 SRA records to extract study info...")
    
    if not search2["webenv"]:
        return {}
    
    # Fetch summaries in batches
    study_runs = defaultdict(lambda: {"runs": 0, "organisms": set(), "platforms": set()})
    fetched = 0
    batch_size = 500
    to_fetch = min(50000, int(total_sc))
    
    while fetched < to_fetch:
        time.sleep(DELAY)
        url = f"{EUTILS_BASE}/esummary.fcgi"
        params = {
            "db": "sra",
            "query_key": search2["query_key"],
            "WebEnv": search2["webenv"],
            "retstart": fetched,
            "retmax": batch_size,
            "retmode": "json",
        }
        
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            data = r.json()
            
            result = data.get("result", {})
            uids = result.get("uids", [])
            if not uids:
                break
            
            for uid in uids:
                rec = result.get(uid, {})
                if not isinstance(rec, dict):
                    continue
                
                exp_xml = rec.get("expxml", "")
                if not exp_xml:
                    continue
                
                try:
                    root = ET.fromstring(f"<root>{exp_xml}</root>")
                    
                    study_el = root.find(".//Study")
                    study_acc = study_el.get("acc", "") if study_el is not None else ""
                    
                    bioproject_el = root.find(".//Bioproject")
                    bioproject = bioproject_el.text if bioproject_el is not None else ""
                    
                    organism_el = root.find(".//Organism")
                    organism = organism_el.get("CommonName", "") if organism_el is not None else ""
                    
                    platform_el = root.find(".//Platform")
                    platform = platform_el.get("instrument_model", "") if platform_el is not None else ""
                    
                    key = study_acc or bioproject or "unknown"
                    study_runs[key]["runs"] += 1
                    study_runs[key]["organisms"].add(organism)
                    study_runs[key]["platforms"].add(platform)
                    if bioproject:
                        study_runs[key]["bioproject"] = bioproject
                    if study_acc:
                        study_runs[key]["study_acc"] = study_acc
                        
                except ET.ParseError:
                    pass
            
            fetched += len(uids)
            if fetched % 5000 == 0:
                print(f"  Processed {fetched:,}/{to_fetch:,}, {len(study_runs)} unique studies")
        
        except Exception as e:
            print(f"  Error at {fetched}: {e}")
            fetched += batch_size
    
    # Convert sets for JSON
    for k in study_runs:
        study_runs[k]["organisms"] = list(study_runs[k]["organisms"])
        study_runs[k]["platforms"] = list(study_runs[k]["platforms"])
    
    print(f"\n  Total unique SC studies in SRA: {len(study_runs)}")
    print(f"  Total runs sampled: {fetched:,}")
    
    return dict(study_runs)


def cross_reference_with_geo(sra_studies, geo_catalog_path):
    """
    Cross-reference SRA studies with our GEO catalog to identify
    studies that are NOT in GEO (likely controlled-access only).
    """
    print("\nCross-referencing with GEO catalog...")
    
    try:
        import pandas as pd
        geo = pd.read_parquet(geo_catalog_path)
        geo_runs = set(geo["run_accession"].dropna().unique())
        geo_samples = set(geo["sample_accession"].dropna().unique())
        print(f"  GEO catalog: {len(geo_runs):,} runs, {len(geo_samples):,} samples")
    except Exception as e:
        print(f"  Could not load GEO catalog: {e}")
        return {}
    
    # Check which SRA study runs overlap with GEO
    in_geo = 0
    not_in_geo = 0
    controlled_candidates = {}
    
    for study_key, info in sra_studies.items():
        # A study primarily in GEO will have most runs in our catalog
        # Studies NOT in GEO are likely controlled-access
        # We can't check individual runs without fetching them, but
        # BioProject accessions starting with certain prefixes indicate dbGaP
        
        bp = info.get("bioproject", "")
        sa = info.get("study_acc", "")
        
        # Studies not typically in GEO
        if not sa.startswith("SRP"):  # Non-SRP studies are often controlled
            controlled_candidates[study_key] = info
            not_in_geo += 1
        else:
            in_geo += 1
    
    print(f"  Likely in GEO (SRP): {in_geo}")
    print(f"  Likely controlled-access (non-SRP): {not_in_geo}")
    
    return controlled_candidates


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    geo_catalog = "/mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet"
    
    print("=" * 70)
    print("dbGaP DETAILED STUDY METADATA")
    print("=" * 70)
    
    dbgap_studies = fetch_dbgap_study_details()
    
    print("\n" + "=" * 70)
    print("SRA SINGLE-CELL STUDY LANDSCAPE")
    print("=" * 70)
    
    sra_studies = search_sra_sc_by_biosample()
    
    print("\n" + "=" * 70)
    print("CROSS-REFERENCE WITH GEO CATALOG")
    print("=" * 70)
    
    controlled = cross_reference_with_geo(sra_studies, geo_catalog)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"dbGaP SC studies: {len(dbgap_studies)}")
    print(f"SRA SC studies (total): {len(sra_studies)}")
    print(f"SRA SC studies (controlled candidates): {len(controlled)}")
    
    total_controlled_runs = sum(s["runs"] for s in controlled.values())
    print(f"Controlled candidate runs: {total_controlled_runs:,}")
    
    # Organism breakdown for controlled
    org_counts = defaultdict(int)
    for s in controlled.values():
        for org in s.get("organisms", []):
            if org:
                org_counts[org] += s["runs"]
    
    if org_counts:
        print("\nControlled-access organism breakdown:")
        for org, cnt in sorted(org_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {org}: {cnt:,} runs")
    
    output = {
        "dbgap_studies": dbgap_studies,
        "sra_sc_studies": sra_studies,
        "controlled_candidates": controlled,
        "summary": {
            "dbgap_study_count": len(dbgap_studies),
            "sra_study_count": len(sra_studies),
            "controlled_candidate_count": len(controlled),
            "controlled_runs": total_controlled_runs,
            "organism_breakdown": dict(org_counts),
        }
    }
    
    outfile = os.path.join(output_dir, "dbgap_detailed_studies.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
