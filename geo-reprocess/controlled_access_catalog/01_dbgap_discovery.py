#!/usr/bin/env python3
"""
Discover single-cell studies in dbGaP via NCBI E-utilities.

dbGaP metadata is publicly queryable through NCBI's databases:
- gap (dbGaP study metadata)
- sra (Sequence Read Archive — includes library strategy/source)
- biosample (sample metadata)

Strategy:
1. Search SRA for controlled-access single-cell runs (library_strategy + access=controlled)
2. Extract study-level metadata (BioProject, dbGaP phs accession)
3. Get protocol, organism, cell counts from SRA metadata
4. Cross-reference with our GEO catalog to identify non-duplicates
"""

import requests
import xml.etree.ElementTree as ET
import json
import time
import sys
import os
from collections import defaultdict

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Rate limit: 3 requests/sec without API key, 10/sec with key
DELAY = 0.35  # seconds between requests

def esearch(db, term, retmax=100000, retstart=0):
    """Search NCBI database, return list of IDs."""
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db": db,
        "term": term,
        "retmax": retmax,
        "retstart": retstart,
        "retmode": "json",
        "usehistory": "y",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    result = data["esearchresult"]
    return {
        "count": int(result["count"]),
        "ids": result["idlist"],
        "webenv": result.get("webenv"),
        "query_key": result.get("querykey"),
    }

def esummary_batch(db, webenv, query_key, retstart=0, retmax=500):
    """Fetch summaries using web history."""
    url = f"{EUTILS_BASE}/esummary.fcgi"
    params = {
        "db": db,
        "query_key": query_key,
        "WebEnv": webenv,
        "retstart": retstart,
        "retmax": retmax,
        "retmode": "json",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()

def search_dbgap_single_cell():
    """
    Search dbGaP for single-cell related studies.
    Uses the 'gap' database which indexes dbGaP studies.
    """
    queries = [
        # Direct single-cell terms in dbGaP
        '"single cell"[All Fields] OR "single-cell"[All Fields] OR "scRNA"[All Fields] OR "snRNA"[All Fields]',
        '"10x genomics"[All Fields] OR "10x chromium"[All Fields] OR "Drop-seq"[All Fields]',
        '"scATAC"[All Fields] OR "single cell ATAC"[All Fields] OR "multiome"[All Fields]',
        '"single nucleus"[All Fields] OR "snATAC"[All Fields] OR "CITE-seq"[All Fields]',
        '"spatial transcriptomics"[All Fields] OR "Visium"[All Fields] OR "MERFISH"[All Fields]',
    ]
    
    all_ids = set()
    for q in queries:
        time.sleep(DELAY)
        result = esearch("gap", q)
        print(f"  dbGaP gap search: {result['count']} hits for: {q[:60]}...")
        all_ids.update(result["ids"])
    
    print(f"\nTotal unique dbGaP study IDs: {len(all_ids)}")
    return all_ids

def search_sra_controlled_sc():
    """
    Search SRA for controlled-access single-cell sequencing runs.
    SRA public metadata includes access type, library strategy, etc.
    """
    # SRA queries for controlled-access single-cell data
    # library_strategy: RNA-Seq with library_source: TRANSCRIPTOMIC SINGLE CELL
    queries = {
        "sc_rnaseq": (
            '"TRANSCRIPTOMIC SINGLE CELL"[Source] AND '
            '"controlled access"[Access]'
        ),
        "sc_atacseq": (
            '("ATAC-seq"[Strategy] OR "ATAC-Seq"[Strategy]) AND '
            '"GENOMIC SINGLE CELL"[Source] AND '
            '"controlled access"[Access]'
        ),
        "sc_generic": (
            '("single cell"[Selection] OR "single cell"[Title]) AND '
            '"controlled access"[Access] AND '
            '("RNA-Seq"[Strategy] OR "ATAC-seq"[Strategy])'
        ),
        "10x_controlled": (
            '("10x"[Platform] OR "ILLUMINA"[Platform]) AND '
            '"TRANSCRIPTOMIC SINGLE CELL"[Source] AND '
            '"controlled access"[Access]'
        ),
    }
    
    results = {}
    for label, query in queries.items():
        time.sleep(DELAY)
        try:
            result = esearch("sra", query, retmax=0)  # just get count first
            count = result["count"]
            print(f"  SRA [{label}]: {count:,} runs")
            results[label] = {"count": count, "query": query}
        except Exception as e:
            print(f"  SRA [{label}]: ERROR - {e}")
            results[label] = {"count": 0, "query": query, "error": str(e)}
    
    return results

def get_sra_controlled_sc_details(max_records=10000):
    """
    Fetch detailed SRA records for controlled-access single-cell runs.
    Returns study-level aggregated information.
    """
    query = (
        '"TRANSCRIPTOMIC SINGLE CELL"[Source] AND '
        '"controlled access"[Access]'
    )
    
    print(f"\nFetching SRA controlled-access SC records (up to {max_records:,})...")
    time.sleep(DELAY)
    search = esearch("sra", query, retmax=max_records)
    total = search["count"]
    print(f"  Total matching: {total:,}")
    
    if not search["webenv"]:
        print("  No web history available, fetching IDs directly...")
        return {}
    
    # Fetch in batches of 500
    studies = defaultdict(lambda: {
        "runs": 0, "samples": set(), "organisms": set(),
        "platforms": set(), "strategies": set(),
        "bioproject": None, "title": None,
    })
    
    fetched = 0
    batch_size = 500
    to_fetch = min(int(total), max_records)
    
    while fetched < to_fetch:
        time.sleep(DELAY)
        try:
            data = esummary_batch(
                "sra", search["webenv"], search["query_key"],
                retstart=fetched, retmax=batch_size
            )
        except Exception as e:
            print(f"  Error at batch {fetched}: {e}")
            break
        
        result = data.get("result", {})
        uids = result.get("uids", [])
        if not uids:
            break
        
        for uid in uids:
            rec = result.get(uid, {})
            if not isinstance(rec, dict):
                continue
            
            # Extract experiment XML snippet
            exp_xml = rec.get("expxml", "")
            runs_xml = rec.get("runs", "")
            
            # Parse key fields from the summary
            # The expxml contains study accession, organism, etc.
            try:
                # Wrap in root element for parsing
                if exp_xml:
                    root = ET.fromstring(f"<root>{exp_xml}</root>")
                    study_el = root.find(".//Study")
                    study_acc = study_el.get("acc", "unknown") if study_el is not None else "unknown"
                    
                    organism_el = root.find(".//Organism")
                    organism = organism_el.get("CommonName", "") if organism_el is not None else ""
                    taxid = organism_el.get("taxid", "") if organism_el is not None else ""
                    
                    platform_el = root.find(".//Platform")
                    platform = platform_el.get("instrument_model", "") if platform_el is not None else ""
                    
                    library_el = root.find(".//Library_descriptor")
                    strategy = ""
                    if library_el is not None:
                        strat_el = library_el.find("LIBRARY_STRATEGY")
                        strategy = strat_el.text if strat_el is not None else ""
                    
                    bioproject_el = root.find(".//Bioproject")
                    bioproject = bioproject_el.text if bioproject_el is not None else None
                    
                    summary_el = root.find(".//Summary")
                    title = summary_el.get("Title", "") if summary_el is not None else ""
                    
                    studies[study_acc]["runs"] += 1
                    studies[study_acc]["organisms"].add(organism or taxid)
                    studies[study_acc]["platforms"].add(platform)
                    studies[study_acc]["strategies"].add(strategy)
                    if bioproject:
                        studies[study_acc]["bioproject"] = bioproject
                    if title and not studies[study_acc]["title"]:
                        studies[study_acc]["title"] = title
                    
                    # Count runs
                    if runs_xml:
                        run_root = ET.fromstring(f"<root>{runs_xml}</root>")
                        for run in run_root.findall(".//Run"):
                            sample = run.get("acc", "")
                            if sample:
                                studies[study_acc]["samples"].add(sample)
                        
            except ET.ParseError:
                pass
        
        fetched += len(uids)
        if fetched % 2000 == 0 or fetched >= to_fetch:
            print(f"  Processed {fetched:,}/{to_fetch:,} records, {len(studies)} studies so far")
    
    # Convert sets to lists for JSON serialization
    for acc in studies:
        studies[acc]["samples"] = list(studies[acc]["samples"])
        studies[acc]["organisms"] = list(studies[acc]["organisms"])
        studies[acc]["platforms"] = list(studies[acc]["platforms"])
        studies[acc]["strategies"] = list(studies[acc]["strategies"])
    
    return dict(studies)

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 70)
    print("PHASE 1: dbGaP Study Discovery (gap database)")
    print("=" * 70)
    dbgap_ids = search_dbgap_single_cell()
    
    print("\n" + "=" * 70)
    print("PHASE 2: SRA Controlled-Access Single-Cell Counts")
    print("=" * 70)
    sra_counts = search_sra_controlled_sc()
    
    print("\n" + "=" * 70)
    print("PHASE 3: SRA Controlled-Access SC Study Details")
    print("=" * 70)
    studies = get_sra_controlled_sc_details(max_records=50000)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"dbGaP study IDs found: {len(dbgap_ids)}")
    print(f"SRA controlled-access SC studies: {len(studies)}")
    total_runs = sum(s["runs"] for s in studies.values())
    total_samples = sum(len(s["samples"]) for s in studies.values())
    print(f"Total SRA runs: {total_runs:,}")
    print(f"Total SRA run accessions collected: {total_samples:,}")
    
    # Organism breakdown
    org_counts = defaultdict(int)
    for s in studies.values():
        for org in s["organisms"]:
            org_counts[org] += s["runs"]
    print("\nOrganism breakdown (by runs):")
    for org, cnt in sorted(org_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {org}: {cnt:,}")
    
    # Save results
    output = {
        "dbgap_study_ids": list(dbgap_ids),
        "sra_query_counts": sra_counts,
        "studies": studies,
        "summary": {
            "dbgap_studies": len(dbgap_ids),
            "sra_studies": len(studies),
            "total_runs": total_runs,
            "total_run_accessions": total_samples,
            "organism_breakdown": dict(org_counts),
        }
    }
    
    outfile = os.path.join(output_dir, "dbgap_sra_controlled_sc.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {outfile}")

if __name__ == "__main__":
    main()
