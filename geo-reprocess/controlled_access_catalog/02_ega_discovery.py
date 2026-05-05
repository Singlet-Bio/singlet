#!/usr/bin/env python3
"""
Discover single-cell datasets in the European Genome-phenome Archive (EGA).

EGA has a public metadata API that allows browsing studies, datasets, and
file metadata without requiring authorized access. The actual data downloads
require DAC approval, but the catalog is open.

API docs: https://ega-archive.org/api/
Metadata API: https://www.ebi.ac.uk/ega/api/
"""

import requests
import json
import time
import os
from collections import defaultdict

EGA_API = "https://ega-archive.org/metadata/v2"
# Alternative: EBI search API
EBI_SEARCH = "https://www.ebi.ac.uk/ebisearch/ws/rest"

DELAY = 0.5  # Rate limiting


def search_ega_studies_via_ebi(query, max_results=1000):
    """
    Search EGA studies via EBI Search API.
    This searches across EGA study metadata.
    """
    url = f"{EBI_SEARCH}/ega"
    all_entries = []
    start = 0
    page_size = min(100, max_results)
    
    while start < max_results:
        time.sleep(DELAY)
        params = {
            "query": query,
            "format": "json",
            "size": page_size,
            "start": start,
            "fields": "id,name,description,STUDY_TYPE,CENTER_NAME",
        }
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            entries = data.get("entries", [])
            if not entries:
                break
            all_entries.extend(entries)
            total = int(data.get("hitCount", 0))
            start += page_size
            if start >= total:
                break
        except Exception as e:
            print(f"  EBI search error at offset {start}: {e}")
            break
    
    return all_entries


def search_ega_datasets_via_ebi(query, max_results=5000):
    """
    Search EGA datasets via EBI Search API.
    Datasets contain the actual files and are linked to studies.
    """
    url = f"{EBI_SEARCH}/ega"
    all_entries = []
    start = 0
    page_size = min(100, max_results)
    
    while start < max_results:
        time.sleep(DELAY)
        params = {
            "query": query,
            "format": "json",
            "size": page_size,
            "start": start,
        }
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            entries = data.get("entries", [])
            if not entries:
                break
            all_entries.extend(entries)
            total = int(data.get("hitCount", 0))
            start += page_size
            if start >= total:
                break
        except Exception as e:
            print(f"  EBI search error at offset {start}: {e}")
            break
    
    return all_entries


def search_ena_controlled_sc():
    """
    Search ENA (European Nucleotide Archive) for controlled-access
    single-cell studies. ENA is the data backend for EGA sequencing.
    
    Uses the ENA Portal API which is more structured.
    """
    # ENA Portal API - search for studies with single-cell library source
    url = "https://www.ebi.ac.uk/ena/portal/api/search"
    
    queries = {
        "sc_transcriptomic": {
            "result": "study",
            "query": 'library_source="TRANSCRIPTOMIC SINGLE CELL"',
            "fields": "study_accession,secondary_study_accession,study_title,tax_id,scientific_name,center_name,first_public,study_description",
            "limit": 0,  # count only
        },
        "sc_genomic": {
            "result": "study", 
            "query": 'library_source="GENOMIC SINGLE CELL"',
            "fields": "study_accession,study_title",
            "limit": 0,
        },
    }
    
    results = {}
    for label, params in queries.items():
        time.sleep(DELAY)
        try:
            params_copy = dict(params)
            params_copy["format"] = "json"
            r = requests.get(url, params=params_copy, timeout=60)
            if r.status_code == 200:
                try:
                    data = r.json()
                    results[label] = len(data) if isinstance(data, list) else 0
                except:
                    # Count from text
                    results[label] = r.text.count("\n")
            else:
                results[label] = f"HTTP {r.status_code}"
            print(f"  ENA [{label}]: {results[label]}")
        except Exception as e:
            print(f"  ENA [{label}]: ERROR - {e}")
            results[label] = str(e)
    
    return results


def get_ega_study_details(study_id):
    """Get details for a specific EGA study."""
    time.sleep(DELAY)
    url = f"{EGA_API}/studies/{study_id}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def search_arrayexpress_sc():
    """
    Search ArrayExpress/BioStudies for single-cell studies.
    Some EGA studies are cross-referenced here.
    """
    url = "https://www.ebi.ac.uk/biostudies/api/v1/search"
    queries = [
        "single cell RNA-seq",
        "scRNA-seq controlled access",
        "10x Chromium controlled",
        "single nucleus RNA",
    ]
    
    all_results = []
    for q in queries:
        time.sleep(DELAY)
        params = {"query": q, "pageSize": 25}
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                total = data.get("totalHits", 0)
                print(f"  BioStudies [{q[:30]}...]: {total} hits")
                all_results.append({"query": q, "total": total})
        except Exception as e:
            print(f"  BioStudies [{q[:30]}...]: ERROR - {e}")
    
    return all_results


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 70)
    print("PHASE 1: EGA Study Discovery via EBI Search")
    print("=" * 70)
    
    sc_queries = [
        "single cell RNA",
        "scRNA-seq",
        "single cell ATAC",
        "10x Chromium",
        "single nucleus RNA",
        "snRNA-seq",
        "CITE-seq",
        "spatial transcriptomics",
        "Drop-seq",
        "Smart-seq2 single cell",
        "single cell multiome",
    ]
    
    all_ega_entries = {}
    total_hits = 0
    for q in sc_queries:
        entries = search_ega_studies_via_ebi(q, max_results=500)
        new_count = 0
        for entry in entries:
            eid = entry.get("id", "")
            if eid and eid not in all_ega_entries:
                all_ega_entries[eid] = entry
                new_count += 1
        total_hits += len(entries)
        print(f"  EGA search [{q:30s}]: {len(entries):4d} hits, {new_count} new unique")
    
    print(f"\nTotal unique EGA entries: {len(all_ega_entries)}")
    
    print("\n" + "=" * 70)
    print("PHASE 2: ENA Single-Cell Study Counts")
    print("=" * 70)
    ena_results = search_ena_controlled_sc()
    
    print("\n" + "=" * 70)
    print("PHASE 3: BioStudies / ArrayExpress Search")
    print("=" * 70)
    biostudies = search_arrayexpress_sc()
    
    # Categorize EGA entries
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    ega_by_prefix = defaultdict(list)
    for eid, entry in all_ega_entries.items():
        prefix = eid.split("A")[0] + "A" if "A" in eid else eid[:4]
        ega_by_prefix[prefix].append(entry)
    
    print(f"EGA unique entries: {len(all_ega_entries)}")
    for prefix, entries in sorted(ega_by_prefix.items()):
        print(f"  {prefix}*: {len(entries)}")
    
    # Save
    output = {
        "ega_entries": {k: v for k, v in all_ega_entries.items()},
        "ena_counts": ena_results,
        "biostudies": biostudies,
        "summary": {
            "total_ega_entries": len(all_ega_entries),
            "by_prefix": {k: len(v) for k, v in ega_by_prefix.items()},
        }
    }
    
    outfile = os.path.join(output_dir, "ega_controlled_sc.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
