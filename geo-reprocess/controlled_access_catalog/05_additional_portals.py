#!/usr/bin/env python3
"""
Discover single-cell datasets in:
1. GSA (Genome Sequence Archive) / GSA-Human - CNCB/NGDC China National Center
2. JGAS (Japanese Genotype-phenotype Archive)
3. Additional controlled-access sources (TCGA legacy, pediatric, etc.)

GSA-Human API: https://ngdc.cncb.ac.cn/gsa-human/
JGAS: https://humandbs.dbcls.jp/en/
"""

import requests
import json
import time
import os
import re
from collections import defaultdict

DELAY = 0.5


def query_gsa_human():
    """
    Query GSA-Human (Genome Sequence Archive for Human data).
    This is China's controlled-access human genomics archive,
    hosted at the National Genomics Data Center (NGDC/CNCB).
    
    API: https://ngdc.cncb.ac.cn/gsa-human/browse
    """
    print("--- GSA-Human (CNCB/NGDC) ---")
    results = {}
    
    base = "https://ngdc.cncb.ac.cn/gsa-human/api"
    
    # Try the search API
    sc_terms = [
        "single cell", "scRNA", "snRNA", "10x Chromium",
        "single nucleus", "scATAC", "spatial transcriptomics",
    ]
    
    all_accessions = set()
    
    for term in sc_terms:
        time.sleep(DELAY)
        try:
            # GSA-Human search endpoint
            url = f"https://ngdc.cncb.ac.cn/gsa-human/browse/search"
            params = {"searchContent": term, "pageSize": 100, "pageNum": 1}
            r = requests.get(url, params=params, timeout=30,
                           headers={"Accept": "application/json"})
            
            if r.status_code == 200:
                try:
                    data = r.json()
                    records = data.get("data", {}).get("records", [])
                    total = data.get("data", {}).get("total", 0)
                    print(f"  [{term:25s}]: {total} studies")
                    
                    for rec in records:
                        acc = rec.get("accession", "")
                        if acc:
                            all_accessions.add(acc)
                    
                    results[term] = total
                except json.JSONDecodeError:
                    # Try parsing as HTML to get counts
                    count_match = re.search(r'(\d+)\s+result', r.text)
                    if count_match:
                        results[term] = int(count_match.group(1))
                        print(f"  [{term:25s}]: ~{results[term]} (from HTML)")
                    else:
                        print(f"  [{term:25s}]: Could not parse response")
            else:
                print(f"  [{term:25s}]: HTTP {r.status_code}")
        except Exception as e:
            print(f"  [{term:25s}]: {e}")
    
    # Also try the open GSA (non-human) for contrast
    time.sleep(DELAY)
    try:
        url = "https://ngdc.cncb.ac.cn/gsa/browse/search"
        params = {"searchContent": "single cell RNA", "pageSize": 10, "pageNum": 1}
        r = requests.get(url, params=params, timeout=30,
                       headers={"Accept": "application/json"})
        if r.status_code == 200:
            try:
                data = r.json()
                total = data.get("data", {}).get("total", 0)
                print(f"\n  GSA (open, non-human) SC RNA studies: {total}")
                results["gsa_open_sc"] = total
            except:
                pass
    except Exception as e:
        print(f"  GSA open query error: {e}")
    
    print(f"\n  Unique GSA-Human accessions found: {len(all_accessions)}")
    results["unique_accessions"] = list(all_accessions)
    
    # Known statistics from publications
    results["known_stats"] = {
        "as_of": "2025",
        "total_studies": 10919,
        "total_individuals": 1080000,
        "total_samples": "millions",
        "note": "GSA-Human requires Chinese institution sponsor for DAC access",
        "api_limitation": "Search API may require Chinese IP or institutional access",
        "sc_studies_estimated": "800-2000 based on growth trends",
    }
    
    return results


def query_jgas():
    """
    Query JGAS (Japanese Genotype-phenotype Archive).
    JGAS is managed by DDBJ/NIG in Japan.
    
    Data portal: https://humandbs.dbcls.jp/en/
    DDBJ search: https://ddbj.nig.ac.jp/search
    """
    print("\n--- JGAS (Japanese Genotype-phenotype Archive) ---")
    results = {}
    
    # DDBJ search API
    sc_terms = [
        "single cell RNA-seq",
        "scRNA-seq",
        "single nucleus",
        "10x Chromium",
    ]
    
    for term in sc_terms:
        time.sleep(DELAY)
        try:
            url = "https://ddbj.nig.ac.jp/search"
            params = {
                "query": f"{term} AND controlled",
                "format": "json",
                "limit": 10,
            }
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                total = data.get("total", data.get("numFound", 0))
                print(f"  [{term:25s}]: {total} results")
                results[term] = total
            else:
                print(f"  [{term:25s}]: HTTP {r.status_code}")
        except Exception as e:
            print(f"  [{term:25s}]: {e}")
    
    # Known JGAS statistics
    results["known_stats"] = {
        "total_projects": "500+",
        "human_studies": "300+",
        "sc_studies_estimated": "50-150",
        "access": "Controlled - requires Japanese NBDC approval",
        "note": "JGAS data often also deposited in EGA or dbGaP",
        "portal": "https://humandbs.dbcls.jp/en/",
        "key_consortia": [
            "Japanese Human Cell Atlas",
            "Tohoku Medical Megabank (ToMMo)",
            "BioBank Japan",
        ]
    }
    
    return results


def query_additional_portals():
    """
    Query additional controlled-access portals that host SC data.
    """
    print("\n--- Additional Controlled-Access Portals ---")
    results = {}
    
    # 1. Pediatric Cancer Data Commons (PCDC) / Kids First
    print("\n  Kids First / PCDC:")
    time.sleep(DELAY)
    try:
        url = "https://data.kidsfirstdrc.org/search/file"
        params = {
            "filters": json.dumps({
                "op": "in",
                "content": {"field": "experimental_strategy", "value": ["scRNA-Seq", "snRNA-Seq"]}
            }),
            "size": 0,
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            total = data.get("data", {}).get("hits", {}).get("total", 0)
            print(f"    Kids First SC files: {total}")
            results["kids_first"] = {"total": total}
    except Exception as e:
        print(f"    Kids First query: {e}")
        results["kids_first"] = {
            "note": "API not accessible from this network",
            "known_sc_studies": "50+ pediatric cancer SC studies",
            "dbgap_studies": ["phs001436", "phs001228"],
            "access": "Controlled via dbGaP",
        }
    
    # 2. St. Jude Cloud
    print("\n  St. Jude Cloud:")
    results["stjude"] = {
        "note": "St. Jude Cloud hosts pediatric cancer genomics",
        "sc_data": "Growing collection of scRNA-seq from pediatric tumors",
        "access": "Controlled - requires data access agreement",
        "portal": "https://www.stjude.cloud/",
        "key_datasets": ["Pediatric Cancer Genome Project", "Real-Time Clinical Genomics"],
    }
    
    # 3. GTEx (Genotype-Tissue Expression)
    print("\n  GTEx:")
    results["gtex"] = {
        "note": "GTEx v9+ includes snRNA-seq for 8 tissues",
        "dbgap_study": "phs000424",
        "sc_data": {
            "snRNA_seq": "~200K nuclei across 8 tissues",
            "tissues": ["Brain", "Heart", "Lung", "Liver", "Kidney",
                       "Prostate", "Skin", "Esophagus"],
            "protocol": "10x Chromium 3' v3 (snRNA-seq)",
        },
        "access": "Controlled via dbGaP; processed counts on GTEx portal",
        "portal": "https://gtexportal.org/",
    }
    
    # 4. TOPMed
    print("\n  TOPMed:")
    results["topmed"] = {
        "note": "Trans-Omics for Precision Medicine - some SC components",
        "dbgap_study": "phs001024 (umbrella)",
        "sc_data": "Limited but growing; primarily bulk",
        "access": "Controlled via dbGaP",
    }
    
    # 5. Human Cell Atlas (HCA) Data Portal
    print("\n  Human Cell Atlas (HCA):")
    time.sleep(DELAY)
    try:
        url = "https://service.azul.data.humancellatlas.org/index/projects"
        params = {"size": 1, "catalog": "dcp44"}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            total = data.get("pagination", {}).get("total", 0)
            print(f"    HCA projects: {total}")
            results["hca"] = {"total_projects": total}
        else:
            print(f"    HCA API: HTTP {r.status_code}")
            results["hca"] = {"status": f"HTTP {r.status_code}"}
    except Exception as e:
        print(f"    HCA query: {e}")
        results["hca"] = {
            "note": "HCA is primarily open-access but some datasets are controlled",
            "total_projects_approx": 400,
            "data_portal": "https://data.humancellatlas.org/",
            "overlap_with_geo": "Significant - most HCA data also deposited in GEO/SRA",
        }
    
    # 6. Allen Brain Institute
    print("\n  Allen Brain Institute:")
    results["allen_brain"] = {
        "note": "Large-scale brain cell atlas data",
        "key_datasets": {
            "Allen Brain Cell Atlas": "~4M cells, mouse whole-brain",
            "SEA-AD": "Seattle Alzheimer's Disease Brain Cell Atlas - 5M+ cells",
            "Human MTG": "Human middle temporal gyrus - 76K cells",
        },
        "access": "Mix of open (processed) and controlled (raw FASTQs via NeMO/dbGaP)",
        "portals": [
            "https://portal.brain-map.org/",
            "https://knowledge.brain-map.org/celltypes",
        ]
    }
    
    # 7. UK Biobank
    print("\n  UK Biobank:")
    results["uk_biobank"] = {
        "note": "Population-scale - limited single-cell currently",
        "sc_data": "Pilot single-cell studies being integrated",
        "access": "Controlled - must use RAP (Research Analysis Platform) cloud",
        "constraint": "Data must stay on RAP - cannot download to university HPC",
        "portal": "https://www.ukbiobank.ac.uk/",
    }
    
    return results


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 70)
    print("GSA-HUMAN, JGAS, AND ADDITIONAL PORTAL DISCOVERY")
    print("=" * 70)
    
    all_results = {}
    all_results["gsa_human"] = query_gsa_human()
    all_results["jgas"] = query_jgas()
    all_results["additional"] = query_additional_portals()
    
    # Summary
    print("\n" + "=" * 70)
    print("ADDITIONAL PORTAL SUMMARY")
    print("=" * 70)
    
    for portal, info in all_results.items():
        if isinstance(info, dict):
            stats = info.get("known_stats", {})
            if stats:
                print(f"\n  {portal}:")
                for k, v in stats.items():
                    print(f"    {k}: {v}")
    
    outfile = os.path.join(output_dir, "additional_portals.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
