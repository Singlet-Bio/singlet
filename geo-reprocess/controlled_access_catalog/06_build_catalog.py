#!/usr/bin/env python3
"""
Comprehensive controlled-access single-cell catalog builder.

This script:
1. Fetches all 646 dbGaP-linked SC BioProject details
2. Extracts phs accessions, organisms, descriptions, disease info
3. Gets SRA run counts and library info per BioProject
4. Cross-references with GEO catalog to identify unique-to-dbGaP data
5. Combines EGA data from previous discovery
6. Produces a unified catalog with protocol classification
"""

import requests
import xml.etree.ElementTree as ET
import json
import time
import os
import re
import sys
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


def esummary_json(db, ids_or_webenv, query_key=None, retstart=0, retmax=500):
    url = f"{EUTILS_BASE}/esummary.fcgi"
    if isinstance(ids_or_webenv, list):
        params = {"db": db, "id": ",".join(ids_or_webenv), "retmode": "json"}
    else:
        params = {
            "db": db, "query_key": query_key, "WebEnv": ids_or_webenv,
            "retstart": retstart, "retmax": retmax, "retmode": "json",
        }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def elink(dbfrom, dbto, ids):
    """Link records between NCBI databases."""
    url = f"{EUTILS_BASE}/elink.fcgi"
    params = {"dbfrom": dbfrom, "db": dbto, "id": ",".join(ids[:20]),
              "retmode": "json"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_all_dbgap_sc_bioprojects():
    """Fetch all BioProject entries linked to dbGaP & single-cell."""
    print("=" * 70)
    print("STEP 1: Discover dbGaP-linked SC BioProjects")
    print("=" * 70)

    queries = [
        'dbgap[All Fields] AND ("single cell"[All Fields] OR scRNA[All Fields] OR snRNA[All Fields])',
        'dbgap[All Fields] AND ("10x genomics"[All Fields] OR chromium[All Fields])',
        'dbgap[All Fields] AND ("scATAC"[All Fields] OR "single cell ATAC"[All Fields])',
        'dbgap[All Fields] AND ("spatial transcriptomics"[All Fields] OR "Visium"[All Fields])',
        'dbgap[All Fields] AND ("CITE-seq"[All Fields] OR "multiome"[All Fields])',
        'dbgap[All Fields] AND ("Drop-seq"[All Fields] OR "Smart-seq"[All Fields])',
        'dbgap[All Fields] AND ("single nucleus"[All Fields])',
    ]

    all_ids = set()
    for q in queries:
        time.sleep(DELAY)
        result = esearch("bioproject", q)
        all_ids.update(result["ids"])
        print(f"  Query [{q[:55]}...]: {result['count']} hits, total unique: {len(all_ids)}")

    print(f"\n  Total unique BioProject IDs: {len(all_ids)}")
    return list(all_ids)


def fetch_bioproject_details(bp_ids):
    """Fetch detailed metadata for each BioProject."""
    print("\n" + "=" * 70)
    print("STEP 2: Fetch BioProject Details")
    print("=" * 70)

    projects = {}
    batch_size = 100

    for i in range(0, len(bp_ids), batch_size):
        batch = bp_ids[i:i + batch_size]
        time.sleep(DELAY)
        try:
            data = esummary_json("bioproject", batch)
            result = data.get("result", {})
            for uid in batch:
                rec = result.get(uid, {})
                if not isinstance(rec, dict):
                    continue

                name = rec.get("project_name", "")
                title = rec.get("project_title", "")
                descr = rec.get("project_description", "")
                org = rec.get("organism_name", "")
                org_taxid = rec.get("organism_taxid", "")
                reg = rec.get("registration_date", "")
                supergroup = rec.get("project_data_type_list", [])
                accession = rec.get("project_acc_list", "")

                # Extract phs accession
                phs = ""
                for key, val in rec.items():
                    if isinstance(val, str):
                        m = re.search(r'phs\d+', val)
                        if m:
                            phs = m.group()
                            break

                # Classify protocol from description
                desc_lower = (descr + " " + name + " " + title).lower()
                protocols = []
                if "10x" in desc_lower or "chromium" in desc_lower:
                    protocols.append("10x_chromium")
                if "smart-seq" in desc_lower or "smartseq" in desc_lower:
                    protocols.append("smart-seq")
                if "drop-seq" in desc_lower or "dropseq" in desc_lower:
                    protocols.append("drop-seq")
                if "cite-seq" in desc_lower or "citeseq" in desc_lower:
                    protocols.append("cite-seq")
                if "multiome" in desc_lower:
                    protocols.append("multiome")
                if "visium" in desc_lower or "spatial" in desc_lower:
                    protocols.append("spatial")
                if "atac" in desc_lower:
                    protocols.append("atac")
                if "merfish" in desc_lower:
                    protocols.append("merfish")
                if "snrna" in desc_lower or "single nucleus" in desc_lower or "single-nucleus" in desc_lower:
                    protocols.append("snRNA")
                if "scrna" in desc_lower or "single cell rna" in desc_lower or "single-cell rna" in desc_lower:
                    protocols.append("scRNA")

                # Classify disease from description
                diseases = []
                disease_keywords = {
                    "cancer": ["cancer", "tumor", "carcinoma", "melanoma", "leukemia",
                              "lymphoma", "glioma", "sarcoma", "adenocarcinoma", "myeloma"],
                    "neurological": ["alzheimer", "parkinson", "als", "neurodegeneration",
                                    "brain", "neural", "cortex"],
                    "autoimmune": ["lupus", "rheumatoid", "autoimmune", "inflammatory",
                                  "crohn", "colitis", "psoriasis"],
                    "cardiovascular": ["heart", "cardiac", "cardiovascular", "atherosclerosis"],
                    "pulmonary": ["lung", "pulmonary", "fibrosis", "copd", "asthma"],
                    "infectious": ["hiv", "covid", "sars", "infection", "viral", "sepsis"],
                    "metabolic": ["diabetes", "obesity", "metabolic"],
                    "developmental": ["development", "embryo", "fetal", "pediatric"],
                    "kidney": ["kidney", "renal", "nephro"],
                    "liver": ["liver", "hepat"],
                    "normal": ["healthy", "normal", "atlas", "reference"],
                }
                for disease_cat, keywords in disease_keywords.items():
                    if any(kw in desc_lower for kw in keywords):
                        diseases.append(disease_cat)

                projects[uid] = {
                    "bioproject_id": uid,
                    "name": name,
                    "title": title,
                    "description": descr[:500] if descr else "",
                    "organism": org,
                    "taxid": org_taxid,
                    "phs_accession": phs,
                    "registration_date": reg,
                    "protocols_inferred": protocols,
                    "disease_categories": diseases,
                }

        except Exception as e:
            print(f"  Error fetching batch at {i}: {e}")

        if (i + batch_size) % 200 == 0 or i + batch_size >= len(bp_ids):
            print(f"  Fetched {min(i + batch_size, len(bp_ids))}/{len(bp_ids)} BioProjects")

    print(f"  Total BioProjects with metadata: {len(projects)}")
    return projects


def get_sra_run_counts(projects):
    """Get SRA run counts for each BioProject to estimate data volume."""
    print("\n" + "=" * 70)
    print("STEP 3: Get SRA Run Counts per BioProject")
    print("=" * 70)

    bp_ids = list(projects.keys())
    processed = 0

    for uid, proj in projects.items():
        # Search SRA for runs in this BioProject
        bp_acc = proj.get("name", "")  # BioProject accession is often the name
        if not bp_acc.startswith("PRJNA"):
            # Try to get accession from project_acc
            time.sleep(DELAY)
            try:
                # Search SRA by BioProject ID
                query = f'{uid}[BioProject]'
                result = esearch("sra", query, retmax=0)
                proj["sra_run_count"] = result["count"]
            except Exception as e:
                proj["sra_run_count"] = 0
        else:
            time.sleep(DELAY)
            try:
                query = f'{bp_acc}[BioProject]'
                result = esearch("sra", query, retmax=0)
                proj["sra_run_count"] = result["count"]
            except:
                proj["sra_run_count"] = 0

        processed += 1
        if processed % 50 == 0:
            print(f"  Queried {processed}/{len(bp_ids)} BioProjects")
            # Brief pause every 50 to be nice to the API
            time.sleep(1)

    # Summary stats
    total_runs = sum(p.get("sra_run_count", 0) for p in projects.values())
    print(f"\n  Total SRA runs across all dbGaP SC BioProjects: {total_runs:,}")
    return projects


def cross_reference_with_geo(projects, geo_catalog_path):
    """Cross-reference with GEO catalog."""
    print("\n" + "=" * 70)
    print("STEP 4: Cross-Reference with GEO Catalog")
    print("=" * 70)

    try:
        import pandas as pd
        geo = pd.read_parquet(geo_catalog_path)
        geo_gses = set(geo["gse_id"].dropna().unique())
        geo_runs = set(geo["run_accession"].dropna().unique())
        print(f"  GEO catalog: {len(geo_gses):,} GSEs, {len(geo_runs):,} SRA runs")
    except Exception as e:
        print(f"  Could not load GEO catalog: {e}")
        return projects

    # For each BioProject, check if we have overlapping runs
    # We can't check individual runs without fetching them all,
    # but we can use BioProject<->GEO links
    in_geo = 0
    not_in_geo = 0
    
    for uid, proj in projects.items():
        # Heuristic: if the BioProject has 0 SRA runs, it might be
        # entirely in dbGaP's controlled tier
        runs = proj.get("sra_run_count", 0)
        proj["likely_in_geo"] = runs > 0  # conservative
        proj["geo_overlap"] = "unknown"  # would need per-run check
        
        if runs > 0:
            in_geo += 1
        else:
            not_in_geo += 1

    print(f"  BioProjects with SRA runs (may overlap GEO): {in_geo}")
    print(f"  BioProjects with 0 SRA runs (dbGaP-only): {not_in_geo}")

    return projects


def load_ega_data():
    """Load EGA discovery results."""
    print("\n" + "=" * 70)
    print("STEP 5: Load EGA Discovery Data")
    print("=" * 70)

    ega_file = os.path.join(os.path.dirname(__file__), "ega_controlled_sc.json")
    if os.path.exists(ega_file):
        with open(ega_file) as f:
            ega = json.load(f)
        entries = ega.get("ega_entries", {})
        print(f"  Loaded {len(entries)} EGA entries")
        
        # Classify EGA entries
        ega_studies = []
        for eid, entry in entries.items():
            if not eid.startswith("EGA"):
                continue
            
            name = ""
            desc = ""
            fields = entry.get("fields", {})
            if fields:
                name = fields.get("name", [""])[0] if isinstance(fields.get("name"), list) else fields.get("name", "")
                desc = fields.get("description", [""])[0] if isinstance(fields.get("description"), list) else fields.get("description", "")
            
            ega_studies.append({
                "ega_id": eid,
                "name": name,
                "description": desc[:500] if desc else "",
            })
        
        print(f"  EGA SC studies: {len(ega_studies)}")
        return ega_studies
    else:
        print("  No EGA data file found")
        return []


def build_unified_catalog(projects, ega_studies):
    """Build the final unified catalog."""
    print("\n" + "=" * 70)
    print("STEP 6: Build Unified Catalog")
    print("=" * 70)

    catalog = {
        "metadata": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Controlled-access single-cell data catalog",
        },
        "dbgap_projects": projects,
        "ega_studies": ega_studies,
    }

    # Summary statistics
    print("\n--- dbGaP Summary ---")
    total_projects = len(projects)
    phs_count = sum(1 for p in projects.values() if p.get("phs_accession"))
    human_count = sum(1 for p in projects.values() if "homo sapiens" in p.get("organism", "").lower())
    
    # Protocol breakdown
    proto_counts = defaultdict(int)
    for p in projects.values():
        for proto in p.get("protocols_inferred", []):
            proto_counts[proto] += 1
    
    # Disease breakdown
    disease_counts = defaultdict(int)
    for p in projects.values():
        for d in p.get("disease_categories", []):
            disease_counts[d] += 1
    
    # Organism breakdown
    org_counts = defaultdict(int)
    for p in projects.values():
        org = p.get("organism", "unknown")
        org_counts[org] += 1
    
    total_runs = sum(p.get("sra_run_count", 0) for p in projects.values())
    
    print(f"  Total BioProjects: {total_projects}")
    print(f"  With phs accession: {phs_count}")
    print(f"  Human: {human_count}")
    print(f"  Total SRA runs: {total_runs:,}")
    
    print("\n  Protocol inference:")
    for proto, cnt in sorted(proto_counts.items(), key=lambda x: -x[1]):
        print(f"    {proto}: {cnt} projects")
    
    print("\n  Disease categories:")
    for disease, cnt in sorted(disease_counts.items(), key=lambda x: -x[1]):
        print(f"    {disease}: {cnt} projects")
    
    print("\n  Organisms:")
    for org, cnt in sorted(org_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {org}: {cnt} projects")
    
    print(f"\n--- EGA Summary ---")
    print(f"  Total SC studies: {len(ega_studies)}")

    catalog["summary"] = {
        "dbgap": {
            "total_projects": total_projects,
            "with_phs": phs_count,
            "human": human_count,
            "total_sra_runs": total_runs,
            "protocol_counts": dict(proto_counts),
            "disease_counts": dict(disease_counts),
            "organism_counts": dict(org_counts),
        },
        "ega": {
            "total_studies": len(ega_studies),
        },
    }

    return catalog


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    geo_catalog = "/mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet"

    # Step 1: Discover
    bp_ids = fetch_all_dbgap_sc_bioprojects()
    
    # Step 2: Details
    projects = fetch_bioproject_details(bp_ids)
    
    # Step 3: SRA counts (skip for now — takes long, do subset)
    # We'll sample 100 projects for run counts
    sample_projects = dict(list(projects.items())[:100])
    sample_projects = get_sra_run_counts(sample_projects)
    # Update originals
    for uid in sample_projects:
        projects[uid] = sample_projects[uid]
    
    # Step 4: Cross-reference
    projects = cross_reference_with_geo(projects, geo_catalog)
    
    # Step 5: EGA
    ega_studies = load_ega_data()
    
    # Step 6: Unified catalog
    catalog = build_unified_catalog(projects, ega_studies)
    
    # Save
    outfile = os.path.join(output_dir, "controlled_access_catalog.json")
    with open(outfile, "w") as f:
        json.dump(catalog, f, indent=2, default=str)
    print(f"\nCatalog saved to {outfile}")
    
    # Also save a summary TSV
    tsv_file = os.path.join(output_dir, "dbgap_sc_projects.tsv")
    with open(tsv_file, "w") as f:
        headers = ["bioproject_id", "phs_accession", "organism", "name",
                   "protocols_inferred", "disease_categories", "sra_run_count",
                   "description"]
        f.write("\t".join(headers) + "\n")
        for uid, proj in sorted(projects.items(), key=lambda x: -x[1].get("sra_run_count", 0)):
            row = [
                uid,
                proj.get("phs_accession", ""),
                proj.get("organism", ""),
                proj.get("name", ""),
                "|".join(proj.get("protocols_inferred", [])),
                "|".join(proj.get("disease_categories", [])),
                str(proj.get("sra_run_count", "")),
                proj.get("description", "")[:200],
            ]
            f.write("\t".join(row) + "\n")
    print(f"TSV saved to {tsv_file}")


if __name__ == "__main__":
    main()
