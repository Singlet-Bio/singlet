#!/usr/bin/env python3
"""
Get SRA run counts for ALL dbGaP-linked SC BioProjects,
and enrich EGA entries with dataset/study detail.
"""

import requests
import json
import time
import os
from collections import defaultdict

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DELAY = 0.35


def esearch_count(db, term):
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {"db": db, "term": term, "retmax": 0, "retmode": "json"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return int(r.json()["esearchresult"]["count"])


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load existing catalog
    with open(os.path.join(output_dir, "controlled_access_catalog.json")) as f:
        cat = json.load(f)
    
    projects = cat["dbgap_projects"]
    
    # Get SRA counts for all projects not yet queried
    to_query = [(uid, p) for uid, p in projects.items() 
                if p.get("sra_run_count") is None]
    
    print(f"Projects needing SRA run count: {len(to_query)}")
    
    for i, (uid, proj) in enumerate(to_query):
        time.sleep(DELAY)
        try:
            count = esearch_count("sra", f"{uid}[BioProject]")
            proj["sra_run_count"] = count
        except Exception as e:
            proj["sra_run_count"] = 0
        
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(to_query)} queried")
            time.sleep(1)  # Be nice to the API
    
    # Recalculate totals
    total_runs = sum(p.get("sra_run_count", 0) for p in projects.values())
    runs_by_project = sorted(
        [(uid, p.get("sra_run_count", 0)) for uid, p in projects.items()],
        key=lambda x: -x[1]
    )
    
    print(f"\n{'='*60}")
    print(f"COMPLETE SRA RUN COUNTS")
    print(f"{'='*60}")
    print(f"Total BioProjects: {len(projects)}")
    print(f"Total SRA runs: {total_runs:,}")
    print(f"Projects with >0 runs: {sum(1 for _, c in runs_by_project if c > 0)}")
    print(f"Projects with >100 runs: {sum(1 for _, c in runs_by_project if c > 100)}")
    print(f"Projects with >1000 runs: {sum(1 for _, c in runs_by_project if c > 1000)}")
    print(f"Projects with >10000 runs: {sum(1 for _, c in runs_by_project if c > 10000)}")
    
    print(f"\nTop 20 by run count:")
    for uid, count in runs_by_project[:20]:
        p = projects[uid]
        name = p.get("name", "")[:50]
        phs = p.get("phs_accession", "")
        protos = "|".join(p.get("protocols_inferred", []))
        print(f"  {count:>8,}  {phs:12s}  {protos:25s}  {name}")
    
    # Protocol breakdown weighted by run count
    proto_runs = defaultdict(int)
    disease_runs = defaultdict(int)
    for p in projects.values():
        runs = p.get("sra_run_count", 0)
        for proto in p.get("protocols_inferred", []):
            proto_runs[proto] += runs
        for d in p.get("disease_categories", []):
            disease_runs[d] += runs
    
    print(f"\nProtocol inference (by SRA runs):")
    for proto, runs in sorted(proto_runs.items(), key=lambda x: -x[1]):
        print(f"  {proto:20s}: {runs:>8,} runs")
    
    print(f"\nDisease categories (by SRA runs):")
    for d, runs in sorted(disease_runs.items(), key=lambda x: -x[1]):
        print(f"  {d:20s}: {runs:>8,} runs")
    
    # Also try to get the number of SC runs specifically (not all runs)
    # Many of these BioProjects have both SC and bulk data
    print(f"\n{'='*60}")
    print(f"ESTIMATING SINGLE-CELL SPECIFIC RUNS")
    print(f"{'='*60}")
    
    # Sample 30 top projects to check SC-specific run fraction
    sample_projects = [uid for uid, count in runs_by_project[:30] if count > 50]
    sc_fractions = []
    
    for uid in sample_projects[:15]:  # Sample 15
        total = projects[uid].get("sra_run_count", 0)
        if total == 0:
            continue
        time.sleep(DELAY)
        try:
            sc_count = esearch_count("sra", 
                f'{uid}[BioProject] AND "TRANSCRIPTOMIC SINGLE CELL"[Source]')
            frac = sc_count / total if total > 0 else 0
            sc_fractions.append(frac)
            print(f"  {uid}: {sc_count:,}/{total:,} runs are SC ({frac:.1%})")
        except:
            pass
    
    if sc_fractions:
        avg_frac = sum(sc_fractions) / len(sc_fractions)
        print(f"\n  Average SC fraction: {avg_frac:.1%}")
        print(f"  Estimated total SC runs: ~{int(total_runs * avg_frac):,}")
    
    # Save updated catalog
    cat["dbgap_projects"] = projects
    cat["summary"]["dbgap"]["total_sra_runs"] = total_runs
    
    with open(os.path.join(output_dir, "controlled_access_catalog.json"), "w") as f:
        json.dump(cat, f, indent=2, default=str)
    
    print(f"\nUpdated catalog saved")


if __name__ == "__main__":
    main()
