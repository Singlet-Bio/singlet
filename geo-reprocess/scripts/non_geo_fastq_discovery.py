#!/usr/bin/env python3
"""
Tier 1+2 Non-GEO FASTQ Source Discovery

Discovers SRA-accessible datasets from non-GEO sources:
  - Tier 1: HCA projects with BioProject but no GEO, CellxGene SRA-direct collections
  - Tier 2: E-MTAB experiments from CellxGene and SCEA → ERP via BioStudies → ENA

Outputs:
  - tier1_sra_accessions.tsv: BioProject → SRR mapping for direct SRA download
  - tier2_emtab_accessions.tsv: E-MTAB → ERP mapping with ENA FASTQ URLs
  - non_geo_acquisition_summary.json: Full structured output
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

CELLXGENE_BASE = "https://api.cellxgene.cziscience.com/curation/v1"
BIOSTUDIES_BASE = "https://www.ebi.ac.uk/biostudies/api/v1/studies"
ENA_BASE = "https://www.ebi.ac.uk/ena/portal/api/filereport"
HCA_BASE = "https://service.azul.data.humancellatlas.org/index/projects"
SCEA_URL = "https://www.ebi.ac.uk/gxa/sc/json/experiments"

GSE_RE = re.compile(r"GSE\d{4,9}")
EMTAB_RE = re.compile(r"E-MTAB-\d+")
ERP_RE = re.compile(r"ERP\d+")
BIOPROJECT_RE = re.compile(r"PRJ[A-Z]{2}\d+|SRP\d+|ERP\d+")


def _get_json(url, params=None, timeout=30):
    """GET request with retry."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"Accept": "application/json"})
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.RequestException:
            time.sleep(1)
    return None


# ---------------------------------------------------------------------------
# Tier 1: SRA-accessible non-GEO sources
# ---------------------------------------------------------------------------

def discover_hca_sra_projects():
    """Find HCA projects with BioProject accessions but no GEO link."""
    print("  Fetching HCA projects...")
    url = HCA_BASE
    params = {"catalog": "dcp57", "size": 75}
    all_projects = []
    page = 0

    while url and page < 12:
        data = _get_json(url, params=params if page == 0 else None, timeout=60)
        if not data:
            break
        all_projects.extend(data.get("hits", []))
        url = data.get("pagination", {}).get("next")
        page += 1
        time.sleep(0.2)

    results = []
    for p in all_projects:
        accs = {}
        title = ""
        organism = ""
        for proj in p.get("projects", []):
            title = title or proj.get("projectTitle", "")
            for a in proj.get("accessions", []):
                ns = a.get("namespace", "")
                acc = a.get("accession", "")
                accs.setdefault(ns, []).append(acc)
        for sp in p.get("samples", []):
            for o in sp.get("organ", []):
                organism = organism or str(o)

        has_geo = bool(accs.get("geo_series"))
        bioprojects = accs.get("insdc_project", [])

        if bioprojects and not has_geo:
            results.append({
                "source": "HCA",
                "title": title,
                "bioprojects": bioprojects,
                "all_accessions": {k: v for k, v in accs.items()},
                "organism": organism,
            })

    print(f"  Found {len(results)} HCA projects with BioProject but no GEO")
    return results


def discover_cellxgene_sra_direct():
    """Find CellxGene collections with direct SRA/BioProject links but no GEO."""
    print("  Fetching CellxGene collections...")
    collections = _get_json(CELLXGENE_BASE + "/collections", timeout=60) or []

    results = []
    for c in collections:
        link_text = " ".join(
            l.get("link_url", "") + " " + l.get("link_name", "")
            for l in c.get("links", [])
        )

        if GSE_RE.search(link_text):
            continue  # already GEO-linked

        bioprojects = BIOPROJECT_RE.findall(link_text)
        if bioprojects:
            results.append({
                "source": "CellxGene",
                "collection_id": c.get("collection_id", ""),
                "title": c.get("name", ""),
                "bioprojects": bioprojects,
                "n_datasets": len(c.get("datasets", [])),
            })

    print(f"  Found {len(results)} CellxGene collections with SRA but no GEO")
    return results


# ---------------------------------------------------------------------------
# Tier 2: E-MTAB via BioStudies → ENA
# ---------------------------------------------------------------------------

def discover_emtab_sources():
    """Collect all E-MTAB accessions from CellxGene and SCEA."""
    emtabs = {}

    # CellxGene E-MTAB collections
    print("  Scanning CellxGene for E-MTAB...")
    collections = _get_json(CELLXGENE_BASE + "/collections", timeout=60) or []
    for c in collections:
        link_text = " ".join(
            l.get("link_url", "") + " " + l.get("link_name", "")
            for l in c.get("links", [])
        )
        for m in EMTAB_RE.finditer(link_text):
            acc = m.group(0)
            emtabs.setdefault(acc, {"sources": set(), "title": ""})
            emtabs[acc]["sources"].add("CellxGene")
            emtabs[acc]["title"] = emtabs[acc]["title"] or c.get("name", "")

    # SCEA E-MTAB experiments
    print("  Scanning SCEA for E-MTAB...")
    scea = _get_json(SCEA_URL, timeout=30)
    if scea:
        for exp in scea.get("experiments", []):
            acc = exp.get("experimentAccession", "")
            if acc.startswith("E-MTAB"):
                emtabs.setdefault(acc, {"sources": set(), "title": ""})
                emtabs[acc]["sources"].add("SCEA")
                emtabs[acc]["title"] = (
                    emtabs[acc]["title"] or exp.get("experimentDescription", "")
                )

    print(f"  Found {len(emtabs)} unique E-MTAB accessions")
    return emtabs


def map_emtab_to_erp(emtab_acc):
    """Map E-MTAB accession to ERP via BioStudies API."""
    data = _get_json(f"{BIOSTUDIES_BASE}/{emtab_acc}", timeout=15)
    if not data:
        return None

    # Extract ERP from links — BioStudies wraps links as list-of-lists
    links = data.get("section", {}).get("links", [])
    for link_group in links:
        items = link_group if isinstance(link_group, list) else [link_group]
        for item in items:
            if isinstance(item, dict):
                url = item.get("url", "")
                m = ERP_RE.search(url)
                if m:
                    return m.group(0)
    return None


def query_ena_runs(erp_acc):
    """Query ENA for run-level file information."""
    params = {
        "accession": erp_acc,
        "result": "read_run",
        "fields": "run_accession,experiment_accession,sample_accession,"
                  "library_strategy,library_source,instrument_model,"
                  "fastq_ftp,submitted_ftp,library_layout",
        "format": "json",
        "limit": 5,  # just sample first 5 for discovery
    }
    data = _get_json(ENA_BASE, params=params, timeout=30)
    if not data:
        return None

    return {
        "n_runs_sampled": len(data),
        "has_fastq_ftp": any(r.get("fastq_ftp", "") for r in data),
        "has_submitted_ftp": any(r.get("submitted_ftp", "") for r in data),
        "library_strategies": list(set(r.get("library_strategy", "") for r in data)),
        "sample_run": data[0] if data else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--tier", choices=["1", "2", "both"], default="both",
                        help="Which tier to discover")
    parser.add_argument("--map-erp", action="store_true",
                        help="For Tier 2, also map E-MTAB → ERP via BioStudies (slow)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}

    # --- Tier 1 ---
    if args.tier in ("1", "both"):
        print("\n=== Tier 1: SRA-accessible non-GEO sources ===")
        hca_projects = discover_hca_sra_projects()
        cxg_sra = discover_cellxgene_sra_direct()

        # Combine into DataFrame
        rows = []
        for p in hca_projects:
            for bp in p["bioprojects"]:
                rows.append({
                    "source": "HCA",
                    "accession": bp,
                    "title": p["title"][:120],
                })
        for c in cxg_sra:
            for bp in c["bioprojects"]:
                rows.append({
                    "source": "CellxGene",
                    "accession": bp,
                    "title": c["title"][:120],
                })

        df1 = pd.DataFrame(rows).drop_duplicates(subset=["accession"])
        df1.to_csv(outdir / "tier1_sra_accessions.tsv", sep="\t", index=False)
        print(f"  Wrote {len(df1)} unique accessions to tier1_sra_accessions.tsv")

        summary["tier1"] = {
            "hca_projects": len(hca_projects),
            "cellxgene_collections": len(cxg_sra),
            "unique_bioprojects": len(df1),
        }

    # --- Tier 2 ---
    if args.tier in ("2", "both"):
        print("\n=== Tier 2: E-MTAB via ENA ===")
        emtabs = discover_emtab_sources()

        rows = []
        for acc, info in emtabs.items():
            row = {
                "emtab": acc,
                "sources": ",".join(sorted(info["sources"])),
                "title": info["title"][:120],
                "erp": "",
                "has_fastq_ftp": "",
                "has_submitted_ftp": "",
                "library_strategies": "",
            }

            if args.map_erp:
                erp = map_emtab_to_erp(acc)
                row["erp"] = erp or ""
                if erp:
                    ena_info = query_ena_runs(erp)
                    if ena_info:
                        row["has_fastq_ftp"] = str(ena_info["has_fastq_ftp"])
                        row["has_submitted_ftp"] = str(ena_info["has_submitted_ftp"])
                        row["library_strategies"] = ",".join(ena_info["library_strategies"])
                time.sleep(0.3)  # rate limit BioStudies

            rows.append(row)

        df2 = pd.DataFrame(rows)
        df2.to_csv(outdir / "tier2_emtab_accessions.tsv", sep="\t", index=False)
        print(f"  Wrote {len(df2)} E-MTAB accessions to tier2_emtab_accessions.tsv")

        summary["tier2"] = {
            "unique_emtabs": len(df2),
            "from_cellxgene": sum(1 for _, r in df2.iterrows() if "CellxGene" in r["sources"]),
            "from_scea": sum(1 for _, r in df2.iterrows() if "SCEA" in r["sources"]),
            "from_both": sum(1 for _, r in df2.iterrows()
                            if "CellxGene" in r["sources"] and "SCEA" in r["sources"]),
        }

    # --- Save summary ---
    with open(outdir / "non_geo_acquisition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to non_geo_acquisition_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
