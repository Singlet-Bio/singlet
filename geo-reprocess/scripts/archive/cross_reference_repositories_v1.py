#!/usr/bin/env python3
"""Cross-reference our GEO single-cell catalog against major single-cell repositories.

Queries:
  1. CellxGene (CZI) — collections API, extract GSE IDs from links
  2. EBI Single Cell Expression Atlas (SCEA) — experiments API, E-GEOD-* → GSE*
  3. HCA Data Portal — Azul service API, extract GEO accessions from projects
  4. Broad Single Cell Portal — study API

Outputs:
  - repository_gse_crossref.parquet: one row per GSE with boolean columns for each repo
  - repository_crossref_summary.txt: human-readable summary
"""

import json
import re
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────────
CATALOG_DIR = Path("/mnt/projects/debruinz_project/cellarium/catalog")
OUTPUT_DIR = CATALOG_DIR
TIMEOUT = 60

GSE_RE = re.compile(r"GSE\d{4,9}")


def load_catalog_gses():
    """Load all unique GSE IDs from our catalogs."""
    gse_sets = {}
    files = {
        "geo_single_cell_catalog": "geo_single_cell_catalog.parquet",
        "all_gse_descriptions": "all_gse_descriptions.parquet",
        "processing_catalog": "processing_catalog.parquet",
        "stage7_multimodal": "stage7_multimodal_catalog.parquet",
    }
    all_gses = set()
    for name, fname in files.items():
        p = CATALOG_DIR / fname
        if p.exists():
            df = pd.read_parquet(str(p), columns=["gse_id"])
            s = set(df["gse_id"].dropna().unique())
            gse_sets[name] = s
            all_gses.update(s)
            print(f"  {name}: {len(s):,} unique GSEs")
    print(f"  Union: {len(all_gses):,} unique GSEs\n")
    return all_gses, gse_sets


# ── Repository fetchers ─────────────────────────────────────────────────────────

def fetch_cellxgene_gses():
    """Fetch all GSE IDs referenced in CellxGene collections."""
    print("Fetching CellxGene collections...")
    url = "https://api.cellxgene.cziscience.com/curation/v1/collections"
    try:
        r = requests.get(url, timeout=TIMEOUT, headers={"Accept": "application/json"})
        r.raise_for_status()
        collections = r.json()
    except Exception as e:
        print(f"  ERROR: {e}")
        return {}, {}

    gse_to_collections = defaultdict(list)
    collection_info = {}

    for coll in collections:
        coll_id = coll.get("collection_id", "unknown")
        coll_name = coll.get("name", "")
        links = coll.get("links", [])

        found_gses = set()
        for link in links:
            link_url = link.get("link_url", "")
            link_name = link.get("link_name", "")
            # Extract GSE IDs from URLs and names
            for text in [link_url, link_name]:
                for m in GSE_RE.finditer(text):
                    found_gses.add(m.group())

        for gse in found_gses:
            gse_to_collections[gse].append(coll_id)
        if found_gses:
            collection_info[coll_id] = {
                "name": coll_name,
                "n_gses": len(found_gses),
                "gses": sorted(found_gses),
            }

    print(f"  Found {len(gse_to_collections):,} unique GSEs across {len(collection_info):,} collections")
    return dict(gse_to_collections), collection_info


def fetch_scea_gses():
    """Fetch all GSE IDs from EBI Single Cell Expression Atlas (E-GEOD-* accessions)."""
    print("Fetching EBI SCEA experiments...")
    url = "https://www.ebi.ac.uk/gxa/sc/json/experiments"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ERROR: {e}")
        return {}, {}

    experiments = data.get("experiments", [])
    gse_to_experiments = {}
    experiment_info = {}

    for exp in experiments:
        acc = exp.get("experimentAccession", "")
        # E-GEOD-NNNNN → GSE NNNNN
        if acc.startswith("E-GEOD-"):
            gse_num = acc.replace("E-GEOD-", "")
            gse_id = f"GSE{gse_num}"
            gse_to_experiments[gse_id] = acc
            experiment_info[acc] = {
                "gse_id": gse_id,
                "description": exp.get("experimentDescription", "")[:120],
                "n_assays": exp.get("numberOfAssays", 0),
                "species": exp.get("species", ""),
                "technology": ", ".join(exp.get("technologyType", [])) if isinstance(exp.get("technologyType"), list) else str(exp.get("technologyType", "")),
            }

    all_accessions = [e.get("experimentAccession", "") for e in experiments]
    n_total = len(experiments)
    n_geod = len(gse_to_experiments)
    n_other = n_total - n_geod
    print(f"  {n_total} total experiments, {n_geod} E-GEOD-* (have GSE), {n_other} other accessions")
    return gse_to_experiments, experiment_info


def _parse_hca_hits(hits, gse_to_projects, project_info):
    """Parse HCA project hits, extracting GSE IDs from accessions."""
    for hit in hits:
        project_id = hit.get("entryId", "unknown")
        projects_arr = hit.get("projects", [{}])
        proj = projects_arr[0] if projects_arr else {}
        title = proj.get("projectTitle", "")
        accessions = proj.get("accessions", [])

        found_gses = set()
        for acc_block in accessions:
            namespace = acc_block.get("namespace", "")
            accession = acc_block.get("accession", "")
            # Extract all GSE IDs (handles comma-separated values)
            if namespace == "geo_series" or GSE_RE.search(accession):
                for m in GSE_RE.finditer(accession):
                    found_gses.add(m.group())

        for gse in found_gses:
            gse_to_projects[gse].append(project_id)
        if found_gses:
            project_info[project_id] = {
                "title": title[:120],
                "n_gses": len(found_gses),
                "gses": sorted(found_gses),
            }


def fetch_hca_gses():
    """Fetch GSE IDs from HCA Data Portal via Azul service."""
    print("Fetching HCA Data Portal projects...")
    gse_to_projects = defaultdict(list)
    project_info = {}

    # Try multiple catalog versions (latest first); max page size is 75
    catalogs = ["dcp57", "dcp56", "dcp55", "dcp54", "dcp53"]
    base_url = "https://service.azul.data.humancellatlas.org"
    PAGE_SIZE = 75

    for catalog in catalogs:
        url = f"{base_url}/index/projects?catalog={catalog}&size={PAGE_SIZE}"
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                hits = data.get("hits", [])
                print(f"  Catalog {catalog}: {len(hits)} projects returned")
                _parse_hca_hits(hits, gse_to_projects, project_info)

                if hits:
                    # Check for pagination
                    pagination = data.get("pagination", {})
                    next_url = pagination.get("next", None)
                    page = 1
                    while next_url:
                        page += 1
                        try:
                            r2 = requests.get(next_url, timeout=TIMEOUT)
                            if r2.status_code == 200:
                                data2 = r2.json()
                                hits2 = data2.get("hits", [])
                                _parse_hca_hits(hits2, gse_to_projects, project_info)
                                pagination2 = data2.get("pagination", {})
                                next_url = pagination2.get("next", None)
                                print(f"    Page {page}: {len(hits2)} more projects")
                            else:
                                break
                        except Exception:
                            break
                    break  # Found a working catalog, stop trying others
            elif r.status_code == 404:
                print(f"  Catalog {catalog}: 404")
                continue
            else:
                print(f"  Catalog {catalog}: HTTP {r.status_code}")
                continue
        except Exception as e:
            print(f"  Catalog {catalog}: ERROR {e}")
            continue

    if not gse_to_projects:
        # Fallback: try the projects endpoint without catalog param
        try:
            url = f"{base_url}/index/projects?size={PAGE_SIZE}"
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                hits = data.get("hits", [])
                print(f"  Default catalog: {len(hits)} projects")
                _parse_hca_hits(hits, gse_to_projects, project_info)
        except Exception as e:
            print(f"  Fallback ERROR: {e}")

    print(f"  Found {len(gse_to_projects):,} unique GSEs across {len(project_info):,} projects")
    return dict(gse_to_projects), project_info


def fetch_scp_gses():
    """Fetch GSE IDs from Broad Single Cell Portal."""
    print("Fetching Broad Single Cell Portal studies...")
    # SCP API docs: https://singlecell.broadinstitute.org/single_cell/api
    url = "https://singlecell.broadinstitute.org/single_cell/api/v1/studies"
    gse_to_studies = defaultdict(list)
    study_info = {}

    try:
        # The public API may require auth or have rate limits
        r = requests.get(
            url,
            timeout=TIMEOUT,
            headers={"Accept": "application/json"},
            params={"page": 1, "per_page": 1000},
            verify=False,
        )
        if r.status_code == 200:
            studies = r.json()
            if isinstance(studies, list):
                for study in studies:
                    accession = study.get("accession", "")
                    name = study.get("name", "")
                    description = study.get("description", "")
                    # Look for GSE IDs in accession, name, description
                    for text in [accession, name, description]:
                        for m in GSE_RE.finditer(str(text)):
                            gse_to_studies[m.group()].append(accession)
                            study_info[accession] = {"name": name[:120]}
                print(f"  {len(studies)} studies, {len(gse_to_studies)} with GSE references")
            elif isinstance(studies, dict):
                # Might be paginated response
                study_list = studies.get("studies", studies.get("data", []))
                for study in study_list:
                    accession = study.get("accession", "")
                    name = study.get("name", "")
                    description = study.get("description", "")
                    for text in [accession, name, description]:
                        for m in GSE_RE.finditer(str(text)):
                            gse_to_studies[m.group()].append(accession)
                            study_info[accession] = {"name": name[:120]}
                print(f"  {len(study_list)} studies, {len(gse_to_studies)} with GSE references")
        else:
            print(f"  HTTP {r.status_code} — SCP API may require authentication")
    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"  Found {len(gse_to_studies):,} unique GSEs")
    return dict(gse_to_studies), study_info


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Cross-referencing catalog GSEs against single-cell repositories")
    print("=" * 70)
    print()

    # Load our catalog GSEs
    print("Loading catalog GSEs...")
    our_gses, gse_sets = load_catalog_gses()

    # Fetch from each repository
    repos = {}

    cxg_gses, cxg_info = fetch_cellxgene_gses()
    repos["cellxgene"] = set(cxg_gses.keys())
    print()

    scea_gses, scea_info = fetch_scea_gses()
    repos["scea"] = set(scea_gses.keys())
    print()

    hca_gses, hca_info = fetch_hca_gses()
    repos["hca"] = set(hca_gses.keys())
    print()

    scp_gses, scp_info = fetch_scp_gses()
    repos["scp"] = set(scp_gses.keys())
    print()

    # Build cross-reference dataframe
    print("=" * 70)
    print("Building cross-reference table...")
    print("=" * 70)

    all_external_gses = set()
    for name, gses in repos.items():
        all_external_gses.update(gses)

    # Union of our GSEs and external GSEs
    all_gses_union = our_gses | all_external_gses

    rows = []
    for gse in sorted(all_gses_union):
        row = {"gse_id": gse}
        row["in_our_catalog"] = gse in our_gses
        for name in repos:
            row[f"in_{name}"] = gse in repos[name]
        row["n_repos"] = sum(1 for name in repos if gse in repos[name])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save
    out_path = OUTPUT_DIR / "repository_gse_crossref.parquet"
    df.to_parquet(str(out_path), index=False)
    print(f"Saved cross-reference table: {out_path}")
    print(f"  {len(df):,} total GSEs, {df['in_our_catalog'].sum():,} in our catalog")
    print()

    # ── Summary statistics ──────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    summary_lines = []

    for name in repos:
        col = f"in_{name}"
        n_total = df[col].sum()
        n_also_ours = df[df["in_our_catalog"] & df[col]].shape[0]
        n_only_them = df[~df["in_our_catalog"] & df[col]].shape[0]
        n_only_us = df[df["in_our_catalog"] & ~df[col]].shape[0]
        pct_overlap = 100 * n_also_ours / n_total if n_total > 0 else 0
        label = name.upper()

        line = (
            f"{label:12s}: {n_total:5,} GSEs total | "
            f"{n_also_ours:5,} overlap with us ({pct_overlap:5.1f}%) | "
            f"{n_only_them:5,} they have, we don't | "
            f"{n_only_us:5,} we have, they don't"
        )
        print(line)
        summary_lines.append(line)

    print()

    # Multi-repo presence
    multi = df[df["n_repos"] >= 2]
    multi_ours = multi[multi["in_our_catalog"]]
    multi_missing = multi[~multi["in_our_catalog"]]
    print(f"GSEs in ≥2 external repos: {len(multi):,}")
    print(f"  Already in our catalog: {len(multi_ours):,}")
    print(f"  NOT in our catalog: {len(multi_missing):,}")
    summary_lines.append(f"\nGSEs in ≥2 external repos: {len(multi):,}")
    summary_lines.append(f"  Already in our catalog: {len(multi_ours):,}")
    summary_lines.append(f"  NOT in our catalog: {len(multi_missing):,}")

    # High-value missing datasets (in external repos but NOT in our catalog)
    print()
    print("=" * 70)
    print("HIGH-VALUE MISSING: GSEs in external repos but NOT in our catalog")
    print("=" * 70)

    missing = df[~df["in_our_catalog"] & (df["n_repos"] > 0)].sort_values("n_repos", ascending=False)
    print(f"\nTotal missing GSEs found in external repos: {len(missing):,}")
    summary_lines.append(f"\nTotal missing GSEs in external repos: {len(missing):,}")

    if len(missing) > 0:
        print(f"\nTop 50 missing GSEs (by number of repos):")
        for _, row in missing.head(50).iterrows():
            repo_names = [n for n in repos if row[f"in_{n}"]]
            print(f"  {row['gse_id']:12s}  repos={row['n_repos']}  [{', '.join(repo_names)}]")

        # Also show by repo
        for name in repos:
            col = f"in_{name}"
            missing_in_repo = missing[missing[col]]
            if len(missing_in_repo) > 0:
                print(f"\n  {name.upper()}: {len(missing_in_repo):,} GSEs not in our catalog")
                for _, row in missing_in_repo.head(20).iterrows():
                    gse = row["gse_id"]
                    # Add context from repo metadata
                    extra = ""
                    if name == "cellxgene" and gse in cxg_gses:
                        coll_ids = cxg_gses[gse]
                        if coll_ids and coll_ids[0] in cxg_info:
                            extra = f" — {cxg_info[coll_ids[0]]['name'][:60]}"
                    elif name == "scea" and gse in scea_gses:
                        acc = scea_gses[gse]
                        if acc in scea_info:
                            extra = f" — {scea_info[acc]['description'][:60]}"
                    elif name == "hca" and gse in hca_gses:
                        proj_ids = hca_gses[gse]
                        if proj_ids and proj_ids[0] in hca_info:
                            extra = f" — {hca_info[proj_ids[0]]['title'][:60]}"
                    print(f"    {gse}{extra}")

    # ── Pattern analysis ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("PATTERN ANALYSIS: What do external repos have that we might be missing?")
    print("=" * 70)

    # SCEA species/technology breakdown for missing GSEs
    if scea_info:
        missing_scea = [acc for acc, info in scea_info.items() if info["gse_id"] not in our_gses]
        if missing_scea:
            species_counts = defaultdict(int)
            tech_counts = defaultdict(int)
            for acc in missing_scea:
                info = scea_info[acc]
                species_counts[info["species"]] += 1
                tech_counts[info["technology"]] += 1
            print("\nSCEA missing GSEs by species:")
            for sp, n in sorted(species_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  {sp}: {n}")
            print("\nSCEA missing GSEs by technology:")
            for tech, n in sorted(tech_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  {tech}: {n}")

    # Save summary
    summary_path = OUTPUT_DIR / "repository_crossref_summary.txt"
    with open(str(summary_path), "w") as f:
        f.write("Cross-reference Summary\n")
        f.write("=" * 70 + "\n")
        f.write("\n".join(summary_lines))
        f.write("\n")
    print(f"\nSaved summary: {summary_path}")

    # ── Also save detailed JSON for each repo ───────────────────────────────────
    detail = {
        "cellxgene": {
            "n_gses": len(cxg_gses),
            "n_collections": len(cxg_info),
            "gses": sorted(cxg_gses.keys()),
        },
        "scea": {
            "n_gses": len(scea_gses),
            "n_experiments": len(scea_info),
            "gses": sorted(scea_gses.keys()),
        },
        "hca": {
            "n_gses": len(hca_gses),
            "n_projects": len(hca_info),
            "gses": sorted(hca_gses.keys()),
        },
        "scp": {
            "n_gses": len(scp_gses),
            "gses": sorted(scp_gses.keys()),
        },
    }
    detail_path = OUTPUT_DIR / "repository_crossref_detail.json"
    with open(str(detail_path), "w") as f:
        json.dump(detail, f, indent=2)
    print(f"Saved detail JSON: {detail_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
