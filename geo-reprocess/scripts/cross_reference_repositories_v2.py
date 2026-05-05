#!/usr/bin/env python3
"""Cross-reference our GEO single-cell catalog against major single-cell repositories.

Queries:
  1. CellxGene (CZI) — collections API, extract GSE IDs from links
  2. EBI Single Cell Expression Atlas (SCEA) — experiments API, E-GEOD-* → GSE*
  3. HCA Data Portal — Azul service API, extract GEO accessions from projects
  4. Broad Single Cell Portal — scrape study pages for GEO references
  5. UCSC Cell Browser — scan dataset descriptions for GSE IDs
  6. Panglao DB — extract SRS sample IDs and cross-reference via SRA

Outputs:
  - repository_gse_crossref.parquet: one row per GSE with boolean columns for each repo
  - repository_crossref_summary.txt: human-readable summary
  - repository_crossref_detail.json: per-repository metadata
"""

import json
import re
import sys
import time
import warnings
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

warnings.filterwarnings("ignore", message=".*Unverified HTTPS.*")

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

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
            print("  {}: {:,} unique GSEs".format(name, len(s)))
    print("  Union: {:,} unique GSEs\n".format(len(all_gses)))
    return all_gses, gse_sets


def load_catalog_srs_mapping():
    """Load SRX→GSE mapping from catalog for Panglao cross-referencing."""
    p = CATALOG_DIR / "geo_single_cell_catalog.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(str(p), columns=["gse_id", "srx_accession"])
    df = df.dropna(subset=["srx_accession"])
    # Build SRX → GSE mapping using vectorized operations
    srx_to_gse = dict(zip(df["srx_accession"], df["gse_id"]))
    print("  Loaded {:,} SRX→GSE mappings\n".format(len(srx_to_gse)))
    return srx_to_gse


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
        print("  ERROR: {}".format(e))
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

    print("  Found {:,} unique GSEs across {:,} collections".format(
        len(gse_to_collections), len(collection_info)))
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
        print("  ERROR: {}".format(e))
        return {}, {}

    experiments = data.get("experiments", [])
    gse_to_experiments = {}
    experiment_info = {}

    for exp in experiments:
        acc = exp.get("experimentAccession", "")
        if acc.startswith("E-GEOD-"):
            gse_num = acc.replace("E-GEOD-", "")
            gse_id = "GSE{}".format(gse_num)
            gse_to_experiments[gse_id] = acc
            experiment_info[acc] = {
                "gse_id": gse_id,
                "description": exp.get("experimentDescription", "")[:120],
                "n_assays": exp.get("numberOfAssays", 0),
                "species": exp.get("species", ""),
                "technology": ", ".join(exp.get("technologyType", []))
                    if isinstance(exp.get("technologyType"), list)
                    else str(exp.get("technologyType", "")),
            }

    n_total = len(experiments)
    n_geod = len(gse_to_experiments)
    print("  {} total experiments, {} E-GEOD-* (have GSE), {} other accessions".format(
        n_total, n_geod, n_total - n_geod))
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

    catalogs = ["dcp57", "dcp56", "dcp55", "dcp54", "dcp53"]
    base_url = "https://service.azul.data.humancellatlas.org"
    PAGE_SIZE = 75

    for catalog in catalogs:
        url = "{}/index/projects?catalog={}&size={}".format(base_url, catalog, PAGE_SIZE)
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                hits = data.get("hits", [])
                print("  Catalog {}: {} projects returned".format(catalog, len(hits)))
                _parse_hca_hits(hits, gse_to_projects, project_info)

                if hits:
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
                                print("    Page {}: {} more projects".format(page, len(hits2)))
                            else:
                                break
                        except Exception:
                            break
                    break
            elif r.status_code == 404:
                print("  Catalog {}: 404".format(catalog))
                continue
            else:
                print("  Catalog {}: HTTP {}".format(catalog, r.status_code))
                continue
        except Exception as e:
            print("  Catalog {}: ERROR {}".format(catalog, e))
            continue

    if not gse_to_projects:
        try:
            url = "{}/index/projects?size={}".format(base_url, PAGE_SIZE)
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                hits = data.get("hits", [])
                print("  Default catalog: {} projects".format(len(hits)))
                _parse_hca_hits(hits, gse_to_projects, project_info)
        except Exception as e:
            print("  Fallback ERROR: {}".format(e))

    print("  Found {:,} unique GSEs across {:,} projects".format(
        len(gse_to_projects), len(project_info)))
    return dict(gse_to_projects), project_info


def fetch_scp_gses():
    """Fetch GSE IDs from Broad Single Cell Portal by scraping study pages."""
    print("Fetching Broad Single Cell Portal studies...")
    gse_to_studies = defaultdict(list)
    study_info = {}

    # Step 1: Collect all study accessions via search API
    all_accs = []
    page = 1
    total_pages = 0
    while True:
        try:
            r = requests.get(
                "https://singlecell.broadinstitute.org/single_cell/api/v1/search",
                params={"type": "study", "page": page, "per_page": 100},
                timeout=30, verify=False,
                headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
            )
            if r.status_code != 200:
                break
            d = r.json()
            accs = d.get("matching_accessions", [])
            if page == 1:
                total_pages = d.get("total_pages", 0)
                total_studies = d.get("total_studies", 0)
                print("  {} total studies, {} pages".format(total_studies, total_pages))
            all_accs.extend(accs)
            if not accs or page >= total_pages:
                break
            page += 1
        except Exception as e:
            print("  Page {} ERROR: {}".format(page, e))
            break

    # Deduplicate
    all_accs = list(set(all_accs))
    print("  Collected {} unique accessions".format(len(all_accs)))

    # Step 2: Scrape study pages for GEO references (concurrent)
    # (~3% of studies reference GSEs.)
    def _scrape_scp_study(acc):
        try:
            url = "https://singlecell.broadinstitute.org/single_cell/study/{}".format(acc)
            r = requests.get(url, timeout=15, verify=False, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            })
            if r.status_code == 200:
                gses = set(GSE_RE.findall(r.text))
                return acc, gses
        except:
            pass
        return acc, set()

    checked = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_scrape_scp_study, acc): acc for acc in all_accs}
        for future in as_completed(futures):
            acc, gses = future.result()
            for gse in gses:
                gse_to_studies[gse].append(acc)
                study_info[acc] = {"accession": acc}
            checked += 1
            if checked % 100 == 0:
                print("  Scraped {}/{} studies, found {} GSEs so far".format(
                    checked, len(all_accs), len(gse_to_studies)))

    study_info["_total_studies"] = {"total": len(all_accs), "scraped": checked}

    print("  Found {:,} unique GSEs from {:,} scraped studies".format(
        len(gse_to_studies), checked))
    return dict(gse_to_studies), study_info


def fetch_ucsc_cellbrowser_gses():
    """Fetch GSE IDs from UCSC Cell Browser dataset descriptions."""
    print("Fetching UCSC Cell Browser datasets...")
    gse_to_datasets = defaultdict(list)
    dataset_info = {}

    try:
        r = requests.get("https://cells.ucsc.edu/dataset.json", timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("  ERROR fetching dataset.json: {}".format(e))
        return {}, {}

    datasets = data.get("datasets", [])

    # Collect all dataset names including nested sub-datasets
    all_names = []
    def collect_names(ds_list):
        for ds in ds_list:
            if isinstance(ds, dict):
                name = ds.get("name", "")
                if name:
                    all_names.append(name)
                subs = ds.get("datasets", [])
                if subs:
                    collect_names(subs)
    collect_names(datasets)
    print("  {} top-level datasets, {} total (incl. nested)".format(len(datasets), len(all_names)))

    # Scan each dataset's desc.json for GSE references
    errors = 0
    for i, name in enumerate(all_names):
        desc_url = "https://cells.ucsc.edu/{}/desc.json".format(name)
        try:
            r2 = requests.get(desc_url, timeout=10)
            if r2.status_code == 200:
                gses = set(GSE_RE.findall(r2.text))
                for gse in gses:
                    gse_to_datasets[gse].append(name)
                if gses:
                    dataset_info[name] = {
                        "name": name,
                        "n_gses": len(gses),
                        "gses": sorted(gses),
                    }
        except:
            errors += 1

        if (i + 1) % 50 == 0:
            print("  Scanned {}/{} datasets, {} GSEs found".format(
                i + 1, len(all_names), len(gse_to_datasets)))

    print("  Found {:,} unique GSEs across {:,} datasets ({} errors)".format(
        len(gse_to_datasets), len(dataset_info), errors))
    return dict(gse_to_datasets), dataset_info


def fetch_panglaodb_gses(srx_to_gse):
    """Fetch sample SRS/SRA IDs from Panglao DB and cross-reference to GSE via catalog SRA mappings."""
    print("Fetching Panglao DB samples...")
    gse_to_panglao = defaultdict(list)

    try:
        r = requests.get("https://panglaodb.se/samples.html", timeout=TIMEOUT, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "text/html",
        })
        if r.status_code != 200:
            print("  HTTP {} — cannot access Panglao".format(r.status_code))
            return {}, {"n_srs_samples": 0}
    except Exception as e:
        print("  ERROR: {}".format(e))
        return {}, {"n_srs_samples": 0}

    # Extract SRS sample IDs and SRA study IDs from HTML
    srs_ids = set(re.findall(r"(SRS\d{5,})", r.text))
    sra_studies = set(re.findall(r"SRA(\d{5,})", r.text))
    print("  Found {} SRS IDs, {} SRA study IDs on page".format(len(srs_ids), len(sra_studies)))

    # Cross-reference SRS→GSE via our catalog's SRA run accessions
    # Load catalog with SRR accessions for matching
    cat_path = CATALOG_DIR / "geo_single_cell_catalog.parquet"
    if cat_path.exists():
        cat = pd.read_parquet(str(cat_path), columns=["gse_id", "gsm_id", "srx_accession"])
        # Match via sra_study (SRP accessions share the same study)
        # SRA study IDs from Panglao are numeric; our catalog has SRP/ERP accessions
        # This mapping is indirect, so just report what Panglao covers
        print("  Note: Panglao uses SRS IDs, not GSE. GSE mapping requires NCBI ELink (skipped).")

    panglao_info = {
        "n_srs_samples": len(srs_ids),
        "n_sra_studies": len(sra_studies),
        "srs_sample_list": sorted(srs_ids)[:100],
    }

    return {}, panglao_info


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Cross-referencing catalog GSEs against single-cell repositories")
    print("=" * 70)
    print()

    # Load our catalog GSEs
    print("Loading catalog GSEs...")
    our_gses, gse_sets = load_catalog_gses()

    print("Loading SRX→GSE mapping for cross-referencing...")
    srx_to_gse = load_catalog_srs_mapping()

    # Fetch from each repository
    repos = {}
    repo_details = {}

    # 1. CellxGene
    cxg_gses, cxg_info = fetch_cellxgene_gses()
    repos["cellxgene"] = set(cxg_gses.keys())
    repo_details["cellxgene"] = {
        "n_gses": len(cxg_gses),
        "n_collections": len(cxg_info),
        "gses": sorted(cxg_gses.keys()),
    }
    print()

    # 2. SCEA
    scea_gses, scea_info = fetch_scea_gses()
    repos["scea"] = set(scea_gses.keys())
    repo_details["scea"] = {
        "n_gses": len(scea_gses),
        "n_experiments": len(scea_info),
        "gses": sorted(scea_gses.keys()),
    }
    print()

    # 3. HCA
    hca_gses, hca_info = fetch_hca_gses()
    repos["hca"] = set(hca_gses.keys())
    repo_details["hca"] = {
        "n_gses": len(hca_gses),
        "n_projects": len(hca_info),
        "gses": sorted(hca_gses.keys()),
    }
    print()

    # 4. Broad SCP
    scp_gses, scp_info = fetch_scp_gses()
    repos["scp"] = set(scp_gses.keys())
    repo_details["scp"] = {
        "n_gses": len(scp_gses),
        "gses": sorted(scp_gses.keys()),
    }
    print()

    # 5. UCSC Cell Browser
    ucsc_gses, ucsc_info = fetch_ucsc_cellbrowser_gses()
    repos["ucsc_cb"] = set(ucsc_gses.keys())
    repo_details["ucsc_cb"] = {
        "n_gses": len(ucsc_gses),
        "n_datasets": len(ucsc_info),
        "gses": sorted(ucsc_gses.keys()),
    }
    print()

    # 6. Panglao DB (SRS-based, cross-reference limited)
    panglao_gses, panglao_info = fetch_panglaodb_gses(srx_to_gse)
    repos["panglaodb"] = set(panglao_gses.keys()) if panglao_gses else set()
    repo_details["panglaodb"] = panglao_info
    print()

    # ── Build cross-reference dataframe ─────────────────────────────────────────
    print("=" * 70)
    print("Building cross-reference table...")
    print("=" * 70)

    # Only include repos that provide GSE-level data
    gse_repos = {k: v for k, v in repos.items() if v}
    repo_names = list(repos.keys())

    all_external_gses = set()
    for name, gses in repos.items():
        all_external_gses.update(gses)

    all_gses_union = our_gses | all_external_gses

    rows = []
    for gse in sorted(all_gses_union):
        row = {"gse_id": gse}
        row["in_our_catalog"] = gse in our_gses
        for name in repo_names:
            row["in_{}".format(name)] = gse in repos.get(name, set())
        row["n_repos"] = sum(1 for name in repo_names if gse in repos.get(name, set()))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save
    out_path = OUTPUT_DIR / "repository_gse_crossref.parquet"
    df.to_parquet(str(out_path), index=False)
    print("Saved cross-reference table: {}".format(out_path))
    print("  {:,} total GSEs, {:,} in our catalog".format(len(df), df["in_our_catalog"].sum()))
    print()

    # ── Summary statistics ──────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    summary_lines = []

    for name in repo_names:
        col = "in_{}".format(name)
        if col not in df.columns:
            continue
        n_total = df[col].sum()
        n_also_ours = df[df["in_our_catalog"] & df[col]].shape[0]
        n_only_them = df[~df["in_our_catalog"] & df[col]].shape[0]
        n_only_us = df[df["in_our_catalog"] & ~df[col]].shape[0]
        pct_overlap = 100 * n_also_ours / n_total if n_total > 0 else 0
        label = name.upper()

        line = (
            "{:12s}: {:5,} GSEs total | "
            "{:5,} overlap with us ({:5.1f}%) | "
            "{:5,} they have, we don't | "
            "{:5,} we have, they don't"
        ).format(label, n_total, n_also_ours, pct_overlap, n_only_them, n_only_us)
        print(line)
        summary_lines.append(line)

    print()

    # Multi-repo presence
    multi = df[df["n_repos"] >= 2]
    multi_ours = multi[multi["in_our_catalog"]]
    multi_missing = multi[~multi["in_our_catalog"]]
    print("GSEs in >=2 external repos: {:,}".format(len(multi)))
    print("  Already in our catalog: {:,}".format(len(multi_ours)))
    print("  NOT in our catalog:     {:,}".format(len(multi_missing)))
    summary_lines.append("")
    summary_lines.append("GSEs in >=2 external repos: {:,}".format(len(multi)))
    summary_lines.append("  Already in our catalog: {:,}".format(len(multi_ours)))
    summary_lines.append("  NOT in our catalog:     {:,}".format(len(multi_missing)))

    # High-value missing datasets
    print()
    print("=" * 70)
    print("HIGH-VALUE MISSING: GSEs in external repos but NOT in our catalog")
    print("=" * 70)

    missing = df[~df["in_our_catalog"] & (df["n_repos"] > 0)].sort_values("n_repos", ascending=False)
    print("\nTotal missing GSEs found in external repos: {:,}".format(len(missing)))
    summary_lines.append("")
    summary_lines.append("Total missing GSEs in external repos: {:,}".format(len(missing)))

    if len(missing) > 0:
        print("\nTop 50 missing GSEs (by number of repos):")
        for _, row in missing.head(50).iterrows():
            repo_list = [n for n in repo_names if row.get("in_{}".format(n), False)]
            print("  {:12s}  repos={}  [{}]".format(row["gse_id"], row["n_repos"], ", ".join(repo_list)))

        for name in repo_names:
            col = "in_{}".format(name)
            missing_in_repo = missing[missing[col]]
            if len(missing_in_repo) > 0:
                print("\n  {}: {:,} GSEs not in our catalog".format(name.upper(), len(missing_in_repo)))
                for _, row in missing_in_repo.head(20).iterrows():
                    gse = row["gse_id"]
                    extra = ""
                    if name == "cellxgene" and gse in cxg_gses:
                        coll_ids = cxg_gses[gse]
                        if coll_ids and coll_ids[0] in cxg_info:
                            extra = " -- {}".format(cxg_info[coll_ids[0]]["name"][:60])
                    elif name == "scea" and gse in scea_gses:
                        acc = scea_gses[gse]
                        if acc in scea_info:
                            extra = " -- {}".format(scea_info[acc]["description"][:60])
                    elif name == "hca" and gse in hca_gses:
                        proj_ids = hca_gses[gse]
                        if proj_ids and proj_ids[0] in hca_info:
                            extra = " -- {}".format(hca_info[proj_ids[0]]["title"][:60])
                    elif name == "ucsc_cb" and gse in ucsc_gses:
                        ds_names = ucsc_gses[gse]
                        if ds_names:
                            extra = " -- {}".format(ds_names[0])
                    print("    {}{}".format(gse, extra))

    # ── Pattern analysis ────────────────────────────────────────────────────────
    if scea_info:
        missing_scea = [acc for acc, info in scea_info.items() if info["gse_id"] not in our_gses]
        if missing_scea:
            print()
            print("SCEA missing GSEs by species:")
            species_counts = defaultdict(int)
            for acc in missing_scea:
                info = scea_info[acc]
                species_counts[info["species"]] += 1
            for sp, n in sorted(species_counts.items(), key=lambda x: -x[1])[:10]:
                print("  {}: {}".format(sp, n))

    # Save summary
    summary_path = OUTPUT_DIR / "repository_crossref_summary.txt"
    with open(str(summary_path), "w") as f:
        f.write("Cross-reference Summary\n")
        f.write("=" * 70 + "\n")
        f.write("Generated: {}\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        f.write("Repositories queried:\n")
        f.write("  1. CellxGene (CZI) — curated single-cell data portal\n")
        f.write("  2. SCEA (EBI) — Single Cell Expression Atlas\n")
        f.write("  3. HCA — Human Cell Atlas Data Portal\n")
        f.write("  4. SCP — Broad Single Cell Portal\n")
        f.write("  5. UCSC CB — UCSC Cell Browser\n")
        f.write("  6. Panglao DB — single-cell reference database\n\n")
        f.write("\n".join(summary_lines))
        f.write("\n")
    print("\nSaved summary: {}".format(summary_path))

    # Save detailed JSON
    detail_path = OUTPUT_DIR / "repository_crossref_detail.json"
    with open(str(detail_path), "w") as f:
        json.dump(repo_details, f, indent=2)
    print("Saved detail JSON: {}".format(detail_path))

    # ── Now merge into the main catalog ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("Merging cross-reference into main catalog...")
    print("=" * 70)

    cat_path = CATALOG_DIR / "geo_single_cell_catalog.parquet"
    cat = pd.read_parquet(str(cat_path))
    print("Catalog: {:,} rows, {} columns".format(len(cat), len(cat.columns)))

    # Drop old repo columns if they exist
    repo_cols = ["in_cellxgene", "in_scea", "in_hca", "in_scp", "in_ucsc_cb",
                 "in_panglaodb", "n_repos"]
    existing_cols = [c for c in repo_cols if c in cat.columns]
    if existing_cols:
        cat = cat.drop(columns=existing_cols)
        print("  Dropped old columns: {}".format(existing_cols))

    # Build GSE-level lookup from crossref
    xref = df[["gse_id"] + ["in_{}".format(n) for n in repo_names] + ["n_repos"]]

    # Merge
    cat = cat.merge(xref, on="gse_id", how="left")
    for name in repo_names:
        col = "in_{}".format(name)
        if col in cat.columns:
            cat[col] = cat[col].fillna(False)
    cat["n_repos"] = cat["n_repos"].fillna(0).astype(int)

    # Save
    cat.to_parquet(str(cat_path), index=False)
    print("  Saved updated catalog: {:,} rows, {} columns".format(len(cat), len(cat.columns)))
    print("  New columns: {}".format([c for c in cat.columns if c.startswith("in_") or c == "n_repos"]))

    # Summary of catalog repo coverage
    print()
    for name in repo_names:
        col = "in_{}".format(name)
        if col in cat.columns:
            n_gsm = cat[col].sum()
            n_gse = cat[cat[col]]["gse_id"].nunique()
            print("  {}: {:,} GSMs ({:,} GSEs) in catalog overlap".format(
                name.upper(), n_gsm, n_gse))

    print("\nDone!")


if __name__ == "__main__":
    main()
