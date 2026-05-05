#!/usr/bin/env python3
"""
Non-GEO Sample Ingestion Pipeline

Enumerates run-level data from non-GEO sources (HCA BioProjects, E-MTAB experiments),
deduplicates against the existing GEO catalog, and adds new samples in catalog-compatible format.

Outputs:
  - non_geo_samples.parquet: New samples ready for catalog merge
  - non_geo_controlled_access.json: Controlled access datasets for EGA/dbGaP catalog
  - non_geo_ingestion_log.txt: Processing log

Usage:
  python3 ingest_non_geo_samples.py [--outdir DIR] [--dry-run] [--max-per-source N]
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants matching catalog schema
# ---------------------------------------------------------------------------
CATALOG_COLUMNS = [
    "gse_id", "gsm_id", "series_title", "organism", "taxon_id",
    "protocol_inferred", "protocol_confidence", "species_ref_genome",
    "species_annotation", "srx_accession", "srr_accessions",
    "library_strategy", "library_source", "library_selection",
    "library_layout", "instrument_platform", "instrument_model",
    "sra_study", "bioproject", "read_count", "base_count",
    "ena_fastq_r1", "ena_fastq_r2", "ncbi_sra_s3",
    "submission_date", "last_update_date", "pubmed_ids",
    "contact_name", "contact_institute", "summary", "overall_design",
    "sample_organism", "sample_source", "sample_characteristics",
    "supplementary_files", "n_gsm_in_series", "processing_status", "notes",
    "in_cellxgene", "in_scea", "in_hca", "in_scp", "in_ucsc_cb",
    "in_panglaodb", "n_repos",
]

# Species reference genomes (must match existing pipeline)
SPECIES_REF = {
    "homo sapiens": ("GRCh38", "GENCODE v45"),
    "mus musculus": ("GRCm39", "GENCODE vM35"),
    "rattus norvegicus": ("mRatBN7.2", "Ensembl 112"),
    "danio rerio": ("GRCz11", "Ensembl 112"),
    "macaca mulatta": ("Mmul_10", "Ensembl 112"),
    "macaca fascicularis": ("Macaca_fascicularis_6.0", "Ensembl 112"),
    "sus scrofa": ("Sscrofa11.1", "Ensembl 112"),
    "gallus gallus": ("bGalGal1.mat.broiler.GRCg7b", "Ensembl 112"),
    "bos taurus": ("ARS-UCD1.3", "Ensembl 112"),
    "canis lupus familiaris": ("ROS_Cfam_1.0", "Ensembl 112"),
    "drosophila melanogaster": ("BDGP6.46", "Ensembl 112"),
    "caenorhabditis elegans": ("WBcel235", "Ensembl 112"),
    "pan troglodytes": ("Pan_tro_3.0", "Ensembl 112"),
    "callithrix jacchus": ("mCalJac1.pat.X", "Ensembl 112"),
}

# ENA API
ENA_FILEREPORT = "https://www.ebi.ac.uk/ena/portal/api/filereport"
ENA_SEARCH = "https://www.ebi.ac.uk/ena/portal/api/search"
BIOSTUDIES_API = "https://www.ebi.ac.uk/biostudies/api/v1/studies"
HCA_API = "https://service.azul.data.humancellatlas.org/index/projects"
CELLXGENE_API = "https://api.cellxgene.cziscience.com/curation/v1"
SCEA_API = "https://www.ebi.ac.uk/gxa/sc/json/experiments"

ERP_RE = re.compile(r"ERP\d+")
GSE_RE = re.compile(r"GSE\d{4,9}")
EMTAB_RE = re.compile(r"E-MTAB-\d+")


def _get(url, params=None, timeout=30):
    """GET with retry."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"Accept": "application/json"})
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
        except requests.RequestException:
            time.sleep(1)
    return None


def log(msg, logfile=None):
    print(msg)
    if logfile:
        logfile.write(msg + "\n")
        logfile.flush()


# ---------------------------------------------------------------------------
# ENA run enumeration
# ---------------------------------------------------------------------------

def enumerate_ena_runs(accession, limit=0):
    """Get all runs for a study/project accession from ENA.
    
    accession: ERP*, SRP*, PRJNA*, PRJEB* etc.
    Returns list of dicts with run-level metadata.
    """
    fields = (
        "run_accession,experiment_accession,sample_accession,study_accession,"
        "library_strategy,library_source,library_selection,library_layout,"
        "instrument_platform,instrument_model,read_count,base_count,"
        "fastq_ftp,fastq_bytes,submitted_ftp,tax_id,scientific_name,"
        "experiment_title,sample_title,study_title,sample_description,"
        "center_name,first_public,last_updated"
    )
    params = {
        "accession": accession,
        "result": "read_run",
        "fields": fields,
        "format": "json",
    }
    if limit > 0:
        params["limit"] = limit
    
    r = _get(ENA_FILEREPORT, params=params, timeout=60)
    if r is None:
        return []
    try:
        return r.json()
    except Exception:
        return []


def build_ena_fastq_urls(run):
    """Build R1/R2 FASTQ URLs from ENA run metadata."""
    # Prefer fastq_ftp (pre-processed FASTQs)
    ftp = run.get("fastq_ftp", "")
    if ftp:
        parts = ftp.split(";")
        r1 = ""
        r2 = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if "_1.fastq.gz" in p:
                r1 = "ftp://" + p if not p.startswith("ftp://") else p
            elif "_2.fastq.gz" in p:
                r2 = "ftp://" + p if not p.startswith("ftp://") else p
        if r1 or r2:
            return r1, r2

    # Fall back to submitted_ftp
    sub = run.get("submitted_ftp", "")
    if sub:
        parts = sub.split(";")
        r1 = ""
        r2 = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            pl = p.lower()
            if "r1" in pl or "_1.fastq" in pl or "_1.fq" in pl:
                r1 = "ftp://" + p if not p.startswith("ftp://") else p
            elif "r2" in pl or "_2.fastq" in pl or "_2.fq" in pl:
                r2 = "ftp://" + p if not p.startswith("ftp://") else p
        if r1 or r2:
            return r1, r2

    # Construct standard ENA path from run accession
    srr = run.get("run_accession", "")
    if srr:
        prefix = srr[:6]
        suffix = srr[9:] if len(srr) > 9 else ""
        suffix_part = "/" + suffix.zfill(3) if suffix else ""
        base = f"ftp://ftp.sra.ebi.ac.uk/vol1/fastq/{prefix}{suffix_part}/{srr}"
        return f"{base}_1.fastq.gz", f"{base}_2.fastq.gz"
    
    return "", ""


def infer_protocol(run):
    """Infer single-cell protocol from ENA metadata."""
    title = (run.get("experiment_title", "") + " " + 
             run.get("sample_title", "") + " " +
             run.get("study_title", "") + " " +
             run.get("sample_description", "")).lower()
    
    strategy = run.get("library_strategy", "").upper()
    
    # Protocol keywords
    if "10x" in title and "v3" in title:
        return "10xv3", "medium"
    elif "10x" in title and "v2" in title:
        return "10xv2", "medium"
    elif "10x" in title or "chromium" in title:
        return "10x_suspect", "low"
    elif "drop-seq" in title or "dropseq" in title:
        return "dropseq", "medium"
    elif "smart-seq" in title or "smartseq" in title:
        return "smartseq2", "medium"
    elif "celseq" in title or "cel-seq" in title:
        return "celseq2", "medium"
    elif "sci-rna" in title:
        return "scirna", "medium"
    elif "parse" in title and "bioscience" in title:
        return "parse", "medium"
    elif "bd rhapsody" in title:
        return "bd_rhapsody", "medium"
    elif "indrop" in title:
        return "indrop", "medium"
    elif "seq-well" in title or "seqwell" in title:
        return "seqwell", "medium"
    elif "split-seq" in title:
        return "splitseq", "medium"
    elif "dnbelab" in title:
        return "dnbelab", "medium"
    elif strategy == "RNA-SEQ":
        return "unknown_sc", "low"
    else:
        return "unknown", "low"


def get_species_ref(organism):
    """Get reference genome for organism."""
    org_lower = organism.lower().strip()
    if org_lower in SPECIES_REF:
        return SPECIES_REF[org_lower]
    return ("", "")


# ---------------------------------------------------------------------------
# Source: HCA BioProjects (Tier 1)
# ---------------------------------------------------------------------------

def discover_hca_bioprojects():
    """Get HCA projects with BioProject accessions but no GEO."""
    url = HCA_API
    params = {"catalog": "dcp57", "size": 75}
    all_projects = []
    page = 0
    
    while url and page < 12:
        r = _get(url, params=params if page == 0 else None, timeout=60)
        if not r:
            break
        data = r.json()
        all_projects.extend(data.get("hits", []))
        url = data.get("pagination", {}).get("next")
        page += 1
        time.sleep(0.2)
    
    results = []
    for p in all_projects:
        accs = {}
        title = ""
        for proj in p.get("projects", []):
            title = title or proj.get("projectTitle", "")
            for a in proj.get("accessions", []):
                accs.setdefault(a.get("namespace", ""), []).append(a.get("accession", ""))
        
        has_geo = bool(accs.get("geo_series"))
        bioprojects = accs.get("insdc_project", [])
        geo_series = accs.get("geo_series", [])
        
        if bioprojects and not has_geo:
            results.append({
                "title": title,
                "bioprojects": bioprojects,
                "geo_series": geo_series,
                "all_accessions": accs,
            })
    
    return results


# ---------------------------------------------------------------------------
# Source: CellxGene non-GEO (Tier 1 SRA + Tier 2 E-MTAB + Controlled)
# ---------------------------------------------------------------------------

def discover_cellxgene_non_geo():
    """Categorize all CellxGene collections by data source and access."""
    r = _get(CELLXGENE_API + "/collections", timeout=60)
    if not r:
        return {"sra_direct": [], "emtab": [], "ega": [], "dbgap": []}
    
    collections = r.json()
    result = {"sra_direct": [], "emtab": [], "ega": [], "dbgap": []}
    
    PHS_RE = re.compile(r"phs\d{6}")
    EGA_RE = re.compile(r"ega-archive|EGAD\d+|EGAS\d+")
    BP_RE = re.compile(r"PRJNA\d+|PRJEB\d+|SRP\d+|ERP\d+")
    
    for c in collections:
        link_text = " ".join(
            l.get("link_url", "") + " " + l.get("link_name", "")
            for l in c.get("links", [])
        )
        
        has_geo = bool(GSE_RE.search(link_text))
        has_emtab = bool(EMTAB_RE.search(link_text))
        has_ega = bool(EGA_RE.search(link_text.lower()))
        has_dbgap = bool(PHS_RE.search(link_text))
        has_sra = bool(BP_RE.search(link_text)) and not has_geo
        
        info = {
            "collection_id": c.get("collection_id", ""),
            "name": c.get("name", ""),
            "links": c.get("links", []),
            "n_datasets": len(c.get("datasets", [])),
        }
        
        if has_geo:
            continue  # already in our catalog
        
        if has_ega:
            info["ega_accessions"] = list(dict.fromkeys(re.findall(r"EGAD\d+|EGAS\d+", link_text)))
            result["ega"].append(info)
        
        if has_dbgap:
            info["phs_accessions"] = list(dict.fromkeys(PHS_RE.findall(link_text)))
            result["dbgap"].append(info)
        
        if has_emtab:
            info["emtab_accessions"] = list(dict.fromkeys(EMTAB_RE.findall(link_text)))
            result["emtab"].append(info)
        elif has_sra and not has_ega and not has_dbgap:
            info["sra_accessions"] = list(dict.fromkeys(BP_RE.findall(link_text)))
            result["sra_direct"].append(info)
    
    return result


# ---------------------------------------------------------------------------
# Source: SCEA E-MTAB (Tier 2)
# ---------------------------------------------------------------------------

def discover_scea_emtabs():
    """Get all SCEA E-MTAB experiments."""
    r = _get(SCEA_API, timeout=30)
    if not r:
        return []
    
    results = []
    for exp in r.json().get("experiments", []):
        acc = exp.get("experimentAccession", "")
        if acc.startswith("E-MTAB"):
            sp = exp.get("species", "")
            species_list = [sp] if isinstance(sp, str) and sp else (
                [s.get("val", "") if isinstance(s, dict) else str(s) for s in sp]
                if isinstance(sp, list) else []
            )
            results.append({
                "accession": acc,
                "title": exp.get("experimentDescription", ""),
                "species": species_list,
                "n_assays": exp.get("numberOfAssays", 0),
            })
    return results


# ---------------------------------------------------------------------------
# E-MTAB → ERP mapping
# ---------------------------------------------------------------------------

def map_emtab_to_erp(emtab_acc):
    """Map E-MTAB to ERP via BioStudies API."""
    r = _get(f"{BIOSTUDIES_API}/{emtab_acc}", timeout=15)
    if not r:
        return None
    data = r.json()
    links = data.get("section", {}).get("links", [])
    for link_group in links:
        items = link_group if isinstance(link_group, list) else [link_group]
        for item in items:
            if isinstance(item, dict):
                m = ERP_RE.search(item.get("url", ""))
                if m:
                    return m.group(0)
    return None


# ---------------------------------------------------------------------------
# Build catalog rows from ENA runs
# ---------------------------------------------------------------------------

def runs_to_catalog_rows(runs, source_id, source_type, source_title, 
                         is_cellxgene=False, is_scea=False, is_hca=False):
    """Convert ENA runs to catalog-compatible rows."""
    rows = []
    
    # Group runs by experiment (SRX) to create sample-level entries
    by_experiment = {}
    for run in runs:
        srx = run.get("experiment_accession", run.get("run_accession", ""))
        by_experiment.setdefault(srx, []).append(run)
    
    for srx, srx_runs in by_experiment.items():
        # Take metadata from first run
        first = srx_runs[0]
        organism = first.get("scientific_name", "")
        ref_genome, annotation = get_species_ref(organism)
        
        # Skip non-supported species
        if not ref_genome and organism.lower().strip() not in SPECIES_REF:
            # Still include but mark
            pass
        
        # Aggregate SRR accessions
        srr_list = [r.get("run_accession", "") for r in srx_runs]
        total_reads = sum(int(r.get("read_count", 0) or 0) for r in srx_runs)
        total_bases = sum(int(r.get("base_count", 0) or 0) for r in srx_runs)
        
        # Build FASTQ URLs from first run (representative)
        r1_url, r2_url = build_ena_fastq_urls(first)
        
        # Infer protocol
        protocol, confidence = infer_protocol(first)
        
        # Build catalog-format ID
        # For non-GEO, use source_type prefix instead of GSE/GSM
        if source_type == "hca_bioproject":
            gse_equiv = f"HCA_{source_id}"
        elif source_type == "emtab":
            gse_equiv = source_id  # E-MTAB-NNNNN
        elif source_type == "cellxgene_sra":
            gse_equiv = f"CXG_{source_id[:8]}"
        else:
            gse_equiv = f"EXT_{source_id}"
        
        gsm_equiv = srx  # Use SRX as sample ID
        
        row = {
            "gse_id": gse_equiv,
            "gsm_id": gsm_equiv,
            "series_title": source_title[:500] if source_title else "",
            "organism": organism,
            "taxon_id": str(first.get("tax_id", "")),
            "protocol_inferred": protocol,
            "protocol_confidence": confidence,
            "species_ref_genome": ref_genome,
            "species_annotation": annotation,
            "srx_accession": srx,
            "srr_accessions": ";".join(srr_list),
            "library_strategy": first.get("library_strategy", ""),
            "library_source": first.get("library_source", ""),
            "library_selection": first.get("library_selection", ""),
            "library_layout": first.get("library_layout", ""),
            "instrument_platform": first.get("instrument_platform", ""),
            "instrument_model": first.get("instrument_model", ""),
            "sra_study": first.get("study_accession", ""),
            "bioproject": source_id if "PRJ" in source_id else "",
            "read_count": float(total_reads) if total_reads else float("nan"),
            "base_count": float(total_bases) if total_bases else float("nan"),
            "ena_fastq_r1": r1_url,
            "ena_fastq_r2": r2_url,
            "ncbi_sra_s3": "",
            "submission_date": first.get("first_public", ""),
            "last_update_date": first.get("last_updated", ""),
            "pubmed_ids": "",
            "contact_name": first.get("center_name", ""),
            "contact_institute": first.get("center_name", ""),
            "summary": source_title[:1000] if source_title else "",
            "overall_design": "",
            "sample_organism": organism,
            "sample_source": first.get("sample_title", ""),
            "sample_characteristics": first.get("sample_description", ""),
            "supplementary_files": "",
            "n_gsm_in_series": len(by_experiment),
            "processing_status": "pending",
            "notes": f"non_geo_{source_type}",
            "in_cellxgene": is_cellxgene,
            "in_scea": is_scea,
            "in_hca": is_hca,
            "in_scp": False,
            "in_ucsc_cb": False,
            "in_panglaodb": False,
            "n_repos": sum([is_cellxgene, is_scea, is_hca]),
        }
        rows.append(row)
    
    return rows


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def load_existing_accessions(catalog_path):
    """Load all SRX and SRR accessions from existing catalog for dedup."""
    cat = pd.read_parquet(catalog_path)
    
    existing_srx = set(cat["srx_accession"].dropna().unique())
    existing_srr = set()
    for srrs in cat["srr_accessions"].dropna():
        for srr in str(srrs).split(";"):
            srr = srr.strip()
            if srr:
                existing_srr.add(srr)
    
    existing_bioproject = set(cat["bioproject"].dropna().unique())
    existing_gse = set(cat["gse_id"].dropna().unique())
    
    return {
        "srx": existing_srx,
        "srr": existing_srr,
        "bioproject": existing_bioproject,
        "gse": existing_gse,
        "sra_study": set(cat["sra_study"].dropna().unique()),
    }


def is_duplicate(row, existing):
    """Check if a row is already in the catalog."""
    srx = row.get("srx_accession", "")
    if srx and srx in existing["srx"]:
        return True
    
    srrs = row.get("srr_accessions", "")
    for srr in srrs.split(";"):
        if srr.strip() and srr.strip() in existing["srr"]:
            return True
    
    return False


# ---------------------------------------------------------------------------
# Controlled access catalog builder
# ---------------------------------------------------------------------------

def build_controlled_access_entries(cellxgene_data):
    """Build controlled access entries from CellxGene EGA/dbGaP collections,
    formatted to match existing controlled_access_catalog.json schema."""
    ega_entries = []
    dbgap_entries = {}
    
    # EGA collections from CellxGene
    for coll in cellxgene_data.get("ega", []):
        for ega_id in coll.get("ega_accessions", []):
            ega_entries.append({
                "ega_id": ega_id,
                "name": coll["name"],
                "description": f"CellxGene collection: {coll['name']}. "
                              f"Contains {coll['n_datasets']} dataset(s). "
                              f"Data Access: Requires EGA Data Access Committee (DAC) approval. "
                              f"Apply at https://ega-archive.org/access/. "
                              f"NOT compatible with dbGaP DAR — separate application required. "
                              f"Typical approval: 2-8 weeks with institutional ethics approval.",
                "source": "cellxgene",
                "collection_id": coll["collection_id"],
                "dar_compatible": False,
            })
    
    # dbGaP collections from CellxGene
    for coll in cellxgene_data.get("dbgap", []):
        for phs in coll.get("phs_accessions", []):
            dbgap_entries[phs] = {
                "bioproject_id": phs,
                "name": coll["name"],
                "title": coll["name"],
                "description": f"CellxGene collection: {coll['name']}. "
                              f"Contains {coll['n_datasets']} dataset(s). "
                              f"Data Access: Requires dbGaP Authorized Access via eRA Commons. "
                              f"COMPATIBLE with existing dbGaP DAR if study is included in "
                              f"the approved project scope. Apply at https://dbgap.ncbi.nlm.nih.gov/. "
                              f"Requires PI eRA Commons account, IRB approval, Data Use "
                              f"Certification, and IT security plan. Typical: 1-3 months.",
                "source": "cellxgene",
                "collection_id": coll["collection_id"],
                "dar_compatible": True,
            }
    
    return {"ega_studies": ega_entries, "dbgap_projects": dbgap_entries}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default="/mnt/projects/debruinz_project/cellarium/catalog",
                        help="Output directory")
    parser.add_argument("--catalog", 
                        default="/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet",
                        help="Existing catalog parquet")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write to catalog, just discover")
    parser.add_argument("--max-per-source", type=int, default=0,
                        help="Max runs to enumerate per source (0=unlimited)")
    parser.add_argument("--skip-tier1", action="store_true")
    parser.add_argument("--skip-tier2", action="store_true")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logf = open(outdir / "non_geo_ingestion_log.txt", "w")
    
    # Load existing catalog for dedup
    log("Loading existing catalog for deduplication...", logf)
    existing = load_existing_accessions(args.catalog)
    log(f"  Existing: {len(existing['srx']):,} SRX, {len(existing['srr']):,} SRR, "
        f"{len(existing['bioproject']):,} BioProjects, {len(existing['gse']):,} GSEs", logf)
    
    all_new_rows = []
    stats = Counter()
    
    # ===== TIER 1: SRA-accessible non-GEO =====
    if not args.skip_tier1:
        log("\n" + "=" * 60, logf)
        log("TIER 1: SRA-accessible non-GEO sources", logf)
        log("=" * 60, logf)
        
        # 1a. HCA BioProjects
        log("\n--- HCA BioProjects ---", logf)
        hca_projects = discover_hca_bioprojects()
        log(f"Found {len(hca_projects)} HCA projects with BioProject, no GEO", logf)
        
        for proj in hca_projects:
            for bp in proj["bioprojects"]:
                if bp in existing["bioproject"] or bp in existing["sra_study"]:
                    log(f"  SKIP {bp}: already in catalog as bioproject/sra_study", logf)
                    stats["tier1_dup_bioproject"] += 1
                    continue
                
                log(f"  Enumerating {bp}: {proj['title'][:60]}...", logf)
                limit = args.max_per_source if args.max_per_source > 0 else 0
                runs = enumerate_ena_runs(bp, limit=limit)
                
                if not runs:
                    log(f"    No runs found for {bp}", logf)
                    stats["tier1_no_runs"] += 1
                    continue
                
                # Filter to RNA-Seq
                rna_runs = [r for r in runs if r.get("library_strategy", "").upper() 
                           in ("RNA-SEQ", "OTHER", "SSRNA-SEQ", "")]
                non_rna = len(runs) - len(rna_runs)
                if non_rna > 0:
                    log(f"    Filtered {non_rna} non-RNA runs", logf)
                
                rows = runs_to_catalog_rows(
                    rna_runs, bp, "hca_bioproject", proj["title"],
                    is_hca=True
                )
                
                # Dedup at sample level
                new_rows = [r for r in rows if not is_duplicate(r, existing)]
                dup_rows = len(rows) - len(new_rows)
                
                if dup_rows > 0:
                    log(f"    {dup_rows} samples already in catalog", logf)
                    stats["tier1_dup_samples"] += dup_rows
                
                # Track new entries for dedup within run
                for r in new_rows:
                    srx = r.get("srx_accession", "")
                    if srx:
                        existing["srx"].add(srx)
                    for srr in r.get("srr_accessions", "").split(";"):
                        if srr.strip():
                            existing["srr"].add(srr.strip())
                existing["bioproject"].add(bp)
                
                all_new_rows.extend(new_rows)
                stats["tier1_new_samples"] += len(new_rows)
                log(f"    {len(new_rows)} new samples from {len(rna_runs)} runs", logf)
                time.sleep(0.3)
        
        # 1b. CellxGene SRA-direct
        log("\n--- CellxGene SRA-direct ---", logf)
        cxg_data = discover_cellxgene_non_geo()
        log(f"Found {len(cxg_data['sra_direct'])} CellxGene collections with SRA (no GEO)", logf)
        
        seen_tier1_accs = set()
        for coll in cxg_data["sra_direct"]:
            for acc in coll.get("sra_accessions", []):
                if acc in seen_tier1_accs:
                    continue
                seen_tier1_accs.add(acc)
                if acc in existing["bioproject"] or acc in existing["sra_study"]:
                    stats["tier1_dup_bioproject"] += 1
                    continue
                
                log(f"  Enumerating {acc}: {coll['name'][:60]}...", logf)
                limit = args.max_per_source if args.max_per_source > 0 else 0
                runs = enumerate_ena_runs(acc, limit=limit)
                
                if not runs:
                    stats["tier1_no_runs"] += 1
                    continue
                
                rna_runs = [r for r in runs if r.get("library_strategy", "").upper() 
                           in ("RNA-SEQ", "OTHER", "SSRNA-SEQ", "")]
                
                rows = runs_to_catalog_rows(
                    rna_runs, acc, "cellxgene_sra", coll["name"],
                    is_cellxgene=True
                )
                new_rows = [r for r in rows if not is_duplicate(r, existing)]
                # Track new entries for dedup within run
                for r in new_rows:
                    srx = r.get("srx_accession", "")
                    if srx:
                        existing["srx"].add(srx)
                    for srr in r.get("srr_accessions", "").split(";"):
                        if srr.strip():
                            existing["srr"].add(srr.strip())
                existing["bioproject"].add(acc)
                
                all_new_rows.extend(new_rows)
                stats["tier1_new_samples"] += len(new_rows)
                log(f"    {len(new_rows)} new samples", logf)
                time.sleep(0.3)
    
    # ===== TIER 2: E-MTAB via ENA =====
    if not args.skip_tier2:
        log("\n" + "=" * 60, logf)
        log("TIER 2: E-MTAB via ENA (BioStudies → ERP)", logf)
        log("=" * 60, logf)
        
        # Collect all E-MTAB accessions
        emtab_sources = {}  # acc -> {sources, title}
        
        # From CellxGene
        if not args.skip_tier1:  # already have cxg_data
            for coll in cxg_data.get("emtab", []):
                for acc in coll.get("emtab_accessions", []):
                    emtab_sources.setdefault(acc, {"sources": set(), "title": ""})
                    emtab_sources[acc]["sources"].add("cellxgene")
                    emtab_sources[acc]["title"] = emtab_sources[acc]["title"] or coll["name"]
        else:
            cxg_data = discover_cellxgene_non_geo()
            for coll in cxg_data.get("emtab", []):
                for acc in coll.get("emtab_accessions", []):
                    emtab_sources.setdefault(acc, {"sources": set(), "title": ""})
                    emtab_sources[acc]["sources"].add("cellxgene")
                    emtab_sources[acc]["title"] = emtab_sources[acc]["title"] or coll["name"]
        
        # From SCEA
        scea_emtabs = discover_scea_emtabs()
        for exp in scea_emtabs:
            acc = exp["accession"]
            emtab_sources.setdefault(acc, {"sources": set(), "title": ""})
            emtab_sources[acc]["sources"].add("scea")
            emtab_sources[acc]["title"] = emtab_sources[acc]["title"] or exp["title"]
        
        log(f"Total unique E-MTAB accessions: {len(emtab_sources)}", logf)
        
        # Process each E-MTAB
        erp_map_success = 0
        erp_map_fail = 0
        
        for emtab_acc, info in sorted(emtab_sources.items()):
            # Map to ERP
            erp = map_emtab_to_erp(emtab_acc)
            if not erp:
                erp_map_fail += 1
                stats["tier2_no_erp"] += 1
                continue
            erp_map_success += 1
            
            # Check if ERP is already in catalog
            if erp in existing["sra_study"] or erp in existing["bioproject"]:
                stats["tier2_dup_study"] += 1
                continue
            
            log(f"  {emtab_acc} -> {erp}: {info['title'][:50]}...", logf)
            limit = args.max_per_source if args.max_per_source > 0 else 0
            runs = enumerate_ena_runs(erp, limit=limit)
            
            if not runs:
                stats["tier2_no_runs"] += 1
                continue
            
            # Filter to RNA-seq
            rna_runs = [r for r in runs if r.get("library_strategy", "").upper() 
                       in ("RNA-SEQ", "OTHER", "SSRNA-SEQ", "")]
            
            is_cxg = "cellxgene" in info["sources"]
            is_scea = "scea" in info["sources"]
            
            rows = runs_to_catalog_rows(
                rna_runs, emtab_acc, "emtab", info["title"],
                is_cellxgene=is_cxg, is_scea=is_scea
            )
            new_rows = [r for r in rows if not is_duplicate(r, existing)]
            
            # Track new SRXs to avoid future dups within this run
            for r in new_rows:
                srx = r.get("srx_accession", "")
                if srx:
                    existing["srx"].add(srx)
                for srr in r.get("srr_accessions", "").split(";"):
                    if srr.strip():
                        existing["srr"].add(srr.strip())
            
            all_new_rows.extend(new_rows)
            stats["tier2_new_samples"] += len(new_rows)
            
            if len(new_rows) > 0:
                log(f"    {len(new_rows)} new samples from {len(rna_runs)} RNA runs "
                    f"({len(runs)} total)", logf)
            
            time.sleep(0.3)  # rate limit
        
        log(f"\nERP mapping: {erp_map_success} success, {erp_map_fail} failed", logf)
    
    # ===== Build controlled access entries =====
    log("\n" + "=" * 60, logf)
    log("CONTROLLED ACCESS DATASETS", logf)
    log("=" * 60, logf)
    
    if not args.skip_tier1:
        controlled = build_controlled_access_entries(cxg_data)
    else:
        cxg_data = discover_cellxgene_non_geo()
        controlled = build_controlled_access_entries(cxg_data)
    
    log(f"EGA studies: {len(controlled['ega_studies'])}", logf)
    log(f"dbGaP projects: {len(controlled['dbgap_projects'])}", logf)
    
    # ===== Write outputs =====
    log("\n" + "=" * 60, logf)
    log("WRITING OUTPUTS", logf)
    log("=" * 60, logf)
    
    if all_new_rows:
        new_df = pd.DataFrame(all_new_rows, columns=CATALOG_COLUMNS)
        
        # Type conversions to match catalog
        new_df["read_count"] = pd.to_numeric(new_df["read_count"], errors="coerce")
        new_df["base_count"] = pd.to_numeric(new_df["base_count"], errors="coerce")
        new_df["n_gsm_in_series"] = pd.to_numeric(new_df["n_gsm_in_series"], errors="coerce").fillna(0).astype(int)
        new_df["n_repos"] = pd.to_numeric(new_df["n_repos"], errors="coerce").fillna(0).astype(int)
        for col in ["in_cellxgene", "in_scea", "in_hca", "in_scp", "in_ucsc_cb", "in_panglaodb"]:
            new_df[col] = new_df[col].astype(bool)
        
        # Save standalone
        out_path = outdir / "non_geo_samples.parquet"
        new_df.to_parquet(out_path, index=False)
        log(f"Wrote {len(new_df)} new samples to {out_path}", logf)
        
        # Summary stats
        log(f"\nNew samples by source:", logf)
        for notes_val, count in new_df["notes"].value_counts().items():
            log(f"  {notes_val}: {count}", logf)
        
        log(f"\nNew samples by organism:", logf)
        for org, count in new_df["organism"].value_counts().head(15).items():
            log(f"  {org}: {count}", logf)
        
        log(f"\nNew samples by protocol:", logf)
        for proto, count in new_df["protocol_inferred"].value_counts().items():
            log(f"  {proto}: {count}", logf)
        
        log(f"\nNew samples by library_strategy:", logf)
        for strat, count in new_df["library_strategy"].value_counts().items():
            log(f"  {strat}: {count}", logf)
        
        # Merge into main catalog if not dry-run
        if not args.dry_run:
            log("\nMerging into main catalog...", logf)
            cat = pd.read_parquet(args.catalog)
            merged = pd.concat([cat, new_df], ignore_index=True)
            merged.to_parquet(args.catalog, index=False)
            log(f"Catalog updated: {len(cat)} -> {len(merged)} rows", logf)
        else:
            log("\nDRY RUN: not merging into catalog", logf)
    else:
        log("No new samples found", logf)
    
    # Write controlled access — merge into existing catalog if not dry-run
    if controlled["ega_studies"] or controlled["dbgap_projects"]:
        ctrl_path = outdir / "non_geo_controlled_access.json"
        with open(ctrl_path, "w") as f:
            json.dump(controlled, f, indent=2)
        n_total = len(controlled["ega_studies"]) + len(controlled["dbgap_projects"])
        log(f"\nWrote {n_total} controlled access entries to {ctrl_path}", logf)
        
        if not args.dry_run:
            # Merge into existing controlled access catalog
            existing_ca_path = Path("/mnt/home/debruinz/Singlet-AI/geo-reprocess/"
                                    "controlled_access_catalog/controlled_access_catalog.json")
            if existing_ca_path.exists():
                with open(existing_ca_path) as f:
                    existing_ca = json.load(f)
                
                # Add new EGA entries (avoid duplicates by ega_id)
                existing_ega_ids = {e["ega_id"] for e in existing_ca.get("ega_studies", [])}
                new_ega = [e for e in controlled["ega_studies"] 
                          if e["ega_id"] not in existing_ega_ids]
                existing_ca["ega_studies"].extend(new_ega)
                
                # Add new dbGaP entries (avoid duplicates by key)
                for k, v in controlled["dbgap_projects"].items():
                    if k not in existing_ca.get("dbgap_projects", {}):
                        existing_ca["dbgap_projects"][k] = v
                
                # Update timestamp
                existing_ca["metadata"]["generated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open(existing_ca_path, "w") as f:
                    json.dump(existing_ca, f, indent=2)
                log(f"Merged {len(new_ega)} EGA + {len(controlled['dbgap_projects'])} dbGaP "
                    f"into {existing_ca_path}", logf)
    
    # Final stats
    log("\n" + "=" * 60, logf)
    log("FINAL STATISTICS", logf)
    log("=" * 60, logf)
    for k, v in sorted(stats.items()):
        log(f"  {k}: {v}", logf)
    log(f"  TOTAL NEW SAMPLES: {len(all_new_rows)}", logf)
    
    logf.close()


if __name__ == "__main__":
    main()
