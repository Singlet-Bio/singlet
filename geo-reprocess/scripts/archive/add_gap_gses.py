#!/usr/bin/env python3
"""Add verified gap GSEs to the catalog and prepare for pipeline processing.

Fetches GSM-level metadata from GEO SOFT + SRA metadata from ENA for
the 49 missing GSEs identified by gap verification, then:
1. Adds rows to geo_single_cell_catalog.parquet
2. Writes batch CSV for pipeline scheduling
"""

import csv
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter

import pandas as pd
import requests

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

CATALOG_DIR = Path("/mnt/projects/debruinz_project/cellarium/catalog")
PIPELINE_DIR = Path("/mnt/projects/debruinz_project/cellarium/pipeline")
CATALOG_PATH = CATALOG_DIR / "geo_single_cell_catalog.parquet"
BATCH_OUTPUT = PIPELINE_DIR / "batches_gap_fill"

# ENA API
ENA_URL = "https://www.ebi.ac.uk/ena/portal/api/filereport"
ENA_FIELDS = ",".join([
    "run_accession", "experiment_accession", "study_accession",
    "secondary_study_accession", "sample_accession",
    "library_name", "library_strategy", "library_source",
    "library_selection", "library_layout", "instrument_platform",
    "instrument_model", "read_count", "base_count",
    "fastq_ftp", "fastq_md5", "tax_id", "scientific_name",
])


# Protocol inference keywords
PROTOCOL_PATTERNS = {
    "10xv3": [r"10x chromium.*v3", r"chromium.*3'.*v3", r"10x.*3'.*v3", r"10xv3"],
    "10xv2": [r"10x chromium.*v2", r"chromium.*3'.*v2", r"10x.*3'.*v2", r"10xv2"],
    "10x_multiome": [r"10x.*multiome", r"chromium.*multiome", r"multiome.*atac\+gex"],
    "10xv3_5prime": [r"10x.*5'.*v3", r"5'.*v3.*10x"],
    "dropseq": [r"drop-seq", r"dropseq"],
    "smartseq2": [r"smart-seq2", r"smartseq2", r"smart seq 2"],
    "celseq2": [r"cel-seq2", r"celseq2"],
    "indrop": [r"indrop"],
    "parse": [r"parse.*biosciences", r"split-pool"],
    "bd_rhapsody": [r"bd.*rhapsody"],
    "scirna": [r"sci-rna"],
}


def infer_protocol(text):
    """Infer SC protocol from combined text."""
    text_lower = text.lower()
    for proto, patterns in PROTOCOL_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return proto, "medium"
    
    # Check for general 10x mentions without version
    if re.search(r"10x (chromium|genomics)", text_lower):
        return "10x_suspect", "low"
    if any(x in text_lower for x in ["single-cell rna", "scrna", "snrna", "single cell rna", "single-nucleus rna"]):
        return "unknown_sc", "low"
    return "unknown", "low"


def _get_with_retry(url, params, timeout=60, retries=3):
    """GET with retry on timeout."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            return r
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt+1}/{retries} after timeout...")
                time.sleep(2 * (attempt + 1))
            else:
                raise
    return None


def fetch_gse_soft(gse_id):
    """Fetch GSE SOFT file and parse series + sample metadata."""
    url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    
    # Full SOFT for the series (includes GSM stubs)
    r = _get_with_retry(url, params={
        "acc": gse_id, "targ": "gsm", "form": "text", "view": "brief"
    }, timeout=90)
    
    series_info = {"title": "", "summary": "", "overall_design": "", "type": "",
                   "pubmed_ids": "", "contact_name": "", "contact_institute": "",
                   "submission_date": "", "last_update_date": ""}
    
    gsm_data = {}
    current_gsm = None
    
    if r is None or r.status_code != 200:
        return series_info, gsm_data
    
    # Also fetch series-level metadata
    r_series = _get_with_retry(url, params={
        "acc": gse_id, "targ": "self", "form": "text", "view": "brief"
    }, timeout=60)
    
    if r_series and r_series.status_code == 200:
        for line in r_series.text.split("\n"):
            line = line.strip()
            if line.startswith("!Series_title"):
                series_info["title"] = line.split("=", 1)[-1].strip()
            elif line.startswith("!Series_summary"):
                series_info["summary"] += line.split("=", 1)[-1].strip() + " "
            elif line.startswith("!Series_overall_design"):
                series_info["overall_design"] += line.split("=", 1)[-1].strip() + " "
            elif line.startswith("!Series_pubmed_id"):
                pid = line.split("=", 1)[-1].strip()
                series_info["pubmed_ids"] = (series_info["pubmed_ids"] + ";" + pid).strip(";")
            elif line.startswith("!Series_contact_name"):
                series_info["contact_name"] = line.split("=", 1)[-1].strip()
            elif line.startswith("!Series_contact_institute"):
                series_info["contact_institute"] = line.split("=", 1)[-1].strip()
            elif line.startswith("!Series_submission_date"):
                series_info["submission_date"] = line.split("=", 1)[-1].strip()
            elif line.startswith("!Series_last_update_date"):
                series_info["last_update_date"] = line.split("=", 1)[-1].strip()
    
    time.sleep(0.3)
    
    # Parse GSM data from the gsm target
    for line in r.text.split("\n"):
        line = line.strip()
        if line.startswith("^SAMPLE"):
            current_gsm = line.split("=", 1)[-1].strip()
            gsm_data[current_gsm] = {
                "organism": "", "source": "", "characteristics": "",
                "supplementary_files": "",
            }
        elif current_gsm:
            if line.startswith("!Sample_organism_ch1"):
                org = line.split("=", 1)[-1].strip()
                if gsm_data[current_gsm]["organism"]:
                    gsm_data[current_gsm]["organism"] += "; " + org
                else:
                    gsm_data[current_gsm]["organism"] = org
            elif line.startswith("!Sample_source_name_ch1"):
                gsm_data[current_gsm]["source"] = line.split("=", 1)[-1].strip()
            elif line.startswith("!Sample_characteristics_ch1"):
                val = line.split("=", 1)[-1].strip()
                if gsm_data[current_gsm]["characteristics"]:
                    gsm_data[current_gsm]["characteristics"] += "; " + val
                else:
                    gsm_data[current_gsm]["characteristics"] = val
            elif line.startswith("!Sample_supplementary_file"):
                val = line.split("=", 1)[-1].strip()
                if gsm_data[current_gsm]["supplementary_files"]:
                    gsm_data[current_gsm]["supplementary_files"] += "; " + val
                else:
                    gsm_data[current_gsm]["supplementary_files"] = val
            elif line.startswith("!Sample_relation"):
                val = line.split("=", 1)[-1].strip()
                # SRA: https://www.ncbi.nlm.nih.gov/sra?term=SRX...
                m = re.search(r'term=(SRX\d+)', val)
                if m:
                    gsm_data[current_gsm]["srx"] = m.group(1)
                # BioProject
                m2 = re.search(r'BioProject:?\s*(PRJNA\d+)', val)
                if m2:
                    gsm_data[current_gsm]["bioproject"] = m2.group(1)
    
    return series_info, gsm_data


def _fetch_single_ena(srx):
    """Fetch ENA data for a single SRX."""
    try:
        r = requests.get(ENA_URL, params={
            "accession": srx,
            "result": "read_run",
            "fields": ENA_FIELDS,
            "format": "tsv",
            "limit": 1000,
        }, timeout=30)
        if r.status_code == 200 and r.text.strip() and "\t" in r.text:
            rows = list(csv.DictReader(io.StringIO(r.text), delimiter='\t'))
            return srx, rows
    except Exception:
        pass
    return srx, []


def fetch_ena_sra(srx_ids):
    """Fetch SRA metadata from ENA for a list of SRX IDs."""
    results = {}
    
    # Use ThreadPoolExecutor for parallel fetching (10 concurrent)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_single_ena, srx): srx for srx in srx_ids}
        done = 0
        for future in as_completed(futures):
            srx_key = futures[future]
            try:
                srx_result, rows = future.result()
                for row in rows:
                    exp_acc = row.get("experiment_accession", srx_key)
                    if exp_acc not in results:
                        results[exp_acc] = {
                            "srr_accessions": [],
                            "library_strategy": row.get("library_strategy", ""),
                            "library_source": row.get("library_source", ""),
                            "library_selection": row.get("library_selection", ""),
                            "library_layout": row.get("library_layout", ""),
                            "instrument_platform": row.get("instrument_platform", ""),
                            "instrument_model": row.get("instrument_model", ""),
                            "read_count": 0,
                            "base_count": 0,
                            "fastq_ftp": row.get("fastq_ftp", ""),
                            "tax_id": row.get("tax_id", ""),
                            "scientific_name": row.get("scientific_name", ""),
                            "study_accession": row.get("study_accession", ""),
                            "secondary_study_accession": row.get("secondary_study_accession", ""),
                        }
                    results[exp_acc]["srr_accessions"].append(row.get("run_accession", ""))
                    try:
                        results[exp_acc]["read_count"] += int(row.get("read_count", 0) or 0)
                        results[exp_acc]["base_count"] += int(row.get("base_count", 0) or 0)
                    except (ValueError, TypeError):
                        pass
            except Exception as e:
                print(f"  ENA error for {srx_key}: {e}")
            
            done += 1
            if done % 100 == 0:
                print(f"    ENA: {done}/{len(srx_ids)} fetched, {len(results)} found")
    
    # Collapse SRR lists
    for srx in results:
        results[srx]["srr_accessions"] = ";".join(sorted(results[srx]["srr_accessions"]))
    
    return results


def organism_to_ref(organism):
    """Map organism to reference genome/annotation."""
    org_lower = (organism or "").lower()
    if "homo sapiens" in org_lower:
        return "GRCh38", "GENCODE v45"
    elif "mus musculus" in org_lower:
        return "GRCm39", "GENCODE M34"
    elif "macaca" in org_lower:
        return "Mmul_10", "Ensembl 109"
    elif "danio rerio" in org_lower or "zebrafish" in org_lower:
        return "GRCz11", "Ensembl 109"
    elif "drosophila" in org_lower:
        return "BDGP6", "Ensembl 109"
    elif "caenorhabditis" in org_lower or "c. elegans" in org_lower:
        return "WBcel235", "Ensembl 109"
    elif "rattus" in org_lower:
        return "mRatBN7.2", "Ensembl 109"
    else:
        return "", ""


def parse_fastq_urls(ftp_str):
    """Parse ENA fastq_ftp field into R1/R2 URLs."""
    if not ftp_str:
        return "", ""
    parts = ftp_str.split(";")
    r1 = ""
    r2 = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        url = "ftp://" + p if not p.startswith("ftp://") else p
        if "_1.fastq" in p:
            r1 = url
        elif "_2.fastq" in p:
            r2 = url
    return r1, r2


def main():
    print("=" * 80)
    print("ADD GAP GSEs TO CATALOG")
    print("=" * 80)
    
    # Load the list of GSEs to add
    gap_file = CATALOG_DIR / "gap_gses_to_add.json"
    with open(str(gap_file)) as f:
        gap_data = json.load(f)
    
    gses_to_add = gap_data["all_gses_to_add"]
    print(f"\nGSEs to add: {len(gses_to_add)}")
    
    # Load existing catalog
    print("Loading existing catalog...")
    cat = pd.read_parquet(str(CATALOG_PATH))
    existing_gses = set(cat["gse_id"].unique())
    existing_gsms = set(cat["gsm_id"].unique())
    print(f"  Existing: {len(cat):,} rows, {len(existing_gses):,} GSEs")
    
    # Filter out any GSEs already in catalog
    gses_to_add = [g for g in gses_to_add if g not in existing_gses]
    print(f"  After filtering existing: {len(gses_to_add)} GSEs to add")
    
    if not gses_to_add:
        print("Nothing to add!")
        return
    
    # Fetch metadata for each GSE
    new_rows = []
    no_srx = []
    skipped_plate = 0
    
    for idx, gse in enumerate(gses_to_add):
        print(f"\n[{idx+1}/{len(gses_to_add)}] Fetching {gse}...")
        try:
            series_info, gsm_data = fetch_gse_soft(gse)
        except Exception as e:
            print(f"  ERROR fetching {gse}: {e}")
            continue
        time.sleep(0.35)
        
        if not gsm_data:
            print(f"  WARNING: No GSM data for {gse}")
            continue
        
        print(f"  Found {len(gsm_data)} GSMs, title: {series_info['title'][:60]}")
        
        # Infer protocol from series metadata
        combined_text = " ".join([
            series_info["title"], series_info["summary"], 
            series_info["overall_design"]
        ])
        protocol, confidence = infer_protocol(combined_text)
        
        # Get SRX IDs
        srx_ids = [gsm_data[g].get("srx", "") for g in gsm_data if gsm_data[g].get("srx")]
        print(f"  SRX IDs: {len(srx_ids)}, protocol: {protocol} ({confidence})")
        
        # Fetch ENA SRA metadata
        sra_data = {}
        if srx_ids:
            sra_data = fetch_ena_sra(srx_ids)
            print(f"  ENA: got {len(sra_data)} SRX records")
        
        n_gsm_in_series = len(gsm_data)
        
        for gsm_id, gsm_info in gsm_data.items():
            if gsm_id in existing_gsms:
                continue
            
            organism = gsm_info.get("organism", "")
            ref_genome, ref_annot = organism_to_ref(organism)
            
            srx = gsm_info.get("srx", "")
            sra = sra_data.get(srx, {})
            
            r1, r2 = parse_fastq_urls(sra.get("fastq_ftp", ""))
            
            # Build S3 URL from SRR
            srr_list = sra.get("srr_accessions", "")
            first_srr = srr_list.split(";")[0] if srr_list else ""
            s3_url = f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{first_srr}/{first_srr}" if first_srr else ""
            
            # Refine protocol per-sample from library characteristics
            sample_protocol = protocol
            sample_confidence = confidence
            chars = gsm_info.get("characteristics", "").lower()
            if chars:
                sp, sc = infer_protocol(chars + " " + gsm_info.get("source", ""))
                if sc == "medium":
                    sample_protocol = sp
                    sample_confidence = sc
            
            # Check if this GSM is ATAC-only (skip it)
            if sra.get("library_strategy", "").upper() in ("ATAC-SEQ", "CHIP-SEQ", "DNASE-HYPERSENSITIVITY"):
                continue
            
            row = {
                "gse_id": gse,
                "gsm_id": gsm_id,
                "series_title": series_info["title"],
                "organism": organism,
                "taxon_id": sra.get("tax_id", ""),
                "protocol_inferred": sample_protocol,
                "protocol_confidence": sample_confidence,
                "species_ref_genome": ref_genome,
                "species_annotation": ref_annot,
                "srx_accession": srx,
                "srr_accessions": srr_list,
                "library_strategy": sra.get("library_strategy", ""),
                "library_source": sra.get("library_source", ""),
                "library_selection": sra.get("library_selection", ""),
                "library_layout": sra.get("library_layout", ""),
                "instrument_platform": sra.get("instrument_platform", ""),
                "instrument_model": sra.get("instrument_model", ""),
                "sra_study": sra.get("study_accession", ""),
                "bioproject": gsm_info.get("bioproject", sra.get("secondary_study_accession", "")),
                "read_count": float(sra.get("read_count", 0)),
                "base_count": float(sra.get("base_count", 0)),
                "ena_fastq_r1": r1,
                "ena_fastq_r2": r2,
                "ncbi_sra_s3": s3_url,
                "submission_date": series_info.get("submission_date", ""),
                "last_update_date": series_info.get("last_update_date", ""),
                "pubmed_ids": series_info.get("pubmed_ids", ""),
                "contact_name": series_info.get("contact_name", ""),
                "contact_institute": series_info.get("contact_institute", ""),
                "summary": series_info.get("summary", "")[:500],
                "overall_design": series_info.get("overall_design", "")[:500],
                "sample_organism": organism,
                "sample_source": gsm_info.get("source", ""),
                "sample_characteristics": gsm_info.get("characteristics", "")[:500],
                "supplementary_files": gsm_info.get("supplementary_files", "")[:500],
                "n_gsm_in_series": n_gsm_in_series,
                "processing_status": "pending",
                "notes": "gap_fill_from_repo_crossref",
                "in_cellxgene": False,
                "in_scea": False,
                "in_hca": False,
                "in_scp": False,
                "in_ucsc_cb": False,
                "in_panglaodb": False,
                "n_repos": 0,
            }
            
            if not srx:
                no_srx.append(gsm_id)
            
            new_rows.append(row)
    
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"  New rows to add: {len(new_rows)}")
    print(f"  GSMs without SRX: {len(no_srx)}")
    
    if not new_rows:
        print("No new rows to add!")
        return
    
    new_df = pd.DataFrame(new_rows)
    
    # Summary by GSE
    gse_counts = new_df.groupby("gse_id").size()
    print(f"\n  GSEs added: {len(gse_counts)}")
    for gse, n in gse_counts.sort_values(ascending=False).head(20).items():
        proto = new_df[new_df["gse_id"] == gse]["protocol_inferred"].iloc[0]
        org = new_df[new_df["gse_id"] == gse]["organism"].iloc[0][:30]
        print(f"    {gse}: {n:>5} GSMs, protocol={proto}, organism={org}")
    if len(gse_counts) > 20:
        print(f"    ... and {len(gse_counts) - 20} more")
    
    # Protocol distribution
    print(f"\n  Protocol distribution:")
    for proto, n in new_df["protocol_inferred"].value_counts().items():
        print(f"    {proto}: {n}")
    
    # Organism distribution  
    print(f"\n  Organism distribution:")
    for org, n in new_df["organism"].value_counts().head(10).items():
        print(f"    {org}: {n}")
    
    # Library layout
    print(f"\n  Library layout:")
    for layout, n in new_df["library_layout"].value_counts().items():
        print(f"    {layout}: {n}")
    
    # Has FASTQ URLs
    has_r1 = (new_df["ena_fastq_r1"] != "").sum()
    has_r2 = (new_df["ena_fastq_r2"] != "").sum()
    has_srr = (new_df["srr_accessions"] != "").sum()
    print(f"\n  Has ENA R1: {has_r1}, R2: {has_r2}, SRR: {has_srr}")
    
    # Update repo columns from crossref
    xref_path = CATALOG_DIR / "repository_gse_crossref.parquet"
    if xref_path.exists():
        xref = pd.read_parquet(str(xref_path))
        for repo_col in ["in_cellxgene", "in_scea", "in_hca", "in_scp", "in_ucsc_cb", "in_panglaodb", "n_repos"]:
            if repo_col in xref.columns:
                gse_map = dict(zip(xref["gse_id"], xref[repo_col]))
                new_df[repo_col] = new_df["gse_id"].map(gse_map).fillna(
                    False if repo_col != "n_repos" else 0
                )
                if repo_col == "n_repos":
                    new_df[repo_col] = new_df[repo_col].astype(int)
    
    # Append to catalog
    print(f"\nAppending {len(new_df)} rows to catalog...")
    combined = pd.concat([cat, new_df], ignore_index=True)
    
    # Verify no duplicate GSMs
    dup_gsms = combined["gsm_id"].duplicated().sum()
    if dup_gsms > 0:
        print(f"  WARNING: {dup_gsms} duplicate GSMs detected, removing...")
        combined = combined.drop_duplicates(subset="gsm_id", keep="first")
    
    # Save backup
    backup_path = CATALOG_PATH.with_suffix(".backup_pre_gapfill.parquet")
    cat.to_parquet(str(backup_path), index=False)
    print(f"  Backup saved to {backup_path}")
    
    # Save updated catalog
    combined.to_parquet(str(CATALOG_PATH), index=False)
    print(f"  Updated catalog: {len(combined):,} rows, {combined['gse_id'].nunique():,} GSEs")
    
    # Write batch CSVs for pipeline
    BATCH_OUTPUT.mkdir(parents=True, exist_ok=True)
    batch_size = 50
    processable = new_df[
        (new_df["srr_accessions"] != "") & 
        (new_df["library_layout"] == "PAIRED") &
        (new_df["species_ref_genome"] != "")
    ].copy()
    
    print(f"\n  Processable (PAIRED + has SRR + known genome): {len(processable)}")
    
    # Sort by protocol tier
    TIER_MAP = {
        "10xv3": 1, "10xv2": 1, "10xv3_5prime": 1, "10xv4": 1, "10x_multiome": 1,
        "dropseq": 2, "seqwell": 2, "dnbelab": 2,
        "10x_suspect": 3, "unknown_sc": 4, "unknown": 5,
    }
    processable["_tier"] = processable["protocol_inferred"].map(TIER_MAP).fillna(6).astype(int)
    processable = processable.sort_values(["_tier", "gse_id", "gsm_id"])
    
    n_batches = 0
    for i in range(0, len(processable), batch_size):
        batch = processable.iloc[i:i+batch_size]
        batch_file = BATCH_OUTPUT / f"gap_batch_{n_batches:03d}.csv"
        
        # Write in same format as existing batches
        batch_out = batch[["gse_id", "gsm_id", "srx_accession", "srr_accessions",
                           "protocol_inferred", "protocol_confidence", "organism",
                           "species_ref_genome", "species_annotation",
                           "ena_fastq_r1", "ena_fastq_r2", "ncbi_sra_s3",
                           "library_layout", "read_count", "base_count"]].copy()
        batch_out.to_csv(str(batch_file), index=False)
        n_batches += 1
    
    print(f"  Wrote {n_batches} batch files to {BATCH_OUTPUT}")
    
    # Also save gap-fill summary
    summary = {
        "total_gses_added": int(len(gse_counts)),
        "total_gsms_added": len(new_rows),
        "processable_gsms": len(processable),
        "gsms_without_srx": len(no_srx),
        "protocol_distribution": dict(new_df["protocol_inferred"].value_counts()),
        "organism_distribution": dict(new_df["organism"].value_counts()),
        "n_batches_written": n_batches,
    }
    summary_path = CATALOG_DIR / "gap_fill_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
