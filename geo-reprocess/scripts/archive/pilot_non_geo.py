#!/usr/bin/env python3
"""
Run pilot processing for diverse non-GEO samples.

Selects a diverse set of samples across protocols, species, and sources,
then processes each through the full pipeline (download → detect → quantify).
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scgeo.pipeline.api import process_sample

CATALOG = "/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet"
QUANT_DIR = Path("/mnt/projects/debruinz_project/cellarium/pipeline/quant")


def select_pilot_samples(cat):
    """Select diverse pilot samples across protocols, species and data sources."""
    non_geo = cat[cat["notes"].str.startswith("non_geo", na=False)]
    processable = non_geo[non_geo["protocol_inferred"].isin(["10xv2", "10xv3", "dropseq"])]
    processable = processable[processable["species_ref_genome"].fillna("").str.len() > 0]
    
    pilots = []
    
    # Strategy: 2 samples per (protocol x source) for human/mouse, 1 per other species
    for proto in ["10xv3", "10xv2", "dropseq"]:
        proto_df = processable[processable["protocol_inferred"] == proto]
        
        for source in ["non_geo_hca_bioproject", "non_geo_emtab", "non_geo_cellxgene_sra"]:
            source_df = proto_df[proto_df["notes"] == source]
            if len(source_df) == 0:
                continue
            
            # Human
            human = source_df[source_df["organism"] == "Homo sapiens"]
            if len(human) > 0:
                # Pick sample with smallest read_count (fastest to process)
                sample = human.nsmallest(1, "read_count", keep="first")
                pilots.append(sample.iloc[0])
            
            # Mouse
            mouse = source_df[source_df["organism"] == "Mus musculus"]
            if len(mouse) > 0:
                sample = mouse.nsmallest(1, "read_count", keep="first")
                pilots.append(sample.iloc[0])
        
        # Other species — one per species per protocol
        for org in processable["organism"].unique():
            if org in ("Homo sapiens", "Mus musculus"):
                continue
            org_df = proto_df[proto_df["organism"] == org]
            if len(org_df) > 0:
                sample = org_df.nsmallest(1, "read_count", keep="first")
                pilots.append(sample.iloc[0])
    
    # Deduplicate by gsm_id
    seen = set()
    unique_pilots = []
    for p in pilots:
        if p["gsm_id"] not in seen:
            seen.add(p["gsm_id"])
            unique_pilots.append(p)
    
    return unique_pilots


def run_pilot(sample, dry_run=False):
    """Process a single pilot sample."""
    gsm = sample["gsm_id"]
    gse = sample["gse_id"]
    organism = sample["organism"]
    r1_url = sample.get("ena_fastq_r1", "")
    r2_url = sample.get("ena_fastq_r2", "")
    srr = sample.get("srr_accessions", "")
    protocol = sample.get("protocol_inferred", "")
    
    print(f"\n{'='*70}")
    print(f"PILOT: {gse}/{gsm}")
    print(f"  Organism: {organism}")
    print(f"  Protocol: {protocol}")
    print(f"  Source: {sample.get('notes', '')}")
    print(f"  SRR: {srr}")
    print(f"  R1 URL: {str(r1_url)[:80]}...")
    print(f"  Read count: {sample.get('read_count', 0):,.0f}")
    
    if dry_run:
        print("  DRY RUN — skipping processing")
        return {"gsm_id": gsm, "status": "dry_run"}
    
    # Check if already processed
    manifest_path = QUANT_DIR / gse / gsm / "sample_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            m = json.load(f)
        print(f"  Already processed: status={m.get('status')}, cells={m.get('n_cells', 0)}")
        return {"gsm_id": gsm, "status": "already_processed", **m}
    
    t0 = time.time()
    try:
        result = process_sample(
            gsm_id=gsm,
            gse_id=gse,
            organism=organism,
            ena_r1_url=r1_url if r1_url else None,
            ena_r2_url=r2_url if r2_url else None,
            srr_accession=srr.split(";")[0].strip() if srr else None,
            protocol_hint=protocol,
        )
        elapsed = time.time() - t0
        
        # Read manifest
        if manifest_path.exists():
            with open(manifest_path) as f:
                m = json.load(f)
            print(f"  SUCCESS in {elapsed:.0f}s: cells={m.get('n_cells', 0)}, "
                  f"mapping={m.get('mapping_rate', 0):.1%}, "
                  f"qc={m.get('qc_status', '?')}")
            return {"gsm_id": gsm, "status": "success", "time_s": elapsed, **m}
        else:
            print(f"  COMPLETED in {elapsed:.0f}s but no manifest found")
            return {"gsm_id": gsm, "status": "no_manifest", "time_s": elapsed}
    
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED in {elapsed:.0f}s: {e}")
        return {"gsm_id": gsm, "status": "error", "error": str(e), "time_s": elapsed}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pilots", type=int, default=0,
                        help="Max pilot samples (0=all selected)")
    args = parser.parse_args()
    
    print("Loading catalog...")
    cat = pd.read_parquet(CATALOG)
    
    print("Selecting diverse pilot samples...")
    pilots = select_pilot_samples(cat)
    print(f"Selected {len(pilots)} pilot samples")
    
    if args.max_pilots > 0:
        pilots = pilots[:args.max_pilots]
        print(f"Limited to {len(pilots)} pilots")
    
    # Print selection summary
    print(f"\nPilot selection:")
    protocols = {}
    organisms = {}
    sources = {}
    for p in pilots:
        protocols[p["protocol_inferred"]] = protocols.get(p["protocol_inferred"], 0) + 1
        organisms[p["organism"]] = organisms.get(p["organism"], 0) + 1
        sources[p["notes"]] = sources.get(p["notes"], 0) + 1
    
    print(f"  Protocols: {protocols}")
    print(f"  Organisms: {organisms}")
    print(f"  Sources: {sources}")
    
    # Process each
    results = []
    for p in pilots:
        result = run_pilot(p, dry_run=args.dry_run)
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("PILOT PROCESSING SUMMARY")
    print(f"{'='*70}")
    status_counts = {}
    for r in results:
        s = r.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
    
    for s, n in sorted(status_counts.items()):
        print(f"  {s}: {n}")
    
    # Save results
    outpath = Path("/mnt/projects/debruinz_project/cellarium/catalog/pilot_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
