#!/usr/bin/env python3
"""Find optimal small pilot samples with valid URLs."""
import pandas as pd

cat = pd.read_parquet("/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet")
non_geo = cat[cat["notes"].str.startswith("non_geo", na=False)]
processable = non_geo[non_geo["protocol_inferred"].isin(["10xv2", "10xv3", "dropseq"])]
processable = processable[processable["species_ref_genome"].fillna("").str.len() > 0]

pilots = []

for proto in ["10xv3", "10xv2", "dropseq"]:
    pf = processable[processable["protocol_inferred"] == proto]
    for org in ["Homo sapiens", "Mus musculus", "Rattus norvegicus", 
                "Drosophila melanogaster", "Pan troglodytes", "Danio rerio"]:
        of = pf[pf["organism"] == org]
        if len(of) == 0:
            continue
        # Prefer single-SRR with both R1+R2 and small read count
        single_srr = of[~of["srr_accessions"].str.contains(";", na=False)]
        with_urls = single_srr[
            (single_srr["ena_fastq_r1"].fillna("").str.len() > 10) & 
            (single_srr["ena_fastq_r2"].fillna("").str.len() > 10)
        ]
        if len(with_urls) > 0:
            best = with_urls.nsmallest(1, "read_count").iloc[0]
        elif len(single_srr) > 0:
            best = single_srr.nsmallest(1, "read_count").iloc[0]
        elif len(of) > 0:
            best = of.nsmallest(1, "read_count").iloc[0]
        else:
            continue
        
        r1_ok = len(str(best["ena_fastq_r1"])) > 10
        r2_ok = len(str(best["ena_fastq_r2"])) > 10
        n_srr = len(str(best["srr_accessions"]).split(";"))
        
        print(f"{proto:10s} {org:25s} {best['gse_id']:20s} {best['gsm_id']:15s} "
              f"reads={best['read_count']:>12,.0f} src={best['notes']}  "
              f"r1={r1_ok} r2={r2_ok} n_srr={n_srr}")
        
        pilots.append({
            "gsm_id": best["gsm_id"],
            "gse_id": best["gse_id"],
            "organism": org,
            "protocol": proto,
            "read_count": best["read_count"],
            "r1_url": best.get("ena_fastq_r1", ""),
            "r2_url": best.get("ena_fastq_r2", ""),
            "srr": best.get("srr_accessions", ""),
        })

print(f"\nTotal pilot candidates: {len(pilots)}")

# Recommend the top 8 smallest
print("\n--- RECOMMENDED PILOTS (8 smallest) ---")
pilots.sort(key=lambda x: x["read_count"])
for p in pilots[:8]:
    print(f"  {p['protocol']:8s} {p['organism']:20s} {p['gse_id']}/{p['gsm_id']}  reads={p['read_count']:,.0f}")
