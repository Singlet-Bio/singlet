#!/usr/bin/env python3
"""
Run specific pilot samples through the processing pipeline.
Selected for diversity: small read counts, different protocols/species/sources.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scgeo.pipeline.api import process_sample

QUANT = Path("/mnt/projects/debruinz_project/cellarium/pipeline/quant")

# Selected pilot samples — smallest per protocol/species/source with valid URLs
PILOTS = [
    # 10xv2, Human, HCA — TINY: 131K reads
    {"gsm_id": "ERX4126790", "gse_id": "HCA_ERP114453", "organism": "Homo sapiens",
     "r1": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR416/005/ERR4161455_1.fastq.gz",
     "r2": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR416/005/ERR4161455_2.fastq.gz",
     "srr": "ERR4161455", "protocol": "10xv2", "reads": 131307},
    # 10xv2, Mouse, EMTAB — 13.6M reads
    {"gsm_id": "ERX7028527", "gse_id": "E-MTAB-8145", "organism": "Mus musculus",
     "r1": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR745/005/ERR7457785/ERR7457785_1.fastq.gz",
     "r2": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR745/005/ERR7457785/ERR7457785_2.fastq.gz",
     "srr": "ERR7457785", "protocol": "10xv2", "reads": 13612012},
    # 10xv3, Rat, EMTAB — 38M reads
    {"gsm_id": "ERX4970963", "gse_id": "E-MTAB-10030", "organism": "Rattus norvegicus",
     "r1": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR516/001/ERR5166021_1.fastq.gz",
     "r2": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR516/001/ERR5166021_2.fastq.gz",
     "srr": "ERR5166021", "protocol": "10xv3", "reads": 37975287},
    # dropseq, Drosophila, EMTAB — 25.6M reads
    {"gsm_id": "ERX4409346", "gse_id": "E-MTAB-9444", "organism": "Drosophila melanogaster",
     "r1": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR449/004/ERR4494004_1.fastq.gz",
     "r2": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR449/004/ERR4494004_2.fastq.gz",
     "srr": "ERR4494004", "protocol": "dropseq", "reads": 25642534},
    # 10xv3, Human, HCA — 163M reads
    {"gsm_id": "ERX10670386", "gse_id": "HCA_ERP146619", "organism": "Homo sapiens",
     "r1": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR112/088/ERR11262888/ERR11262888_1.fastq.gz",
     "r2": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR112/088/ERR11262888/ERR11262888_2.fastq.gz",
     "srr": "ERR11262888", "protocol": "10xv3", "reads": 163583708},
    # 10xv2, Chimp, HCA — 179M reads
    {"gsm_id": "ERX3494777", "gse_id": "HCA_ERP116749", "organism": "Pan troglodytes",
     "r1": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR347/006/ERR3473126_1.fastq.gz",
     "r2": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR347/006/ERR3473126_2.fastq.gz",
     "srr": "ERR3473126", "protocol": "10xv2", "reads": 179579272},
]


def main():
    results = []
    
    for p in PILOTS:
        gsm = p["gsm_id"]
        gse = p["gse_id"]
        
        print(f"\n{'='*70}")
        print(f"PILOT: {gse}/{gsm} — {p['protocol']} {p['organism']} ({p['reads']:,} reads)")
        print(f"{'='*70}")
        
        # Check if already processed
        manifest = QUANT / gse / gsm / "sample_manifest.json"
        if manifest.exists():
            with open(manifest) as f:
                m = json.load(f)
            status = m.get("status", "?")
            cells = m.get("n_cells", 0)
            print(f"  Already processed: status={status}, n_cells={cells}")
            results.append({"gsm_id": gsm, "gse_id": gse, "status": f"already_{status}", 
                          "n_cells": cells, "organism": p["organism"], "protocol": p["protocol"]})
            continue
        
        t0 = time.time()
        try:
            result = process_sample(
                gsm_id=gsm,
                gse_id=gse,
                organism=p["organism"],
                ena_r1_url=p["r1"],
                ena_r2_url=p["r2"],
                srr_accession=p["srr"],
                protocol_hint=p["protocol"],
            )
            elapsed = time.time() - t0
            
            # Read manifest if created
            if manifest.exists():
                with open(manifest) as f:
                    m = json.load(f)
                n_cells = m.get("n_cells", 0)
                mapping_rate = m.get("mapping_rate", 0)
                qc_status = m.get("qc_status", "?")
                status = m.get("status", "?")
                print(f"  {status.upper()} in {elapsed:.0f}s: "
                      f"n_cells={n_cells:,}, mapping={mapping_rate:.1%}, qc={qc_status}")
                results.append({
                    "gsm_id": gsm, "gse_id": gse, "status": status,
                    "n_cells": n_cells, "mapping_rate": mapping_rate,
                    "qc_status": qc_status, "time_s": elapsed,
                    "organism": p["organism"], "protocol": p["protocol"],
                })
            else:
                print(f"  Completed in {elapsed:.0f}s but no manifest")
                results.append({"gsm_id": gsm, "gse_id": gse, "status": "no_manifest",
                              "time_s": elapsed, "organism": p["organism"], "protocol": p["protocol"]})
                
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED in {elapsed:.0f}s: {e}")
            results.append({"gsm_id": gsm, "gse_id": gse, "status": "error",
                          "error": str(e), "time_s": elapsed,
                          "organism": p["organism"], "protocol": p["protocol"]})
    
    # Summary
    print(f"\n{'='*70}")
    print("PILOT RESULTS SUMMARY")
    print(f"{'='*70}")
    for r in results:
        cells = r.get("n_cells", "?")
        mr = r.get("mapping_rate", "?")
        t = r.get("time_s", "?")
        if isinstance(mr, float):
            mr = f"{mr:.1%}"
        if isinstance(t, float):
            t = f"{t:.0f}s"
        print(f"  {r['protocol']:8s} {r['organism']:25s} {r['gse_id']}/{r['gsm_id']}  "
              f"status={r['status']:12s} cells={cells}  mapping={mr}  time={t}")
    
    # Save
    with open("/mnt/projects/debruinz_project/cellarium/catalog/pilot_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved")


if __name__ == "__main__":
    main()
