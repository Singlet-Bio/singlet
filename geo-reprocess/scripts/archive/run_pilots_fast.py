#!/usr/bin/env python3
"""
Fast pilot processing using samples with confirmed ENA FASTQ availability.
These samples have verified fastq_ftp entries on ENA — no fasterq-dump needed.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scgeo.pipeline.api import process_sample

QUANT = Path("/mnt/projects/debruinz_project/cellarium/pipeline/quant")

# Samples with CONFIRMED ENA FASTQ availability (verified via filereport API)
PILOTS = [
    # 10xv2, Mouse, EMTAB — 13.6M reads — ERR7457785 has fastq_ftp
    {"gsm_id": "ERX7028527", "gse_id": "E-MTAB-8145", "organism": "Mus musculus",
     "r1": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR745/005/ERR7457785/ERR7457785_1.fastq.gz",
     "r2": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR745/005/ERR7457785/ERR7457785_2.fastq.gz",
     "srr": "ERR7457785", "protocol": "10xv2", "reads": 13612012},
    # 10xv3, Human, HCA — 163M reads — ERR11262888 has fastq_ftp
    {"gsm_id": "ERX10670386", "gse_id": "HCA_ERP146619", "organism": "Homo sapiens",
     "r1": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR112/088/ERR11262888/ERR11262888_1.fastq.gz",
     "r2": "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR112/088/ERR11262888/ERR11262888_2.fastq.gz",
     "srr": "ERR11262888", "protocol": "10xv3", "reads": 163583708},
]


def main():
    results = []
    for p in PILOTS:
        gsm, gse = p["gsm_id"], p["gse_id"]
        print(f"\n{'='*70}")
        print(f"PILOT: {gse}/{gsm} — {p['protocol']} {p['organism']} ({p['reads']:,} reads)")
        print(f"{'='*70}")

        manifest = QUANT / gse / gsm / "sample_manifest.json"
        if manifest.exists():
            with open(manifest) as f:
                m = json.load(f)
            print(f"  Already processed: status={m.get('status')}, n_cells={m.get('n_cells', 0)}")
            results.append({"gsm_id": gsm, "gse_id": gse, "status": f"already_{m.get('status')}",
                          "n_cells": m.get("n_cells", 0), "organism": p["organism"], "protocol": p["protocol"]})
            continue

        t0 = time.time()
        try:
            result = process_sample(
                gsm_id=gsm, gse_id=gse, organism=p["organism"],
                ena_r1_url=p["r1"], ena_r2_url=p["r2"],
                srr_accession=p["srr"], protocol_hint=p["protocol"],
            )
            elapsed = time.time() - t0
            if manifest.exists():
                with open(manifest) as f:
                    m = json.load(f)
                n_cells = m.get("n_cells", 0)
                mapping_rate = m.get("mapping_rate", 0)
                qc_status = m.get("qc_status", "?")
                status = m.get("status", "?")
                print(f"  {status.upper()} in {elapsed:.0f}s: n_cells={n_cells:,}, "
                      f"mapping={mapping_rate:.1%}, qc={qc_status}")
                results.append({"gsm_id": gsm, "gse_id": gse, "status": status,
                    "n_cells": n_cells, "mapping_rate": mapping_rate,
                    "qc_status": qc_status, "time_s": elapsed,
                    "organism": p["organism"], "protocol": p["protocol"]})
            else:
                print(f"  Completed in {elapsed:.0f}s but no manifest created")
                results.append({"gsm_id": gsm, "gse_id": gse, "status": "no_manifest",
                    "time_s": elapsed, "organism": p["organism"], "protocol": p["protocol"]})
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED in {elapsed:.0f}s: {e}")
            results.append({"gsm_id": gsm, "gse_id": gse, "status": "error",
                "error": str(e), "time_s": elapsed,
                "organism": p["organism"], "protocol": p["protocol"]})

    print(f"\n{'='*70}")
    print("FAST PILOT RESULTS SUMMARY")
    print(f"{'='*70}")
    for r in results:
        cells = r.get("n_cells", "?")
        mr = r.get("mapping_rate", "?")
        t = r.get("time_s", "?")
        if isinstance(mr, float): mr = f"{mr:.1%}"
        if isinstance(t, float): t = f"{t:.0f}s"
        print(f"  {r['protocol']:8s} {r['organism']:25s} {r['gse_id']}/{r['gsm_id']}  "
              f"status={r['status']:12s} cells={cells}  mapping={mr}  time={t}")

    with open("/mnt/projects/debruinz_project/cellarium/catalog/pilot_fast_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to pilot_fast_results.json")

if __name__ == "__main__":
    main()
