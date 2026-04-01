# Preprocessing Pipeline

The preprocessing module provides tools for processing raw FASTQ data into
compressed .spz files. These are the same tools used to build the SingletDB
atlas.

## Overview

The pipeline processes samples through 5 stages:

1. **Download** — FASTQ files from ENA (parallel byte-range) or SRA (fasterq-dump)
2. **Detect** — Identify the sequencing protocol (10x v2/v3, Drop-seq, Smart-seq2)
3. **Quantify** — Count matrix generation with simpleaf + piscem
4. **QC** — Quality control metrics and thresholds
5. **Export** — Compress to .spz format

## Supported Species

```python
from singlet.preprocessing import list_supported_species

for sp in list_supported_species():
    print(f"{sp['name']:25s} {sp['assembly']:20s} (txid {sp['taxon_id']})")
```

24 species with pre-built indices, including human, mouse, zebrafish, rat,
drosophila, C. elegans, and arabidopsis.

## Example: Process a Sample

```python
from singlet.preprocessing import (
    download_fastq, detect_protocol, quantify, run_qc, export_to_spz
)

# 1. Download FASTQs
dl = download_fastq(
    "GSM1234567",
    ena_r1_url="https://ftp.sra.ebi.ac.uk/..._1.fastq.gz",
    ena_r2_url="https://ftp.sra.ebi.ac.uk/..._2.fastq.gz",
    srr_accession="SRR1234567",  # SRA fallback
)
assert dl.success

# 2. Detect protocol
protocol = detect_protocol(dl.r1_paths[0], dl.r2_paths[0])
print(f"Protocol: {protocol.protocol} ({protocol.confidence})")

# 3. Quantify
result = quantify(
    dl.r1_paths, dl.r2_paths,
    protocol=protocol.chemistry,
    organism="human",
    output_dir="/tmp/quant_output",
)
print(f"Mapped {result.mapping_rate:.1%}, {result.n_cells} cells")

# 4. QC
metrics = run_qc("/tmp/quant_output")
print(f"QC: {metrics.qc_status} ({metrics.n_cells} cells, "
      f"{metrics.median_genes_per_cell} genes/cell)")

# 5. Export
export_to_spz("/tmp/quant_output", "GSM1234567.spz")
```

## External Dependencies

The preprocessing pipeline requires these command-line tools:

- **simpleaf** — Quantification (includes alevin-fry)
- **fasterq-dump** — SRA downloads (fallback)
- **pigz** — Parallel gzip compression (optional, falls back to gzip)
- **curl** — HTTP parallel downloads
