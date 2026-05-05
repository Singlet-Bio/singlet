# Pipeline Guide — Catalog v1.0

> How to run the full catalog-building pipeline: discovery → quantification → merge → QC → index.

---

## 1. Architecture

The catalog pipeline has four stages:

```
GEO Discovery → FASTQ Quantification → Per-GSE Merge → Catalog Index
     scgeo           simpleaf              merge_gse.py     build_catalog_index.py
                     + kraken2             + build_kraken2.py
```

**Stage 1 (Discovery)** identifies single-cell datasets on GEO and builds a processing catalog. **Stage 2 (Quantification)** claims samples, downloads FASTQs, detects chemistry, and quantifies using simpleaf/alevin-fry with optional Kraken2 microbiome classification. **Stage 3 (Merge)** combines per-sample outputs into per-study `.1pz` files. **Stage 4 (Index)** builds catalog-wide Parquet indexes.

---

## 2. Prerequisites

- Python ≥ 3.10
- `sc-geo` and `singlepress` packages installed
- `simpleaf`, `piscem`, `fasterq-dump`, optional `kraken2`
- Access to SLURM job scheduler
- Reference genome indices built (see `docs/install.md`)

### 2.1 Environment Variables

```bash
export SCGEO_BASE="/path/to/pipeline"
export SCGEO_WORKSPACE="/path/to/geo-reprocess"
export ALEVIN_FRY_HOME="$SCGEO_BASE/af_home"
```

---

## 3. Stage 1 — Catalog Discovery

### 3.1 Build the Processing Catalog

```bash
python scripts/build_processing_catalog.py
```

This queries NCBI GEO for single-cell RNA-seq datasets, enriches with SRA/ENA metadata, infers protocols, and writes `$SCGEO_BASE/catalog/processing_catalog.parquet`.

### 3.2 Build Filtered Batches

```bash
python scripts/build_v10_filtered_batches.py
```

Generates batch files from the catalog, filtering by phase, protocol confidence, and species. Writes batch Parquet files to `$SCGEO_BASE/catalog/batches/`.

### 3.3 Phases

Processing is organized into phases ordered by confidence and volume:

| Phase | Description | Organisms |
|-------|-------------|-----------|
| 1 | Human droplet RNA (10x, Drop-seq, InDrop) | Human |
| 2a | Multiome GEX components | Human |
| 2b | CITE-seq GEX components | Human |
| 2c | Reclassified droplet (from catalog curation) | Human |
| 2d | Ambiguous human (auto-detect chemistry) | Human |
| 3 | Screen-flagged recovery | Human |
| 4a | Mouse | Mouse |
| 4b | Other organisms | 50+ species |

---

## 4. Stage 2 — FASTQ Quantification

### 4.1 How It Works

Each SLURM worker:
1. **Claims** a batch of samples atomically via `grab_batch.py`
2. **Downloads** FASTQs from ENA (primary) or SRA (fallback) with budget limits
3. **Detects** chemistry (10x v2/v3, Drop-seq, etc.) by read structure analysis
4. **Quantifies** using simpleaf with species-appropriate reference index
5. **Classifies** non-host reads with Kraken2 (optional)
6. **Reports** per-sample results to a CSV in `$SCGEO_BASE/pipeline/results/`
7. **Cleans up** downloaded FASTQs after successful quantification

### 4.2 Claim System

`grab_batch.py` implements an atomic claim system to prevent duplicate processing:

- Claims are recorded in `$SCGEO_BASE/pipeline/claims/` as JSON ledger files
- Each worker claims a batch of unclaimed GSMs
- If a worker crashes, unclaimed samples can be picked up by other workers
- `reconcile_catalog.py` merges all result CSVs back into the catalog

### 4.3 Launch SLURM Array

The `launch.sh` script submits a SLURM array:

```bash
# Process phase 1 (human droplet RNA)
bash scripts/launch.sh --phase 1

# Process phase 4a (mouse)
bash scripts/launch.sh --phase 4a
```

Each array task runs `slurm_worker.sh` which:
1. Activates the conda environment
2. Sources environment variables
3. Calls `grab_batch.py` to claim and process samples

### 4.4 Worker Script Structure

```bash
#!/bin/bash
#SBATCH --job-name=scgeo_p1
#SBATCH --array=0-99
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/p1_%A_%a.out
#SBATCH --error=logs/p1_%A_%a.err

module load miniconda3
conda activate scgeo

export SCGEO_BASE="/path/to/pipeline"
export SCGEO_WORKSPACE="/path/to/geo-reprocess"
export ALEVIN_FRY_HOME="$SCGEO_BASE/af_home"

cd "$SCGEO_WORKSPACE"
python scripts/grab_batch.py \
  --phase 1 \
  --batch-size 20 \
  --worker-id "$SLURM_ARRAY_TASK_ID"
```

### 4.5 Monitoring

```bash
# Quick SLURM status
squeue -u $USER

# Pipeline monitor with failure analysis
python scripts/pipeline_monitor.py --phase 1

# Reconcile results into catalog
python scripts/reconcile_catalog.py
```

### 4.6 Sweep Orphaned Downloads

After processing, clean up stale FASTQ files from crashed or timed-out workers:

```bash
# Dry run first — shows what would be deleted
python scripts/sweep_orphan_downloads.py --dry-run

# Actually delete
python scripts/sweep_orphan_downloads.py
```

### 4.7 Rebuild Batches

If catalog curation changes the eligible sample set:

```bash
# Full rebuild (rescans manifests)
python scripts/rebuild_batches_v10.py

# Fast rebuild (skips manifest scan — uses existing catalog status)
python scripts/rebuild_batches_v10_fast.py
```

---

## 5. Stage 3 — Per-GSE Merge

### 5.1 What It Does

`scripts/merge_gse.py` reads per-GSM outputs within each GSE directory and produces:

1. **`counts.1pz`** — horizontally stacked expression matrix (genes × all cells)
2. **`metadata.parquet`** — per-cell annotations (barcode, gsm_id, organism, total_counts)
3. **`feature_metadata.parquet`** — per-gene annotations (gene_name, reference)
4. **`study_metadata.json`** — study-level metadata from GEO catalog
5. **`provenance.json`** — merge provenance with per-GSM details

For multi-species GSEs, separate outputs are written to species subdirectories (e.g., `Homo_sapiens/`, `Mus_musculus/`).

### 5.2 Single GSE

```bash
python scripts/merge_gse.py GSE117795
```

### 5.3 SLURM Array (Full Catalog)

The merge distributes work across SLURM array tasks. Each task processes a deterministic subset of GSE directories:

```bash
#!/bin/bash
#SBATCH --job-name=merge_gse
#SBATCH --array=0-499
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/merge_%A_%a.out
#SBATCH --error=logs/merge_%A_%a.err

source /tmp/catalog_env/bin/activate
cd /path/to/Singlet-AI
python scripts/merge_gse.py --task-id $SLURM_ARRAY_TASK_ID --n-tasks 500
```

Progress is logged to stderr with per-GSE timing and status.

### 5.4 Handling OOM Failures

Large GSEs (>1M cells) may exceed the default 16GB memory. Use a retry strategy with increasing memory:

1. Identify failed GSEs from logs:
   ```bash
   grep -l "oom-kill\|Killed\|CANCELLED" logs/merge_*.err | \
     xargs grep "Processing" | grep -oP "GSE\d+" | sort -u > retry_gses.txt
   ```

2. Retry with more memory:
   ```bash
   #SBATCH --mem=48G  # or 128G, 250G for the largest
   #SBATCH --array=0-$(( $(wc -l < retry_gses.txt) - 1 ))
   
   GSE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" retry_gses.txt)
   python scripts/merge_gse.py "$GSE"
   ```

**Observed memory tiers for catalog v1.0:**
| Tier | Memory | GSEs | Outcome |
|------|--------|------|---------|
| Initial | 16 GB | 3,309 | 2,851 OK, 77 OOM |
| Medium | 48 GB | 262 | 260 OK, 2 OOM |
| Large | 128 GB | 13 | 11 OK, 2 OOM |
| XL (bigmem) | 250 GB | 4 | 4 OK |

### 5.5 Kraken2 Matrix Build

After merging expression data, build per-GSE Kraken2 matrices:

```bash
#!/bin/bash
#SBATCH --job-name=kraken2_build
#SBATCH --array=0-99
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

source /tmp/catalog_env/bin/activate
cd /path/to/Singlet-AI
python scripts/build_kraken2.py --task-id $SLURM_ARRAY_TASK_ID --n-tasks 100
```

`build_kraken2.py` reads `kraken2_cell_taxa.parquet` from each GSM subdirectory, unions all taxon IDs, and writes `kraken2.1pz` + `kraken2_features.parquet`.

---

## 6. Stage 4 — Catalog Index

### 6.1 Build Index Files

```bash
python scripts/build_catalog_index.py
```

Scans all GSE directories with `provenance.json` and produces:

- **`catalog/catalog_v1.parquet`** — one row per GSE (3,309 rows)
- **`catalog/sample_index.parquet`** — one row per GSM with column offsets (21,006 rows)

### 6.2 SLURM Submission

```bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00

python scripts/build_catalog_index.py
```

Typically completes in under 5 minutes.

---

## 7. Validation

### 7.1 Check Merge Completeness

```python
import pyarrow.parquet as pq
from pathlib import Path

catalog = pq.read_table("catalog/catalog_v1.parquet").to_pandas()
print(f"GSEs in catalog: {len(catalog)}")
print(f"Total cells: {catalog.n_cells.sum():,}")
print(f"With kraken2: {catalog.has_kraken2.sum()}")
```

### 7.2 Validate .1pz Files

```python
import singlepress as sp

# Single file
assert sp.validate_1pz("pipeline/quant/GSE117795/counts.1pz")

# Quick header check
info = sp.info_1pz("pipeline/quant/GSE117795/counts.1pz")
print(f"Shape: {info['m']} × {info['n']}, nnz: {info['nnz']}")
```

### 7.3 Spot-Check Metadata

```python
mat = sp.read_1pz("pipeline/quant/GSE117795/counts.1pz")
assert mat.obs is not None
assert "barcode" in mat.obs.columns
assert "gsm_id" in mat.obs.columns
assert mat.var is not None
assert mat.uns["gse_id"] == "GSE117795"
```

---

## 8. Directory Structure

```
cellarium/
├── catalog/
│   ├── catalog_v1.parquet          # Study-level index
│   ├── sample_index.parquet        # Sample-level index with offsets
│   ├── all_gse_descriptions.parquet  # GEO metadata cache
│   └── geo_single_cell_catalog.parquet  # Discovery catalog
│
└── pipeline/
    └── quant/
        ├── GSE100274/              # Single-species GSE
        │   ├── counts.1pz
        │   ├── metadata.parquet
        │   ├── feature_metadata.parquet
        │   ├── study_metadata.json
        │   ├── provenance.json
        │   ├── kraken2.1pz
        │   ├── kraken2_features.parquet
        │   ├── GSM2676916/
        │   └── ...
        ├── GSE100384/              # Multi-species GSE
        │   ├── Homo_sapiens/
        │   │   └── (merged files)
        │   ├── Mus_musculus/
        │   │   └── (merged files)
        │   └── GSM*/
        └── ...
```

---

## 9. Operational Notes

- **Memory**: Most GSEs merge in <16 GB. The top ~2% need 48–250 GB.
- **Wall time**: 500 SLURM tasks process all 3,309 GSEs in ~1 hour (16 GB tier).
- **Kraken2**: 100 SLURM tasks build all kraken2 matrices in ~1-2 hours.
- **Catalog index**: Single task, <5 minutes.
- **Idempotent**: Re-running merge on a GSE overwrites existing outputs.
- **Multi-species detection**: Automatic — if source GSMs have different organisms, outputs are split into species subdirectories.
