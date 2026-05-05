# geo-reprocess

[![Documentation](https://img.shields.io/badge/docs-singlet--ai.github.io-blue)](https://singlet-ai.github.io/geo-reprocess/)

HPC-native toolkit for building single-cell RNA-seq catalogs from GEO and reprocessing FASTQs at scale using simpleaf/alevin-fry.

## Features

- **Catalog Discovery** — Query NCBI GEO, enrich with SRA/ENA metadata, infer protocols
- **FASTQ Processing** — Download, detect chemistry, quantify, QC, and classify non-host RNA
- **SLURM Orchestration** — Phase-based array jobs with atomic claim system, SIGTERM-safe cleanup
- **Controlled-Access Discovery** — Catalog datasets across dbGaP, EGA, GDC, HuBMAP, and more
- **55+ Species** — Human, mouse, rat, zebrafish, fruit fly, plus 50 additional organisms with auto-resolved reference genomes
- **Status Reporting** — Production analytics, failure analysis, orphan download sweeping

## Installation

### From source (recommended for HPC)

```bash
git clone https://github.com/Singlet-AI/geo-reprocess.git
cd geo-reprocess
pip install -e ".[dev]"
```

### From PyPI

```bash
pip install sc-geo
```

## New Cluster Setup

Setting up on a new SLURM cluster requires: (1) the Python package, (2) external
bioinformatics tools, and (3) reference genome indices built once per species.

### 1. Python environment

```bash
# Create a dedicated conda environment
conda create -n scgeo python=3.11 -y
conda activate scgeo

# Install the package in editable mode
git clone https://github.com/Singlet-AI/geo-reprocess.git
cd geo-reprocess
pip install -e .

# Install runtime dependencies
pip install singlepress pandas pyarrow scipy numpy requests aiohttp tqdm
```

### 2. External tools

```bash
# simpleaf + piscem (required for quantification)
conda install -c bioconda -c conda-forge simpleaf

# fasterq-dump (SRA fallback downloads)
conda install -c bioconda sra-tools

# kraken2 (optional — non-host RNA classification)
conda install -c bioconda kraken2

# pigz (parallel gzip — used by fasterq-dump post-processing)
conda install -c conda-forge pigz
```

### 3. Reference genome indices

simpleaf requires pre-built indices per species. Build them once:

```bash
# Set simpleaf home for index storage
export ALEVIN_FRY_HOME="/path/to/indices"

# Example: build human + mouse (the two most common)
simpleaf index \
  --output "$ALEVIN_FRY_HOME/human" \
  --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz \
  --gtf Homo_sapiens.GRCh38.113.gtf.gz \
  --threads 16

simpleaf index \
  --output "$ALEVIN_FRY_HOME/mouse" \
  --fasta Mus_musculus.GRCm39.dna.primary_assembly.fa.gz \
  --gtf Mus_musculus.GRCm39.113.gtf.gz \
  --threads 16
```

Species-to-index mapping is configured in `scgeo/config/species.py`. The pipeline
supports 55+ organisms — see that file for the full list and expected assembly names.

### 4. Pipeline directory structure

```bash
export SCGEO_BASE="/path/to/pipeline"

mkdir -p "$SCGEO_BASE/pipeline"/{claims,downloads,quant,results,logs}
mkdir -p "$SCGEO_BASE/catalog"
```

Layout after setup:
```
$SCGEO_BASE/
├── af_home/               # simpleaf index home (ALEVIN_FRY_HOME)
│   ├── human/             # Per-species indices
│   ├── mouse/
│   └── ...
├── catalog/
│   ├── processing_catalog.parquet     # Main sample catalog
│   └── curation/                      # Manual curation CSVs
├── pipeline/
│   ├── claims/            # Atomic claim ledger (grab_batch.py)
│   ├── downloads/         # Temporary FASTQ storage (cleaned after quant)
│   ├── quant/             # Per-GSE quantification outputs
│   │   └── GSE*/GSM*/     # Per-sample: counts.1pz, sample_manifest.json
│   ├── results/           # Per-job result CSVs
│   └── logs/              # SLURM stdout/stderr
```

### 5. Environment variables

Set these in your SLURM worker script or `.bashrc`:

```bash
export SCGEO_BASE="/path/to/pipeline"           # Pipeline root
export SCGEO_WORKSPACE="/path/to/geo-reprocess" # Source code
export ALEVIN_FRY_HOME="$SCGEO_BASE/af_home"    # simpleaf indices
```

### 6. Running the pipeline

```bash
# Build/update the processing catalog from GEO
python scripts/build_processing_catalog.py

# Launch SLURM array processing (phase 1 = human droplet RNA)
bash scripts/launch.sh --phase 1

# Monitor progress
python scripts/pipeline_monitor.py --phase 1

# Sweep orphaned downloads (safe with --dry-run first)
python scripts/sweep_orphan_downloads.py --dry-run
python scripts/sweep_orphan_downloads.py
```

See [docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md) for the full operational guide.

## Catalog v1.0

The pipeline produces a standardized catalog of reprocessed single-cell datasets:

- **3,309 GSE datasets** — 354M cells, 337B non-zeros
- **`.1pz` format** — per-GSE merged expression matrices with embedded obs/var/uns metadata
- **Kraken2 microbiome** — per-GSE taxon count matrices for 3,240 datasets
- **Parquet indexes** — catalog-wide study and sample discovery

See [docs/CATALOG_SPEC.md](docs/CATALOG_SPEC.md) for the directory layout and metadata schemas.

## Pipeline Architecture

```
GEO Discovery → FASTQ Quantification → Per-GSE Merge → Catalog Index
     scgeo           simpleaf              merge_gse.py     build_catalog_index.py
                     + kraken2             + build_kraken2.py
```

Processing phases:
| Phase | Description |
|-------|-------------|
| 1 | Human droplet RNA (10x, Drop-seq, InDrop) |
| 2a | Multiome GEX components |
| 2b | CITE-seq GEX components |
| 2c | Reclassified droplet (from catalog curation) |
| 2d | Ambiguous human (auto-detect chemistry) |
| 3 | Screen-flagged recovery |
| 4a | Mouse |
| 4b | Other organisms |

## Part of Singlet AI

| Repository | Purpose |
|-----------|---------|
| **geo-reprocess** | HPC pipeline, catalog, status reporting |
| [singlet](https://github.com/Singlet-AI/singlet) | Python client library |
| [singlepress](https://github.com/Singlet-AI/singlepress) | Sparse matrix compression |
| [singlet-intelligence](https://github.com/Singlet-AI/singlet-intelligence) | ML models & architecture |
| [singlet-strategy](https://github.com/Singlet-AI/singlet-strategy) | Strategic planning |
| [singletai-website](https://github.com/Singlet-AI/singletai-website) | Website & dashboard |
| [papers](https://github.com/Singlet-AI/papers) | Manuscripts & reports |

## License

GPL-3.0-or-later
