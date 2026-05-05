# geo-reprocess

HPC-native toolkit for building single-cell RNA-seq catalogs from GEO and reprocessing FASTQs at scale using simpleaf/alevin-fry.

## Overview

**geo-reprocess** is the data engine powering [SingletDB](https://singletdb.com). It provides:

- **Catalog Discovery** — Query NCBI GEO for single-cell RNA-seq series, enrich with SRA/ENA metadata, and infer protocols
- **FASTQ Processing** — Download, detect chemistry, quantify with simpleaf, run QC, and classify non-host RNA
- **SLURM Orchestration** — Submit array jobs, monitor progress, and aggregate results across HPC clusters
- **Status Reporting** — Analytics scripts for production monitoring, failure analysis, and coverage dashboards
- **Controlled-Access Discovery** — Catalog controlled-access datasets across dbGaP, EGA, GDC, HuBMAP, and more

## Installation

```bash
pip install sc-geo
```

## Quick Start

```python
import scgeo

# Build a catalog of single-cell GEO series
catalog = scgeo.build_catalog(output="catalog.parquet")

# Filter for human 10x Chromium datasets
filtered = scgeo.filter_catalog(catalog, organisms=["Homo sapiens"])

# Submit a SLURM batch job
job_id = scgeo.submit_batch(filtered, partition="general", cpus=36)

# Monitor progress
scgeo.monitor_job(job_id)
```

```{toctree}
:maxdepth: 2
:caption: Contents

install
quickstart
guides/index
api/index
reference/index
```

## Related Projects

- [singlet](https://singlet-ai.github.io/singlet/) — Python client for SingletDB
- [singlepress](https://singlet-ai.github.io/singlepress/) — Sparse matrix compression (.1pz format)
- [singlet-intelligence](https://singlet-ai.github.io/singlet-intelligence/) — ML models and architecture
