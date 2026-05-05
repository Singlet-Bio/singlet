# Status Reporting & Analytics

## Overview

The `scripts/` directory contains operational analytics tools for monitoring production pipeline runs.

## Available scripts

### `analyze_catalog.py`
Build priority-annotated catalog from the stage 7 multimodal catalog. Classifies species, assigns priorities (P1: human transcriptomics/Visium → P5: other multi-omic), and generates modality × species matrices.

### `analyze_v5_status.py`
Status report on v5 pipeline production runs. Shows top failure errors from batch result CSV files, protocol distribution, and success/failure counts.

### `comprehensive_status.py`
Production status dashboard combining stage 7 catalog, batch metadata, failure directory, and GEO catalog. Reports species/protocol/disease breakdown by SRA run counts.

### `check_provenance.py`
Discovery modality analysis tracking which query methods discovered assays (Visium, Multiome, CITE-seq, Perturb-seq). Identifies coverage gaps.

### `test_failures.py`
Categorizes failure modes from pipeline failure JSON files. Counts by category, generates examples, and checks if failed samples were retried successfully.

### `test_failure_recovery.py`
Tests whether the current codebase handles previous failure modes. Checks protocol detection fixes, fallback logic, and retry queues.

## Running analytics

All scripts are designed for compute nodes:

```bash
ssh compute_node "cd /path/to/geo-reprocess && python scripts/comprehensive_status.py"
```
