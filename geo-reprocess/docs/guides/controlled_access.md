# Controlled-Access Data Discovery

## Overview

The `controlled_access_catalog/` scripts discover and catalog single-cell datasets behind access controls across multiple repositories.

## Data sources

| Source | Discovery Method | Key Data |
|--------|-----------------|----------|
| **dbGaP** | NCBI E-utilities, SRA queries | Study accessions, disease info, BioProject links |
| **EGA** | EBI Search API | Study/dataset metadata |
| **GDC** | REST API faceted queries | Cancer genomics projects |
| **HTAN** | REST API | Human Tumor Atlas Network |
| **HuBMAP** | REST API | Human BioMolecular Atlas |
| **Brain/NeMO** | REST API | Neuroscience data |
| **GTEx** | REST API | Genotype-Tissue Expression |
| **GSA-Human** | Web API | Genome Sequence Archive |
| **JGAS** | Web API | Japanese Genome Archive |

## Pipeline stages

The 8-step discovery pipeline:

1. `01_dbgap_discovery.py` — Find dbGaP studies with single-cell data
2. `02_ega_discovery.py` — Discover EGA studies/datasets
3. `03_portal_discovery.py` — Query GDC, HTAN, HuBMAP, etc.
4. `04_dbgap_detailed.py` — Fetch detailed dbGaP study records
5. `05_additional_portals.py` — GSA-Human, JGAS, others
6. `06_build_catalog.py` — Unify into single catalog with SRA cross-reference
7. `07_enrich_counts.py` — Add run counts, disease categories
8. `08_final_analysis.py` — Deduplication, priority categorization, summary statistics

## Key findings

- ~646 dbGaP-linked BioProjects with 180,000+ SRA runs
- Estimated 50K–100K single-cell specific runs
- Dominated by human samples (P1 priority)
