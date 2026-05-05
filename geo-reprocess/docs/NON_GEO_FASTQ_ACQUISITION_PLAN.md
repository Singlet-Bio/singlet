# Non-GEO FASTQ Acquisition Plan

## Executive Summary

Our GEO single-cell catalog contains **22,601 GSEs** (1.49M samples). Cross-referencing against CellxGene, SCEA, and HCA reveals enormous amounts of single-cell data that live outside GEO or lack GEO FASTQ deposits. CellxGene alone hosts **~180M cells** across 2,083 datasets — only **31M cells (~17%)** come from GEO-linked collections with SRA data.

This document lays out a tiered plan to acquire FASTQ data from non-GEO sources, ordered by feasibility and pipeline compatibility.

---

## Data Landscape Census

### CellxGene (359 collections, 2,083 datasets)

| FASTQ Access Tier | Collections | Datasets | Est. Cells | Notes |
|---|---|---|---|---|
| SRA via GEO | 167 | 869 | ~31M | Already tracked in our catalog |
| SRA direct (BioProject) | 6 | 30 | ~830K | No GEO link, but FASTQ on SRA |
| ENA via E-MTAB | 32 | 270 | ~21.2M | Needs BioStudies→ERP→ENA pipeline |
| HCA portal | 15 | 71 | ~8.7M | Some have SRA, some need HCA download |
| NEMO archive | 13 | 247 | ~17.7M | BICCN brain data, asset landing pages |
| Controlled (EGA) | 25 | 144 | ~17M | Requires EGA data access application |
| Controlled (dbGaP) | 15 | 99 | overlap | Requires dbGaP application |
| Processed only | 68 | 254 | ~49M | h5ad available from CellxGene |
| Zenodo | 14 | 40 | ~18.4M | h5ad/count matrices on Zenodo |
| Synapse | 4 | 59 | ~14M | h5ad via Synapse portal |

### SCEA (383 experiments)

| Type | Experiments | Assays | Notes |
|---|---|---|---|
| E-GEOD (GEO-linked) | 97 | 3.78M | Already tracked via GEO |
| E-MTAB (ArrayExpress) | 185 | ~8M+ | FASTQ via ENA/BioStudies |
| E-CURD | 52 | ~2M+ | Curated data, various sources |
| E-HCAD (HCA-derived) | 26 | ~1M+ | HCA Data Portal |
| E-ENAD | 18 | varies | ENA submissions |
| E-ANND (AnnData) | 4 | ~1M+ | Processed data |

### HCA Data Portal (528 projects)

| Category | Projects | Notes |
|---|---|---|
| Has GEO link | 384 | Already tracked |
| Has SRA/BioProject, no GEO | 48 | **Directly processable** |
| Has neither | 90 | Need alternative access |

---

## Tier 1: SRA-Accessible Non-GEO Data (IMMEDIATE — use existing pipeline)

**Validated: 83 unique BioProject accessions (71 from HCA, 12 from CellxGene)**

These datasets have FASTQ on SRA via BioProject accessions but are not in GEO. Our existing SRA download pipeline can handle them directly.

### Sources
1. **48 HCA projects with BioProject but no GEO** — 71 unique accessions (some projects have multiple)
2. **14 CellxGene collections with direct SRA/BioProject links** — 12 unique accessions

### Generated Artifact
`tier1_sra_accessions.tsv` — 83 rows with `source`, `accession`, `title`

### Pipeline
```
BioProject accession → SRA Run Selector → prefetch + fasterq-dump → FASTQ
```

### Action Items
- [ ] Map BioProject → SRR run accessions via ENA/SRA (enumerate runs per project)
- [ ] Filter for single-cell RNA-seq library strategies
- [ ] Download FASTQs using existing `fasterq-dump` infrastructure
- [ ] Feed into standard `simpleaf`/`salmon` quantification pipeline

---

## Tier 2: ENA-Accessible via ArrayExpress/BioStudies (HIGH PRIORITY)

**Yield: ~32 CellxGene collections + 185 SCEA experiments ≈ 21M+ cells**
**Validated: 236 unique E-MTAB accessions, 232 mapped to ERP, 200 are RNA-seq with FASTQ on ENA**

European Nucleotide Archive hosts FASTQs for ArrayExpress (E-MTAB-*) experiments. These are high-quality, curated single-cell datasets not present in GEO.

### Validated Pipeline (tested end-to-end)
```
E-MTAB-NNNNN
  → BioStudies API: https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-NNNNN
  → Extract ERP accession from .section.links[] (list-of-lists structure)
  → ENA File Report: https://www.ebi.ac.uk/ena/portal/api/filereport?accession=ERP...&result=read_run
  → Download from submitted_ftp (FASTQ.gz or BAM) — 230/232 have submitted_ftp
  → Download from fastq_ftp (direct FASTQ.gz) — 92/232 have fastq_ftp
```

### Validated Results from 236-Accession Scan
- **232/236** E-MTABs successfully mapped to ERP via BioStudies API
- **230/232** have files on `submitted_ftp` (FASTQ.gz or BAMs)
- **92/232** have direct FASTQ.gz on `fastq_ftp` (cleanest path)
- **200/232** are RNA-seq library strategy (the rest: OTHER, ssRNA-seq, ATAC, CLONEEND, etc.)
- **Sources**: 172 SCEA-only, 49 CellxGene-only, 11 in both

### Generated Artifact
`tier2_emtab_accessions.tsv` — 236 rows with `emtab`, `sources`, `title`, `erp`, `has_fastq_ftp`, `has_submitted_ftp`, `library_strategies`

### Key Findings from API Testing
- **BioStudies → ERP mapping works reliably** (232/236 = 98.3% success)
- **ENA `fastq_ftp` is often empty**, but `submitted_ftp` almost always has files
- **File types vary**: some have FASTQ.gz directly (e.g., E-MTAB-8142), others have BAMs (e.g., E-MTAB-10187)
- BAMs can be converted: `samtools fastq -1 R1.fq.gz -2 R2.fq.gz input.bam`

### CellxGene E-MTAB Collections (32 collections, 270 datasets, ~21.2M cells)
These are already curated with cell type annotations — high-value targets.

### SCEA E-MTAB Experiments (185 experiments, ~8M+ assays)
The largest non-GEO SCEA category. These have standardized metadata via Expression Atlas.

### Action Items
- [x] Discover all E-MTAB accessions from CellxGene + SCEA (`non_geo_fastq_discovery.py --tier 2`)
- [x] Map E-MTAB → ERP via BioStudies API (`--map-erp` flag)
- [x] Query ENA for FASTQ availability (fastq_ftp vs submitted_ftp)
- [ ] Enumerate all SRR run accessions per ERP (full ENA filereport, not just 5-sample)
- [ ] Classify submitted_ftp files as FASTQ vs BAM for each experiment
- [ ] Build bulk download script (wget/curl from ENA FTP)
- [ ] Handle BAM→FASTQ conversion for applicable datasets (`samtools fastq`)
- [ ] Deduplicate against GEO catalog (some E-MTABs may share data with GEO GSEs)
- [ ] Filter to human + mouse, single-cell RNA-seq only

---

## Tier 3: NEMO Archive (MEDIUM PRIORITY — BICCN brain data)

**Yield: 13 collections, 247 datasets, ~17.7M cells**

NEMO (Neuroscience Multi-Omic Archive) hosts BICCN brain atlas data. These are some of the largest, highest-quality brain single-cell datasets.

### Access Patterns
- **`assets.nemoarchive.org/dat-*`** — landing pages with file listings (HTML, needs scraping)
- **`data.nemoarchive.org/biccn/...`** — direct HTTP file tree (Apache-style directory listing)
- **4 of 18 NEMO collections also have GEO links** (GSE132489, GSE185862, GSE207334, GSE215353) — may already be in our catalog
- **14 NEMO collections have NO GEO link** — need direct NEMO/data.nemoarchive.org access

### Complications
- Many NEMO datasets are **snATAC-seq, methylation, or multi-omic** — not all are RNA-seq
- Need to filter for 10x scRNA-seq / snRNA-seq specifically
- Download may require NEMO account or Aspera/Globus transfer
- Some use non-standard data organization (grant/PI/modality hierarchy)

### Action Items
- [ ] Scrape `assets.nemoarchive.org` landing pages for file manifests
- [ ] Filter for RNA-seq / snRNA-seq modalities only
- [ ] Test direct HTTP download from `data.nemoarchive.org`
- [ ] Check if NEMO supports `wget`/`curl` bulk download or requires Globus

---

## Tier 4: Controlled Access (LONG-TERM — requires applications)

**Yield: ~25 EGA collections (~17M cells) + 24 dbGaP collections**

### EGA (European Genome-phenome Archive)
- 25 CellxGene collections with EGA accessions (EGAD*, EGAS*)
- Requires formal Data Access Application per dataset/study
- Approval timeline: weeks to months
- Once approved, download via `pyEGA3` or EGA download client

### dbGaP
- 24 CellxGene collections reference dbGaP studies (phs*)
- 10 unique phs accessions (phs002371 = HTAN is most common with 7 collections)
- Requires dbGaP authorized access application
- We already have a `controlled_access_catalog/` directory — extend this

### Action Items
- [ ] Compile full list of EGA study/dataset accessions from CellxGene
- [ ] Compile full list of phs accessions from CellxGene + any from SCEA
- [ ] Prioritize applications by cell count and scientific value
- [ ] Cross-reference with existing `controlled_access_catalog/` entries

---

## Tier 5: Processed Data Only (FALLBACK — h5ad/count matrices)

**Yield: ~81M cells (no raw FASTQs available)**

These datasets have no raw FASTQ deposit anywhere — only processed count matrices or h5ad files. However, CellxGene provides standardized h5ad downloads for ALL its datasets. From our perspective, **these would need to be processed through our pipeline from counts rather than FASTQs**.

### Sub-categories
| Source | Collections | Cells | Data Format |
|---|---|---|---|
| GitHub-linked only | 30 | ~30.8M | h5ad from CellxGene |
| Other miscellaneous | 22 | ~13.3M | h5ad from CellxGene |
| Zenodo-linked | 20 | ~18.4M | h5ad from CellxGene or Zenodo |
| Synapse-linked | 5 | ~14M | h5ad from CellxGene or Synapse |
| No external links | 19 | ~3.7M | h5ad from CellxGene only |
| SCP-linked | 3 | ~660K | h5ad from CellxGene |

### Key Consideration
CellxGene h5ad files contain:
- Normalized expression in `.X` (265 datasets) or raw counts in `raw.X` (502 datasets)
- The `raw_data_location` field indicates which layer has raw counts
- Cell type annotations, UMAP embeddings, donor metadata
- Standardized to CELLxGENE schema (ontology terms for tissue, disease, etc.)

### Action Items
- [ ] Determine which h5ad files have raw counts (`raw.X`) vs normalized only (`X`)
- [ ] Build pipeline to ingest h5ad count matrices (skip quantification)
- [ ] Prioritize datasets with `raw_data_location = "raw.X"` (have unnormalized counts)

---

## The 69 "Missing" GEO GSEs

Our cross-reference found 69+ GSEs present in external repositories (CellxGene, SCEA, HCA) but absent from our catalog. Investigation revealed:

- **ALL 69 have 0 SRA records** — they exist in GEO with supplementary files only (RAW.tar, count matrices, h5ad files) but no FASTQ deposits
- **These cannot be processed through our standard FASTQ→salmon pipeline**
- **They are effectively Tier 5 candidates** — processed data ingestion only
- Some are ATAC-seq rather than RNA-seq (GSE154027, GSE161383, etc.)

---

## Priority Implementation Order

| Phase | Tier | Effort | Yield | Status |
|---|---|---|---|---|
| **Phase 1** | Tier 1: SRA direct | Low | 83 BioProjects | Discovery complete ✓ |
| **Phase 2** | Tier 2: E-MTAB via ENA | Medium | 200 RNA-seq experiments (~21M+ cells) | Discovery + ERP mapping complete ✓ |
| **Phase 3** | Tier 5: h5ad ingestion | Medium | ~81M cells | Needs new ingestion path |
| **Phase 4** | Tier 3: NEMO archive | Medium-High | ~17.7M cells (RNA-seq subset) | Needs custom scraping |
| **Phase 5** | Tier 4: Controlled access | High (bureaucratic) | ~17M+ cells | Needs access applications |

### Estimated Total Gain
- **Tier 1+2 (FASTQ-accessible)**: 83 BioProjects + 200 E-MTAB RNA-seq experiments → millions of cells via standard pipeline
- **Tier 3 (NEMO, partial FASTQ)**: ~10-17M cells (RNA-seq subset of BICCN brain data)
- **Tier 5 (processed data)**: ~81M cells if we build h5ad ingestion from CellxGene
- **Tier 4 (controlled)**: ~17M cells pending EGA/dbGaP access approvals

---

## Generated Artifacts

| File | Location | Contents |
|---|---|---|
| `tier1_sra_accessions.tsv` | `/mnt/projects/debruinz_project/cellarium/catalog/` | 83 BioProject accessions (71 HCA, 12 CellxGene) |
| `tier2_emtab_accessions.tsv` | `/mnt/projects/debruinz_project/cellarium/catalog/` | 236 E-MTAB accessions with ERP, FASTQ availability, library strategy |
| `non_geo_acquisition_summary.json` | `/mnt/projects/debruinz_project/cellarium/catalog/` | Structured summary |
| `non_geo_fastq_discovery.py` | `geo-reprocess/scripts/` | Discovery script (Tier 1+2) |

---

## Technical Requirements

### New Infrastructure Needed
1. **ENA bulk download** — `submitted_ftp` FASTQ/BAM download + BAM→FASTQ conversion for Tier 2
2. **h5ad ingestion pipeline** — extract raw counts from CellxGene h5ad, integrate with catalog
3. **NEMO scraper** — parse asset landing pages for file manifests
4. **Run enumeration** — expand BioProject/ERP accessions to individual SRR/ERR run lists

### Already Built
- `non_geo_fastq_discovery.py` — discovers Tier 1+2 accessions, maps E-MTAB → ERP → ENA
- `cross_reference_repositories.py` — tracks GSE presence across CellxGene, SCEA, HCA

### Existing Infrastructure to Leverage
- `fasterq-dump` / SRA toolkit for Tier 1
- `simpleaf` / `salmon` quantification for Tier 1-2 FASTQs
- `controlled_access_catalog/` framework for Tier 4
