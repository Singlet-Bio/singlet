# Changelog

## [1.1.0] — 2026-04-06

### Added
- **Orphan download sweeper** (`scripts/sweep_orphan_downloads.py`) — categorizes and cleans stale/completed FASTQ downloads with `--dry-run` safety
- **Fast batch rebuilder** (`scripts/rebuild_batches_v10_fast.py`) — confidence-filtered batch generation without slow manifest scans
- **SIGTERM cleanup handler** in `process_samples()` — catches SLURM preemption signals and removes partial downloads before exit
- **Zero-mapping-rate early exit** — aborts QC for non-GEX libraries (ADT/HTO/VDJ) that map 0% to transcriptome
- **Permit-list strategy tracking** — `QuantResult.permit_strategy` records which simpleaf strategy succeeded (unfiltered-pl, knee, expect-5000, expect-500)
- **Fail-stage annotation** — `SampleResult.fail_stage` records where processing failed (download, detect, quant, qc, export)
- **Download budget propagation** — `budget_remaining` parameter threads through ENA/SRA downloaders so per-file timeouts respect the overall sample time budget
- **20 new species** — vertebrates (Astyanax mexicanus, Cricetulus griseus, Capra hircus, Tupaia belangeri, Scophthalmus maximus, Microtus ochrogaster, Ciona savignyi), invertebrates (Apis mellifera, Nematostella vectensis, Strongylocentrotus purpuratus, Ascaris suum, Mnemiopsis leidyi), plants (Glycine max, Oryza sativa, Sorghum bicolor, Medicago truncatula, Populus trichocarpa), protists (Toxoplasma gondii)
- **Species aliases** — common names (goat, honeybee, rice, soybean, prairie vole, etc.) resolve to taxon IDs
- Removed `skip: True` from Ambystoma mexicanum, Pleurodeles waltl, Schmidtea mediterranea (now processable)

### Changed
- **Protocol detection hardening** — R1-too-short (<24bp), cDNA-too-short (<31bp for piscem), R2-too-short checks all downgrade confidence to "low" and reject
- **Chemistry auto-correction** — catalog says 10xv3 but R1=26bp → auto-downgrade to 10xv2; reverse for 10xv2 with R1=28bp → upgrade to 10xv3
- **Barcode whitelist gating** — when neither R1 nor R2 matches 10x whitelist (0% match), confidence downgrades to "low" (rejects non-10x data)
- **Catalog confidence cap** — `catalog_confidence` parameter propagates through detection; output confidence cannot exceed catalog's own level
- **CITE-seq chemistry** — CITE-seq GEX libraries now map to `10xv3` chemistry (ADT libraries correctly fail detection or QC)
- **InDrop** added to `LOW_SENSITIVITY_PROTOCOLS` (relaxed min_genes QC threshold)
- **Visium/SlideSeq inference priority** — spatial protocols detected before droplet protocols to prevent "10x Visium" matching 10x droplet rules
- **Download timeout recalibrated** — 80 bytes/read (was 1), 4 MB/s sustained (was 5), 300s overhead (was 120)
- **simpleaf timeout** increased to 7200s (2 hours) for large samples with 200M+ reads
- **Atomic file concatenation** — parallel download segments written to temp file then renamed, preventing partial files on crash
- **Expanded cleanup patterns** — removes `.fastq`, `.fq`, `.sra`, `.sra.cache`, `.dl_segments_*` in addition to `.fastq.gz`/`.fq.gz`
- **Kraken2 swapped-read support** — uses corrected R1/R2 paths when detection flags orientation mismatch
- **Permit-list retry intelligence** — skips retries when mapping rate is below QC threshold (no strategy can rescue low-mapping samples); skips retries on known alevin-fry exit-25856 bug
- **Failed quant mapping rate** — extracts and reports mapping rate even from failed simpleaf runs (aids failure triage)
- **Build script organism filter** — `build_processing_catalog.py` now imports from `scgeo.config.species` to stay in sync with species registry
- **Batch builders filter low-confidence** — `build_v10_filtered_batches.py` and `rebuild_batches_v10.py` exclude `protocol_confidence=low` and `unknown`/`10x_suspect`/`snRNA_unknown` protocols
- Download failure manifests — download failures now write `sample_manifest.json` and clean up files (was silently dropping)

### Fixed
- R1/R2 file count mismatch now returns proper error instead of cryptic simpleaf crash
- Download manifest written on failure (previously only written on quant/QC failure, leaving orphaned downloads)

## [1.0.0] — 2026-04-04

### Added
- Per-GSE merge pipeline (`scripts/merge_gse.py`) — horizontally stacks per-GSM `.1pz` matrices into per-study files with embedded obs/var/uns metadata
- Kraken2 matrix builder (`scripts/build_kraken2.py`) — assembles per-GSE microbiome count matrices from per-GSM taxonomic assignments
- Catalog index builder (`scripts/build_catalog_index.py`) — generates `catalog_v1.parquet` and `sample_index.parquet` for catalog-wide discovery
- SLURM array launchers for merge, retry, and kraken2 build
- Catalog specification document (`docs/CATALOG_SPEC.md`)
- Pipeline guide document (`docs/PIPELINE_GUIDE.md`)
- Multi-species GSE support — automatic detection and split into species subdirectories

### Changed
- Pipeline output format: `.1pz` v3 exclusively (no `.spz`)
- Version bumped to 1.0.0
- README updated with catalog v1.0 summary and doc links

### Removed
- Legacy scripts archived to `scripts/archive/`

### Catalog v1.0 Statistics
- 3,309 GSE datasets (3,377 with multi-species splits)
- 21,006 GSM samples
- 354M cells, 337B non-zeros
- 3,240 datasets with Kraken2 microbiome matrices

## [0.2.0] — 2026-02

### Added
- GEO catalog discovery (NCBI GEO + SRA/ENA metadata enrichment)
- FASTQ quantification pipeline (simpleaf/alevin-fry + Kraken2)
- SLURM job orchestration and monitoring
- Controlled-access dataset discovery (dbGaP, EGA, GDC, HuBMAP)
- Production analytics and failure analysis
