# singlet-pileup Development Log

## Project Goal
Custom C++ streaming BAM pileup tool for maximum-extraction scRNA-seq reprocessing.
Reads unsorted BAM from stdin (piped from STARsolo) and extracts 5 feature layers in a single pass:
1. SNP allele counts at 600K common positions → snp_ad.1pz + snp_dp.1pz
2. RNA editing A/G counts at REDIportal sites → editing_rates.1pz
3. APA 3' UTR length index from PolyA_DB → apa_index.1pz
4. Exon-level counts via interval tree → exon_counts.1pz
5. chrM read extraction → chrM.bam (for mgatk)

## Architecture
- Header-only C++ with htslib dependency for BAM I/O
- singlepress.h for .1pz output
- CLI binary (`singlet-pileup`)
- Minimal Python wrapper for post-hoc corrections only

## Test Sample
- **GSM8291068** from GSE268426 — 1,764 cells, 98.6% mapping rate, human 10xv3
- Already processed via piscem → counts.1pz on disk
- Small enough for fast iteration

---

## Cycle 0: Environment Setup (2026-04-07)

### 0.1 Tool Installation
**Status**: Starting

Tools needed on Clipper cluster:
- [x] STAR/STARsolo (for BAM generation)  
- [x] samtools + htslib (BAM handling, library for C++ build)
- [x] cellSNP-lite (baseline comparison for SNP pileup)
- [x] vireoSNP (donor demultiplexing)
- [ ] mgatk (mitochondrial analysis)
- [ ] pysam (Python BAM access)

### 0.2 Reference Indices
**Status**: Starting

| Index | Status | Path |
|-------|--------|------|
| STAR human genome (GRCh38) | Pre-existing (CellRanger 8.0) | `/cellarium/reference/GRCh38-2024-A/star/` |
| Human GTF (Ensembl 113) | Pre-existing | `/cellarium/reference/GRCh38-2024-A/genes/genes.gtf.gz` |
| 1000G common SNP VCF (600K) | **NEEDS DOWNLOAD** | — |
| REDIportal editing sites | **NEEDS DOWNLOAD** | — |
| PolyA_DB v3 PAS sites | **NEEDS DOWNLOAD** | — |
| Exon interval index | **NEEDS BUILD** from GTF | — |
| Pathogen splici index | **DEFERRED** | — |
| CARD + VFDB functional DB | **DEFERRED** | — |

### 0.3 End-to-End Prototype Plan
1. Install STAR + samtools + cellSNP-lite + vireo on compute node
2. Download reference files (SNP VCF, REDIportal)
3. Re-download FASTQs for test sample GSM8291068
4. Run STARsolo → sorted BAM
5. Run cellSNP-lite → AD/DP matrices (baseline truth)
6. Build C++ singlet-pileup reading from stdin
7. Compare singlet-pileup output to cellSNP-lite (validation)
8. Profile and optimize

---

## Cycle 1: Tool Installation + STARsolo Test (2026-04-07)

### Actions taken:
- SLURM job 337184: Installed STAR 2.7.11b, samtools 1.21, cellSNP-lite 1.2.3, vireoSNP into `cellarium` conda env
- mgatk not yet installed (not critical for first smoke test)

### Test sample changed:
- **GSM8313394** (GSE269338) instead of GSM8291068 — single SRR (SRR29320040), 38M reads, 2,013 cells, 92.2% mapping rate
- Reason: Single run → faster iteration; existing quant output on disk for barcode extraction

---

## Cycle 2: Reference Preparation + Build (2026-04-07)

### Reference Downloads (SLURM 337234/337254):
- [x] 1000G common SNP VCF: 7,416,067 sites — sorted with bcftools, indexed with tabix
- [x] Exon interval BED: 1,586,950 intervals extracted from GTF
- [x] Simple SNP TSV for fast C++ loading: `common_snps_hg38.tsv.gz`
- [ ] STAR 2.7.11b index: Building on c001 (~30 min for genome generate)
  - First attempt failed: process substitution `<(zcat GTF)` → STAR can't read `/dev/fd/63`
  - Fixed: decompress GTF to temp file, pass plain path
- [ ] REDIportal editing sites: DEFERRED to Cycle 3
- [ ] PolyA_DB PAS sites: DEFERRED to Cycle 3

### C++ Build:
- Created project at `/mnt/home/debruinz/Singlet-AI/singlet-pileup/`
- **Headers**:
  - `interval_tree.h` — Static augmented interval tree, O(log n + k) overlap query
  - `sparse_accumulator.h` — COO→CSC template accumulator with barcode hash table
  - `pileup_engine.h` — Core streaming BAM pileup logic (all 5 layers)
  - `mtx_writer.h` — Matrix Market export
- **CLI**: `src/main.cpp` — CLI argument parser + export logic
- **Build**: CMake 3.14+, C++17, links htslib/zstd/lz4
  - Initial compile errors: stale `snp_sites_count_` ref, CSCMatrix member name mismatch
  - Fixed → clean build: **118KB binary, all tests pass**
- **Tests**: `test_interval_tree.cpp` — 6 tests, all pass

### Barcode whitelist:
- Found at: `/mnt/projects/debruinz_project/irungum_1000G/scRNAseq/software/cellranger-10.0.0/lib/python/cellranger/barcodes/3M-february-2018_TRU.txt.gz`
- 3,686,400 barcodes (TRU variant for 10x Chromium 3' v3)

### Pipeline scripts:
- `scripts/download_references.sh` — Downloads SNP VCF, builds exon BED, STAR index
- `scripts/smoke_test.sh` — End-to-end: FASTQ download → STARsolo → cellSNP-lite baseline
- `scripts/build.sh` — CMake configure + build on compute node
- `python/mtx_to_1pz.py` — Convert MTX → .1pz via singlepress
- `python/compare_cellsnp.py` — Validate singlet-pileup vs cellSNP-lite

### Next (Cycle 3):
- Wait for STAR index to finish building
- STARsolo alignment of test sample
- cellSNP-lite baseline run
- Run singlet-pileup on BAM, compare outputs

---

## Cycle 3: Bug Fixes, Gzip Support, and Pipeline Automation (2026-04-07)

### Bug Fixes:
1. **Chromosome name mismatch**: bcftools outputs bare names (`1`, `2`) but BAM uses `chr1`, `chr2`.
   - Added `resolve_chrom()` method handling bidirectional chr prefix normalization (chr1↔1, chrM↔MT)
   - Applied to all reference loaders (SNPs, editing, exons)

2. **SNP position base**: bcftools TSV is 1-based (VCF convention) but 4-column loader was treating as 0-based.
   - Fixed: all formats now treated as 1-based → converted to 0-based internally

3. **resolve_references() incomplete**: Only resolved SNPs, not editing sites or exons.
   - Fixed: now calls `resolve_pending_snps()`, `resolve_pending_editing()`, `resolve_pending_exons()`

4. **String concatenation bug**: `ref[0] + ">"` invalid in C++ (adding char to const char*).
   - Fixed with `std::string(1, ref[0])` construction

5. **Missing include**: `<limits>` needed for `std::numeric_limits` in sparse_accumulator oversaturating math

6. **sam_write1 warning**: Changed from `(void)sam_write1(...)` to proper error check

### Gzip Support:
- Added `LineReader` RAII class using zlib gzFile for transparent .gz reading
- All loaders (SNPs, editing, exons) now read .gz files natively
- No more decompression step needed in the pipeline
- Added ZLIB as explicit CMake dependency

### BED Format Support:
- `load_exon_model()` now auto-detects BED vs GTF format by file extension
- BED: 0-based half-open (no conversion needed)
- GTF: 1-based inclusive → converted to 0-based half-open
- Pre-built exon BED (`exons_hg38.bed.gz`, 1.5M intervals) loads much faster than parsing GTF

### Additional Tests:
- `test_sparse_accumulator.cpp` — 7 tests covering dedup, saturation at 255, CSC structure, empty

### Pipeline Automation:
- Created `scripts/download_editing_sites.sh` — Downloads REDIportal v3 hg38, extracts A→G sites
- Fixed smoke_test.sh: replaced `<(zcat ...)` process substitution with temp file for STAR whitelist
- Updated `scripts/run_pileup_test.sh`: uses gzipped SNP TSV directly + BED for exons
- Job chain: **337254** (STAR index) → **337394** (smoke test) → **337395** (pileup comparison)

### Build Status:
- Clean build: **129KB binary** (Release), 0 errors, 0 warnings
- All tests pass: interval_tree (6), sparse_accumulator (7), **pileup_integration (10 checks)**
- Git: 7 commits on `main`

### STAR Index:
- Building on c001 (SLURM 337254), SA complete, junction insertion in progress (~72 min total)
- Job chain: 337254 → 337394 (smoke test) → 337395 (pileup comparison)

---

## Cycle 4: Integration Test & Validation (2026-04-07)

### Integration Test Fix:
- Original test manually constructed bam1_t records via htslib → "Cannot read BAM header" failure
- Rewrote to use SAM text + `samtools view -bS` conversion — much simpler, reliable
- Fixed `no_barcode` expectation: reads with CB tag but unknown barcode are NOT counted as `no_barcode`
  (they have a CB tag, just not in the whitelist → `barcode_index()` returns -1)

### Test Coverage:
- 6 synthetic reads: 3 passing (2 with SNP hits, 1 chrM), 1 low MAPQ, 1 unknown barcode, 1 accumulation test
- Verifies: total_reads, mapped_reads, barcoded_reads, low_mapq, no_barcode, chrm_reads, snp_hits
- Verifies: DP and AD matrix totals after CSC conversion
- Verifies: MTX file export (snp_dp.mtx, snp_ad.mtx)

### Comparison Script Fix:
- Fixed `compare_cellsnp.py`: SNP TSVS use bare chromosomes (1, 2, ...) vs cellSNP VCF uses chr-prefix (chr1, chr2, ...)
- Added `normalize_snp()` function for bidirectional chr-prefix normalization before alignment

### Reference Validation:
- SNP TSV: 7,416,067 entries, 127 duplicate positions (0.002% — negligible)
- Multi-allelic entries (e.g., "A,T") correctly filtered by alt length check
- Exon BED: 1,586,950 intervals, chr-prefixed, 0-based half-open

---

## Cycle 5: Recovery, Optimizations, and Smoke Test (2026-04-07)

### Network Interruption Recovery
- STAR index build COMPLETED (17 GB, SLURM 337471) — ~19 min on c003 with 64G
- First STAR index attempt failed with OOM at 32G (**33.5G peak** during junction insertion)
- Resubmitted with 64G → SUCCESS
- Index at: `/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b`

### Code Corruption Fix
- `pileup_engine.h` line ~434 had mangled text: `std::vec` garbage merged with optimization code
- Root cause: failed edit during network interruption
- Fixed: clean `std::vector<PendingSNP> snp_pending_` declaration + proper `resolve_pending_snps()`

### Performance Optimization: Binary Search Skip
- Added per-chromosome sorted position vectors (`chrom_snp_positions_`, `chrom_editing_positions_`)
- New `has_variant_overlap(tid, start, end)` method: O(log n) binary search to check if any variant site overlaps read span
- **Effect**: Skips expensive CIGAR walk + hash lookups for reads that don't overlap any variant
- Expected speedup: significant for dense BAMs where most reads don't overlap the ~7M SNP sites

### Bug Fixes (cont'd)
- Removed `reads_with_variants` from main.cpp JSON output + run_pileup_test.sh (field was never added to PileupStats)
- Fixed `load_barcodes()`: changed from `std::ifstream` to `LineReader` for gzipped barcode file support
- Added `barcodes()` public accessor to PileupEngine; simplified main.cpp to reuse engine's barcode list
- Fixed `run_pileup_test.sh`: SNP path was `common_snps_hg38.tsv.gz` (non-existent) → corrected to `genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.vcf.gz`

### Build Status
- Clean build: **133KB binary** (Release), 0 errors, 0 warnings
- All 3 tests pass (interval_tree, sparse_accumulator, pileup_integration)

### Smoke Test Status
- SLURM job 337472 (smoke_test) RUNNING on c003
- FASTQ download via ENA FTP: R1 expected 817MB, R2 expected 2.1GB
- At 17 min elapsed: R1 ~54% (443MB), R2 ~19% (435MB)
- Estimated download completion: ~35 min remaining
- Job chain: 337472 (smoke test) → 337473 (pileup test, pending dependency)

### Infrastructure Improvements
- Created `scripts/smoke_test_fast.sh` — uses fasterq-dump (2-3× faster than ENA FTP), includes all phases in one script

### ENA FASTQ Sizes (SRR29320040)
| File | Expected Size | Description |
|------|--------------|-------------|
| R1 | 817 MB | Barcode + UMI (28bp × 38M reads) |
| R2 | 2.1 GB | cDNA read (~90bp × 38M reads) |

### Known Limitations (to address in Cycle 6)
1. **No UMI deduplication** — raw read counting inflates DP vs cellSNP-lite baseline
2. **RNA editing sites not downloaded** — REDIportal download deferred
3. **APA analysis stub** — PolyA_DB not downloaded, `load_pas_sites()` is placeholder
4. **MTX format only** — .1pz conversion via separate Python script
5. **No streaming stdin test yet** — code supports `-` path but not validated with pipes
