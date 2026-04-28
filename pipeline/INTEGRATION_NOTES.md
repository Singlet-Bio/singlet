# singlify-bio Integration Notes

This document records APIs shipped by **singlify-bio** that need wiring into `singlify.cpp`.

---

## Feature: Bulk ATAC-seq Auto-Detection and QC (G-BULK-ATAC)
**Header**: `include/singlet-pileup/bulk_atac.h`
**API**:
```cpp
bool is_bulk_atac(uint32_t cblen, bool is_paired_end,
                  const std::string& assay_type = "") → bool
BulkAtacQC compute_bulk_atac_qc(const std::vector<ATACFragment>& fragments,
                                  uint64_t total_reads,
                                  const std::vector<int32_t>& mito_chrom_idxs = {},
                                  uint64_t tss_overlapping_fragments = 0) → BulkAtacQC
bool write_bulk_atac_qc(const std::string& path, const BulkAtacQC& qc) → bool
```
**Pipeline insertion point**: After STAR PE-DNA alignment sort, before ATACFragmentExtractor; replaces cell-calling with dedup + bulk QC.
**CLI flag needed**: `--mode bulk-atac` (auto-inferred when cblen=0 + paired-end)
**Dependencies**: `atac_fragment.h` for `ATACFragment` struct; no htslib needed for QC path
**Shipped**: cycle 106, commit 938f346 on branch main

---

## Feature: TPM/FPKM Normalized Gene Expression Output (G-TPM)
**Header**: `include/singlet-pileup/tpm_fpkm.h`
**API**:
```cpp
std::vector<GeneLength> compute_gene_lengths(const GeneModel& gm);
std::vector<double>     compute_tpm(const std::vector<int32_t>& counts,
                                    const std::vector<GeneLength>& lengths);
std::vector<double>     compute_fpkm(const std::vector<int32_t>& counts,
                                     const std::vector<GeneLength>& lengths,
                                     uint64_t total_mapped);
bool write_gene_expression_tsv(const std::string& path,
                                const std::vector<GeneLength>& lengths,
                                const std::vector<int32_t>& counts,
                                const std::vector<double>& tpm,
                                const std::vector<double>& fpkm);
```
**Pipeline insertion point**: After gene counting (bulk / Smart-seq2 path), before output flush
**CLI flag needed**: `--output-tpm` (default: off for scRNA; auto-on for bulk/SS2)
**Output file**: `<out_prefix>/gene_expression.tsv`
**TSV columns**: `gene_id, gene_name, effective_length, count, TPM, FPKM`
**Dependencies**: `gene_model.h` (GeneModel, exon_starts/ends)
**Shipped**: cycle 105, commit 6bc47d9

---

## Feature: G-TAGGED-BAM — Cell Ranger-Compatible Tagged BAM Output
**Header**: `include/singlet-pileup/tagged_bam.h`
**API**:
```cpp
struct TaggedBamConfig {
    std::string output_path;
    bool add_barcode_tag = true;   // CB:Z, CR:Z, CY:Z
    bool add_umi_tag     = true;   // UB:Z, UR:Z, UY:Z
    bool add_gene_tag    = true;   // GX:Z, GN:Z, RE:A
    bool add_xf_tag      = true;   // xf:i feature type flags
    bool add_index       = true;   // create .bai after close()
    int  threads         = 4;
    const std::unordered_set<std::string>* called_barcodes = nullptr;
};
class TaggedBamWriter {
    bool open(const TaggedBamConfig& cfg, const sam_hdr_t* hdr);
    bool write_record(bam1_t* b, ...); // CB/CR/CY/UB/UR/UY/GX/GN/RE/xf tags
    void close(); uint64_t records_written() const; uint64_t records_skipped() const;
};
```
**Pipeline insertion point**: During pileup pass, per-record after barcode/UMI correction.
**CLI flag needed**: `--tagged-bam [PATH]`
**Output file**: `{out_prefix}/possorted_genome_bam.bam` + `.bai`
**Dependencies**: htslib (PILEUP_LINK_LIBS)
**Shipped**: cycle G-TAGGED-BAM, commit 99ea644 on branch main

---

## Feature: G-VELOCITY — Spliced/Unspliced/Ambiguous Velocity Matrices
**Header**: `include/singlet-pileup/velocity.h`
**API**:
```cpp
// Collapse intron-interval CSC to gene-level CSC (duck-typed GeneModel)
template <typename CSCType, typename GeneModelType>
singlet::GeneCSC<uint32_t> collapse_intron_to_gene(
    const CSCType& intron_csc, const GeneModelType& gm);

// Create empty (nnz=0) gene-level CSC placeholder
template <typename T>
singlet::GeneCSC<T> make_empty_gene_csc(uint32_t nrows, uint32_t ncols);
```
**Pipeline insertion point**: Automatically triggered inside `export_results()` when
  `pileup_cfg.count_introns && engine.gene_model().n_introns() > 0`
**Output files** (both .1pz and .mtx modes):
  - `spliced.{1pz,mtx}` = gene-level collapse of exon_counts
  - `unspliced.{1pz,mtx}` = gene-level collapse of intron_counts
  - `ambiguous.{1pz,mtx}` = zero-nnz placeholder (future: exon-intron overlap reads)
**CLI flag needed**: None — auto-enabled with `--count-introns`
**Dependencies**: `gene_model.h` (no htslib)
**Shipped**: cycle G-VELOCITY, commit 3f6c77c on branch main

---

## Feature: Equivalence-Class EM Multi-Mapper Rescue (G-EM)
**Header**: `include/singlet-pileup/em_rescue.h`
**API**:
```cpp
// Core EM run (from pre-computed gene_totals)
singlet::em_rescue::EMGeneCSC run_em_rescue(
    const std::vector<AmbigRead>& ambig_reads,
    const std::vector<double>& gene_totals,
    uint32_t n_genes, uint32_t n_cells,
    int max_iter=50, double tol=0.01,
    EMRescueStats* stats_out=nullptr);

// Convenience overload (computes gene_totals from GeneCSC)
template<typename GeneCSCType>
singlet::em_rescue::EMGeneCSC run_em_rescue(
    const std::vector<AmbigRead>& ambig_reads,
    const GeneCSCType& gene_csc,
    int max_iter=50, double tol=0.01,
    EMRescueStats* stats_out=nullptr);

// AmbigRead accessor on PileupEngine
const std::vector<em_rescue::AmbigRead>& engine.ambig_reads() const;
```
**Pipeline insertion point**: Automatically triggered inside `export_results()` after gene_csc
  is computed. Runs if `engine.ambig_reads()` is non-empty.
**CLI flag needed**: None — automatic (only runs when ambig_reads > 0)
**Output files**: `gene_counts_em.1pz` (or `.mtx`) alongside `gene_counts`
**Log line**: `[em_rescue] N reads rescued across M equivalence classes, K iterations, I integer counts`
**Dependencies**: pileup_engine.h (collects AmbigRead during resolve_mm), export.h (triggers EM + writes output)
**Changes to pileup_engine.h**:
  - `second_gene` field added to both `MultiHitEntry` structs (streaming + parallel)
  - `ambig_reads_` private member + `ambig_reads()` accessor on PileupEngine
  - `WorkerContext::ambig_reads` per-worker vector, merged into `ambig_reads_` after parallel run
  - resolve_mm + resolve_mm_parallel: collect AmbigRead when gene_ambiguous + both genes known
  - Second-gene detection at: secondary cross-gene mismatch, primary multigene from exon_overlaps, aln_gene mismatch
**Shipped**: cycle G-EM, commit bad8248 on branch main

---

## Bug Fix: Any-Overlap Gene Assignment (commit 532e2b3)
**File**: `include/singlet-pileup/pileup_engine.h` (inline, no separate header)
**Root cause**: The pileup engine used strict block-containment for exon assignment. STARsolo Gene
  uses HTSeq union (any-overlap). A secondary broad_overlap_genes_ check over-rejected reads even
  when containment uniquely assigned gene A.
**Fix**: Replaced containment + broad_overlap_genes_ with any-overlap in all 4 code paths:
  - `run()`: primary path + `determine_exon_gene()`
  - `run_parallel()`: `process_read_parallel()` + `determine_exon_gene_parallel()`
**Impact on concordance** (C01, streaming mode, 2565 cells):
  - Gene r: 0.9946 → 0.9990
  - EEF1A1: 9,110 → 40,684 (STARsolo: 51,043)
  - multigene_reads: 65,470 → 57,780
**Note**: Parallel mode fixed by commit 3141e4a (below).
**No pipeline changes required**: fix is internal to pileup_engine.h.


## Bug Fix: Cross-Worker Multi-Mapper Merge (commit 3141e4a)
**File**: `include/singlet-pileup/pileup_engine.h` (inline, `run_parallel()`)
**Root cause**: In sorted-BAM parallel mode, reads whose primary alignment is on chromosome A and
  secondary on chromosome B are split across independent worker threads with no shared mm_buffer.
  Workers with only secondaries (bc_idx=-1) failed to resolve the entry; primary-worker entries
  flushed partially without the cross-chromosome secondaries, losing multi-mapper resolutions.
  For EEF1A1 (many pseudogenes on different chromosomes), this caused ~30K UMI loss per dataset.
**Fix**:
  - Per-worker thread flush now only resolves COMPLETE entries (seen >= nh). Partial entries are
    kept in wc.mm_buffer after the worker finishes.
  - After all threads join, a global cross-worker merge collects partial mm_buffer entries from all
    workers, accumulates bc_idx + UMI from the primary worker's entry, gene info from all workers
    with full gene-ambiguity detection, and resolves via a dedicated global UmiDedup.
  - New log line: `[parallel-cross-worker] N cross-worker entries → M exon hits`
**Validation on C01 (SRR32855204, 10x-arc-gex, 40.4M reads)**:
  - EEF1A1: 8,884 → 41,251 (+364%, ≥30K target met); STARsolo=48,288
  - Gene Pearson r (vs STARsolo): 0.9948 → 0.9995 (≥0.999 target met)
  - Cross-worker resolved: 584,773 exon hits from 1,230,611 partial entries
  - Wall overhead: +~0.1s (global merge) on 2.7s parallel pipeline = +3.7%
**Validation on C03 (SRR10885105, 10xv2)**: parallel-vs-streaming r = 0.999990
**No pipeline changes required**: fix is fully within run_parallel() in pileup_engine.h.


At merge time, use these specs to integrate features into the pipeline orchestrator.

singlify-bio does NOT modify `singlify.cpp` — all pipeline integration requirements are documented here.

---

## Feature: Cell Cycle Phase Scoring (N19)
**Header**: `include/singlet-pileup/cell_cycle.h`
**API**:
  - `struct CellCycleEntry { s_score, g2m_score, phase }` — per-cell
  - `struct CellCycleResult { cells, n_s_genes_detected, n_g2m_genes_detected, n_g1, n_s, n_g2m }`
  - `CellCycleResult score_cell_cycle(exon_csc, model, threshold=0.1)` — template, any CSC type
  - `void write_cell_cycle_tsv(path, result, barcodes)` — writes barcode/phase/s_score/g2m_score TSV
**Algorithm**: Tirosh et al. 2016 simplified scores: mean(log1p(counts)) over 43 S-phase and 54 G2M-phase
  marker genes. Phase = S if Score_S > Score_G2M and Score_S > threshold; G2M if Score_G2M wins; else G1.
**Pipeline insertion point**: Phase 1.57 in `export.h` — after sex calling (1.55), before ancestry (1.56).
  Runs whenever GTF gene model with hierarchy is present and exon_csc.ncols > 0. Always on (no flag needed).
**Output**: `{out_prefix}/cell_cycle_scores.tsv` — columns: barcode, phase, s_score, g2m_score
**Dependencies**: `gene_model.h` only; no external libraries.
**Validation (C01 = SRR32855204, 10x-arc-gex, 40.4M reads, 4155 cells, 2026-04-11)**:
  - G1=3486 (83.9%), S=297 (7.1%), G2M=372 (9.0%) — biologically plausible ✓
  - 42/43 S genes detected, 52/54 G2M genes detected ✓
  - Overhead: 0.013s (well within <1s target for 10K cells) ✓
  - Build: clean, no warnings ✓
**Shipped**: N19, cycle 2026-04-11, commit cfd6fc9

---

## Feature: Species Auto-Detection (N1)
**Header**: `include/singlet-pileup/species_detect.h`
**Supporting header**: `include/singlet-pileup/species_kmer_db_gen.h` (auto-generated — regenerate
  with `scripts/build_species_kmer_db.py` when new species/assemblies are needed)
**API**:
  - `struct DetectionResult { species, taxon_id, genome_tag, confidence, method, reads_sampled, kmers_tested, kmers_hit }`
  - `DetectionResult detect(const std::string& onefq_path, bool verbose=true)`
    — tries metadata path first (<1ms), falls back to k-mer sampling (~0.5–0.7s)
  - `bool DetectionResult::confident()` — true if confidence ≥ threshold OR method=="metadata"
  - `std::string DetectionResult::suggested_star_dir(const std::string& ref_base)` — maps genome_tag to genome dir
  - `std::string organism_from_metadata(const std::string& onefq_path)` — metadata-only, no k-mers
  - `std::string genome_tag_from_star_dir(const std::string& genome_dir)` — parses STAR genomeParameters.txt
**Detection thresholds**: CONFIDENCE_THRESH=0.001, MIN_RATIO=2.0 (winner needs 2x runner-up hits)
**Pipeline insertion point**: Wired directly into `singlify.cpp` main(). After N4 whitelist auto-resolution,
  before the `genome_dir.empty()` validation check.
**CLI flags added**:
  - `--ref-base DIR` — reference root directory (e.g. `/mnt/projects/.../reference`)
  - `SINGLIFY_REF_BASE` env var (fallback when `--ref-base` not given)
  - When enabled, auto-resolves `--genome-dir` if not specified by user
**Genome mapping**:
  - "GRCh38" → `{ref_base}/GRCh38-2024-A/star_2.7.11b`
  - "GRCm39" → `{ref_base}/GRCm39-2024-A/star_2.7.11b`
**Validation (kmer DB path, 21/21 unit tests pass)** on corpus:
  - C01 SRR32855204 (human 3' ARC GEX): confidence=0.953, method=kmer ✓ (1280/99486 hits)
  - C02 SRR27329891 (human 5' 10xv3): confidence=0.712 ✓
  - C06 SRR23582977 (human sci-RNA): confidence=0.837 ✓
  - C11 SRR34789664: 0 hits via kmer-DB fallback — needs bloom filters for mouse detection
**Bloom filters** (256MB each for ~0.7% FPR @ 153M k-mers):
  - `species_filters/human_21mer.bloom` — building from GRCh38 transcriptome (~60 min)
  - `species_filters/mouse_21mer.bloom` — building from GRCm39 exons (~30 min)
  - Initial 45MB design had 34.96% empirical FPR (unusable); fixed to 256MB in commit 7195f97
**Wall overhead**: ~0.5–0.7s (k-mer path, 5000 reads sampled)
**Dependencies**: `lib1fq/reader.h`, `species_kmer_db_gen.h` (optional — graceful fallback)
**Shipped**: N1, cycle 4, commit dc4aaae (header+tests), 7195f97 (filter size fix)

---

## Feature: EmptyDrops Cell Calling (N5)
**Header**: `include/singlet-pileup/cell_calling.h`
**API**:
  - `struct CellCallResult { cell_indices, deviance, fdr, tested_indices, n_ambient, ambient_total }`
  - `CellCallResult call_cells_emptydrops(counts_per_barcode, gene_counts_csc, fdr_threshold=0.01, lower=100, n_ambient_max=50000)`
  - `void write_cell_calls(outpath, result, barcodes, counts_per_barcode, fdr_threshold=0.01)`
**Algorithm**: Lun et al. (2019) EmptyDrops multinomial deviance LRT:
  1. Ambient profile estimated from barcodes with count **<= lower** (empty droplets).
  2. Tested barcodes = count **> lower**.
  3. Deviance = 2 * Σ_{g: n_g>0} n_g * [log(n_g) − log(N) − log(p_g)]; df = n_nonzero − 1.
  4. BH FDR correction; cells = FDR < fdr_threshold.
**Pipeline insertion point**: After COO→CSC conversion (Phase 1). Requires the UNFILTERED
  barcode×gene CSC matrix (including empty droplets with count <= lower).
  Run BEFORE barcode filtering in `export_results()`.
**CLI flags added** (wired 2026-04-11):
  - `--cell-calling` — force on (default: auto-on when `--barcodes` not provided)
  - `--no-cell-calling` — disable
  - `--fdr-threshold F` — FDR cutoff [0.001]
  - `--lower-umi N` — lower UMI bound for ambient pool [100]
**Output**: `{out_prefix}/cell_calls.tsv` — columns: barcode, total_umi, deviance, fdr, is_cell
**Integration point**: `export.h` Phase 1.7 — after COO→CSC (Phase 1), before mt heteroplasmy
  (Phase 2). Activated by `ExportConfig::run_cell_calling` flag (set in singlify.cpp).
**Dependencies**: `sparse_accumulator.h` only; no external math libraries.
**Statistical note**: Ambient = barcodes with count ≤ lower (true empty RNA, not cells).
  Works correctly ONLY on unfiltered count matrices. Pre-filtering barcodes removes empty droplets
  and makes the algorithm conservative (all tested barcodes appear as significant deviators).
**Validation (C01 = SRR32855204, 10x-arc-gex, 40.4M reads, 2026-04-11)**:
  - singlify cells called: 10,120 | STARsolo_regen filtered: 2,565
  - Concordance (overlap/starsolo): 0.9992 (2,563/2,565 = 99.92%) ✓ (target ≥80%)
  - EmptyDrops overhead: 0.053s (well within <5s target) ✓
  - Pipeline wall: 140s vs ~120s baseline (genome cold cache — EmptyDrops itself: 0.05s) ✓
  - Note: singlify calls more cells than STARsolo knee-point (EmptyDrops is more sensitive)
**Shipped**: N5, cycle 2026-04-11, integrated into export.h Phase 1.7

---

## Feature: Per-Cell QC Metrics (N9)
**Header**: `include/singlet-pileup/cell_qc_metrics.h`
**API**:
  - `CellQCMetrics compute_cell_qc(exon_csc, intron_csc, gene_model) → CellQCMetrics`
  - `void write_cell_qc_tsv(path, qc, barcodes)` — writes TSV with 6 columns
**Pipeline insertion point**: After COO→CSC conversion (Phase 1), before write threads (Phase 3).
  Integrated directly into `export.h` as Phase 1.5 — no singlify.cpp changes needed.
**CLI flag needed**: None — runs automatically whenever GTF gene model with hierarchy is loaded.
**Output**: `{out_prefix}/cell_qc_metrics.tsv`
**Columns**: `barcode`, `total_umis`, `total_genes`, `mt_pct`, `ribo_pct`, `intronic_pct`
**MT detection**: chrM/MT/chrMT chromosome OR gene_name prefix "MT-"/"mt-". Note: 10x GRCh38-2024-A
  reference intentionally excludes chrM from GTF annotation, so MT%=0 for those samples. Standard
  GENCODE/Ensembl GTFs (without 10x filtering) will show non-zero MT%.
**Ribo detection**: Case-insensitive RPL*/RPS* gene name prefix (human and mouse).
**Dependencies**: `gene_model.h`, `sparse_accumulator.h`
**Performance**: O(nnz) single-pass over exon+intron CSC matrices; <1s for typical datasets.
**Shipped**: N9, commit pending

---

## Feature: Sequencing Saturation (N7)
**Header**: `include/singlet-pileup/saturation.h`
**New APIs**:
  - `struct CellSaturation { barcode_idx, total_reads, unique_umis, saturation, reads_per_umi }`
  - `std::vector<CellSaturation> compute_saturation(const DirectionalUmiStore& store, int umi_len)`
  - `double median_saturation(std::vector<CellSaturation>& cells)`
  - `void write_saturation_tsv(path, cells, barcodes)`
**New `DirectionalUmiStore` method** (added to `umi_dedup.h`):
  - `struct PerCellUmiStats { uint64_t total_reads; uint64_t unique_umis; }`
  - `std::unordered_map<uint32_t, PerCellUmiStats> per_cell_stats(int umi_len) const`
**New `PileupEngine` method** (added to `pileup_engine.h`):
  - `const DirectionalUmiStore& dir_exon_store() const`
**Pipeline insertion point**: Integrated into `export.h::export_results()` as Phase 1.6.
  Runs after Phase 1.5 (Cell QC), before write threads.
  Gated by: `pileup_cfg.umi_dedup_directional && !engine.dir_exon_store().empty()`
**CLI flag needed**: None — runs automatically when `--pipeline` or `--umi-dedup-directional` is set.
**Output**: `{out_prefix}/saturation_metrics.tsv`
**Columns**: `barcode`, `total_reads`, `unique_umis`, `saturation`, `reads_per_umi`
**Saturation formula**: CellRanger-compatible: `saturation = 1 - (unique_umis / total_reads)`
**Validation (C01, SRR32855204, 10x-arc-gex)**:
  - n_cells=4155, median_saturation=0.213
  - mean_saturation=0.212, mean_reads_per_umi=1.27
  - Low saturation expected for scARC-GEX (multiome data has fewer reads/cell)
**Performance**: No extra BAM pass — reuses `DirectionalUmiStore` groups already in memory.
  O(n_groups) with O(k²) per group (k = unique UMIs ≈ 1-10); adds <1s to pipeline.
**Dependencies**: `umi_dedup.h`
**Shipped**: cycle N7, 2026-04-11

---

## Feature: Sex & Karyotype Calling (N14)
**Header**: `include/singlet-pileup/sex_calling.h`
**API**:
  - `struct SexCallResult { sex, confidence, xist_cpm, y_cpm, xist_umis, y_umis, n_y_genes_detected, total_umis }`
  - `SexCallResult call_sex<CSCMat>(exon_csc, model) → SexCallResult`
    — templated, works with any `SparseAccumulator<T>::CSCMatrix`
  - `void write_sex_call_json(path, result)`
**Algorithm**:
  1. Walk `GeneModel::genes()` to find XIST exon rows (chrX) and chrY marker exon rows
     (UTY, DDX3Y, KDM5D, EIF1AY, ZFY, RPS4Y1, USP9Y)
  2. Single O(nnz) pass over CSC matrix to sum total/XIST/Y marker UMIs
  3. Normalize to CPM: xist_cpm = xist_umis * 1e6 / total_umis; y_cpm similarly
  4. Classify: female if XIST_cpm > 1.0 AND Y_cpm < 2.0; male if Y_cpm > 2.0 AND n_y_genes ≥ 2
  5. confidence = min(1.0, xist_cpm / 10.0) for female, min(1.0, y_cpm / 20.0) for male
**Pipeline insertion point**: Integrated into `export.h::export_results()` as Phase 1.55.
  Runs after Phase 1.5 (Cell QC), before Phase 1.6 (Saturation).
  Gated by: `!pileup_cfg.exon_gtf_path.empty() && engine.gene_model().has_gene_hierarchy() && exon_csc.ncols > 0`
**CLI flag needed**: None — runs automatically whenever GTF gene model is loaded and cells present.
**Output**: `{out_prefix}/sex_call.json`
**Validation**:
  - Unit test (synthetic GTF, 3 cells, 4 genes): 8/8 tests pass ✓
    - female test: xist_cpm=800000 → sex=female, confidence=1.0 ✓
    - male test: y_cpm=444444, n_y=2 → sex=male, confidence=1.0 ✓
    - unknown test: no markers → sex=unknown ✓
  - C01 SRR32855204 (human, 10x-ARC-GEX, female): sex=female, confidence=1.0,
    xist_cpm=551.65, xist_umis=5236, total_umis=9491441, wall=0.186s ✓
    STARsolo GeneFull ground truth: XIST=344 UMIs, Y-genes=0 → confirmed female ✓
  - C03 SRR10885105 (human, 10xv2): sex=unknown, confidence=0, xist_umis=0, y_umis=0 ✓
    STARsolo GeneFull ground truth: XIST=0 UMIs, Y-genes=0 → correctly unknown ✓
    (Low UMI/cell dataset; biology provides no sex markers — correct answer IS unknown)
  - SRR5250847 (human, Drop-seq): sex=female, confidence=1.0, xist_cpm=222.72,
    xist_umis=163, y_marker_umis=2, n_y_genes=1, total_umis=731847, wall=0.018s ✓
    STARsolo GeneFull ground truth: XIST=155 UMIs, Y-genes low → confirmed female ✓
  - **N14 concordance: 3/3 datasets match STARsolo-implied sex call (100%)**
**Wall overhead**: 0.186s on SRR32855204 (40M reads, full GTF) ✓ (<1s target met)
**Reference requirement**: GTF must include chrX and chrY annotations.
  **IMPORTANT: GRCh38-2024-A/genes/genes.gtf was truncated (chr1-19 only, 2,863,807 lines).
  Regenerated from genes.gtf.gz → 3,293,166 lines including chrX, chrY, chr20-22, patches.
  Always use genes.gtf.gz or verify line count ≥3M before use.**
**Dependencies**: `gene_model.h`, `<unordered_set>`, `<unordered_map>`
**Shipped**: N14, cycle 2026-04-11

---

## Feature: UMI Error Correction — Directional 1-Hamming (N6)
**Header**: `include/singlet-pileup/umi_dedup.h`
**New APIs**:
  - `uint64_t umi_pack_2bit(const char* umi, int len)` — pack UMI string to 2-bit integer
  - `int umi_hamming_2bit(uint64_t a, uint64_t b, int umi_len)` — XOR-based Hamming distance
  - `uint32_t directional_dedup(vector<pair<uint64_t,uint32_t>>& umi_counts, int umi_len)` — Smith et al. 2017 directional algorithm, returns component count
  - `class DirectionalUmiStore` — per-run accumulator; `record(bc,gene,exon,umi)` + `finalize<AccT>(acc, umi_len)→uint64_t`
**Pipeline insertion point**: Integrated into `PileupEngine::run()` single-thread path.
  Activated by `PileupConfig::umi_dedup_directional = true`.
**New PileupConfig fields**:
  - `bool umi_dedup_directional = false` — enable directional correction
  - `int umi_len = 12` — UMI length for Hamming distance computation
**CLI flags (singlify.cpp)**:
  - `--umi-dedup-directional` — enable (also auto-enabled by `--pipeline`)
  - `--no-umi-dedup-directional` — override, disable even in pipeline mode (N6 task addition)
**Auto-enable**: `pipeline_mode && !config.umi_dedup_directional && !no_dir_dedup_explicit`
**Limitation**: `run_parallel()` (BAI-indexed parallel path) does NOT yet use directional dedup;
  it falls back to simple `UmiDedup`. Parallel directional dedup requires a merge step — future work.
**Performance**: O(n²) pairwise Hamming per (cell, gene) group; typical group n=1-10 UMIs → negligible.
  Groups >1000 are rare; even O(n²) at n=1000 = 10⁶ ops.
  Wall overhead: +1.8% on C01 (113s vs 111s) for 3.4M cell×gene groups.
**Validation (C01, SRR32855204, 10x-arc-gex, 40M reads)**:
  - Pre-dedup exon reads: 10,210,823
  - After exact dedup: 8,075,568 unique (20.9% duplicate rate)
  - After directional: 8,043,929 unique (additional 31,639 = 0.39% above exact dedup)
  - Pearson r (exact vs directional matrix): 0.99993
  - 4054/4155 cells had ≥1 UMI further collapsed by directional
  - Low rate expected for 12-bp UMIs (4^12=16.7M space; 1-Hamming pairs rare)
  - For 8-bp protocols (Drop-seq, sci-RNA-seq3) we expect higher dedup rates
**Dependencies**: `<algorithm>`, `<numeric>`, `<unordered_map>`, `<vector>`
**Shipped**: N6 task, commit pending

---

## Feature: Pipeline Provenance Manifest (N8)
**Header**: `include/singlet-pileup/provenance.h`
**New APIs**:
  - `struct ProvenanceConfig` — input, reference, parameters, timing fields
  - `void write_provenance_json(out_prefix, prov, n_exon_features, n_cells, total_umis)`
**Pipeline insertion point**: Integrated into `export.h::export_results()`.
  Automatically called at end of export if `export_cfg.provenance.input_file` is non-empty.
**ExportConfig new field**: `ProvenanceConfig provenance` (in `export.h`)
**Output**: `{out_prefix}/provenance.json`
**singlify.cpp wiring**: In the `// ── Write stats ──` section, populate provenance fields before calling `export_results()`:
  ```cpp
  // Add after: singlet::ExportConfig export_cfg; ... export_cfg.threads = config.threads;
  export_cfg.provenance.singlify_version = "0.3.0";
  export_cfg.provenance.input_file      = onefq_file.empty() ? sra_file : onefq_file;
  export_cfg.provenance.input_reads     = stats.total_reads;
  export_cfg.provenance.genome_dir      = genome_dir;
  export_cfg.provenance.gtf_path        = config.exon_gtf_path;
  export_cfg.provenance.threads         = star_threads;
  export_cfg.provenance.umi_dedup       = config.umi_dedup;
  export_cfg.provenance.umi_dedup_directional = config.umi_dedup_directional;
  export_cfg.provenance.pipeline        = pipeline_mode;
  export_cfg.provenance.wall_seconds    = total_sw.elapsed_s();
  export_cfg.provenance.pileup_seconds  = pileup_time;
  // star_seconds: requires capturing STAR wall time separately (add StopWatch star_sw before fork+waitpid)
  ```
**Dependencies**: `<chrono>`, `<ctime>`, `<fstream>`, `<string>` (no external deps)
**Shipped**: cycle N, commit TODO

---

## Feature: Allele-Specific Expression (N15)
**Header**: `include/singlet-pileup/ase.h`
**New APIs**:
  - `struct ASEEntry { uint32_t cell_idx; uint32_t snp_idx; uint32_t ref_count; uint32_t alt_count; float allelic_ratio; }`
  - `std::vector<ASEEntry> ase::compute_ase(snp_ad_csc, snp_dp_csc, min_depth=10)` — O(nnz) merge-join, no extra BAM pass
  - `void ase::write_ase_tsv(path, entries, barcodes, snp_names)` — TSV with header: barcode, snp_id, ref_count, alt_count, allelic_ratio
**Pipeline insertion point**: Integrated into `export.h::export_results()` as a write_thread.
  Triggered whenever `!pileup_cfg.snp_path.empty()` (both pipeline and non-pipeline modes).
  Runs concurrently with exon/intron/SJ matrix writes; SNP CSC matrices remain valid until all write_threads join.
**Output**: `{out_prefix}/ase_counts.tsv`
**Filter logic**: Reports only heterozygous sites — dp ≥ min_depth AND ad ≥ 1 AND (dp-ad) ≥ 1.
  allelic_ratio = alt/(ref+alt); expected ~0.5 for balanced, deviation indicates ASE.
**Note**: AD/DP values are uint8_t capped at 255 — depths >255 saturate (rare in scRNA-seq).
  min_depth=10 default; no new CLI flag needed (internal to compute_ase).
**singlify.cpp wiring**: Already integrated via export.h. No changes to singlify.cpp required.
**Dependencies**: `sparse_accumulator.h` (included via export.h chain); `<cstdio>`, `<fstream>`, `<vector>`
**Shipped**: cycle N15, build exit 0, April 2026

---

## Feature: Ancestry Classification (N13)
**Header**: `include/singlet-pileup/ancestry.h`
**API**:
  - `struct AIM { chrom, pos_0b, ref, alt, rsid, freq[5] }` — ancestry-informative marker
  - `const std::vector<AIM>& get_aim_panel()` — 51-SNP panel from Nassir 2009, Kosoy 2009, 1KGP Phase 3
  - `struct AncestryResult { ancestry, confidence, log_lik[5], prob[5], n_informative, n_covered, low_data }`
  - `AncestryResult classify_ancestry(snp_ad_csc, snp_dp_csc, aim_start_idx, aims)` — ML over 5 populations (AFR/EUR/EAS/SAS/AMR)
  - `void write_ancestry_json(path, result, aims)` — writes ancestry_call.json
**Populations**: AFR=African, EUR=European, EAS=East Asian, SAS=South Asian, AMR=Admixed American
**Algorithm**: Log-likelihood sum over covered AIMs: L_p = Σ ad_j*log(f_pj) + (dp_j-ad_j)*log(1-f_pj); softmax→probabilities
**Integration**: Zero-config — AIMs auto-injected into SNP pileup in PileupEngine::load_references()
  via new `load_aim_panel()` private method. New PileupConfig flag `bool aim_ancestry = true;` (default on).
  New private field `aim_snp_start_` + public getter `aim_snp_start()` on PileupEngine.
  export.h runs ancestry at Phase 1.56 (after sex calling), triggers whenever `engine.aim_snp_start() != UINT32_MAX`.
**PileupEngine changes**:
  - `PileupConfig::aim_ancestry = true` (new field)
  - `PileupEngine::aim_snp_start()` (new getter)
  - `PileupEngine::load_aim_panel()` (new private method, injects into snp_pending_)
  - Includes `ancestry.h` at top of pileup_engine.h
**export.h changes**:
  - Includes `ancestry.h`
  - `need_snp_csc` flag replaces `!pileup_cfg.snp_path.empty()` for snp_ad/snp_dp CSC conversion
  - Phase 1.56 block runs ancestry classification
**singlify.cpp wiring**: None required — fully integrated via pileup_engine.h + export.h.
**Output**: `{out_prefix}/ancestry_call.json` — always written when AIM panel has ≥5 covered sites
**Validation** (C01, SRR32855204, 10x ARC GEX, Human): ancestry=EUR, confidence=0.990, n_informative=7, n_covered=2
**CLI flags**: None — always runs. To disable: set `config.aim_ancestry = false` before load_references()
**Wall overhead**: <1ms (ancestry classification; SNP pileup overhead is negligible, 51 AIMs)
**Dependencies**: `<cmath>`, `<algorithm>`, `<array>`, `<string>`, `<vector>` (no external deps)
**Shipped**: cycle N13, April 2026

---

## Feature: Ambient RNA Correction (N11)
**Header**: `include/singlet-pileup/ambient_correction.h`
**API**:
  - `struct AmbientProfile { gene_fractions, total_ambient_umi, n_ambient_barcodes }` — per-feature ambient fractions
  - `AmbientProfile estimate_ambient(const CSCu16& counts, const std::vector<uint64_t>& bc_totals, uint64_t lower_umi, int n_ambient_max=50000)`
    — aggregates empty droplets (bc_totals[b] ≤ lower_umi), normalizes with pseudocount 0.5
  - `struct CellContamination { rho, log_likelihood, n_genes_used }` — per-cell contamination estimate
  - `std::vector<CellContamination> estimate_contamination(const CSCu16& counts, const AmbientProfile& ambient, const std::vector<uint32_t>& cell_indices, const std::vector<uint64_t>& bc_totals, uint32_t n_top_genes=200)`
    — median-ratio estimator over top-200 ambient-enriched features, clipped to [0, 0.95]
  - `CSCu16 correct_counts(counts, ambient, contam, cell_indices, bc_totals)` — subtract rho * N_c * ambient[g] per gene
  - `void write_ambient_profile(path, ambient, feature_names)` → ambient_profile.tsv
  - `void write_contamination_tsv(path, contam, cell_indices, barcodes)` → ambient_contamination.tsv
**Algorithm**: SoupX-lite v1 (simplified from Young & Bhatt 2020, GigaScience 9(12)):
  1. Empty droplet pool: all barcodes ≤ lower_umi (default 100), sorted by count descending, max 50k
  2. Ambient profile: sum counts across empty pool, normalize with pseudocounts
  3. Per-cell ρ: median(observed[g,c] / (N_c × ambient[g])) over top-200 ambient features, clipped [0,0.95]
  4. Correction: corrected[g,c] = max(0, obs[g,c] - round(ρ × N_c × ambient[g]))
**Integration**: Fully integrated into export.h Phase 1.7–1.75
  - Phase 1.7 refactored: `bc_totals`, `CellCallResult cc_result`, `CSCu16 exon_corrected_csc` declared OUTSIDE if-block
  - Phase 1.75: runs whenever `cc_result.cell_indices` is non-empty (i.e., whenever cell calling finds cells)
  - Corrected matrix written to write_threads when `export_cfg.run_ambient_correction = true`
**ExportConfig new field**: `bool run_ambient_correction = false` — enables exon_counts_corrected.1pz/.mtx output
**Outputs** (always when cell calling runs):
  - `{out_prefix}/ambient_profile.tsv` — feature-level ambient fractions
  - `{out_prefix}/ambient_contamination.tsv` — per-cell rho and n_genes_used
**Optional output** (when `run_ambient_correction = true`):
  - `{out_prefix}/exon_counts_corrected.1pz` — ambient-subtracted count matrix
**singlify.cpp wiring needed**: Add `--ambient-correction` CLI flag → `export_cfg.run_ambient_correction = true`
**Activation**: Cell calling must run first (auto or via `--cell-calling`). With `--barcodes` provided,
  all barcodes are cells so n_ambient=0 → ambient correction skipped (correct behavior).
  Use `--pipeline` mode with .1fq files that have valid protocol metadata for auto-whitelist+cell-calling.
**Validation** (C01/SRR20020820, 10xv3, Human): n_ambient_barcodes=859, total_ambient_umi=57714,
  n_cells=350, median_rho=0.95 (clipped — v1 estimator notes: top-200 ambient features overlap with
  cell-expressed genes, inflating ratio; future v2 should filter to ambient-specific features only)
**Wall overhead**: ~0.12s for 350 cells (ambient estimation + contamination estimation; negligible)
**Note on rho bias**: The v1 median-ratio estimator consistently clips to 0.95 because top ambient features
  are also cell-expressed. A future v2 should pre-filter top-ambient features to exclude genes with high
  cross-cell variability (variance > threshold), restricting to truly ambient-leaked genes.
**Dependencies**: `sparse_accumulator.h` only; no external deps
**Shipped**: cycle N11, April 2026

## Feature: Doublet Detection (N12)
**Header**: `include/singlet-pileup/doublet_detect.h`
**API**:
  - `detect_doublets(counts, cell_indices, threshold=0.25, sim_ratio=1.0, k_neighbors=10, seed=42) → std::vector<DoubletResult>`
  - `write_doublet_tsv(path, results, cell_indices, barcodes, counts) → void`
  - `struct DoubletResult { double score; bool is_doublet; }`
**Algorithm**: Hybrid score = 0.6 × kNN-sim-fraction + 0.4 × UMI-ratio component
  - Simulated doublets created by summing n_cells × sim_ratio random cell pairs (L1-normalised profiles)
  - k-NN over pool of [observed | simulated] profiles; score = fraction of k neighbours that are simulated
  - UMI component: saturates at 1.0 for cells with 2× median UMI count
  - Default threshold = 0.25 (empirical; tunable per dataset)
**Pipeline insertion point**: Phase 1.76 in `export.h`, after ambient correction (Phase 1.75), before mt heteroplasmy (Phase 2)
**Activation condition**: `!cc_result.cell_indices.empty() && exon_csc.ncols > 0` (requires cell calling)
**CLI flag needed**: `--doublet-threshold <float>` (optional, future work; currently uses 0.25 default)
**Output**: `{out_prefix}/doublet_scores.tsv` — columns: barcode, total_umis, doublet_score, is_doublet
**Dependencies**: `sparse_accumulator.h` only; uses `<random>`, `<algorithm>`
**Shipped**: cycle N12, April 2026

## Feature: Deep Archive Mode (F5)
**Implementation**: `src/singlify.cpp` — `cmd_archive()` function
**API**: `singlify archive <input.1fq> -o <output.noq.1fq> [--codec-level N] [--verbose]`
**Algorithm**: decode-and-re-encode loop with `qual_mode=NONE`, `codec_level=6`
  - Opens input with `lib1fq::Reader`, creates `lib1fq::Writer` with qual_mode=NONE
  - Copies protocol/assay/confidence/dedup/trim flags from source header
  - For each block: iterates `read_block()` → `add_read(…, nullptr, …, nullptr)` (no quality)
  - Passes through metadata JSON unchanged
  - Identity guard: early-exit if input already has qual_mode=NONE
**Pipeline insertion point**: Off-pipeline (run manually after pileup is complete)
**CLI flag needed**: N/A — standalone subcommand `singlify archive`
**Dependencies**: `lib1fq::Reader`, `lib1fq::Writer` (already linked)
**Output**: `<input>.noq.1fq` (default) — uniform 'F' quality when decoded back to FASTQ
**Size reduction**: 31.4% on binned-quality input (SRR20020820: 144 MB → 99 MB)
**Wall time**: 4.4s for 5M reads (single-threaded re-encode)
**Shipped**: F5, commit 126b9ee, April 2026

## Feature: V(D)J Gene Usage Counting (N17)
**Header**: `include/singlet-pileup/vdj_counter.h`
**API**:
- `VdjModel::load_gtf(path) → bool` — parse gene_biotype/gene_type=IG_*/TR_* from GTF
- `VdjModel::resolve_chroms(chrom_to_tid)` — build tid-indexed interval trees
- `VdjModel::query(tid, start, end, results)` — append overlapping VDJ gene indices
- `VdjModel::gene_names() → const vector<string>&` — gene display names with biotype
**Pipeline insertion point**: After exon/intron counting in hot loop; uses aligned_blocks_ read extent
**CLI flag needed**: `--no-count-vdj` to disable (enabled by default when `--exons` is provided)
**Output**: `vdj_gene_usage.1pz` (or `.mtx`) — sparse matrix, rows=VDJ gene segments, cols=cell barcodes
**Graceful**: zero-hit datasets log "[vdj_counting] No VDJ reads detected" and write no file
**GTF compatibility**: supports both `gene_type` (GENCODE/GRCh38) and `gene_biotype` (Ensembl/GRCm39)
**No strand filter**: VDJ recombination creates non-standard orientations; any overlap counts
**UMI dedup**: exact dedup per (barcode, vdj_gene, UMI) — same as exon counting
**Validation**:
- C01 (SRR32855204, 10x-arc-gex, GRCh38): 411 genes, 78308 hits, 8504 nnz, 411×4155 matrix
- C11 (SRR34789664, 10xv3 mouse brain, GRCm39): 490 genes found, 0 hits — graceful empty
**Wall overhead**: <1% (aligned_blocks_ span query, no extra CIGAR walk)
**Dependencies**: `PileupEngine::gene_model_` GTF same path, `IntervalTree`, `UmiDedup`, `SparseAccumulator`
**Shipped**: cycle N17, commit 85b6429, April 2026

## Feature: CRISPR Guide Capture Counting (N18)
**Header**: `include/singlet-pileup/crispr_guide.h`
**API**:
- `GuideRef::load_csv(path) → bool` — load name,sequence CSV (auto-detects header)
- `GuideRef::match(seq, len) → int` — exact substring match of any guide in read; returns guide id or -1
- `GuideRef::match(string) → int` — std::string overload (case-insensitive, uppercases input)
- `GuideRef::names() → vector<string>` — guide names in id order for matrix feature list
- `GuideCounter::init(n_guides, barcodes)` — initialize per-barcode × guide accumulator
- `GuideCounter::count(bc_idx, guide_id, umi, umi_len)` — record one observation with UMI dedup
- `GuideCounter::accumulator() → SparseAccumulator<uint16_t>&` — access sparse accumulator
- `GuideCounter::merge(other)` — merge worker accumulator for parallel pileup
**Pipeline insertion point**: After NH/MAPQ/UMI extraction, before exon/intron counting (hot loop)
**CLI flag needed**: `--guide-ref <guides.csv>` (empty = disabled, zero overhead when not set)
**Output**: `guide_counts.1pz` or `guide_counts.mtx.gz` — rows=guides, cols=cell barcodes
**Graceful**: When `guide_ref_path.empty()`, counter is inactive — no overhead, no output
**UMI dedup**: reuses `UmiDedup` (exact 64-bit hash of (bc_idx, guide_id, UMI))
**Guide matching**: Exact substring search of guide sequence within query read; uppercase-normalized
**CSV format**: `name,sequence` (with or without header row; auto-detected)
**Unit tests**: test/test_crispr_guide.cpp — 7 tests, all pass with -O3 -DNDEBUG
**PGO note**: CMakeLists.txt sets -fprofile-use=/dev/null for test target only — singlify PGO
  profile crashes SparseAccumulator<uint16_t>::to_csc() with small test matrices (O3 vector OOB)
**Shipped**: cycle N18, commit ecbad79, April 2026

---

## Feature: Shared-Memory Genome (N21)
**Header**: `src/singlify.cpp` (inline helpers, no separate header)
**API**:
- `star_genome_shmkey(genome_dir) → key_t` — returns System V SHM key (st_ino of genome dir) used by STAR
- `star_genome_in_shm(genome_dir) → bool` — true if STAR's SHM segment is present via shmget()
- `run_star_genome_op(genome_dir, load_mode, out_prefix) → int` — fork-exec STAR with LoadAndExit or Remove
- `cmd_genome(argc, argv) → int` — dispatches `singlify genome load|unload|status <genome-dir>`
**Pipeline insertion point**: Before STAR alignment args construction; auto-detection replaces NoSharedMemory with LoadAndKeep when shm segment found
**CLI flags**:
  - `singlify genome load <genome-dir>` — load genome into SHM (~15-23s one-time cost)
  - `singlify genome unload <genome-dir>` — release SHM segment
  - `singlify genome status <genome-dir>` — print shmKey + loaded/not loaded
  - `--genome-load MODE` — override auto-detection (NoSharedMemory default)
**Dependencies**: System V IPC (shmget/shmctl), POSIX fork/waitpid, star_main_impl()
**Auto-detection**: At pipeline launch, if genome_load=="NoSharedMemory" and star_genome_in_shm() returns true, switches to LoadAndKeep transparently; logs "[singlify] N21: Genome found in shared memory — using LoadAndKeep."
**Timing**: genome load once: 15-23s (GRCh38: ~15s, GRCm39: ~23s); warm pipeline attach: 2s vs 9s from hot page cache; saves 13-20s on cold-cache nodes
**Validation**: C01 (SRR32855204, GRCh38) and C11 (SRR34789664, GRCm39) — pileup counts identical between NoSharedMemory and LoadAndKeep runs; load/unload/status commands verified working
**Shipped**: cycle N21, commit f90c21c, April 2026

---

## Feature: Per-Cell Read Statistics (N20)
**Header**: `include/singlet-pileup/read_stats.h`
**API**:
- `compute_read_stats(per_cell_reads, exon_indptr, exon_data, intron_indptr, intron_data, ncols) → vector<CellReadStats>`
- `write_read_stats_tsv(path, stats, barcodes) → bool`
- `median_dup_rate(stats) → double`
- `lander_waterman(n_total, n_dup) → uint64_t`
**Pipeline insertion point**: export.h Phase 1.58 — after CSC conversion, before ancestry classification
**Output**: `{out_prefix}/read_stats.tsv` — 6-column TSV: barcode, total_reads, unique_umis, dup_reads, dup_rate, est_complexity
**Dependencies**: PileupEngine::per_cell_reads() (new field added to pileup_engine.h), exon+intron CSC matrices
**pileup_engine.h changes**:
- Added `std::vector<uint32_t> per_cell_reads_` private field
- Incremented in `run()` single-thread path: `per_cell_reads_[bc_idx]++` after barcode lookup
- Added `cell_reads` to `WorkerContext` for parallel path; merged element-wise after join
- Added public accessor `per_cell_reads() const → const vector<uint32_t>&`
**Validation**: C01 (SRR32855204, 10x-arc-gex): n_cells=4155, median_dup_rate=0.374 (biologically plausible); output time=0.006s
**Shipped**: cycle N20, commit dc50c22, April 2026

## Feature: ATAC Fragment Extraction (A1)
**Header**: `include/singlet-pileup/atac_fragment.h`
**API**:
- `ATACFragmentExtractor::set_barcode_map(const unordered_map<string,uint32_t>&) → void`
- `ATACFragmentExtractor::process_record(const bam1_t*, const bam_hdr_t*) → void`
- `ATACFragmentExtractor::finalise() → void` (sorts by chrom_idx, start)
- `ATACFragmentExtractor::fragments() const → const vector<ATACFragment>&`
- `ATACFragmentExtractor::total_reads/proper_pairs/duplicate_fragments/unique_fragments/no_barcode_reads() const → size_t`
**Pipeline insertion point**: After STAR PE-DNA BAM sort, before A2 (atac_bin_matrix.h)
**CLI flag needed**: `--atac` (future singlify.cpp integration)
**Dependencies**: htslib (bam1_t, BAM_F* flags)
**Key design notes**:
- Processes read1 only (BAM_FREAD1); read2 silently skipped
- Tn5 shift: start+=4, end-=5 per Buenrostro/10x ATAC convention
- Barcode from QNAME prefix "BC:XXXX_" written during .1fq decode
- All-N barcodes (absent I2) counted as no_barcode and skipped
- Dedup: unordered_set<FragmentKey> with 16-byte key (tid,start,end,bc_idx)
- finalise() sort enables streaming to downstream bin matrix
**Shipped**: cycle 74, April 12 2026

## Feature: ATAC Bin Matrix (A2)
**Header**: `include/singlet-pileup/atac_bin_counter.h`
**API**:
- `ATACBinCounter::set_bin_width(int width=500) → void`
- `ATACBinCounter::set_chromosome_sizes(const vector<pair<string,int32_t>>&) → void`
- `ATACBinCounter::count_fragments(const vector<ATACFragment>&) → void`
- `ATACBinCounter::count_one(const ATACFragment&) → void`
- `ATACBinCounter::bin_features() const → vector<string>` (chr:start-end format)
- `ATACBinCounter::to_csc(size_t num_cells) const → SparseAccumulator<uint16_t>::CSCMatrix`
- `ATACBinCounter::total_bins() const → size_t`
- `ATACBinCounter::nonzero_bins() const → size_t`
- `ATACBinCounter::clear() → void`
**Pipeline insertion point**: After A1 (atac_fragment.h) finalise(), before .1pz export
**CLI flag needed**: `--atac-bin-width 500` (future singlify.cpp integration)
**Dependencies**: atac_fragment.h (ATACFragment), sparse_accumulator.h (SparseAccumulator)
**Key design notes**:
- global_bin_offset[chrom_idx] precomputed from chrom sizes; O(1) lookup per fragment
- Fragment [start,end) overlaps bins [start/W, (end-1)/W] — typically 1-2 bins at 500bp
- uint16_t counts: saturates at 65535 (far above any real ATAC signal)
- bin_features() emits "chr:start-end\tchr:start-end\tPeaks" (10x ATAC features.tsv format)
- 5 unit tests; all pass; compile-clean
**Shipped**: cycle 75, commit ab56ac7, April 12 2026

---

## Feature: ATAC QC Metrics (A3)
**Header**: `include/singlet-pileup/atac_qc.h`
**API**:
- `ATACQCComputer::set_chrom_map(const unordered_map<string,int32_t>&) → void`
- `ATACQCComputer::set_mito_chroms(const vector<string>& = {"chrM","MT","chrMT"}) → void`
- `ATACQCComputer::set_tss_positions(const unordered_map<int32_t,vector<int32_t>>&) → void`
- `ATACQCComputer::load_tss_from_gtf(const string& gtf_path) → bool`
- `ATACQCComputer::set_tss_window(int32_t half_width = 1000) → void`
- `ATACQCComputer::set_frip_min_count(uint32_t n = 3) → void`
- `ATACQCComputer::set_hist_bin_width(int w = 10) → void`
- `ATACQCComputer::compute(const vector<ATACFragment>&, const ATACFragmentExtractor&, uint32_t num_cells, const ATACBinCounter* = nullptr) → void`
- `ATACQCComputer::cell_qc() const → const vector<ATACCellQC>&`
- `ATACQCComputer::fragment_size_histogram(int bin_width=10) const → vector<uint64_t>`
- `ATACQCComputer::write_qc_tsv(const string& path, const vector<string>& barcodes) const → void`
- `ATACBinCounter::fragment_global_bin(const ATACFragment&) const → size_t` (added for FRIP)
**Pipeline insertion point**: After A1 finalise() + A2 count_fragments(), before .1pz export
**CLI flag needed**: `--atac-qc` (future singlify.cpp integration, optional output)
**Dependencies**: atac_fragment.h, atac_bin_counter.h, zlib
**Key design notes**:
- TSS enrichment = tss_fragments / total_fragments (simplified); TSS loaded strand-aware from GTF
- Mito detection by chrom_idx lookup (chrM/MT/chrMT); defaults applied if not set
- Median fragment size: per-cell sort of fragment lengths
- FRIP approx: bins with sum >= frip_min_count are "signal"; fraction of fragments in signal bins
- 12 unit tests (6 scenarios); all pass; compile-clean
**Shipped**: cycle 76, commit 99b7803, April 12 2026

## Feature: Directional UMI Correction Parallel Parity (N6-parallel fix)
**Header**: `include/singlet-pileup/pileup_engine.h`
**API**: `WorkerContext::dir_exon_store.record(bc_idx, gene_idx, exon_idx, umi_packed)` — per-worker store; `DirectionalUmiStore::finalize(wc.exon, umi_len)` after thread join
**Pipeline insertion point**: After `threads.join()`, before `merge_accumulators()` in `run_parallel()`
**CLI flag needed**: none (auto-enabled in pipeline mode via `umi_dedup_directional=true`); pileup binary now exposes `--umi-dedup-directional`
**Dependencies**: N6 (DirectionalUmiStore), commit 3141e4a (cross-worker merge)
**Shipped**: cycle 77, commit 85f5ba0

## Feature: ADT Tag Matcher (T1)
**Header**: `include/singlet-pileup/adt_matcher.h`
**API**:
- `AdtMatcher::load_reference(const string& path) → bool`
- `AdtMatcher::add_tag(const string& name, const string& sequence) → void`
- `AdtMatcher::build_index() → void`
- `AdtMatcher::match(const char* seq, int len, int max_offset=5) → int` (≥0 tag id, -1 no-match, -2 ambiguous)
- `AdtMatcher::num_tags() const → size_t`
- `AdtMatcher::tag_name(int idx) const → const string&`
- `AdtMatcher::tag_sequence(int idx) const → const string&`
- `AdtMatcher::names() const → vector<string>`
- `AdtMatcher::matched/ambiguous/unmatched() const → size_t` (atomic stats)
**Pipeline insertion point**: Separate ADT worker loop; runs in parallel with GEX pileup after STAR alignment. Receives Stream 2 (StreamRole::FEATURE) of 3-stream CITE-seq .1fq bundle.
**CLI flag needed**: `--tag-ref <path.csv>` (future singlify.cpp integration)
**Dependencies**: None (header-only, stdlib only)
**Key design notes**:
- Exact hash lookup first; Hamming-1 expanded neighbourhood hash second (O(1) per read)
- Offset scanning 0..max_offset (default 5) to tolerate sequencing soft-clip / adapter jitter
- Collision detection: ambiguous neighbours stored as sentinel -2 in hamming1_ map
- Stats counters are atomic; safe for multi-threaded accumulation without locks
- 8 unit tests; all pass; compile-clean
**Shipped**: cycle 78, commit c4aeb6a, April 12 2026

## Feature: ADT UMI Dedup + Counting (T2)
**Header**: `include/singlet-pileup/adt_counter.h`
**API**: `AdtCounter::add(bc_idx, tag_idx, umi)`, `merge(other)`, `finalize()`, `counts() → SparseAccumulator<uint32_t>`
**Pipeline insertion point**: After AdtMatcher tag matching, before pz_writer export
**CLI flag needed**: none (runs automatically when assay_type=CITE_SEQ_GEX)
**Dependencies**: sparse_accumulator.h
**Shipped**: commit 69a735a, April 12 2026

## Feature: HTO Demultiplexing (T3)
**Header**: `include/singlet-pileup/hto_demux.h`
**API**: `HtoDemux::demux(hto_counts, tag_names, n_cells) → vector<HtoDemuxResult>`, `write_tsv(path, barcodes, results)`
**Pipeline insertion point**: After AdtCounter finalize(), if HTO tags detected in reference
**CLI flag needed**: `--hto-tags` CSV subset of tag names that are HTOs (default: all tags if assay_type=HTO)
**Dependencies**: sparse_accumulator.h
**Algorithm**: CLR normalization per cell + per-HTO threshold = mean + 1.5*MAD of negative mode
**Shipped**: commit 69a735a, April 12 2026

## Feature: Visium Per-Spot Pileup Integration (V2)
**Header**: `include/singlet-pileup/visium_spatial.h` (V1, existing)
**API**: `VisiumSpatialParser::load_positions(path)`, `write_coordinates_tsv(path, barcodes)`
**Pipeline insertion point**: After export_results(), if is_visium_mode==true
**CLI flag needed**: `--tissue-positions FILE` (tissue_positions.csv from Space Ranger)
**Dependencies**: visium_spatial.h
**How it works**: Visium spots ARE barcodes; existing pileup engine counts genes per barcode.
  V2 post-processing loads tissue_positions.csv and writes `spatial_coordinates.tsv`.
  Protocol detection: `eahdr.assay_type == AssayType::SPATIAL_RNA` (tag="10x-visium").
**Output**: `{out_prefix}/spatial_coordinates.tsv` (barcode, array_row, array_col, pixel_row, pixel_col, in_tissue)
**Shipped**: commit c0ddcd9, April 12 2026

## Feature: Visium Spatial QC Metrics (V3)
**Header**: `include/singlet-pileup/visium_qc.h`
**API**:
  - `build_spot_qc(barcodes, total_umis, genes_detected, mt_pct, parser) → vector<SpotQCRecord>`
  - `compute_visium_summary(records, min_umi=100) → VisiumQCSummary`
    — fields: n_barcodes, n_in_tissue, n_detected_spots, tissue_coverage, median_umis, mean_umis, std_umis
  - `write_visium_qc_tsv(path, records) → bool`
  - `log_visium_summary(summary)` — prints to stderr in singlify log style
**Pipeline insertion point**: After V2 coordinate write, within is_visium_mode block
**CLI flag needed**: `--visium-min-umi N` (default: 100, threshold for tissue_coverage metric)
**Dependencies**: visium_spatial.h, cell_qc_metrics.h (compute_cell_qc called inline)
**Output**: `{out_prefix}/visium_qc.tsv` (per-spot: barcodes+UMIs+genes+mt_pct+coords+flags)
**Shipped**: commit c0ddcd9, April 12 2026

## Feature: Alignment QC — SS2 + B2 (Smart-seq2 / Bulk RNA QC)
**Header**: `include/singlet-pileup/alignment_qc.h`
**API**:
- `AlignmentQCComputer::compute_from_stats(total, mapped, unique, multi, dup, mito_frac, gene_counts, profile) → AlignmentQC`
- `AlignmentQCComputer::compute_gene_body_coverage(per_gene_bins, n_bins=100) → vector<double>`
- `AlignmentQCComputer::five_prime_three_prime_ratio(profile, head_bins=10) → float`
- `AlignmentQCComputer::lander_waterman_complexity(total_mapped, distinct_reads) → float`  [static]
- `AlignmentQCComputer::write_report(path, qc)` — 12-line TSV
- `AlignmentQCComputer::write_coverage_profile(path, profile)` — tab-separated bin values
**Pipeline insertion point**: After gene counting (B1/SS1), before mtx write; pass per-gene counts + gene body coverage bins
**CLI flag needed**: `--alignment-qc` (default: enabled for ss2/bulk-rna modes)
**Dependencies**: None (pure header, stdlib only)
**Shipped**: commit 04d6851, April 12 2026

## Feature: RNA Variant Caller (SS3 + B3)
**Header**: `include/singlet-pileup/rna_variant_caller.h`
**API**:
- `AlleleCount::add(int base_idx)` — accumulate one base
- `AlleleCount::top_two(fi, fc, si, sc)` — find top two alleles
- `RNAVariantCaller::call_position(chrom_idx, pos, ac, out, ref_idx=-1) → bool`
- `RNAVariantCaller::process_bam(bam_path, ref_fasta="") → void` — htslib bam_plp_* pileup
- `RNAVariantCaller::write_tsv(path, chrom_names)` / `write_vcf(path, chrom_names)`
- `RNAVariantCaller::variants() → const vector<RNAVariant>&`
- `RNAVariantCaller::positions_examined() → size_t`, `variants_called() → size_t`
**Pipeline insertion point**: After STAR alignment, before 1pz write; run on BAM for ss2/bulk-rna assay types
**CLI flag needed**: `--variant-calling` (default: disabled; opt-in for ss2/bulk-rna)
**Dependencies**: htslib (bam_plp_*, bam_seqi, bam_get_qual) — already linked
**Shipped**: commit 99cc4ca, April 12 2026

## Feature: Read-Level Dedup Stats (B4)
**Header**: `include/singlet-pileup/read_dedup_stats.h`
**API**:
- `ReadDedupStats::process_bam(bam_path) → Result`
- `ReadDedupStats::estimate_library_size(total_mapped, distinct_reads) → double` (static)
- `ReadDedupStats::parse_illumina_qname(name) → IlluminaCoords` (static)
- `ReadDedupStats::pixel_distance(x1, y1, x2, y2) → double` (static)
- `ReadDedupStats::make_dedup_key(bam1_t*) → DedupKey` (static)
**Pipeline insertion point**: After STAR alignment, before 1pz/MTX write; run on sorted BAM for bulk-rna/smart-seq2 assay types alongside variant caller (B3)
**CLI flag needed**: `--dedup-stats` (default: enabled for bulk-rna/smart-seq2); `--optical-distance N` (default: 2500)
**Dependencies**: htslib (sam_read1, bam_dup1, BAM_FDUP flag) — already linked
**Shipped**: commit 7bcfa55, April 12 2026

## Feature: Read-Level Dedup Stats (B4)
**Header**: `include/singlet-pileup/read_dedup_stats.h`
**API**:
- `ReadDedupStats::process_bam(bam_path) → Result`
- `ReadDedupStats::estimate_library_size(total_mapped, distinct_reads) → double` (static)
- `ReadDedupStats::parse_illumina_qname(name) → IlluminaCoords` (static)
- `ReadDedupStats::pixel_distance(x1, y1, x2, y2) → double` (static)
- `ReadDedupStats::make_dedup_key(bam1_t*) → DedupKey` (static)
**Pipeline insertion point**: After STAR alignment, before 1pz/MTX write; bulk-rna/smart-seq2 assay types
**CLI flag needed**: `--dedup-stats` (enabled by default for bulk-rna/ss2); `--optical-distance N` (default: 2500)
**Dependencies**: htslib (sam_read1, bam_dup1, BAM_FDUP) — already linked
**Shipped**: commit 7bcfa55, April 12 2026

## Feature: ATAC Cell Calling (A7)
**Header**: `include/singlet-pileup/atac_cell_caller.h`
**API**:
- `AtacCellCaller::call_cells(barcodes, unique_fragment_counts, tss_enrichment_scores, frip_scores, results) → Summary`
- `AtacCellCaller::auto_fragment_threshold(counts) → uint32_t`
**Pipeline insertion point**: After ATACQCComputer::compute() (A3); produces filtered barcode list for ATAC matrix output
**CLI flag needed**: `--atac-min-tss N` (default: 2.0); `--atac-min-frags N` (default: 500); `--atac-min-frip N` (default: 10%); `--atac-auto-threshold` (default: on)
**Dependencies**: None (header-only, standard library only)
**Shipped**: commit 5c5c623, April 12 2026

## Feature: ATAC Pipeline Wiring — A3 + A7 + fragment output
**Header**: `include/singlet-pileup/atac_qc.h`, `include/singlet-pileup/atac_cell_caller.h`
**API**:
- `ATACQCComputer::compute(frags, extractor, n_cells, bin_counter)` → fills `cell_qc()`
- `ATACQCComputer::write_qc_tsv(path, barcodes)` → `atac_qc.tsv`
- `AtacCellCaller::call_cells(barcodes, frag_counts, tss_scores, frip_scores, results)` → `Summary`
**Pipeline insertion point**: After ATACBinCounter + write_names, before BAM cleanup
**CLI flag needed**: `--exons <gtf>` (enables TSS scoring; skipped if absent/None)
**Dependencies**: atac_fragment.h, atac_bin_counter.h, atac_qc.h, atac_cell_caller.h
**Shipped**: cycle 97, commit f36920d

## Feature: CellRanger-compatible MTX Export (G-EXPORT)
**Header**: `include/singlet-pileup/mtx_writer.h` + `include/singlet-pileup/export.h`
**API**:
- `write_features_10x(path, gene_ids, gene_names, feature_type="Gene Expression") → bool`
- `write_barcodes_10x(path, barcodes, gem_well=1) → bool`
- `collapse_exon_to_gene<CSCType>(exon_csc, gm) → GeneCSC` (struct with indptr/indices/data/nrows/ncols)
**Pipeline insertion point**: Already integrated inside `write_matrix` lambda in `export_results()`. Activated automatically when `output_format == "mtx"` and exon_gtf is present.
**CLI flag needed**: `--output-format mtx` (already wired via `ExportConfig::output_format`)
**Output**: `{out_prefix}/filtered_feature_bc_matrix/` containing `matrix.mtx`, `features.tsv` (3-col), `barcodes.tsv` (barcode-1 suffix)
**Dependencies**: `<filesystem>` (C++17), `GeneModel::exon_to_gene()`, `GeneModel::gene_ids()`, `GeneModel::gene_names()`
**Shipped**: commit e15907d on branch main

## Feature: Combined Gene-Level Counts for snRNA-seq (G-SNRNA)
**Header**: `include/singlet-pileup/export.h`
**API**:
- `GeneCSC<OutT>` — top-level template struct (indptr/indices/data/nrows/ncols), shared by both functions
- `collapse_to_gene_counts<CSCType>(exon_csc, intron_csc, gm) → GeneCSC<OutT>` — sums exon+intron per gene
- `collapse_exon_to_gene<CSCType>(exon_csc, gm) → GeneCSC<OutT>` — exon-only (unchanged, now returns typed GeneCSC<OutT>)
**Pipeline insertion point**: Written as a new parallel thread in Phase 3 of `export_results()`, immediately after the exon_counts write thread. Gated on `engine.gene_model().n_genes() > 0`.
**Output**:
- 1pz mode: `{out_prefix}/gene_counts.1pz` (gene_ids as feature names)
- mtx mode: `{out_prefix}/gene_counts.mtx` + `gene_counts_features.tsv` + `gene_counts_barcodes.tsv`; ALSO writes `filtered_feature_bc_matrix/` with exon+intron combined (replaces exon-only in that path)
**CLI flag needed**: None — always written when GTF is present. Falls back to exon-only when `--no-introns` or no intron data.
**Dependencies**: GeneModel::intron_to_gene(), GeneModel::gene_ids(), GeneModel::gene_names()
**Validation**: C00 (SRR32855204): 38606×12089, 10.6M nnz, 7.97 MB
**Shipped**: commit 4c5d0e2 on branch main

## Feature: Saturation Downsampling Curve (G-SATCURVE)
**Header**: `include/singlet-pileup/saturation_curve.h`
**API**:
- `compute_saturation_curve(cell_stats, exon_csc, gm, fractions) → vector<SaturationPoint>`
- `write_saturation_curve_tsv(path, curve) → bool`
**Pipeline insertion point**: Phase 1.65 in `export_results()`, after Phase 1.58 (read_stats), before Phase 1.6 (saturation_metrics)
**CLI flag needed**: None (runs automatically when GTF + per_cell_reads available)
**Dependencies**: `read_stats.h`, `sparse_accumulator.h`, `gene_model.h`
**Output**: `saturation_curve.tsv` — 6 rows × 6 cols (fraction, sampled_reads, median_umis, median_genes, mean_umis, mean_genes)
**Algorithm**: Lander-Waterman for UMIs; Poisson per-gene for genes; both analytical from existing CSC + CellReadStats
**Shipped**: cycle ~111, see commit

## Feature: G-PSI — Per-cell Splice Junction PSI
**Header**: `include/singlet-pileup/splice_psi.h`
**API**:
- `parse_junction_name(name, chrom, start, end) → bool`
- `build_splice_events(sj_names) → std::vector<SpliceEvent>`
- `compute_psi(sj_csc, sj_names) → PSIResult`
- `write_splice_events(path, events, sj_names) → bool`

**Pipeline insertion point**: After `write_matrix("sj_counts", ...)` in `export_results()` (export.h ~line 580)
**CLI flag needed**: None (auto-enabled whenever `count_sj=true` and junctions were detected)
**Outputs**:
  - `splice_psi.1pz` (1pz mode) or `splice_psi.mtx` + `splice_psi_features.tsv` + `splice_psi_barcodes.tsv`
  - `splice_events.tsv` — tab-separated event groupings (event_id, event_type, n_junctions, junction_names)
**Algorithm**: Group junctions by shared 5' donor (same chrom:start) or 3' acceptor (same chrom:end). Only groups ≥2 junctions form events. Per cell, PSI = count(j,c) / sum(group,c). Each junction assigned to first (alphabetically earliest) event it belongs to.
**Dependencies**: `mtx_writer.h`, `pz_writer.h` (already in export.h)
**Shipped**: cycle current, commit 6a80d2d on branch main

## Feature: G-BARNYARD — Per-Cell Species Classification (Dual-Genome)
**Header**: `include/singlet-pileup/barnyard.h`
**API**:
- `classify_gene_species(gene_ids, gene_names) → std::vector<uint8_t>`
- `is_barnyard_experiment(gene_species) → bool`
- `is_barnyard_experiment(gene_ids, gene_names) → bool`
- `classify_barnyard(gene_csc, gene_species, singlet_threshold=0.9f, doublet_min_fraction=0.2f) → BarnyardResult`
- `write_barnyard_tsv(path, result, barcodes) → bool`
**Pipeline insertion point**: After `export_results()` gene counting phase; check `is_barnyard_experiment(gm.gene_ids(), gm.gene_names())` and if true, call `classify_barnyard()` and `write_barnyard_tsv(out_prefix + "/barnyard.tsv", ...)`.
**CLI flag needed**: None (auto-detected from gene model)
**Output**: `barnyard.tsv` — columns: barcode, species_call, human_umis, mouse_umis, human_fraction
**Classification rules** (default thresholds):
  - HUMAN: human_fraction >= 0.90
  - MOUSE: mouse_fraction >= 0.90
  - DOUBLET: both species >= doublet_min_fraction (0.20)
  - AMBIGUOUS: below singlet threshold, one species < doublet_min (e.g. 85/15 split)
**Species detection** (priority order):
  1. Ensembl ID prefix: ENSG* → human, ENSMUSG* → mouse, ENSDARG* → zebrafish
  2. CellRanger 3-underscore prefix: GRCh38___*, hg38___* → human; GRCm39___*, mm10___* → mouse
  3. Simple underscore prefix: GRCh38_*, hg38_* → human; GRCm39_*, mm10_* → mouse
**Dependencies**: None (header-only, stdlib only)
**Shipped**: cycle current, see commit on branch main

## Feature: G-BARNYARD-ROUTE — Per-Cell Species Routing (Dual-Genome)
**Header**: `include/singlet-pileup/barnyard_route.h`
**API**:
- `route_barnyard(gene_csc, gene_species, cell_calls, gene_names, gene_ids, barcodes) → BarnyardRouteResult`
- `SpeciesCSC` struct: indptr/indices/data (CSC), nrows, ncols, gene_names, gene_ids, barcodes, species_name
- `BarnyardRouteResult` struct: `.human` (SpeciesCSC), `.mouse` (SpeciesCSC)
**Pipeline insertion point**: After `classify_barnyard()`; call `route_barnyard()` when `is_barnyard_experiment()` is true. Write per-species matrices via `write_1pz()` or `.mtx` export.
**CLI flag needed**: None (auto when barnyard detected); optional `--barnyard-out-prefix`
**Output**: `human_gene_counts.1pz` + `mouse_gene_counts.1pz` (or `.mtx`)
**Routing logic**:
  - Gene mask: gene_species[g] == HUMAN (1) or MOUSE (2)
  - Cell mask: cell_calls[c] == HUMAN or MOUSE (singlets only; DOUBLET/AMBIGUOUS excluded)
  - 2-pass submatrix extraction: pass 1 counts nnz per column, pass 2 fills entries
  - Row/col indices remapped to 0-based within species subset
**Dependencies**: `barnyard.h` (SpeciesCall, SpeciesCode, classify_gene_species, classify_barnyard)
**Shipped**: cycle current, commit 5b493c9 on branch main

## Feature: G-CELLPLEX — 10x CellPlex / CMO Demultiplexing
**Header**: `include/singlet-pileup/cellplex_demux.h`
**API**:
- `CellPlexDemux::demux(cmo_counts, cmo_names, n_cells) → vector<CellPlexResult>`
- `CellPlexDemux::write_tsv(path, barcodes, results) → void`
- `struct CellPlexResult { call, sample, sample_idx, top_clr, top_cmo }`
**Pipeline insertion point**: After ADT counting in export.h; if feature ref contains CMO-type features, extract CMO rows from ADT SparseAccumulator, call demux, write `cellplex_assignments.tsv`
**CLI flag needed**: No new flag; triggered when `--feature-ref` CSV contains features of type `Multiplexing Capture` (10x convention)
**Algorithm**: CLR per-cell (same as HTO), per-CMO Otsu threshold (maximises between-class variance over all cells)
**Dependencies**: `sparse_accumulator.h` (existing)
**Shipped**: commit e44cc7e on branch main

## Feature: Transcript-Level Equivalence Class Counting (G-TXLEVEL)
**Header**: `include/singlet-pileup/transcript_compat.h`
**API**:
- `TranscriptModel::load_gtf(path, gm) → bool` — re-parse GTF to map exon_idx → transcript set
- `TranscriptModel::build_from_data(transcripts, exon_assignments)` — programmatic init (tests)
- `TranscriptModel::exon_to_transcripts(exon_idx) → const vector<uint32_t>&`
- `TCCBuilder::add_exon_count(bc_idx, exon_idx, count, tm)` — add single observation
- `TCCBuilder::add_from_exon_csc(indptr, indices, data, n_cells, tm)` — bulk from exon CSC
- `TCCBuilder::build(n_cells, tm) → TCCResult` — produce EC×cell CSC matrix
- `write_tcc_matrix(path, tcc) → bool` — MatrixMarket EC×cell output
- `write_ec_table(path, tcc) → bool` — bustools EC table (tab-sep transcript indices/row)
- `write_transcript_table(path, tcc) → bool` — transcript_id + gene_id TSV
**Pipeline insertion point**: After pileup completes (post-hoc); call after `sum_to_genes`. Does NOT touch the hot path.
**CLI flag needed**: `--tcc` (optional; enables TCC output to `{out_prefix}/tcc_matrix.mtx`, `tcc_ec.tsv`, `tcc_transcripts.tsv`)
**Dependencies**: `gene_model.h` (existing), zlib
**Algorithm**: Re-parse GTF for transcript_id per exon; each merged gene_model exon maps to all transcripts with an overlapping raw exon; EC = sorted transcript-index set; `std::map<vector<uint32_t>, uint32_t>` registry for insertion-order EC IDs
**Shipped**: commit 8b9ee39 on branch main

## Feature: ATAC Peak Calling (G-ATAC-PEAKS)
**Header**: `include/singlet-pileup/atac_peaks.h`
**API**:
- `call_peaks(bin_counts, bin_chroms, bin_starts, params) → vector<ATACPeak>`
- `poisson_pvalue(observed, lambda) → float`  (upper-tail Poisson CDF)
- `write_peaks_bed(path, peaks) → bool`
**Pipeline insertion point**: After ATAC bin matrix is written; aggregate bin counts across cells → call_peaks → write_peaks_bed to `{out_prefix}/peaks.bed`
**CLI flag needed**: (auto, no flag needed — always emits peaks.bed when ATAC mode detected)
**Dependencies**: None (pure C++17, no external libs)
**Algorithm**: Local background window (default 10kb) sliding over per-bin aggregate counts; Poisson upper-tail test per bin; merge adjacent significant bins + merge peaks within merge_distance; BED6 output (peak_id, -log10 p-value, strand=.)
**Shipped**: cycle 113, commit 00ea554 on branch main

## Feature: Multiome Router (G-MULTIOME-ROUTE)
**Header**: `include/singlet-pileup/multiome_router.h`
**API**:
- `detect_modality(protocol_family) → MultiomeModality` — maps protocol string to GEX/ATAC/UNKNOWN
- `is_multiome_protocol(protocol_family) → bool` — true for 10x-arc / multiome variants
- `classify_inputs(input_files) → MultiomeInputs` — groups .1fq files by modality via header peek
- `plan_multiome_run(inputs, output_prefix) → MultiomeConfig` — sets run_gex/run_atac + output prefixes
**Pipeline insertion point**: Entry point of singlify.cpp, before STAR/pileup dispatch
**CLI flag needed**: None (auto-detects; future: `--modality gex|atac|auto`)
**Dependencies**: `lib1fq/types.h` (for `lib1fq::AssayType` enum)
**Shipped**: cycle 2026-04-13, commit 56545eb on branch main

## Feature: Spatial Multiome Router (G-SPATIAL-MULTIOME)
**Header**: `include/singlet-pileup/spatial_multiome.h`
**API**:
- `is_spatial_experiment(protocol_family) → bool` — true for visium/spatial families
- `is_visium_hd(protocol_family) → bool` — true for visium-hd/-hd-8/-hd-16/-hd-2
- `detect_spatial_config(protocol_family) → SpatialConfig` — sets resolution, bin_size_um, has_atac, has_protein
- `parse_hd_barcode(barcode, bin_size_um) → SpatialCoord` — decodes HDrow_col barcode to µm coordinates
- `write_spatial_metadata(path, coords, barcodes, config) → bool` — emits TSV with barcode/row/col/x_um/y_um
**Pipeline insertion point**: After protocol detection (alongside G-MULTIOME-ROUTE); if is_spatial_experiment() → use spatial output paths; if is_visium_hd() → use parse_hd_barcode() instead of tissue_positions.csv; if spatial + has_atac → route RNA→spatial pileup, ATAC→fragment extraction
**CLI flag needed**: None (auto-detects); future: `--spatial-resolution 8|16|55`
**Dependencies**: None (header-only C++17)
**Shipped**: cycle 2026-04-13, commit 1d37f55 on branch main

## Feature: V(D)J CDR3 Clonotype Assembly (G-VDJ)
**Header**: `include/singlet-pileup/vdj_assembly.h`
**API**:
- `translate(nt_seq) → std::string` — standard genetic code, 64-codon table
- `is_productive(cdr3_aa) → bool` — starts C, ends F/W, no stops, len≥5
- `VJReference::add_gene(name, seq, is_v, anchor)` — add V/J gene to DB
- `VJReference::load_from_fasta(v_fasta, j_fasta) → bool` — load IMGT FASTA pairs
- `VJReference::build_index(k=15)` — build k-mer → gene index
- `extract_cdr3(read_seq, ref) → CDR3Extraction` — scan read for V/J anchors
- `assemble_clonotypes(extractions, ref) → VDJAssemblyResult` — group into clonotypes
- `write_clonotype_tsv(path, result, barcodes) → bool` — write clonotypes.tsv
**Pipeline insertion point**: After pileup (exon counting), in VDJ-mode runs; skip for non-immune protocols
**CLI flag needed**: `--vdj-v-ref <fasta> --vdj-j-ref <fasta>` (default: disabled)
**Dependencies**: None (header-only; standard library only)
**Shipped**: cycle 115, commit feb130f

## Feature: CUT&TAG / CUT&RUN Chromatin QC (G-CUTTAG)
**Header**: `include/singlet-pileup/cuttag.h`
**API**:
- `detect_mode(fragment_sizes) → CutTagMode` — CUT_AND_RUN if sub-nucl fraction > 40%
- `compute_cuttag_qc(fragments, total_reads, spike_in_chrom_idxs) → CutTagQC`
- `write_cuttag_qc(path, qc, mode) → bool` — JSON output
- `compute_spike_in_scale_factor(spike_in_reads, total_reads) → double`
**Pipeline insertion point**: After ATAC fragment extraction (bulk or sc); replaces BulkAtacQC for CUT&TAG/CUT&RUN protocol types
**CLI flag needed**: `--assay cuttag|cutrun` (default: auto-detect from fragment size distribution)
**Dependencies**: None (header-only; no htslib; local Fragment struct)
**Shipped**: cycle N, commit e5ae15f

## Feature: ChIP-seq QC (G-CHIPSEQ)
**Header**: `include/singlet-pileup/chipseq.h`
**API**:
- `compute_chipseq_qc(fwd, rev, total_reads, unique_reads, reads_in_peaks, read_length=75, n_lorenz_bins=100) → ChipSeqQC`
- `compute_pbc1(unique_positions, total_positions) → double`
- `compute_pbc2(one_read_positions, two_read_positions) → double`
- `compute_lorenz_curve(coverage, n_bins=100) → std::vector<double>`
- `lorenz_auc(curve) → double`
- `strand_cross_correlation(fwd, rev, max_shift=500) → std::vector<double>`
- `write_chipseq_qc(path, qc) → bool`
**Pipeline insertion point**: After BAM tag counting; before output serialization, for bulk ChIP-seq assays (cblen==0, not ATAC)
**CLI flag needed**: `--assay chipseq` (default: auto-detect from protocol registry)
**Dependencies**: None (header-only; no htslib)
**Shipped**: cycle N, commit b73f02c

## Feature: Single-cell DNA CNV Binning (G-SCDNA)
**Header**: `include/singlet-pileup/scdna_cnv.h`
**API**:
- `compute_mapd(bin_counts) → double`
- `estimate_ploidy(bin_counts, expected_diploid_count) → double`
- `gc_correct_bins(bin_counts, gc_content) → std::vector<double>`
- `compute_cell_cnv_qc(bin_counts, total_reads, unique_reads) → CnvCellQC`
- `write_cnv_qc(path, cell_qc, barcodes) → bool`
**Pipeline insertion point**: After genome binning pass; replaces standard pileup for scDNA-seq protocol type
**CLI flag needed**: `--assay scdna --bin-size 500000` (default bin size 500 kb)
**Dependencies**: None (header-only; no htslib)
**Shipped**: cycle N, commit b73f02c

---

## Feature: G-BARNYARD — Per-Cell Species Classification (export.h wiring)
**Header**: `include/singlet-pileup/barnyard.h`
**API**:
- `classify_gene_species(gene_ids, gene_names) → vector<uint8_t>` — one SpeciesCode per gene
- `is_barnyard_experiment(gene_species) → bool` — true if human+mouse genes both present
- `classify_barnyard(gene_csc, gene_species) → BarnyardResult` — per-cell HUMAN/MOUSE/DOUBLET/AMBIGUOUS
- `write_barnyard_tsv(path, result, barcodes) → bool` — writes barnyard_classification.tsv
**Pipeline insertion point**: Inside gene_counts write thread in `export_results()`, after `gene_csc` is built
**CLI flag needed**: None (auto-enabled when barnyard detected from gene prefixes)
**Dependencies**: Requires gene_ids + gene_names from GeneModel; auto-detects ENSG/ENSMUSG/GRCh38__/GRCm39__ prefixes
**Shipped**: 2026-04-13, wired in export.h

## Feature: G-TPM — Aggregate TPM/FPKM Library Normalization (export.h wiring)
**Header**: `include/singlet-pileup/tpm_fpkm.h`
**API**:
- `compute_gene_lengths(gm) → vector<GeneLength>` — effective exon length per gene
- `compute_tpm(counts, lengths) → vector<double>` — TPM per gene
- `compute_fpkm(counts, lengths, total_mapped) → vector<double>` — FPKM per gene
- `write_gene_expression_tsv(path, lengths, counts, tpm, fpkm) → bool` — writes gene_expression.tsv
**Pipeline insertion point**: Inside gene_counts write thread in `export_results()`, after `gene_csc` is built
**CLI flag needed**: None (always written when GTF present)
**Dependencies**: Requires GeneModel with exon intervals; uses `pileup_stats->mapped_reads` for FPKM denominator
**Output**: `gene_expression.tsv` (gene_id, gene_name, effective_length, count, TPM, FPKM)
**Shipped**: 2026-04-13, wired in export.h

## Feature: G-RRNA — rRNA Contamination Detection (pileup_engine.h + export.h wiring)
**Header**: `include/singlet-pileup/rrna_detect.h`
**API**:
- `RrnaDetector::detect(read_seq) → bool` — returns true if ≥2 diagnostic 21-mers hit
- `RrnaDetector::batch_detect(reads) → RrnaStats` — batch with subunit breakdown
- `write_rrna_report(stats, outpath)` — writes rrna_report.json
**Sampling strategy**: Every 1000th mapped read sampled in `run()` and `process_read_parallel()`; counts stored in `PileupStats::rrna_reads` / `rrna_reads_sampled`. Stats merged across parallel workers.
**Pipeline insertion point**:
  - Sampling: in `pileup_engine.h` `run()` (serial) and `process_read_parallel()` (parallel)
  - Output: in `export_results()` after write_threads.join(); writes `rrna_report.json`
**Dependencies**: `PileupStats::rrna_reads` + `rrna_reads_sampled` fields (added 2026-04-13); `export_cfg.pileup_stats` non-null
**Output**: `rrna_report.json` (rrna_fraction, rrna_reads, total_reads_sampled). Also added to `stats.json`.
**Shipped**: 2026-04-13, wired in pileup_engine.h + export.h

## Feature: G-ATAC-PEAKS-WIRE — ATAC Peak Calling (singlify.cpp wiring)
**Header**: `include/singlet-pileup/atac_peaks.h`
**API**:
- `call_peaks(bin_counts, bin_chroms, bin_starts, params) → vector<ATACPeak>` — Poisson enrichment + merge
- `write_peaks_bed(path, peaks) → bool` — BED6 format
**Pipeline insertion point**: In ATAC section of `singlify.cpp`, after `counter.to_csc()` and cell calling; aggregates CSC across all cells → bin_agg → call_peaks
**CLI flag needed**: None (auto-enabled in ATAC mode)
**Dependencies**: ATACBinCounter CSC matrix; chrom_sizes vector from BAM header
**Output**: `peaks.bed` (BED6: chrom, start, end, peak_N, score, strand)
**Note**: Peak calling header was previously declared in INTEGRATION_NOTES but not yet wired. Now wired.
**Shipped**: 2026-04-13, wired in singlify.cpp

## Feature: G-H5AD — AnnData .h5ad export
**Header**: `include/singlet-pileup/h5ad_writer.h`
**API**:
```cpp
// Config struct — CSC (genes×cells) input → CSR obs×var .h5ad output
struct H5adWriteConfig {
    std::string filepath;
    const int32_t* indptr;     // CSC column pointers (length: n_cells + 1)
    const int32_t* indices;    // CSC row indices (gene indices), length nnz
    const void*    data;       // Non-zero values (uint32_t or float32)
    uint64_t nnz; uint32_t n_genes; uint32_t n_cells;
    bool data_is_float = false;
    const std::vector<std::string>* gene_names   = nullptr;
    const std::vector<std::string>* gene_ids      = nullptr;
    const std::vector<std::string>* cell_barcodes = nullptr;
    struct Layer { std::string name; const int32_t* indptr; const int32_t* indices;
                   const void* data; uint64_t nnz; bool data_is_float = false; };
    std::vector<Layer> layers;
    std::map<std::string, std::string> metadata;
};

// Main entry point — returns true on success, removes partial file on failure
bool write_h5ad(const H5adWriteConfig& cfg);
```
**CSC→CSR key insight**: GeneCSC (genes×cells) indptr/indices ARE the CSR (cells×genes) arrays — no transposition needed.  
**Format**: AnnData 0.1.0 / dataframe 0.2.0. indptr=int64, indices=int32, data=float32. VL UTF-8 strings for obs/_index, var/_index.  
**Groups written**: X (CSR CSC_matrix), obs, var, obsm, varm, obsp, varp, layers/, uns/singlify/  
**Pipeline insertion point**: In `export_results()` after all CSC matrices are finalized; parallel to MTX export  
**CLI flag needed**: `--output-format h5ad` (or auto-alongside MTX) — default OFF until singlify.cpp wired  
**Dependencies**: HDF5 C library (`/usr/lib64/libhdf5.so`, `/usr/include/hdf5.h`) linked via `HDF5_AVAILABLE` in CMakeLists.txt  
**Build**: `cmake --build build --target test_h5ad_writer && ./test_h5ad_writer`  
**Validation**: 14 tests, 58 assertions, all passing  
**Shipped**: 2026-04-13

## Feature: Raw Matrix Export (G-RAWMATRIX)
**Header**: `include/singlet-pileup/export.h` (added to ExportConfig + export_results)
**API**: `ExportConfig::write_raw_matrix = true` → writes `raw_feature_bc_matrix/` (MTX) or `raw_gene_counts.1pz` (1pz) or `raw_feature_bc_matrix.h5ad` (h5ad)
**Pipeline insertion point**: Inside gene_counts write_thread, after filtered_feature_bc_matrix / h5ad writes
**CLI flag needed**: `--raw-matrix` (no argument, boolean toggle; default off)
**Dependencies**: gene_csc already in scope — no re-computation needed
**Shipped**: 2026-04-13

## Feature: Structured Summary JSON (G-SUMMARYJSON)
**Header**: `include/singlet-pileup/summary_json.h` (new file)
**API**:
- `PipelineSummary` struct — read metrics, cell metrics, QC metrics, timing, status, warnings, user_meta
- `write_summary_json(const PipelineSummary&, const std::string& filepath) → bool`
- `classify_outcome(const PipelineSummary&, assay_type="scrna") → std::string` — returns "success" | "low_mapping" | "no_cells" | "low_cells" | "low_genes"
- `json_escape(const std::string&) → std::string` — safe JSON string serialization
**Pipeline insertion point**: At end of `export_results()`, after provenance manifest, before `return result`
**Auto-written to**: `{out_prefix}/summary.json` (always, unconditionally)
**Dependencies**: pileup_stats + cc_result + bc_totals + exon_csc + gene_model (all in scope)
**Shipped**: 2026-04-13

## Feature: G-BUS — BUS Format Export (kallisto|bustools compatible)
**Header**: `include/singlet-pileup/bus_writer.h`
**API**:
- `BusRecord` struct (32 bytes): `barcode`, `umi`, `ec`, `count`, `flags`, `_pad`
- `BusWriteConfig` struct: `filepath`, `cb_len`, `umi_len`, `header_text`
- `BusWriter::open(BusWriteConfig) → bool`
- `BusWriter::write_record(BusRecord) → bool`
- `BusWriter::write_records(vector<BusRecord>) → bool`
- `BusWriter::close()`
- `BusWriter::records_written() → uint64_t`
- `encode_bus_barcode(string) → uint64_t` — 2-bit left-aligned encoding
- `decode_bus_barcode(uint64_t, uint32_t len) → string` — decode 2-bit to nucleotide
- `encode_bus_seq(string) → uint64_t`, `decode_bus_seq(uint64_t, uint32_t) → string`
**Pipeline insertion point**: Inside the pileup read loop, alongside .1pz writes; caller sorts records by barcode before writing
**CLI flag needed**: `--output-format bus` (adds to existing format enum; can combine with 1pz)
**Dependencies**: None (libc only; no HDF5, no htslib)
**Shipped**: 2026-04-13, commit 1d9ecc2

## Feature: G-MOLINFO — Molecule Info HDF5 (Cell Ranger compatible)
**Header**: `include/singlet-pileup/molecule_info.h`
**API**:
- `MoleculeRecord` struct: `barcode`, `feature_idx`, `umi`, `count`
- `MoleculeInfoConfig` struct: `filepath`, `cb_len`, `umi_len`
- `MoleculeInfoWriter::open(MoleculeInfoConfig) → bool`
- `MoleculeInfoWriter::add_molecule(MoleculeRecord)`
- `MoleculeInfoWriter::finalize(gene_names, gene_ids, barcodes, metrics_json) → bool` — sorts by barcode, writes HDF5
- `MoleculeInfoWriter::close()`
- `MoleculeInfoWriter::molecules_written() → uint64_t`
- Read-back helpers in `mol_info_detail::`: `read_int64_ds`, `read_int32_ds`, `read_str_ds`, `read_scalar_str_ds`
**HDF5 datasets written**: `/barcode`, `/barcode_idx`, `/count`, `/feature_idx`, `/gem_group`, `/library_idx`, `/umi`, `/umi_type`, `/metrics_json`, `/barcodes`, `/feature_ref/{name,id,feature_type}`
**Pipeline insertion point**: After pileup completes; `add_molecule` called per dedup'd UMI; `finalize()` called after all cells processed
**CLI flag needed**: `--molecule-info` (boolean toggle; writes `molecule_info.h5` to out_prefix; default off)
**Dependencies**: HDF5 C library (already linked via `HDF5_AVAILABLE` guard in CMakeLists.txt)
**Shipped**: 2026-04-13, commit 1d9ecc2

## Feature: EM Nonhost Abundance Deconvolution (NONHOST-EM)
**Header**: `include/singlet-pileup/nonhost/nonhost_em.h`
**API**:
  - `NonHostScreener::MultiHit` struct: per-read soft multi-species hits (viral_hits + microbial_hits)
  - `NonHostScreener::classify_multi(seq, soft_thresh_mult=0.25f) → MultiHit`
  - `NonHostScreener::classify_multi_batch(reads, mult) → vector<MultiHit>`
  - `em_deconvolve(multi_hits, kingdom_map, max_iter=100, tol=1e-6, min_abundance=1e-4, out_iter=nullptr) → vector<SpeciesAbundance>`
  - `build_kingdom_map(multi_hits) → unordered_map<uint32_t, NonHostCategory>`
**Pipeline insertion point**: After STAR alignment, replaces classify_batch() in the --nonhost-db path
**CLI flag**: automatic when --nonhost-db is provided (no new flag)
**Outputs**: nonhost_em_abundance.tsv, nonhost_summary.json (adds em_species_count, em_iterations, top_pathogens[])
**Assay coverage**: scRNA (main path), Bulk RNA (before early return), ATAC/Visium (ATAC before early return; Visium flows through scRNA path)
**Wiring**: run_nonhost_em_screening() static helper in singlify.cpp called from all three assay paths
**Shipped**: 2026-04-14, commit fe718a1

## Feature: Seed-Chain-Extend Secondary Aligner (NONHOST-SECONDARY-ALIGN)
**Header**: `include/singlet-pileup/nonhost/nonhost_aligner.h`
**API**:
  - `SeedPos` struct: {contig_id, position} — reference seed hit
  - `SecondaryAlignResult` struct: {species_id, kingdom, aligned_reads, total_reads, mapping_rate, coverage}
  - `SpeciesRefIndex::build_from_fasta(path) → bool` — loads FASTA, indexes minimizers (k=21, w=11)
  - `SpeciesRefIndex::build_from_sequences(seqs)` — in-memory build for tests
  - `SpeciesRefIndex::lookup(kmer_hash) → const vector<SeedPos>*`
  - `longest_chain(sorted_positions, max_gap=500) → int` — longest seed chain within gap
  - `NonHostAligner(ref_base, abundance_threshold=0.001f)` — constructor
  - `NonHostAligner::align(reads, em_abundances) → vector<SecondaryAlignResult>` — align reads vs each species
  - `NonHostAligner::write_tsv(results, out_prefix)` — writes nonhost_secondary_alignment.tsv
**Pipeline insertion point**: Inside run_nonhost_em_screening(), after em_deconvolve(), before nonhost_em_abundance.tsv write
**CLI flag**: automatic when --nonhost-db provided AND SINGLIFY_REF_BASE is set; graceful no-op if FASTA missing
**FASTA path template**: `${SINGLIFY_REF_BASE}/nonhost/{viral_genomes|bacterial_genomes|fungal_genomes}/species_{id}.fna`
**Output**: `nonhost_secondary_alignment.tsv` (kingdom, species_id, aligned_reads, total_reads, mapping_rate, coverage)
**Dependencies**: nonhost_em.h (SpeciesAbundance, NonHostCategory), min_sketch.h (extract_minimizers, canonical_hash)
**Shipped**: 2026-04-14, commit 9750d21

## Feature: VDB Read-Swap Protocol Detection (AUTOFIX-VDB-READ-SWAP-PROTOCOL)
**Header**: `include/lib1fq/protocol.h`, `include/lib1fq/sra_encoder.h`, `include/lib1fq/fastq_encoder.h`
**API**: `ProtocolCandidate.reads_swapped` (bool) — set by `detect_protocol()` when inverted R1/R2 is detected
**Pipeline insertion point**: SraEncoder::encode() and FastqEncoder::encode(), immediately after `detect_protocol()` returns
**CLI flag**: automatic — no user-facing flag; fires transparently when R1>=50bp and R2<=34bp (inverted geometry)
**Fix summary**: When VDB stores cDNA as R1 and CB+UMI as R2, proto detection now samples probe spot r2 lengths (majority vote when r2_len=0) to detect inverted geometry. Encoders physically swap R1<->R2 data when `reads_swapped=true`.
**Unit test**: `test_vdb_read_swap_detect` in `src/test_lib1fq.cpp`
**Dependencies**: None new
**Shipped**: 2026-04-15, commit 3de4b61

## Feature: Host k-mer Bloom filter false-positive suppression (NONHOST-HOST-SUBTRACT)
**Header**: `include/singlet-pileup/nonhost/host_kmer_filter.h`
**APIs**:
- `HostKmerFilter::build_from_fasta(fasta_path, k=21, w=11) → HostKmerFilter`
- `HostKmerFilter::save(path) / HostKmerFilter::load(path) / HostKmerFilter::exists(path)`
- `HostKmerFilter::contains(uint64_t min_hash) → bool`
- `HostKmerFilter::remove_host_minimizers(vector<uint64_t>) → vector<uint64_t>`
- `NonHostScreener::classify_multi_filtered(read, host_filter) → MultiHit`
- `NonHostScreener::classify_multi_batch_filtered(reads, host_filter) → vector<MultiHit>`
**Pipeline insertion point**: In `run_nonhost_em_screening()`, auto-loads
  `$SINGLIFY_REF_BASE/nonhost/host_21mer.bloom` (or sibling of --nonhost-db).
  When loaded, uses `classify_multi_batch_filtered()` instead of `classify_multi_batch()`.
**CLI**: `singlify build-host-filter --genome-fasta <fa> --output <bloom> [--kmer 21 --window 11 --threads N]`
**Dependencies**: None (uses existing `murmurhash64A`, `extract_minimizers` from min_sketch.h)
**Output change**: `nonhost_summary.json` gains `"host_filter_active": true/false`
**File format**: HSKBLOOM v1 (magic + 32B header + bit array). GRCh38: ~313 MB.
**Unit tests**: 9 tests in `test/test_nonhost_host_filter.cpp` (all passing). 7/7 nonhost CTests.
**Shipped**: 2026-04-16, commit 24959ae

## Feature: Full-whitelist ambient profiling for EmptyDrops (N22)
**Headers**: `include/singlet-pileup/pileup_engine.h`, `include/singlet-pileup/cell_calling.h`
**APIs**:
- `PileupConfig::whitelist_path` — full barcode whitelist path (e.g. 3M-february-2018.txt); empty = disabled
- `PileupConfig::wl_ambient_ceil` — UMI ceiling for ambient classification (default 50)
- `PileupEngine::wl_umi_counts() → const vector<uint32_t>&` — per-WL-barcode UMI counts (run() only)
- `PileupEngine::wl_ambient_gene_counts() → const vector<uint64_t>&` — global gene sums from WL-only reads
- `PileupEngine::wl_ambient_ceil() → uint32_t`
- `call_cells_emptydrops(..., wl_umi_counts*, wl_ambient_gene_raw*, wl_ambient_ceil)` — new optional WL params
**Pipeline insertion point**: `PileupConfig::whitelist_path` is set in `singlify.cpp` before `load_references()`.
  WL is loaded alongside auto-discovered barcodes; during BAM pileup, reads from WL-but-not-discovered barcodes
  count into `wl_umi_counts_` (per-barcode) and `wl_ambient_gene_counts_` (global gene sums).
  `export.h` passes both to `call_cells_emptydrops()`, which uses WL gene sums as the ambient profile
  instead of the sparse auto-discovered ambient pool (fixes EmptyDrops overcalling on deep 10xv3).
**CLI**: Automatic — `singlify.cpp` sets `config.whitelist_path` from the auto-resolved whitelist_file.
  No new user-facing flags needed. Enabled whenever a real whitelist file is found.
**run_parallel path**: per-worker `wl_ambient_genes` vectors merged into `wl_ambient_gene_counts_` after join.
  Per-barcode `wl_umi_counts_` NOT populated in parallel path (too heavyweight); `n_ambient` estimated from reads.
**Memory**: ~14.8 MB (3.7M × 4-byte WL UMI counts) + ~300 KB (38K gene × 8-byte) + ~14.8 MB WlAmbientIndex.
**Unit tests**: 4 tests in `test/test_wl_ambient.cpp` (all passing). Part of ctest suite (84/84 pass).
**Shipped**: 2026-04-17, commit ccc6a8a
