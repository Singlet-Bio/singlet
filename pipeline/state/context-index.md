# Singlify Context Index

> Maintained by doc-scribe after every code change. Read by orchestrator (system level), Sonnet workers (module level), Haiku scouts (file level).

---

## System Level (Orchestrator reads this — ≤80 lines)

singlify is a self-contained C++ binary: SRA → .1fq → STAR alignment → streaming pileup → .1pz export.

**Current state** (2026-04-12, post scope expansion — Cycle 105):
- Pipeline throughput: 82.0s (32T, PGO+SA prefetch) · **1484s 5-panel total** (C00-C04, cycle 105, -3.4% from /dev/shm fix)
- Counting accuracy: gene r=0.9995 vs matched STARsolo — validated ✅ ; cell calling 99.92% STARsolo recall
- Protocol breadth: 25 protocols, 8/16 families validated (VAL1), 0 crashes; 118/177 downloaded, ~59 processed
- Perf dead-ends closed: S2 (<1% ceiling), S3 (minimizer correctness failure), S7 (interleaved batching). SA prefetch is the ceiling.
- **Assay support**: scRNA-seq ✅, CITE-seq ✅ (T1-T4), scATAC ✅ (A1-A7, fragments + bins + QC + cell calling, PBMC 500 validated), Smart-seq2 ✅ (SS1-SS3), Bulk RNA ✅ (B1-B4), Visium ✅ (V1-V4 complete, E2E synthetic pass). snRNA pending G-SNRNA.
- Auto features: N22 threads ✅, N2 protocol ✅, N4 whitelist ✅, N10 adapter ✅, N1 species ✅ (human+mouse); GATE-A CLOSED
- Biology modules: N6 UMI-dedup, N7 saturation, N5 EmptyDrops, N9 QC, N11 ambient, N12 doublet, N13 ancestry, N14 sex, N15 ASE, N17 V(D)J, N18 CRISPR, N19 cell-cycle (all complete)
- **Open tracks**: Gap Feature Track (9 items: G-EXPORT → G-TXLEVEL), Model Organism (SPECIES-KMER-DB → REF-FETCH-CMD → REF-INSTALL-LOCAL → SPECIES-VAL-PANEL; 57 species from GEO catalog), SCALEVAL (500+ samples, massively parallel SLURM downloads)
- **Biggest unblocking tasks**: G-EXPORT (MEX/h5ad — adoption blocker), G-SNRNA (nuclear RNA broken), G-EM (5-25% reads lost), SPECIES-KMER-DB (unlocks all model organism work)
- **Species registry**: `geo-reprocess/scgeo/config/species.py` — 57 species (NCBI taxon ID → assembly + Ensembl version). Download source: Ensembl FTP (`ftp.ensembl.org/pub/release-111` vertebrates; `ensemblgenomes.ebi.ac.uk/pub/{division}/release-59` plants/metazoa/protists; GENCODE for human/mouse). Pre-built index hosting: TODO — Cloudflare R2 / Zenodo / AWS S3 public.
- Key dead-end pattern: per-read hash overhead >50ns kills caching at ~3μs/read. Minimizer SA narrowing fails (read minimizer ≠ genome minimizer with any sequencing error).

**Architecture**:
- STAR singlet-lite: CMake OBJECT library in `src/star/`, entry `star_main_impl()`
- singlify forks child for STAR, reads BAM pipe for streaming pileup
- .1fq: columnar archive (2-bit packed, barcode-sorted blocks, zstd compressed)
- .1pz: VOCSC compressed sparse matrices (singlepress format)
- Pileup: header-only C++ in `include/singlet-pileup/`

**Production optimizations** (cumulative ~48% faster than stock STAR 2.7.11b):
- 14-mer prefix SA sort (~15%), R2 consecutive dedup (~5%), hash barcode lookup (~3%)
- NUMA interleave (~10%), SA lazy winBin (~7%), SA boundary prefetch (~1.5%)
- Cross-worker multi-mapper merge in parallel pileup (3141e4a: fixes chr A/B multi-mapper drop bug, r=0.9948→0.9995)
- PGO+LTO (~5%), -march=native (~6%)
- S3 minimizer index: DEAD END (14-mer SAi already narrow enough, correctness failure). Index file exists but disabled.

---

## Module Level (Sonnet workers receive relevant module)

### STAR Alignment (src/star/) — perf-exec domain
- Entry: `star_main_impl()` in `STAR.cpp`
- Hot paths: `compareSeqToGenome()` 25% CPU, `matchCBtoWL()` 11%, `stitchWindowAligns()` 21%
- SA binary search: log₂(3.2B) ≈ 32 probes × ~100ns DRAM = ~3μs/seed
- Thread scaling: plateaus at 16T (DRAM bandwidth bound)
- singlify mods: `singlify_cb_match()` replaces barcode binary search, 14-mer sort in ReadAlignChunk
- Build: `cmake --build singlify/build --parallel` or standalone `make` in `STAR/source/`

### .1fq Format (include/lib1fq/) — perf-exec domain
- Codec: `sra_encoder.h` (encode), `fq_decoder.h` (decode), `onefq.h` (format spec)
- Block: 500K reads, columnar (BC, UMI, R1_seq, R2_seq, R1_qual, R2_qual)
- Compression: zstd per column, BINNED4 quality (4 bins from Illumina 8)
- Decode: parallel pread() + std::async, near-linear scaling to 8T
- 40M reads encode: 19.6s (4T), decode: 5.9s (16T)

### Pileup Engine (include/singlet-pileup/) — bio-exec domain
- Core: `pileup_engine.h` — PileupEngine class, streaming BAM via htslib
- Accumulator: `sparse_accumulator.h` — SparseAccumulator<T>, COO→CSC conversion
- Gene model: `gene_model.h` — GTF interval tree, exon/intron classification
- SNP: `snp_pileup.h` — AD/DP allele counting
- Donor: `donor_demux.h` — Vireo-port BinomMixtureVB
- Mitochondrial: `mt_heteroplasmy.h` — per-position base pileup
- Export: `export.h` — ExportConfig, .1pz/MTX dual-path writers
- UMI dedup: `umi_dedup.h` — existing framework (directional correction not yet implemented)
- Profiling: per-op RDTSC breakdown in PROFILING_PIPELINE.md

### Pipeline Orchestrator (src/singlify.cpp) — READ-ONLY for bio-exec
- CLI: .1fq/--reads/download/decode/genome subcommands
- Pipeline: download → encode → decode → align (fork+BAM pipe) → pileup → export
- 40M reads end-to-end: ~104s wall (20T, warm cache, .1fq path)

### Planned Modules (not yet implemented)

#### ATAC Pipeline (include/singlet-pileup/atac_*.h) — bio-exec domain
- `atac_fragment.h` ✅ — Fragment extraction from aligned BAM, paired-end → fragment coordinates, dedup by position (Cycle 75)
- `atac_bin_counter.h` ✅ — Tile genome into bins (default 500bp), per-cell fragment counting, output as .1pz sparse matrix (Cycle 76)
- `atac_qc.h` ✅ — TSS enrichment, fragment size distribution, FRIP, per-cell complexity (Cycle 77)
- Reuses: donor_demux.h (SNP-based demux from ATAC fragments), snp_pileup.h, ancestry, sex calling
- Pipeline wiring ✅ — STAR PE-DNA mode, QNAME barcode injection, three-stream .1fq decode, A1→A2→A3→export routing (Cycle 77)
- **Known issue**: SRA submission process discards index reads (I1/I2) for 10x ATAC libraries. The 16bp barcode in I2 is permanently lost in SRA FRA/FRR submission, making real ATAC E2E validation impossible using SRA data. Real ATAC validation data must come from original 10x mkfastq outputs or ENCODE (which preserves all reads). See Cycle 78 episode.

#### CITE-seq / ADT Module (include/singlet-pileup/adt_*.h) — bio-exec domain [✅ COMPLETE]
- `adt_matcher.h` ✅ — Fast antibody barcode matching (Hamming ≤1), hash-based lookup, CSV reference loading, 8/8 tests (cycle 82)
- `adt_counter.h` ✅ — Per-cell × per-tag UMI dedup using umi_dedup.h, SparseAccumulator, output adt_counts.mtx + features/barcodes TSV (cycle 83)
- `hto_demux.h` ✅ — HTO demultiplexing: CLR normalization, quantile threshold + MAD for singlet/doublet/negative, 5/5 tests (cycle 83)
- Pipeline wiring ✅ — singlify.cpp T4 integration (cycle 84): --feature-ref flag, second .1fq pass for ADT reads, HTO auto-detection from tag names

#### Visium Spatial Module (include/singlet-pileup/visium_*.h) — bio-exec domain [✅ V1-V4 COMPLETE]
- `visium_spatial.h` ✅ — VisiumSpatialParser: barcode → (row, col) via tissue_positions.csv, supports SR 1.x (headerless) + SR ≥2.0 (headered), spatial metadata in .1pz (cycle 84, 27 assertions, 5/5 tests)
- V2 ✅ — per-spot gene pileup, c0ddcd9 (cycle 85); V3 ✅ — per-spot QC metrics, aeafb03 (cycle 85); V4 ✅ — E2E pipeline verified synthetic (cycle 87, 48.2s/5M, spatial_coordinates.tsv)
- Reuses: existing exon pileup engine (spots are cells with coordinates), UMI dedup, gene model

#### Smart-seq2 Module — bio-exec + perf-exec domain
- No new headers needed — reuse gene_model.h with featureCounts-style overlap rules
- Mode switch in singlify.cpp: no UMI, no barcode, per-well counting

#### Bulk RNA-seq Module — perf-exec domain
- No new headers needed — simplified path through alignment + gene counting
- Read-level dedup stats: optical/PCR duplicate marking and library complexity estimation

#### Competitive Feature Parity Track (planned — code-scout + bio-exec)
- **COMP-AUDIT-1** (cycle 106): Fetch STAR CHANGES.md, Cell Ranger docs, alevin-fry simpleaf, kallisto|bustools. Produce gap table vs singlify. Add each new gap as a DAG node. Write to state/competitive_audit.md.
- **Recurring**: every 10 cycles, re-audit for new features in STAR releases and competitor tools. Singlify must match or beat all tools at the Pareto frontier (accuracy within 1%, wall time ≤ competitor on same hardware).
- **Pareto standard**: speed ≤ competitor AND accuracy within 1%. If singlify is slower on any feature, perf-exec profiles + optimizes before shipping.

#### Gap Feature Track (planned — bio-exec domain)
- **G-EXPORT** (priority 1): `export.h` extension for MEX + AnnData h5ad output. `--output-format mex|anndata` flag. ~400 LOC. Unblocks Seurat/Scanpy adoption.
- **G-SNRNA** (priority 2): `gene_model.h` intron-inclusion mode. `--include-introns` flag. ~200 LOC. Required for nuclear RNA-seq workflows.
- **G-EM** (priority 3): EM multi-mapper rescue in `pileup_engine.h`. Equivalence class E-step/M-step. ~600 LOC. Rescues 5-25% silently-dropped reads.
- **G-ATAC-PEAKS**: Poisson peak caller or MACS2 subprocess integration. ~500 LOC.
- **G-PSI**: Per-cell splice junction PSI matrix. ~400 LOC.
- **G-SATCURVE**: Saturation downsampling curve output. ~150 LOC.
- **G-BARNYARD**: Dual-genome barnyard alignment + species assignment. ~300 LOC.
- **G-CELLPLEX**: CMO barcode + Flex probe quantification (reuses adt_matcher.h). ~300 LOC.
- **G-TXLEVEL**: TCC equivalence class output for transcript-level analysis. ~500 LOC.

#### Model Organism Reference Track (planned — bio-exec domain)
- **REF-INDEX-CMD** (first): `singlify index fetch --species <name>` CLI. Ensembl FTP auto-download + STAR genomeGenerate + species registry in ~/.config/singlify/species.json.
- Supported species target: zebrafish (GRCz11), drosophila (BDGP6.46), c.elegans (WBcel235), rat (mRatBN7.2), macaque (Mmul_10).
- Each species: download + index build + k-mer signature for N1 auto-detection + VAL pass on ≥10 SRA samples.

#### Large-Scale Validation Infrastructure (planned — validator domain)
- **SCALEVAL-SLURM**: `scripts/val2_download.sh` SLURM array template. `--array=1-N%100` for 50-100 concurrent downloads. fasterq-dump → singlify encode → log per-sample.
- **SCALEVAL-DESIGN**: 500+ sample draw across all modalities × species × protocol families. Input: val2_samples.csv.
- Pipeline: SCALEVAL-DESIGN → SCALEVAL-SLURM → VAL2 → SCALEVAL-PROCESS → VAL3 → VAL4 → SCALEVAL-EDGECASE.

---

## File Level (updated by code-scout as needed)

### src/star/STAR.cpp
- `star_main_impl(int argc, char* argv[])` — renamed STAR entry, lines 1-50
- `singlify_cb_match()` — hash-based barcode lookup replacing binary search

### include/singlet-pileup/pileup_engine.h
- `PileupEngine` class — main streaming BAM processor
- `process_record(bam1_t*)` — per-read hot path
- `run()` — iteration loop, calls process_record for each BAM record
- `run_parallel()` — parallel worker orchestration with indexed chunk-based BAM read
- Auto-strand detection: pre-probe scan of most-read chromosome (sequential) before thread spawn (Cycle 75, 19f0fc7 fix for C04 87%→0.07% wrong_strand)
- N6 directional UMI correction: now works in both run() and run_parallel() paths (Cycle 78, 85f5ba0). Per-worker finalize() call before merge ensures directional dedup is applied locally before cross-worker multi-mapper merge. Gene r=0.9995 in parallel mode, 0.13% delta vs serial due to cross-worker multi-mapper simple dedup asymmetry.

### include/singlet-pileup/sparse_accumulator.h
- `SparseAccumulator<T>` — COO sparse matrix, `increment()`, `to_csc()`
- `CSCMatrix` struct — compressed sparse column for export

### include/singlet-pileup/gene_model.h
- `GeneModel` — GTF loader, interval tree index
- `classify_position(chrom, pos)` → exonic/intronic/intergenic

### include/singlet-pileup/export.h
- `ExportConfig`, `ExportStats`, `export_results()`, `write_stats_json()`
- Shared between singlify.cpp and test binaries

### include/singlet-pileup/vdj_counter.h (N17)
- `VdjModel` — GTF VDJ gene loader (IG_V_gene, TR_V_gene biotypes)
- `query(tid, rs, re, overlaps)` — interval query for VDJ hits
- `SparseAccumulator<uint16_t>` — per-cell gene usage counting
- Outputs: `vdj_gene_usage_cells.mtx`, `vdj_gene_usage_features.tsv`, `vdj_gene_usage_barcodes.tsv`
- Tested: C01 (human) 411 genes, C11 (mouse) 490 genes, <1% overhead

### include/singlet-pileup/atac_fragment.h (A1, Cycle 75)
- `ATACFragmentExtractor` — Fragment extraction from PE BAM, Tn5 shift (+4/-5 canonical), QNAME barcode parsing
- `process_record(bam1_t*, bam1_t*)` — paired-end processor, extract 5' ends as fragment coords
- `finalise()` — flush per-barcode buffer to fragments.tsv.gz, sorted by position
- `fragments()` — return finalised SkipMap for downstream bins/QC
- Position dedup: unordered_set per chromosome, (barcode, start, end, strand) tuple key
- Unit tests: 8/8 pass (fragment coordinates, Tn5 offset, dedup, QNAME parsing)

### include/singlet-pileup/atac_bin_counter.h (A2, Cycle 76)
- `ATACBinCounter` — Tile genome into configurable bins (default 500bp), per-cell fragment counting
- `increment_bin(barcode, chrom, start, end)` — route fragments to bin matrix
- `to_csc()` → CSCMatrix for .1pz export
- Handles boundary-spanning fragments (fragments crossing bin edges)
- Unit tests: 5/5 pass (bin assignment, boundary checks, sparse accumulation)

### include/singlet-pileup/atac_qc.h (A3, Cycle 77)
- `ATACQCComputer` — Per-cell and global QC metrics for ATAC
- Methods: `compute_tss_enrichment(gene_model, bin_matrix)`, `mito_fraction()`, `median_fragment_size()`, `frip_from_bins(peaks_as_bins)`
- `fragment_global_bin()` accessor on ATACBinCounter for FRIP calculation
- Outputs: per-cell TSS, fragment count, fragment size median; global: TSS histogram (configurable bins), fragment size distribution
- Unit tests: 12/12 pass (TSS calculation, bin queries, histogram generation)

### src/singlify.cpp (ATAC Wiring, Cycle 77, commit f458193)
- Early assay_type detection: 96-byte .1fq header peek → is_atac_mode flag
- STAR conditional: if !is_atac_mode → use soloType CB_samTagOut + GTF, else → PE-DNA mode (alignIntronMax 1, alignMatesGapMax 2000)
- Pileup routing: if is_atac_mode → ATACFragmentExtractor → ATACBinCounter → ATACQCComputer → .1pz; else → existing GEX pipeline
- All engine_ptr creation guarded with !is_atac_mode check (prevents GTF load for ATAC)
- QNAME barcode injection: STAR output pre-processed before pileup (barcode tag → QNAME prefix)
- No protocol auto-detection modification needed — ATAC protocol detected by assay_type in .1fq header

(Additional file-level entries added as code-scout reports them)
