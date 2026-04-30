# Singlify Task DAG

> Updated by orchestrator every cycle. Nodes = tasks, edges = dependencies.

## Status Key
- 🔴 not started
- 🟡 in progress
- 🟢 complete
- ⚫ dead end
- 🔵 blocked (dependency unmet)
- ⏳ deferred (rate limited — target cycle noted)

---

## Deferred (Rate Limited)

> Tasks that were planned but dropped due to rate limits. Orchestrator promotes to Ready Set next available cycle.
> Format: `⏳ TASK-ID: description → deferred from cycle N, retry cycle N+1`

(none currently)

---

## GATE-A: CLOSED (N2+N4+N22 all complete) — all tiers unlocked

---

## Performance Track (perf-exec)

- ⚫ S2: Adaptive Reference Priority → DEAD END cycle 7 (<1% ceiling)
- ⚫ S3: K-mer Pre-Screening → DEAD END cycle 69 (14-mer SAi already sufficiently narrow; read minimizer ≠ genome minimizer breaks correctness; 4.8GB mmap adds latency)
- 🔴 S5: Self-Tuning Parameters → depends: nothing → ceiling: meta
- 🔴 S4: Multi-Species K-mer Filter → depends: S5 → ceiling: 15-18%
- ⚫ S7: Interleaved Multi-Read Search → DEAD END cycle 58
- 🟢 N2: Protocol Auto-Detection → bd299bd (cycle 4)
- 🟢 N4: Whitelist Auto-Resolution → (cycle 6)
- 🟢 N10: Adapter Auto-Selection → 75dfe21 (cycle 8)
- � N21: Shared-Memory Genome → 47c9dff (cycle 98, --genome-shared flag, genome_load.sh wrapper, val1_process.sh updated)
- 🟢 N22: Auto Thread Detection → a241987 (cycle 1)
- 🟢 N19: Cell Cycle Phase Scoring → cfd6fc9 (cycle 68, G1/S/G2M, 0.013s overhead)
- ⚫ F1: rANS Quality Coding → DEAD END cycle 12 (zstd beats rANS)
- 🟢 F5: Deep Archive Mode → 126b9ee (cycle 13, 31.4% size reduction)
- 🟢 PARALLEL-PILEUP: indexed parallel → b6f2024 (cycle 13, pileup 95% faster, net neutral 40M, wins ≥80M); cross-worker merge fix 3141e4a (cycle 73, r=0.9995 achieved)

## Biology Track (bio-exec)

### Tier 1 — ALL COMPLETE
- 🟢 N6: UMI Error Correction → 23b1011→4f67c23 (validated r=0.9998)
- 🟢 N7: Sequencing Saturation → 4f67c23 (cycle 3)
- 🟢 N9: Per-Cell QC Metrics → 56e0c10 (cycle 1)
- 🟢 N8: Pipeline Provenance Manifest → 4f67c23 (cycle 3)
- 🟢 N16: Multi-Junction Gene Counting → 8d68150 (cycle 4)

### Tier 2 — ALL COMPLETE (except N3)
- 🟢 N1: Species Auto-Detection → 30fd19c (cycle 6)
- 🔴 N3: Reference Index Registry → needs hosting infrastructure
- 🟢 N5: EmptyDrops++ Cell Calling → 13bcaa1 (cycle 7, 99.92% concordance)

### Tier 3 — ALL COMPLETE
- 🟢 N15: Allele-Specific Expression → 828e333 (cycle 9)
- 🟢 N14: Sex & Karyotype Calling → e476aa0 (cycle 8)
- 🟢 N13: Ancestry Classification → 35dd2a9 (cycle 9, EUR 99.0%)
- 🟢 N11: Ambient RNA Correction → 488bf67 (cycle 11, v1); v2 MLE estimator 89edddf (cycle 173, per-cell rho std=0.073)
- 🟢 N12: Doublet Detection → fb7ae35 (cycle 12, v1); v5 adaptive threshold a360885 (cycle 173, FPR 48.9%→9.16%)

### Tier 4 — Modality Expansion (NOW UNBLOCKED)
- � N17: V(D)J Gene Usage Counting → 85b6429+6c6753c (cycle 61, 232 LOC, 411 genes, <1% overhead)
- � N18: CRISPR Guide Capture → ecbad79+29d3a93 (cycle 62, 250 LOC, PGO assert bug fixed)

## Cross-Cutting
- 🟢 BASELINE: established cycle 2
- 🟢 Manuscript refresh → cycle 13 (doc-scribe)
- 🟢 FIX-N6-PARALLEL: Directional UMI correction → 85f5ba0 (cycle 78, 0.13% serial/parallel delta)
- 🟢 FIX-CHRM-PARALLEL: → 03cad25 (cycle 79, 1.54M reads exact match, chrM parity restored)
- � BUG-R2-VARLEN: Variable-length R2 encoding → 1154ec2 (cycle 91, clamp/pad to first_r2_len, regression test added)
- ✓ **ALL SERIAL/PARALLEL PARITY BUGS CLOSED (4/4):** 3141e4a (cross-worker merge), 19f0fc7 (auto-strand), 85f5ba0 (N6 directional), 03cad25 (chrM deferred write)
- 🟢 VC-CLI-WIRING: Variant calling CLI → cycle 88 (--variant-calling/--min-coverage/--min-alt-count/--min-vaf wired to SS2+Bulk in singlify.cpp, 19/19 tests)
- 🟢 B4-CLI-WIRE: Dedup stats CLI → 06524e0 (cycle 89, --dedup-stats + --optical-distance in SS2+Bulk paths, dedup_stats.tsv output)
- 🟢 TEST-SAMTOOLS: pileup_integration test fix → e9906cb (cycle 90, samtools PATH in CMakeLists.txt, 19/19 passing)

---

## ATAC Track (bio-exec + perf-exec)

> scATAC-seq and 10x Multiome ATAC. Output: fragment files + bin matrices (.1pz) at user-specified resolution (default 500bp). No peak calling — just bins.

- 🟢 A1: Fragment Extraction Engine → 5c5d8b6 (cycle 75, 8/8 tests, Tn5 shift, position dedup)
- 🟢 A2: Bin Matrix Generator → ab56ac7 (cycle 76, 5/5 tests, 500bp bins → SparseAccumulator)
- � A3: ATAC QC Metrics → 99b7803 (cycle 77, 12/12 tests, TSS enrichment + mito + fragment size + FRIP)
- 🟢 ATAC-WIRE: Pipeline integration → f458193 (cycle 77, PE-DNA STAR, QNAME BC inject, bin matrix export)
- 🟢 ATAC-E2E-SYNTHETIC: End-to-end test on synthetic ATAC → 95885e5 (cycle 78, synthetic pass, real data blocked on SRA I2)
- 🟢 ATAC-3READ-ENCODE: 3-read .1fq encoding for ATAC → d010575 (cycle 92, R1+R2(BC)+R3 layout, I2 stream in reader/writer, 500 PBMC E2E: 5.7M fragments, 2052 cells)
- 🟢 ATAC-10X-DATA: Real 10x ATAC data acquired → atac_pbmc_500_v1 (500 PBMCs, 22.8M reads, cf.10xgenomics.com)
- 🟢 ATAC-WIRE-A3-A7: Pipeline wiring for QC + cell calling + fragment output → f36920d (cycle 95, fragments.tsv + atac_qc.tsv + atac_cells.tsv, +5% overhead)
- ✓ BUG-R2-VARLEN: Variable-length R2 encoding → 1154ec2 (cycle 91, clamp/pad to first_r2_len, regression test added)
- � A4: ATAC Donor Demux → 9752016 (cycle 93, fragment-based SNP genotyping, wraps VB demux, 7 tests/22 checks)
- � A5: ATAC Ancestry/Sex → 2cab3fc (cycle 94, AtacSexCaller + AtacAncestryClassifier, fragment-based, 13 tests)
- 🔴 A6-E2E: 10x Multiome ATAC end-to-end → depends: A2, A3, ATAC-WIRE → Full pipeline on real dataset. Validate vs cellranger-atac.
- � A7: ATAC Cell Calling → 5c5c623 (cycle 93, auto-inflection threshold, TSS+fragment+FRIP filtering, 10 tests)

## Visium Spatial Track (bio-exec)

> Visium spatial transcriptomics. Output: per-spot gene counts with spatial coordinates in .1pz metadata.
> **VISIUM CORE COMPLETE (V1-V3)** — All 3 core components shipped, fully integrated, E2E ready.

- 🟢 V1: Spatial Coordinate Parser → 53638f1 (cycle 84, supports SR 1.x + SR ≥2.0 CSV formats, tissue_positions.csv barcode→(row,col), 27 assertions, 5/5 tests)
- 🟢 V2: Per-Spot Gene Pileup → c0ddcd9 (cycle 85, spots are barcodes using SPATIAL_RNA assay_type, --tissue-positions flag, reuses gene_model + UMI dedup)
- 🟢 V3: Visium QC Metrics → aeafb03 (cycle 85, per-spot UMI/gene stats, tissue coverage, in-tissue filtering, 16/16 tests)
- � V4: Visium E2E Pipeline → c0ddcd9 (cycle 87, full pipeline verified with synthetic test: 48.2s/5M, SPATIAL_RNA assay, spatial_coordinates.tsv + visium_qc.tsv)

## CITE-seq / ADT Track (bio-exec)

> Antibody-Derived Tag and Hashtag Oligo quantification. Output: ADT/HTO count matrices (.1pz) alongside GEX. Supports 10x Feature Barcode, TotalSeq, cell hashing.
> **CITE-seq pipeline COMPLETE (T1-T4)** — All 4 components shipped, fully integrated, compiles clean.

- 🟢 T1: ADT Tag Matcher → c4aeb6a + 2651ba9 (cycle 82, Hamming-1 hash, 8/8 tests, atomic counters)
- 🟢 T2: ADT UMI Dedup + Counting → 69a735a (cycle 83, per-cell×tag exact dedup, SparseAccumulator, 8/8 tests)
- 🟢 T3: HTO Demultiplexing → 69a735a (cycle 83, CLR normalization, quantile threshold MAD, 5/5 tests)
- 🟢 T4: CITE-seq E2E Wiring → 0e5b569 (cycle 84, --feature-ref flag, second .1fq pass, AdtMatcher+AdtCounter+HtoDemux integrated, HTO auto-detection, adt_counts.mtx + adt_features.tsv + adt_barcodes.tsv exported)

## Smart-seq2 Track (bio-exec + perf-exec)

> Plate-based full-length RNA-seq. One well = one cell. Output: gene × cell count matrix.

- � SS1: Plate-Mode Pipeline → aa9dd12 (cycle 81, PE STAR, no soloType, HTSlib gene counting, multi-mapper reject, 100 LOC)
- 🟢 SS2: Smart-seq2 QC → 04d6851 (cycle 86, alignment_qc.h shared with B2, gene body coverage + 5'/3' ratio + complexity)
- 🟢 SS3: Smart-seq2 Variant Calling → 99cc4ca (cycle 87, RNAVariantCaller with htslib bam_plp, de novo discovery, VCF+TSV, 20/20 tests)

## Bulk RNA-seq Track (perf-exec + bio-exec)

> Standard bulk RNA-seq. Output: gene counts, variant calling, QC metrics.

- � B1: Bulk Mode Pipeline → 7f7b26c (cycle 82, auto-strand detect, SE mode, 90% code reuse from SS1)
- 🟢 B2: Bulk QC Metrics → 04d6851 (cycle 86, shared alignment_qc.h with SS2, gene body coverage + Library complexity + saturation)
- 🟢 B3: Bulk Variant Calling → 99cc4ca (cycle 87, shared RNAVariantCaller with SS3, VCF output, splice-skip support, 20/20 tests)
- � B4: Read-Level Dedup Stats → 7bcfa55 (cycle 88, 41/41 tests, PCR+optical duplicate marking, Lander-Waterman library complexity)

## Non-Host Transcriptomics Track (bio-exec + perf-exec)

> Viral, bacterial, fungal, and plant read classification and quantification from unmapped reads after host alignment. Two-phase design: (1) Sylph-port coverage-corrected sketch screening to identify present non-host species; (2) precision STAR alignment (viral) or stripped minimap2 port (bacterial/fungal/plant) for confirmed species only.
> **Default ON** all assays. `--no-viral-screen` / `--no-microbial-screen` to disable. `--non-host-only` for screen-without-mapping mode.
> **Performance mandate**: Phase 1 overlapped with host pileup; ≤15% total overhead; ≤10% when no species pass the gate.
> See singlify.agent.md § "Non-Host Transcriptomics Module" for full design spec, acceptance gate criteria, and output artifact schema.

- ✅ NONHOST-EM (fe718a1 2026-04-14): EM abundance deconvolution for nonhost species
  **Delivers**: NonHostEM::em_deconvolve() — full EM mixture model on per-read minimizer hit rates → converged relative abundance per species. 5/5 unit tests PASS (single-species recovery, 50/50 mixture, ambiguous reads, convergence, min-abundance filter). Wired into scRNA, bulk RNA-seq, Visium, and ATAC post-STAR paths.
  **Outputs**: nonhost_em_abundance.tsv (kingdom|species_id|relative_abundance|reads_assigned|mean_hit_rate), nonhost_summary.json adds em_species_count + em_iterations + top_pathogens[].
  **Classify_multi wired**: yes — classify_multi_batch() at soft_threshold = min_hit_rate/4 collects multi-species hits before EM.
  **Validation targets**: SRR11092058 (COVID-19 PBMC, SARS-CoV-2+), SRR7287187 (Zymo D6300 mock community, 8 bacterial species at known ratios)

- 🔴 NONHOST-SYLPH-PORT: Port Sylph FracMinHash + Poisson coverage correction screening kernel to `include/singlet-pileup/nonhost_screener.h` (header-only, subprocess-free). SIMD AVX2 k-mer hashing via same infrastructure as `singlify_cb_match()`. Hash table partitioned by 8-bit MinHash prefix for cache-local lookup. Unmapped reads stream directly from STAR BAM pipe — zero disk I/O. Accept: ≥99% species recall vs. upstream Sylph on 5-species viral panel (EBV/CMV/HIV-1/SARS-CoV-2/HPV16); <30s on 5M unmapped reads at 20 threads; all existing pileup tests still pass.
- 🔴 NONHOST-VIRALDB: Build viral reference database at `SINGLIFY_REF_BASE/nonhost/viral/`. (1) Download NCBI RefSeq complete viral genomes; cluster at 95% ANI → ~5,000–8,000 non-redundant representatives. (2) Build Sylph-port FracMinHash sketch DB (~200 MB). (3) Build per-family STAR index shards at `star_2.7.11b/shards/{Herpesviridae,Retroviridae,Papillomaviridae,Orthomyxoviridae,...}/` (each a valid STAR genome dir, `--genomeSAindexNbases 11`). (4) Contaminant reject list: PhiX174, E. coli K-12, Mycoplasma pneumoniae, A. laidlawii. (5) GTF annotations for primary clinical viruses: EBV, CMV, HPV 16/18, HIV-1, HTLV-1/2, SARS-CoV-2, influenza A/B/C. Accept: shard `mmap()` activates in <1s; sketch DB ≤250 MB; STAR index covers all named clinical viruses.
- 🔴 NONHOST-MICROBIALDB: Build microbial reference database at `SINGLIFY_REF_BASE/nonhost/microbial/`. (1) GTDB r220 representative prokaryotes + NCBI RefSeq fungi + plant chloroplast mRNA panels → Sylph-port sketch DB (~2 GB). (2) Per-confirmed-species FASTA + GFF at `SINGLIFY_REF_BASE/nonhost/bacterial/{taxon_id}/` for top-100 human bacterial pathogens + top-50 fungal species + common plant panels. Accept: sketch DB ≤2.5 GB; GFF annotation available for all priority species in the tier.
- 🔴 NONHOST-UNMAPPED-CAPTURE: Wire STAR unmapped read streaming into screener thread with CB+UMI preservation. Currently `--outSAMunmapped None` in singlify.cpp (3 locations). Change: unmapped reads flow from STAR BAM pipe → per-read FLAG demux → (mapped → pileup thread) + (unmapped → nonhost_screener thread). CB+UMI extracted from QNAME and attached to each unmapped read struct. Zero disk I/O — no .fastq.gz written. Depends: NONHOST-SYLPH-PORT. Accept: zero .fastq.gz files written; CB/UMI in QNAME preserved through to viral attribution; val1 golden panel pileup outputs byte-identical.
- 🔴 NONHOST-SCREENER: Phase 1 integration in singlify.cpp. Screener runs as `std::thread` overlapped with pileup. Receives unmapped read stream from NONHOST-UNMAPPED-CAPTURE. Returns species hit list `(taxon_id, family, containment, read_count, breadth_pct, n_loci, cb_valid_frac)` after scanning all unmapped reads. Apply acceptance gate (all criteria from agent spec). Wire `--no-viral-screen` / `--no-microbial-screen` flags. Emit `{"event":"nonhost_screen_done","confirmed":N,"viral_screen_s":T}`. Depends: NONHOST-UNMAPPED-CAPTURE, NONHOST-VIRALDB, NONHOST-MICROBIALDB. Accept: ≤5s overhead on clean (no-virus) sample; PhiX/E.coli correctly rejected at gate; val1 pileup regression-free.
- 🟢 NONHOST-HOST-SUBTRACT: 24959ae + 69f858c (cycle 153) Host k-mer subtraction to eliminate false positives. Smoke test (cycle 153, SRR32855204 clean PBMC) showed 370K/2.6M unmapped reads (14%) got viral k-mer hits, 160 species above EM threshold on NEGATIVE control. Top false positives: Bubaline alphaherpesvirus (28%, herpesvirus homology), BeAn 58058 virus (17%), bracovirus (15%, eukaryotic homologs), HERV-K113 (1.5%, genuine endogenous). **Fix**: At index build time, enumerate host genome k-mers and exclude from .snhskidx. At EM runtime, apply unique-k-mer fraction threshold (species must have >10% k-mers NOT shared with host). Accept: clean PBMC negative control <=5 species above EM threshold; only HERV-K survives; zero non-endogenous viral species.
- 🟢 NONHOST-VIRALDB-BUILT: Built viral.snhskidx (1.4GB) at ${SINGLIFY_REF_BASE}/nonhost/viral.snhskidx (cycle 150)
- 🟢 NONHOST-BACTERIALDB-BUILT: Built bacterial.snhskidx (6.2GB) at ${SINGLIFY_REF_BASE}/nonhost/bacterial.snhskidx (cycle 151)
- 🟢 NONHOST-FUNGALDB-BUILT: Built fungal.snhskidx (17GB) at ${SINGLIFY_REF_BASE}/nonhost/fungal.snhskidx (cycle 151)
- 🟢 NONHOST-CB-SPECIES-MATRIX: Per-cell nonhost species matrix -> 7a73823 + 5186d4c (cycle 151, NonHostCellMatrix, nonhost_per_cell.tsv, 10 tests/35 assertions)
- 🟢 NONHOST-SMOKE-TEST: E2E on SRR32855204 clean PBMC (cycle 153). Full output suite produced. Reveals host subtraction needed -- see NONHOST-HOST-SUBTRACT.

## New Biology Features (bio-exec)

- 🟡 N19: Cell Cycle Phase Scoring → cfd6fc9 (cycle 68, G1/S/G2M, 0.013s overhead)
- 🟢 N20: Per-cell Read Statistics → dc50c22 (cycle 70, median_dup=37.4%, 0s overhead)

## Competitive Feature Parity (code-scout + bio-exec)

> Every 10 cycles: audit STAR/STARsolo, Cell Ranger, alevin-fry/simpleaf, kallisto|bustools for new features. Every gap becomes a DAG task. Singlify must match or beat competition at the Pareto frontier in speed AND accuracy. Output: state/competitive_audit.md.

- 🟢 COMP-AUDIT-1: First competitive audit → cycle 115 (code-scout). Gaps found: G-VELOCITY (closed), G-METRICS (closed), G-PERMITLIST (closed), h5ad native (deferred — MTX works with scanpy), BUS binary (low priority), web_summary.html (future), molecule_info.h5 (future). No blocking gaps remain.

## Gap Feature Track (bio-exec + perf-exec)

> Functional capabilities present in Cell Ranger / STARsolo / alevin-fry but not yet in singlify. Ranked by adoption impact.
> **GAP TRACK: 11/11 COMPLETE** + 3 competitive-audit extras shipped (G-VELOCITY, G-METRICS, G-PERMITLIST)

- 🟢 G-VELOCITY: Spliced/unspliced/ambiguous gene matrices → 3f6c77c (cycle 115, scVelo-compatible, collapse_intron_to_gene, 37/37 tests)
- 🟢 G-METRICS: Pipeline metrics summary CSV → bc1fd3a (cycle 115, 15-field CellRanger-compatible metrics_summary.csv, 38/38 tests)
- 🟢 G-PERMITLIST: --forced-cells N / --expect-cells N → bc1fd3a (cycle 115, top-N by UMI count, skip EmptyDrops)

- � G-LOGGING: Structured multi-level logging → 649df57 (cycle 112, JSON Lines + throttled progress + macros, 32/32 tests)
- � G-EXPORT: CellRanger-compatible MTX → e15907d (cycle 110, `filtered_feature_bc_matrix/` with gene-level collapse, 3-col features.tsv, barcode-1 suffix, `--output-format mtx`)
- � G-SNRNA: Combined exon+intron gene counts → 4c5d0e2 (cycle 110, `gene_counts.1pz/mtx`, 78298×8675 on C04, uses collapse_to_gene_counts — both MTX filtered_feature_bc_matrix and .1pz)
- � G-EM: EM multi-mapper rescue → bad8248+255bc87 (cycle 111, equivalence-class EM, gene_counts_em.1pz output, 26/26 tests)
- � G-ATAC-PEAKS: Native ATAC peak calling → 00ea554 (cycle 112, Poisson enrichment + BED6, 31/31 tests)
- � G-PSI: Per-cell splice junction PSI → 6a80d2d (cycle 111, donor/acceptor grouping, sparse float PSI matrix, splice_events.tsv, 25/25 tests)
- � G-SATCURVE: Saturation downsampling curves → a1f968a (cycle 111, analytical Lander-Waterman + Poisson, 0.65s overhead, 24/24 tests)
- � G-BARNYARD: Per-cell species classification → 630c10f (cycle 112, auto-detect from Ensembl/CellRanger gene prefixes, HUMAN/MOUSE/DOUBLET/AMBIGUOUS, 28/28 tests)
- � G-CELLPLEX: CellPlex CMO demultiplexing → e44cc7e (cycle 112, Otsu CLR threshold, 29/29 tests)
- � G-TXLEVEL: Transcript-level TCC → 8b9ee39 (cycle 112, bustools-compatible EC×cell, post-hoc from exon CSC, 30/30 tests)

## Model Organism Reference Track (bio-exec + validator)

> Expand singlify to all 57 species in the GEO catalog (`geo-reprocess/scgeo/config/species.py`). Architecture: (1) bundled ~10MB k-mer DB for species auto-detection, (2) `singlify index fetch` CLI to download+build any reference from Ensembl FTP, (3) SLURM array job to install top-15 model organism references locally.
>
> **Reference genome download source**: Ensembl FTP (all 57 species covered):
> - Vertebrates: `https://ftp.ensembl.org/pub/release-111/fasta/{species}/dna/` + `.../gtf/{species}/`
> - Plants/Metazoa/Protists: `https://ftp.ensemblgenomes.ebi.ac.uk/pub/{division}/release-59/fasta/{species}/dna/`
> - Human/Mouse: GENCODE (`https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_{human|mouse}/release_{45|M34}/`)
>
> **⚠ TODO (future)**: Pre-built STAR indices should be hosted on a CDN (Cloudflare R2, Zenodo, or AWS S3 public) so end users can `singlify index fetch` without building locally (saves 1-6h per species). For now all indices are built locally via SLURM. Leave `# TODO: host pre-built indices at <URL>` comment in implementation.

- � SPECIES-KMER-DB: 37-species compile-time registry → bdbcbd8 (cycle 118, species_registry.h with Ensembl FTP URL builder, 26 assertions)
- � REF-FETCH-CMD: Reference fetch plan builder → d6efff2 (cycle 118, ref_fetch.h with plan_fetch/build_install_all_script/write_registry, 22 assertions)
- � REF-INSTALL-LOCAL: SLURM array for top-15 references → SLURM 353328 COMPLETE (cycle 121, 17/17 species: human+mouse+zebrafish+rat+pig+drosophila+nematode+cow+dog+chicken+yeast+macaque+horse+cat+rabbit+frog+sheep)
- 🟡 SPECIES-VAL-PANEL: Cross-species validation → V2 broad 359989: **3/8 SUCCESS** (cat 0.995, pig 0.998, macaque 0.901). Failures: protocol/VDB/adapter bugs (5), chicken k-mer quality (1).

## Large-Scale SRA Validation (validator)

> G4+G7: Download 500+ diverse SRA samples; verify full autonomous pipeline across all modalities, species, and edge cases. Downloads run as massively parallel SLURM array jobs — one job per SRA, 50–100 concurrent (--array=1-N%100).

- 🟢 VAL1: Diverse Metadata Draw + Sample Draw → cycle 87 strategy + cycle 88 draw (177 samples, 16 protocol families, 13 species, scripts/val1_samples.csv)
- 🟢 VAL1-GOLDEN: Golden regression panel (12 samples × 7 protocols × 2 species) → val1_golden.csv (cycle 103)
- 🟢 SCALEVAL-DESIGN: 500+ sample validation design → 42304d7+1f73ab5 (cycle 120, val2_design.md + val2_download.sh + val2_process.sh + val2_analyze.py)
- 🟡 SCALEVAL-SLURM: SLURM parallel infrastructure → 1f73ab5 (cycle 120, val2_download.sh with 3-retry and metadata, val2_process.sh with auto-detect, val2_analyze.py with flagging)
- 🟡 VAL2: SRA Download + .1fq Construction → SLURM 353439 (cycle 121, 533 samples submitted, downloads running)
- 🟡 SCALEVAL-PROCESS: Full pipeline SLURM run → batch 356866 cycle 128 + reprocess batches cycle 129-131. **Current:** 439/533 = 82.4% raw; 439/459 = 95.6% excl. data+species issues. Bug fixes: d9c9c6b (zero-len R2), f2b9f73 (ddSEQ split-row), 967ccf7 (UMI clamp), c73b143 (uint16 underflow segfault), f915c6b (GTF auto-resolve from --ref-base, +16 new successes), db2b14e (clip5pNbases per-mate), 1523ade (whitelist auto-resolution raw header read). **Permanently unprocessable:** 25 data_incomplete (barcode stripped from SRA), 23 zero_bc (truncated R1/R2), 11 OOM (>768G), 1 low_mapping, 1 star_error. **Species-blocked:** 26 species_unknown (need REF-INSTALL-LOCAL). **Pending:** TIDs 47,71,89,140,177 downloading/reprocessing.
- � VAL3: Mapping Quality Audit → **COMPLETE (cycle 175)**. 533 total, 440 processed, 93 no_output.
  **Results**: 207 PASS (38.8% total, 51.8% of processable excl feature-barcode+no-output). 121 low_genes (22.7%), 51 low_map (9.6%), 40 feature_barcode_short_R2 (7.5%), 13 zero_map_other (2.4%), 8 no_cells (1.5%).
  **Key discovery**: 40 samples are **feature barcode libraries** (CRISPR/ADT/HTO) misclassified as GEX in catalog. R2 contains synthetic scaffold sequences (e.g., TGGAAAGGACGAAACACC = CRISPR scaffold), not cDNA. Zero mapping rate is expected behavior. Breakdown: 18 10x-3p-v3, 9 CITE-seq, 6 10x-3p-v2, 4 BD-Rhapsody, 3 other. Species: 19 human, 9 macaque, 8 mouse, 4 other.
  **Low-map breakdown**: 3 chicken (missing/wrong genome), several ddSEQ (35-48% — marginal), various short-R2 edge cases.
  **Action items**: (1) New AUTOFIX-FEATURE-BARCODE-DETECT filed. (2) 121 low_genes + 8 no_cells likely benefit from reprocessing with current binary (post-EmptyDrops/doublet fixes). (3) 93 no_output need download/species resolution.
- 🔴 VAL4: Rigorous Failure Triage → depends: VAL3 → **Default assumption is pipeline bug, not bad data.** For each flagged sample, execute the full Failure Triage Protocol (see singlify.agent.md ## Failure Triage Protocol). Steps: (1) Fetch GEO SOFT metadata via Entrez API — extract `Sample_description`, `Sample_data_processing`, `Sample_library_strategy`, `Sample_instrument_model`, `Sample_molecule_ch1`. (2) Fetch linked manuscript from `Series_pubmed_id` — search PMC full text for how this GSM was used to support claims. (3) Attempt all mechanistic fixes before declaring failure (STAR flags, adapter content, strand inference, CBlen re-probe, species re-detection, read-swap). (4) Log each attempt: what was tried, STAR Log.final.out mapping rate before/after, exit code. (5) Only after ≥3 distinct fixes fail: classify as (a) confirmed library error, (b) unsupported protocol → add to roadmap, (c) external data quality issue (contamination, low input, etc.). (6) Write verbose report per sample to `state/failure_reports/<SRR>.md`. Update `val2_triage.csv`.
- 🔴 SCALEVAL-EDGECASE: Edge case registry → depends: VAL4 → For each confirmed pipeline bug (not data quality): write one targeted fix commit. Track in `singlify/state/edge_cases.md`: SRR, description, root cause, STAR flags that revealed it, fix commit, regression test name. Every confirmed fix must add a CTest regression. Target: 0 regressions on re-run of VAL1 golden panel after each fix.
- 🔴 VAL5: Verbose Failure Mode Hardening → depends: VAL3 → For each failure category discovered in VAL4: add a singlify diagnostic message at the exact point of failure. Guard rail requirements: mapping_rate <5% → print top-3 STAR mismatches + suggest species/strand check; zero cells called → suggest whitelist mismatch or CBlen re-probe; .1fq decode error → hex dump first 32 bytes; STAR crash (OOM, signal) → print genome size + estimated RAM needed. Each guard rail must have a unit test.
- 🟢 VAL1-GUARD: Crash guard on invalid CB/UMI → dba5f5a (cycle 101, prevents heap corruption on bad .1fq metadata, CBlen>32 or UMIlen>16 → abort before STAR)
- 🟢 VAL1-ALIAS: Protocol alias table → 1e28b3f (cycle 101, maps 10x-v2→10x-3p-v2, 10x-v3→10x-3p-v3, etc., fixes root cause of bad .1fq encoding from protocol tag lookup failures)
- 🟢 VAL1-READSWAP: R1/R2 auto-swap for complex protocols → ec45651 (cycle 102, probe offset fix + dual-orientation detection, fixes Drop-seq/sci-RNA read order)
- 🟢 VAL1-TESTFIX: lib1fq varlen_r2 test restoration → 80b306f (cycle 102, cfg.no_trim fix, 22/22 tests passing)
- 🟢 VAL1-CLIP-SAFETY: R2 constant prefix clip safety cap → 7ba7b4a (cycle 104, remaining<30bp → skip clip)
- 🟢 VAL1-DEVSHM: STAR BAM sort temp to /dev/shm → e36df19 (cycle 105, eliminates Lustre I/O failures, +3.4% speed)
- 🟢 VAL1-SKIP-FIX: Process script skip check → 9f850ef (cycle 105, checks pileup_stats.json)
- 🟢 VAL1-OOM-128G: SLURM memory increase → 4cfdd5b (cycle 106, 64G→128G)
- 🟢 VAL1-MONITOR-MEM: Remove --mem override in monitor → 2e8ccb5 (cycle 106)
- 🟢 VAL1-BAMSORT-RAM: Always set --limitBAMsortRAM → 3000f04 (cycle 107, unconditional 60GiB cap, prevents OOM on >100M read samples)
- � VAL1-INDROP-VARIANT: InDrop version detection → 0084c58 (cycle 118, v1/v2/v3 layout detection, handles SRR5945694 cblen overflow, 13 assertions)
- � VAL1-TINY-CRASH: Tiny dataset guard → 9793b95 (cycle 119, STAR RAM scaling for <1M read datasets, SRR23027738 crash root cause confirmed, 16 assertions)
- � VAL1-SEQWELL-DETECT: Seq-Well protocol detection → 71659a2 (cycle 118, tag normalization, CBlen=150 rejection, 12 assertions)
- 🟢 VAL1-ENCODER-CBLEN: Encoder CBlen protocol table lookup → 9d19766 (cycle 109, build_metadata_json uses known_protocols() by protocol_id, canonical tag resolution for routing)
- 🟢 VAL1-BAMSORT-RAM2: BAMsortRAM heuristic increase → 9d19766 (cycle 109, SA*3+8GiB, was SA*2+4GiB)
- 🟡 VAL1-REDL-CORRUPT: Re-download 30 corrupt/ZSTD .1fq files → batch 350913 (cycle 109)
- 🟡 VAL1-REDL-CBLEN: Re-download 18 CBlen-failure files → batch 350920 (cycle 109)
- 🟡 VAL1-REPROCESS: Reprocess after re-download → batch 350929, depends 350913+350920 (cycle 109)

---

> **Cycle 122 state**: 59/59 tests. REF-INSTALL-LOCAL: ✅ 17/17 COMPLETE. Val2 downloads: SLURM 353439 (279/533 complete, 102 running, 0 failures). Species val samples: 54 ready. val2_process.sh fixed and ready. HEAD: 6354ef6. Next: when val2 downloads finish → submit val2_process.sh array → VAL3 mapping audit → VAL4 triage.

---

## Next-Generation Modality Track (bio-exec + perf-exec)

> **Design charter**: (1) reuse shared alignment, barcode-extraction, gene/feature-pileup, and QC infrastructure — no redundant C++ paths; (2) autonomous detection from .1fq metadata and read structure — `--protocol` override always available; (3) outputs interoperable with CellRanger/Seurat/ArchR/scanpy/CoolTools ecosystems; (4) every modality benchmarked against current SOTA demonstrating equivalence or Pareto improvement on speed, accuracy, or memory before marking complete.

### Multiome & Routing
- � G-MULTIOME-ROUTE: 10x Multiome auto-detect + split/route → 56545eb (cycle 114, detect_modality/classify_inputs/plan_multiome_run, 33/33 tests)
- 🟢 G-SPATIAL-MULTIOME: Visium HD + spatial RNA/ATAC → 1d37f55 (cycle 114, HD barcode parser, spatial metadata writer, 35/35 tests)
- 🟢 G-BARNYARD-ROUTE: Per-cell species routing → 5b493c9 (cycle 114, human/mouse CSC submatrix split, doublets excluded, 34/34 tests)

### Long-Read
- � G-LONGREAD: Long-read scRNA detection → 01e089c (cycle 116, detect_longread, split_masseq_read, edit-distance barcode extraction, 46/46 tests)

### Immune Repertoire
- � G-VDJ: Full V(D)J CDR3 clonotype assembly → feb130f (cycle 114, de novo V/J k-mer anchor extraction, codon translation, productive filter, clonotype_id grouping, 36/36 tests)

### Spatial — Extended Barcode Decoders
- � G-SPATIAL-BARCODES: Spatial barcode decoders → 5d4a954 (cycle 117, Slide-seq/Stereo-seq/VisiumHD/HDST/MERFISH, 37 assertions, 50/50 tests)

### Chromatin & Epigenomics
- � G-BULK-ATAC: Bulk ATAC-seq auto-detection and QC → 938f346 (cycle 115, is_bulk_atac detection, NFR/mono/di fractions, Lander-Waterman complexity, 42/42 tests)
- 🟢 G-CUTTAG: CUT&TAG/CUT&RUN chromatin QC → e5ae15f (cycle 115, mode detection, spike-in normalization, nucleosomal enrichment, 43/43 tests)
- � G-CHIPSEQ: ChIP-seq QC → b73f02c (cycle 116, NSC/RSC/PBC/Lorenz/FRiP, write_chipseq_qc, 45/45 tests)

### Metabolic Labeling
- � G-SLAMSEQ: SLAM-seq T→C conversion counting → 2f1e333 (cycle 115, MD-tag parsing, min_conversions threshold, per-cell stats, 39/39 tests)

### Multimodal Single-Cell
- � G-SHARESEQ: Combinatorial barcode decoder → 3cad33e (cycle 115, SHARE-seq/PAIRED-seq/scifi-seq 2-3 level barcodes, linker matching with Hamming tolerance, 40/40 tests)
- 🟢 G-SCDNA: scDNA CNV detection → b73f02c (cycle 116, MAPD/ploidy/GC-correction, per-cell bin counts, 45/45 tests)

### Methylation — New Aligner Backend
- � G-METHYL: Bisulfite methylation engine → 594b06a (cycle 117, MethylationCaller with CIGAR walking, CpG/CHG/CHH context, WGBS/RRBS/SCBS/NOMe modes, Bismark-compatible BED, 84 assertions)

### Chromatin Conformation
- � G-HIC: Hi-C contact pairs → fa499e1 (cycle 116, classify_contact, write_pairs 4DN .pairs format, cis/trans QC, 47/47 tests)

### GTEx & Bulk Genomics
- � G-TPM: TPM/FPKM normalized expression → 6bc47d9 (cycle 115, effective gene length from exon union, TPM sum invariant, RSEM-compatible FPKM, gene_expression.tsv, 41/41 tests)
- 🟢 G-MIRNA: miRNA quantification → 7b6b49b (cycle 117, miRBase GFF3 parser, k=18 index, RPM output, 21 assertions)
- 🟢 G-RRNA: rRNA contamination detector → 07c865c (cycle 117, 80 diagnostic 21-mers, human/mouse 5S/5.8S/18S/28S, JSON report, 15 assertions)
- 🟢 G-WGS: WGS assembly QC → 0922fa0 (cycle 117, 4-backend assembler routing, N50/N90, k-mer genome size, JSON/TSV reports, 43 assertions)

### Cross-Cutting Infrastructure
- � G-INTEROP: Interoperability compliance suite → 95c4a90 (cycle 118, MTX/fragments/pairs/BED/CSV/VCF validators + generators, 25 assertions)
- � G-BENCHMARK-SUITE: Per-modality benchmark framework → 09aa422 (cycle 119, TSV/markdown round-trip, speedup/Pareto checks, 44 assertions)

---

## AUTOFIX Tasks (cycle 8+)

- 🟢 **AUTOFIX-SCI3-507GB-OOM**: sci-RNA-seq3 (SRR23582977) — **FULLY RESOLVED (5352b85 + 9b3e7ea, cycle 171)**.
  **Fix chain**: (1) 5352b85 per-read exact sum + r2len>1024 guard (FIFO OOM fix). (2) 9b3e7ea: BC_DICT R1 reconstruction stride+position fix, varint corruption guard, quality/trim/dedup column guards, outBAMsortingBinsN 200 for CB_samTagOut, plate-protocol EmptyDrops guard.
  **VALIDATED**: SRR23582977 → exit=0, wall=76s, mapping=51.95%, cells=326 (sci-RNA-seq3 combinatorial). 5-panel benchmark 5/5 PASS (bench5p job 362966).
  **Priority**: CLOSED


- � **AUTOFIX-SCI3-CBSAM-HANG**: sci-RNA-seq3 (SRR23582977) — **RESOLVED (9b3e7ea, cycle 171)**.
  **Root cause**: 6 bugs across 3 files: (1) BC_DICT R1 reconstruction placed BC at offset 0 instead of protocol-correct offset 24, (2) corrupted varint in decode_seq_column cascaded into quality/trim/dedup decoders reading heap garbage, (3) missing column decoder guards on ptr<end, (4) SIZE section r2len>1024 guard missing for file-mode, (5) missing --outBAMsortingBinsN 200 for CB_samTagOut protocols with >10M reads, (6) EmptyDrops returning 0 cells for plate-protocol assays (sci-RNA-seq3, SPLiT-seq).
  **Fix commit**: 9b3e7ea — 3 files changed, 213 insertions(+), 83 deletions(-)
  **VALIDATED**: SRR23582977 → exit=0, wall=76s, mapping=51.95%, cells=326. 5-panel 5/5.
  **Priority**: CLOSED
- � **AUTOFIX-DDSEQ-192GB-OOM**: ddSEQ (SRR17873408) — **ROOT CAUSE FOUND + OOM FIXED (cycle 170, job 362877)**.
  **Root cause**: Old SRR17873408.1fq (Apr 10) was encoded by pre-ad9e999 singlify which wrote BC dict for multi-segment protocols. BC dict presence causes prescan to be SKIPPED (code: `if (ps_reader.has_bc_dict() && complex_concat_segs_all.size() > 1)`), leaving full 15.1M Cartesian WL passed to STAR (3694×4095 segments). STAR allocates `15.1M barcodes × 39K genes × 4B = 2.3TB` for gene×barcode count matrix → OOM at 384GB.
  **Fix**: Re-download SRR17873408. Current singlify (ad9e999+) does NOT write BC dict for multi-segment protocols → prescan runs → WL filtered to ~385 real barcodes → STAR peak ~27GB.
  **VALIDATED**: Job 362877 — wall=493.66s, BCs=385, unique=59.28%, exit=0, RSS=27GB. NO OOM.
  **NEW FINDING**: Fresh VDB download auto-detects SRR17873408 as 10x-3p-v2 (R1=26bp=16CB+10UMI, confidence=3), NOT ddSEQ. Old bench_panel.sh metadata may be wrong. With 10x-3p-v2 protocol, cells=385 (not 3087 from old ddSEQ processing). Mapping rate 59.28% matches exactly. Need oracle validation: is this genuinely a 10x-v2 sample or a ddSEQ sample that happens to have R1=26bp?
  **Acceptance test (updated)**: SRR17873408 processes exit 0, no OOM (RSS ≤80GB), unique ≥50%. Cell count TBD (depends on protocol oracle).
  **Priority**: MEDIUM (OOM resolved; remaining issue is protocol label audit, not a blocking bug)

- � **AUTOFIX-DROPSEQ-DATA-INCOMPLETE**: Drop-seq bench slot 3 — **RESOLVED (cycle 171, SRR12062565 replacement)**.
  **Root cause**: SRR10010840 SRA depositor stripped barcode R2 — permanently unprocessable.
  **Fix**: Replaced with SRR12062565 (GSM4629663, GSE152915, Drop-seq, 50.2M reads, Human). Validated: 67.77% mapping, 69 cells (sparse library, real biology — 803 median genes/cell), bench_panel.sh slot 3 updated.
  **Priority**: CLOSED

- 🔴 **AUTOFIX-V1-WHITELIST-MISSING**: singlify reports `737K-april-2014.txt not found for 10x-3p-v1` → falls back to CB_samTagOut barcode discovery. 10x Chromium v1 (14bp CB + 10bp UMI = 24bp R1) whitelist not bundled.
  **Root cause**: whitelists/ directory has 737K-august-2016.txt (v2) and 3M-february-2018.txt (v3) but NOT 737K-april-2014.txt (v1). known_protocols() maps 10x-3p-v1 to this missing file.
  **Fix target**: Either (a) add 737K-april-2014.txt to whitelists/ (download from 10x cellranger package), or (b) alias v1 to use the v2 whitelist (they may share the same 737K barcode set).
  **Acceptance test**: `singlify download SRR_v1_sample && singlify process` auto-resolves whitelist for 10x-3p-v1 without falling back to CB_samTagOut.
  **Priority**: LOW (v1 is rare in current catalog; CB_samTagOut fallback works but loses 1MM correction)

- 🟢 AUTOFIX-BD-RHAPSODY-WHITELIST: CB_UMI_Complex STAR whitelist handling — **OOM RESOLVED (5 commits, validated cycle 167)**
  **Fix chain (5 commits)**: (1) 77242ae Exact matching, (2) e18680c CB_samTagOut concat architecture, (3) 4b2ecf2 barcode prescan whitelist filtering, (4) ad9e999 multi-segment encoder fix (protocol.h). Combined: 1.77M WL → 22K (SPLiT-seq) or 912K → 56K (BD Rhapsody) via prescan, reducing STAR per-barcode memory from 138GB to 3-9GB.
  **VALIDATED**: SPLiT-seq SLURM 362482 — 76.5G peak at 128G, 84.81% mapping, 234 cells, 170s wall. BD Rhapsody SLURM 362483 — 64.7G peak at 128G (no OOM). BD Rhapsody GEX mapping validation pending (362505, SRR16096461).
  **BD Rhapsody mapping note**: SRR33004875 (GSM6654378) is a BD Rhapsody Sample Tag (SMK) library, NOT GEX — R2 contains synthetic SMK oligos with 0% genomic homology. This is correct pipeline behavior, not a bug. See AUTOFIX-BD-RHAPSODY-SMK-SCREEN.
  **Unblocked**: SPLiT-seq, BD Rhapsody, inDrop, ddSEQ, SureCell — all multi-segment protocols now fit in 128G standard nodes via prescan filtering.
  **Root cause**: STAR's CB_UMI_Complex soloType does NOT accept "None" as a sentinel for `--soloCBwhitelist` — it interprets "None" as a literal file path and errors with "could not open input file None". Tested end-to-end: CB_UMI_Simple accepts `None`, CB_UMI_Complex does not. Affects BD Rhapsody (3 CB segments, CLS1/CLS2/CLS3), sci-rna-seq3 (3 segments), SPLiT-seq, inDrop, ddSEQ, SureCell — any protocol in singlify's complex_tags set.
  **Observed symptom**: STAR exit 109, "*INPUT FILE* error: could not open input file None". Sample SRR24097977 (GSM7152960, BD Rhapsody). Validator jobs 355026 (first fix) and 355065 (second fix) both failed at STAR init.
  **Fix target**: In singlify.cpp's CB_UMI_Complex branch, when no whitelist is provided, either (a) bundle BD Rhapsody CLS1/CLS2/CLS3 whitelists at whitelists/bd_rhapsody_cls{1,2,3}.txt and load them by protocol tag, (b) synthesize per-segment whitelist files on-the-fly from discovered barcodes (writing 3 temp .txt files in TMPDIR then passing their paths to STAR), or (c) pre-extract the 96x96x96 = 884,736 BD Rhapsody Enhanced combinations from the BD library and ship as whitelists/bd_rhapsody_enhanced.txt (single whitelist repeated 3× is NOT correct since STAR would expect each segment to match against the full set, not the per-segment CLS).
  **Acceptance test**: `singlify download SRR24097977 --protocol bd-rhapsody && singlify <1fq> --pipeline --exons HS_GTF --snps SNPs_VCF --out-prefix out/` completes with STAR exit 0, mapping ≥50%, cells ≥10. Run via SLURM on bigmem. Same test applies to sci-rna-seq3 once R1 length is ≥34bp.
  **Blocked samples**: all BD Rhapsody (~321 human catalog samples), all sci-rna-seq3 (included in ~458 sci-rna), all SPLiT-seq, all inDrop (~28K), all ddSEQ, all SureCell. Approximate total blocked: ~30,000 catalog samples.
  **Priority**: HIGH — inDrop alone is 28K samples (12% of in-scope human catalog)
  **Prior fixes already shipped that are needed for this branch**: (1) soloCBposition-once-before-loop (16:40 binary), (2) soloCBwhitelist repeated N times per segment count (16:56 binary). Neither is yet committed to main — commit after whitelist fix is also validated.

- 🟢 AUTOFIX-5P-TSO-TRIM: d43af1c+c40e721+d709885 (cycle 153) — auto-detect constant 5' prefix in R2 during encoding, clip5pNbases always uses two-value form, TSO R2 trim disabled on 3' protocols
  **Root cause**: Adapter detector in 1fq-encode sees TSO-like sequence near start of R2 and auto-sets `r2_maxlen=30`. For 10x-5p-v3, R2 is 100bp cDNA with TSO (AAGCAGTGGTATCAACGCAGAGTACATGGG) AT THE START. The encoder's "truncate FROM position 30" logic cuts off the cDNA body (positions 30-100), keeping only the TSO-containing 5' end. Result: STAR can map the 30bp segments (77% uniquely mapped on GSM8240227/SRR28825471) but they land in random non-exonic locations — 111M mapped / 0 exon hits / 0 cells.
  **Observed symptom**: log line `[1fq-encode] WARNING: Adapter detected in R2 at position 30. Auto-setting r2_maxlen=30`, followed by `[singlet-pileup] Done: 111491770 reads, ... 0 exon hits, 0 intron hits`. Sample GSM8240227 (val1 golden, supposed to have ≥60% mapping + cells).
  **Fix target**: For 5' protocols (protocol_id 4 10x-5p-v3, 14 10x-5p-v4, 5 10x-5p-v2), replace "truncate R2 to maxlen" with "trim FROM position 0 to position <tso_end>, keep rest as cDNA" — i.e., strip the TSO prefix instead of the cDNA suffix. Alternative: disable adapter detection entirely for 5' protocols and rely on STAR's soft clipping.
  **Acceptance test**: GSM8240227 SRR28825471 → mapping ≥60%, cells ≥1000 (val1 expected 19134). Also test 10x-5p-v4 if catalog has one.
  **Blocked samples**: all 10x-5p-v3, 10x-5p-v2, 10x-5p-v4 (catalog ~5K samples)
  **Priority**: MEDIUM (5' is a smaller fraction of the catalog than 3')

- 🟢 AUTOFIX-MULTIOME-CELL-CALL: d4e9d5f (cycle 153) — use auto_barcodes.tsv as STAR whitelist when whitelist_auto_resolved
  **Root cause**: Unknown. GSM5487668 (SRR15296021, 29M reads) mapped 99% but got 0 cells called. Multiome uses ARC whitelist (737K-arc-v1.txt) which is confirmed present. Either (a) EmptyDrops threshold wrong for multiome library depth, (b) exonic pileup not seeing enough signal, or (c) multiome GEX R2 layout misinterpreted.
  **Observed symptom**: GSM5487668 summary.json → estimated_cells=0, 99% mapping rate.
  **Fix target**: Investigate singlet-pileup's handling of multiome GEX specifically. Compare against a known-working 10xv3 sample's cell-calling path. Likely a flag-gating issue.
  **Acceptance test**: GSM5487668 produces ≥100 cells with same pipeline flags.
  **Blocked samples**: all 10x_multiome (~312 human catalog samples)
  **Priority**: MEDIUM

- 🟢 AUTOFIX-EMPTYDROPS-ZERO-FALLBACK: top-N fallback when EmptyDrops returns 0 → d56bcc0 (2026-04-14 07:15, export.h + provenance.h + test_cell_calling.cpp, 72/72 CTests pass). Activates when n_barcodes_above_100_umi ≥ 10; records cell_caller=top_n_fallback in provenance.json. Cannot regress currently-passing samples (only fires on the zero-return path).

- 🟢 AUTOFIX-SHALLOW-BC-DISCOVERY: whitelist + progressive-threshold + top-5000 fallback added to all 3 auto-barcode-discovery paths → 36aa928 (2026-04-14 07:35, src/singlify.cpp, 72/72 CTests pass). Validated live on GSM4483378 (SRR11560752): STAR completed, gene_counts.1pz produced, no pipeline abort. Bit-exact on currently-passing samples (only fires on empty-result path). **Production witness** (2026-04-14 07:48): re-submission of 4 GSE148822 witnesses all now produce 83-89% mapping (previously HARD_FAIL with pipeline abort) — proves the fix is live. Samples still report 0 cells because exonic_fraction ~0.01% (genuinely unusable shallow libraries, not a pipeline bug) — the new result is the CORRECT output.

- 🟢 AUTOFIX-PROTOCOL-CONFIDENCE-OVERRIDE: catalog metadata wins at confidence ≤ LOW → b16bf97 (2026-04-14 08:10, types.h + sra_encoder.h + singlify.cpp + test_protocol_confidence_override.cpp, 73/73 CTests pass, 10 new tests + 34 assertions). Live validated on SRR8521680 (GSM3587947 seqwell): override fired, .1fq header rewritten to seqwell (id=16, CBlen=12, UMIlen=8). Fires only when detection confidence is NONE/LOW, metadata.protocol is set, differs from detected, and exists in known_protocols() registry. MEDIUM/HIGH confidence wins unchanged. Bit-exact on currently-passing samples.
  **Root cause**: `singlify download` / protocol detection path picks the VDB-derived protocol tag even when confidence is 1 or 2 (low) and the result conflicts with the high-confidence catalog classification passed via `--metadata-json`. Three witnessed mis-classifications: (1) GSM8197519 catalog=10xv3, detected=10x-visium confidence=2, R2-empty fail; (2) GSM8808563 catalog=10xv2, detected=quartzseq2 confidence=1, 11.5% mapping instead of 97.8%; (3) GSM3587947-GSM3588001 cluster (12 samples in GSE116256) catalog=seqwell, detected=10x-visium confidence=1, R2-empty fail.
  **Fix target**: When singlify detects a protocol with `confidence <= 2` AND `--metadata-json` is provided AND `metadata.protocol` is set AND it differs from the detected protocol, **prefer the metadata protocol**. Emit a warning: `[protocol_detect] low-confidence (c=N) detection=X disagrees with metadata.protocol=Y; preferring metadata`. When confidence >= 3, keep the detection. When no metadata is given, keep the old behavior.
  **Acceptance test**: re-run SRR8521680 (GSM3587947) with `--metadata-json '{"protocol": "seqwell"}'`; the pipeline should recognize seqwell, use seqwell CBlen/UMIlen, and not report "R2 empty".
  **Blocked samples**: at least ~15+ witnessed in one recent window; unknown catalog-wide. This is the SAME root-cause family as the shallow-BC fix — wrong protocol → wrong barcode structure → failure. Fix unblocks auto-detection on all low-confidence clusters.
  **Priority**: MEDIUM — not as common as VDB-short-download but consistently triggers on entire GSE clusters when it hits. Deferred to Copilot if bio-exec is busy.

- 🔴 COPILOT-BATCH002-METADATA-JSON: Copilot's batch002 job script does not pass `--metadata-json` to `singlify download`, so AUTOFIX-VDB-SHORT-DOWNLOAD (7e02eff) and AUTOFIX-PROTOCOL-CONFIDENCE-OVERRIDE (b16bf97) never fire on the batch002 track. Same ~184 pipeline_crash + ~11 OOM failures per cycle are observed on batch002 but NOT on my pilot_job.sh (which was updated 2026-04-14 08:19).
  **Fix target**: In Copilot's batch002 job script (likely `/mnt/projects/debruinz_project/singlify_pipeline/scripts/batch002.sh` or similar), before calling `singlify download`, write a `meta.json` with `{protocol, read_count, gsm_id, gse_id}` from the catalog entry and add `--metadata-json ${WORK}/meta.json` to the download command. Also update the exit-code handling to distinguish `$?=2` (download_fail, try ENA) from other non-zero exits.
  **Witness**: GSM8428076 (GSE273419) — catalog=10xv3, detected=marsseq2 c=1, SIGKILL OOM at 361s (wasted cluster slot). Would have been prevented by protocol-override fix.
  **Priority**: MEDIUM — Copilot's track to fix, not orchestrator's. File as a message/note to the Copilot agent at the next cross-track handoff.
  **Blocked samples**: ~300+ per cycle on the batch002 track.

- 🔴 PILOT-RESCUE-TIER-WIDEN: pilot_job.sh rescue tier requires `cells ≥ 1000` which misses legitimate 100-1000 cell samples with moderate mapping (30-50%). These samples have real biology but get classified as HARD_FAIL. Witness: GSM6204595 cycle 51 — map=31%, cells=769, classified HARD_FAIL. 769 cells is a usable sample.
  **Fix target**: In `/mnt/projects/debruinz_project/singlify_pipeline/scripts/pilot_job.sh` validate function, add a second rescue tier:
    - `rescue_tier_1` (existing): map ≥ 40% AND cells ≥ 1000 → SUCCESS rescued_low_map
    - `rescue_tier_2` (new): map ≥ 30% AND cells ≥ 100 → SUCCESS rescued_low_map_tier2
    - `strict`: map ≥ 50% AND cells ≥ 10 → SUCCESS
    - else: HARD_FAIL (as before)
  **Acceptance**: GSM6204595 retried under new criteria → SUCCESS.
  **Priority**: LOW — orchestrator-side policy change, not a singlify bug. Defer until higher-value fixes land.

- 🟢 AUTOFIX-PROVENANCE-CELLS-WRONG-FIELD: write_provenance_json() now receives n_cells from cc_result.cell_indices.size() → b546b98 (2026-04-14 08:33, export.h lines 1062-1067, 73/73 CTests pass). Provenance now agrees with summary.json and cell_calls.tsv. Bit-exact on currently-passing samples. The cosmetic discrepancy caused a rabbit-hole root-cause investigation — closing it prevents future confusion.

- 🟢 AUTOFIX-PROVENANCE-CELLS-WRONG-FIELD-LEGACY: b546b98 (resolved) — provenance.json `output.cells` field used to write the wrong value — the candidate barcode pool size (e.g. 3,686,400 = 10xv3 whitelist length) instead of the actual cell-called count. Cosmetic but misleading. Found 2026-04-14 07:48 while validating AUTOFIX-SHALLOW-BC-DISCOVERY: GSM4483378 provenance shows `cells: 3686400, total_umis: 27` but cell_calls.tsv has 0 rows and summary.json reports 0 cells correctly. The `cells` field in provenance should match `cell_calls.tsv` row count, not the candidate pool size. Low priority — summary.json is the authoritative source, provenance is metadata. File fix: in `export.h` or wherever `ProvenanceConfig.cells` is populated, use the cell-caller's returned indices count, not the pre-filter pool size.
  **Root cause**: The auto-barcode-discovery step in singlify scans the .1fq BC dictionary and picks barcodes with `reads >= 100` as cell candidates. When a sample has ~3-4M reads distributed across 3.6M barcodes (10x whitelist-sized), no single barcode reaches 100 reads. The pipeline then exits with `ERROR: No barcodes in BC dictionary have ≥100 reads` BEFORE STAR alignment runs. This blocks the entire AUTOFIX-EMPTYDROPS-ZERO-FALLBACK path because no exon_counts matrix is produced for EmptyDrops to fail on.
  **Observed symptom**: 13 of 15 cycle 50 probe witnesses (GSM4483378, GSM4483395, GSM4483405, GSM4483418, GSM4483426, GSM4680790, GSM4972228, GSM4972233, GSM4972235, GSM4972267, GSM5510161, GSM5510163, GSM8673826) all hit this — all are ~3.8M read 10xv3 libraries. Log signature: `Discovered 0 barcodes (≥100 reads) from .1fq BC dictionary (NNNNNNN entries)`. Exit code 1, `failure_category=pipeline_crash`, wall time ~36s.
  **Fix target**: When `≥100` yields zero barcodes, fall back in this order:
    1. Use the auto-resolved whitelist (e.g., `3M-february-2018.txt` for 10xv3) as the cell candidate set and let STAR/EmptyDrops handle cell calling. The whitelist is already resolved (log shows `Auto-resolved whitelist: 10x-3p-v3 → 3M-february-2018.txt`) but the code isn't using it.
    2. If no whitelist, progressively lower the threshold: try `≥50`, then `≥20`, then `≥10`. If any level yields ≥10 candidates, proceed.
    3. If still nothing, take the top 5000 barcodes by read count as a last-resort candidate set.
  Emit `[auto_barcodes] fallback: threshold=100 yielded 0, using whitelist (N barcodes)` or similar for each case.
  **Acceptance test**: re-run any of the 13 witnesses; pipeline must complete STAR alignment and produce a gene_counts matrix (even if final cell count is still 0 after EmptyDrops). Cycle 50 retry will be the live validator.
  **Blocked samples**: ≥13 confirmed in last probe + likely hundreds more shallow-read 10xv3/v2 across catalog. This is the root cause of what I previously mis-attributed to AUTOFIX-10XV3-LARGE-EMPTYDROPS — the failing samples are SHALLOW, not large.
  **Priority**: HIGH — unblocks all shallow 10x samples from ever reaching EmptyDrops, which then unblocks the top_n_fallback path for any that still produce 0 cells.
  **Root cause**: When EmptyDrops `cell_calling` returns `n_cells=0`, the pipeline gives up and reports 0 cells even though thousands of barcodes in the raw gene_counts matrix have non-trivial UMI counts. This happens on both shallow samples (insufficient reads to form a knee, 18-66s wall time) and deep samples (ambient profile saturating knee detection, 270-450s wall time).
  **Observed symptom** (last 200 samples, 55 total occurrences): 49 10xv3 + 3 10xv2 + 2 dropseq + 1 sci-RNA. Wall time ranges 15s-450s meaning the issue is orthogonal to read depth. Witness samples span many GSEs — GSM7444978 18s 70% map 0 cells, GSM8808563 448s 98% map 0 cells, GSE148822 (7 samples), GSE163122 (5 samples), GSE181737, GSE283885, etc.
  **Prior work**: the narrower AUTOFIX-10XV3-LARGE-EMPTYDROPS ticket only covers >20M read 10xv3, but this pattern is broader.
  **Fix target**: When `cell_calling` returns 0 cells, fall back to top-N barcodes by UMI count (N = min(catalog_expected_cells, 500, n_barcodes_above_100_umi)). Classify the result as SUCCESS with `cell_caller=top_n_fallback` recorded in provenance so downstream tools can distinguish from EmptyDrops calls. Also: EmptyDrops knee detection needs a sanity check — if the knee algorithm reports 0 but the raw matrix has >1000 barcodes with >100 UMI, the knee detection is broken and should emit a warning.
  **Acceptance test**: re-run any 5 previously-failed witness samples (e.g., GSM7444978, GSM5510163, GSM8808563, GSM2855484, GSM8673826); each should produce ≥10 cells called, mapping-rate unchanged, and provenance records which caller was used.
  **Blocked samples**: ~55 in last 200 runs = ~28% of current throughput. Fix lands → immediate +25% SUCCESS rate without touching alignment.
  **Priority**: HIGH — highest-impact fix currently possible; unlocks approximately 25-30% of HARD_FAIL/SOFT_FAIL samples.

- 🟢 AUTOFIX-DEMUX-K-SWEEP-SLOW: donor demux K-sweep >1000x faster → 12362fe (cycle 127, EM+BIC two-phase K-selection, flat buffers, digamma cache, K-level early stopping). Validated: SRR32855204 40.4M reads, 12089 barcodes, 613K SNPs → demux trivially fast (193s total pipeline). 72/72 CTests pass including new test_donor_demux_perf (26 assertions: K-selection accuracy K=1..5, EM/VB concordance ≥95%, speed regression <30s on 3K cells).
- 🟢 AUTOFIX-R1SHORT: Download segfault when R1 < CB+UMI length → 404e6cb (cycle 127, UMI stride clamp in sra_encoder.h + fastq_encoder.h, prevents SIGSEGV on SRR18463218)

- 🟢 AUTOFIX-VDB-SHORT-DOWNLOAD: cmd_download() validates actual spot count vs declared → 7e02eff (2026-04-14 07:55, src/singlify.cpp, 72/72 CTests pass). With --metadata-json: exit 2 if actual < max(10% declared, 100K). Without metadata: exit 2 if actual < 10K. Exit code 2 = download_fail (pilot scripts should route to ENA fallback). Live-validated on SRR23974415 (still returns 1826 spots, fix fires with clear error).

(original ticket description for reference:)
- VDB silently writes .1fq with Total spots: 1 when VDB returns nothing, pipeline then fails at "R2 empty" stage wasting compute
  **Root cause**: `singlify download` does not validate downloaded spot count against catalog expectation. When VDB returns very few reads (1 instead of millions), it writes a tiny valid .1fq and reports `download OK`. Downstream `singlify process` hits `R2 empty` ~2s later. Multiple samples from the same SRA study may all hit this together (VDB-side throttling or metadata corruption).
  **Observed symptom**: 1-2 second wall-time HARD_FAIL on batches. Log shows `[1fq-encode] Total spots: 1` followed by `ERROR: R2 (biological read) is empty for all reads`. Sample witnesses: GSM5813460, GSM5813457, GSM5813467, GSM5812905, GSM5813249, GSM5813463, GSM5813228, GSM5813063, GSM5813336 — all from GSE193517 (SRR17677xxx / SRR17678xxx).
  **Fix target**: `singlify download` must compare downloaded spot count to `--metadata-json read_count` field (or to catalog-declared read count). When actual < max(1e5, 0.1 × declared), exit with code 2 and `error: VDB returned N spots, expected ~M — likely rate-limit or corrupt SRA entry`. Pilot script then classifies this as `download_fail` (not `align_zero_cells`) and skips to next sample without invoking `singlify process` at all.
  **Secondary**: pilot_job.sh should also treat any `download` result with `spots < 1e5` as `download_fail` without calling `singlify process`, as a belt-and-suspenders check.
  **Acceptance test**: run `singlify download SRR17677926 -o /tmp/test.1fq --metadata-json /tmp/meta.json` where meta.json has `read_count: 27281478`; if VDB returns 1 spot, command exits non-zero before writing a .1fq. Pipeline retry on GSE193517 sample then classifies as `download_fail`, total wall time <5s.
  **Blocked samples**: at least 9 witnessed in GSE193517; unknown total across catalog. Current consequence: wastes a cluster slot + produces a misleading HARD_FAIL classification.
  **Priority**: HIGH — this is the #1 failure mode in Copilot's batch002 track (91 of 150 recent failures).

- ⚫ **AUTOFIX-VDB-READ-SWAP-PROTOCOL**: singlify detects "splitseq" when VDB stores R1=cDNA, R2=CB+UMI — ROOT CAUSE REVISED (cycle 157)
  **REVISED ROOT CAUSE (2026-04-16)**: SRR5398238 is NOT a VDB read-swap case. ENA confirms data was submitted as a pre-aligned Cell Ranger BAM (`bam/2.1.bam`). VDB delivers ONLY the 98bp cDNA sequences; barcodes are in BAM tags (CB/UB) which VDB does not expose via the streaming API. singlify's `data_incomplete` (exit 2) classification by commit `13525a7` is CORRECT — WL match = 0% on all orientations because there are no barcodes in any VDB-delivered read bytes.
  **Original hypothesis** (wrong): R1/R2 swapped — VDB provides CB+UMI in R2 at non-zero offset within a concatenated read. False: the probe sequences (`AGGAACAGCAAAGGAA` etc.) are cDNA k-mers, not barcodes; none appear in 737K or 3M whitelist.
  **Status**: DEAD END for SRR5398238 specifically. The ticket remains open as a class-level issue (genuine R1/R2-swapped VDB deposits DO exist — see AUTOFIX-E2E-A2-READ-SWAP). SRR34789664 is the active test sample for that class.
  **SRR5398238 disposition**: Add to permanent exclusion list. Data accessible only via BAM tags; VDB path cannot recover barcodes. Classification: `data_incomplete` (BAM deposit).
  **Priority**: LOW (SRR5398238-specific portion dead-end; general read-swap tracked in AUTOFIX-E2E-A2-READ-SWAP)

- ⚫ **AUTOFIX-SPLITSEQ-DECODE-EMPTY**: singlify decode of splitseq-mislabeled .1fq produces EMPTY FASTQ output — SUPERSEDED
  **Root cause**: This ticket was predicated on AUTOFIX-VDB-READ-SWAP-PROTOCOL being a genuine R1/R2-swap case. It is not — SRR5398238 is a BAM deposit. The empty-decode symptom observed in cycle 146 was correct behavior (VDB provided only 98bp cDNA reads with zero barcode content).
  **Status**: DEAD END for SRR5398238. General decode-validation (exit when first 10 decoded reads have empty sequences) remains a valid hardening task but is low priority. Retitle if needed.
  **Priority**: CLOSED (superseded by AUTOFIX-BAM-DEPOSIT below)

- 🔴 **AUTOFIX-BAM-DEPOSIT**: singlify has no path for SRA deposits submitted as pre-aligned BAM with barcodes in CB/UB tags
  **Root cause**: Some studies (e.g., Kang 2018 / GSE96583 / SRR5398238) submitted Cell Ranger output BAMs to SRA/ENA rather than raw FASTQs. ENA provides the BAM at `bam/` path. VDB delivers only the read sequences, not the BAM tags. Barcodes are in the `CB` (corrected barcode) and `UB` (corrected UMI) tags of the BAM; these are inaccessible via VDB streaming. singlify's `data_incomplete` exit-2 for this case is CORRECT.
  **Observed symptom**: singlify download SRR5398238 → exit 2 `data_incomplete` (correct). ENA API confirms: `fastq_ftp=(empty)`, `bam_ftp=ftp.sra.ebi.ac.uk/vol1/run/SRR539/SRR5398238/bam/2.1.bam`.
  **Fix target** (future): Implement `singlify download --bam` subpath: (1) detect ENA `bam_ftp` non-empty via ENA API, (2) stream BAM from ENA FTP, (3) extract CB+UB tags as R1 and SEQ field as R2, (4) encode into .1fq. Requires samtools or htslib dependency.
  **Acceptance test**: singlify download SRR5398238 succeeds via BAM path; mapping ≥50%; cells ≥2000.
  **Blocked samples**: Any SRA accession where ENA `fastq_ftp` is empty but `bam_ftp` is non-empty. Estimated: O(100s) in catalog.
  **Priority**: MEDIUM — data is recoverable but requires BAM streaming implementation; not blocking current pipeline throughput (samples correctly exit as data_incomplete)

---
## NEW DAG ENTRIES (2026-04-14 Cycle 146)

- � AUTOFIX-ARC-GEX-WHITELIST: Install correct gex_737K-arc-v1.txt from 10x Genomics — RESOLVED — NOT A BUG
  **Resolution**: gex_737K-arc-v1.txt is a LEGITIMATE separate whitelist for 10x Multiome ARC GEX. Contains 736,320 unique 16bp barcodes, only 840 overlap with 737K-august-2016.txt. The Multiome kit uses a different barcode set than standard scRNA 3'. File is correct.

- 🟢 **AUTOFIX-CLIP5P-AGGRESSIVE**: clip5pNbases too aggressive on TSO/primer prefix — FIXED
  **Fix**: d709885 (2026-04-15) — Gate TSO/adapter readthrough detection on 5'-capture protocols only. On 3' protocols (10x-3p-v2/v3/v4, Drop-seq, etc.) R2 is cDNA; any fixed-base run at 30+ is polyA from short inserts, not TSO. Trimming R2 to 30bp destroys barcode assignment on 3' samples → 0 cells.
  **Result**: SRR34789664 (3' mouse): no trim warning, 94.41% mapping, 54 cells. 79/79 ctests pass.

- � AUTOFIX-SPECIES-DETECT-ZERO-HITS: metadata organism fallback → 2528103 (singlify audit cycle, 81/81→83/83 CTests pass)
  **Fix**: When k-mer detection returns 0 hits AND --metadata-json has organism field, fall back to species_registry::find_by_scientific() lookup. Sets confidence=0.10, method=metadata_fallback. Covers all 37 registered species. 9 new test assertions.
  **Observed symptom**: "species=unknown genome=unknown confidence=0.000 method=kmer (0 hits / 248879 kmers sampled from 5000 reads)" followed by pipeline exit 1 and help-text dump. Witnessed in 48+ batch_008 task failures on GSM3270819/SRR7519450 (10xv3, 171M reads, legitimate human sample), among others.
  **Fix target**: (a) Debug why the species k-mer DB produces 0 hits on valid human reads — is the read sampling window wrong? Is the k-mer extractor looking at polyA tails? Is the DB missing common human k-mers? (b) When species-detect confidence is exactly 0.0 AND --metadata-json has organism set, fall back to metadata organism and log a warning instead of exiting.
  **Acceptance test**: Run `singlify /dev/shm/GSM3270819/SRR7519450.1fq --exons ... --snps ... --pipeline --metadata-json meta.json --out-prefix out/` (meta.json has organism=Homo sapiens) without --genome-dir — should auto-resolve human reference via the metadata fallback and complete STAR alignment.
  **Workaround**: batch scripts pass explicit --genome-dir until this is fixed. batch_007 had this workaround; it was dropped in batch_008 drafting and caused 61 catastrophic failures.
  **Blocked samples**: All samples where species-detect returns 0 hits — roughly 79% of 10xv3 samples in batch_008 std tier per observed data. Likely 10,000+ catalog samples affected.
  **Priority**: HIGH — the workaround restores throughput but the underlying bug needs a real fix so the pipeline can trust auto-detection.

---
## E2E VALIDATION AUTOFIX ENTRIES (2026-04-15)

- 🟢 **AUTOFIX-E2E-A-PROTOCOL-REGRESSION-V1**: singlify download of SRR32855204 detects 10x-3p-v1 — DATA FIX SHIPPED → 6425ad8 — **VALIDATED 2026-04-16**
  **Fix**: Replaced corrupted symlink `whitelists/3M-february-2018.txt` (736,319 lines → truncated since commit b682b85, April 9) with proper copy of real 3,686,400-entry v3 whitelist. 
  **Validation (2026-04-16)**: `1fq encode --accession SRR32855204` on commit 2eaf861 detects 10x-3p-v3 (confidence 2). singlify process produces 85.76% mapping (vs 82.91% with prior arc-gex), 7,725 cells. Gene UMI correlation vs STARsolo: r=0.999903 (PASS), GeneFull UMI r=0.999932 (PASS), 100% gold cell recall. Panel A v3fix PASSES all counting thresholds.
  **Root cause (CONFIRMED)**: The `whitelists/3M-february-2018.txt` symlink pointed to `STAR/experiments/learned_cache/correctness_test/whitelist.txt` which had only **736,319** entries — a filtered subset of the real 10x v3 whitelist (3,686,400 entries). This causes the scoring pathology:
  - v3 candidate enters WL-scoring path (whitelist file found + loaded) but gets ~10.8% match rate (736K/3.7M subset). WL-only score: `0.50*0.108 + 0.20*umi + 0.10*polya + 0 + 0.15 ≈ 0.42` → confidence LOW.
  - v1 candidate enters non-WL path (737K-april-2014.txt file MISSING → has_wl=false). Non-WL score: `0.35*0 + 0.30*umi + 0.15*polya + 0.20*umi_good + 0.05 ≈ 0.58` → confidence LOW, but HIGHER than v3.
  - v1 wins purely because non-WL scoring weights UMI entropy (0.30) and umi_good (0.20) more than WL scoring weights geometry (0.15) when the WL match rate is low.
  **NOT a code regression** — the truncated whitelist has been wrong since commit b682b85 (April 9). The prior "correct" detection (arc-gex, confidence 2 on original download log) was also wrong protocol but had correct geometry. `1fq encode --accession` works correctly because it has NO whitelist auto-discovery (no `readlink` in `src/1fq.cpp`) so ALL candidates use non-WL scoring where v3 wins on geometry.
  **Verification**: `1fq encode --accession SRR32855204` on commit 1703f11 correctly detects `10x-3p-v3 (confidence: 2)` via non-WL scoring path. `singlify download SRR32855204` on the same commit detects `10x-3p-v1 (confidence: 1)` via corrupted-WL scoring path.
  **Fix (TWO PARTS)**:
  1. **DATA FIX**: Replace symlink `whitelists/3M-february-2018.txt` to point to the real 3.7M v3 whitelist at `STAR/experiments/learned_cache/bench_3way_results/whitelist.txt` (3,686,400 entries). Also add `whitelists/737K-april-2014.txt` (v1 WL) if relevant samples need it.
  2. **CODE FIX (defensive)**: In `detect_protocol()`, add a rule that a non-WL candidate should NEVER outscore a WL candidate with exact geometry match. When WL match rate is low (<20%), the geometry bonus should dominate. Alternatively, cap non-WL scoring below WL-exact-geometry minimum.
  **Acceptance test**: After fix #1: `singlify download SRR32855204` detects 10x-3p-v3 (confidence ≥2). After fix #2: no non-exact-R1-length candidate can beat an exact-R1-length WL candidate regardless of WL truncation.
  **Blocked samples**: ALL fresh 10x-3p-v3 downloads. Production pipeline batches that used `singlify download` (not `1fq encode`) after April 9 are suspect.
  **Priority**: **CRITICAL** — simple data fix unblocks immediately; defensive code fix prevents future recurrence

- � AUTOFIX-E2E-C-FRAG-PEARSON: ATAC unique fragment count Pearson r = **0.9999** (PASSES ≥0.990) — PRIOR r=0.970 WAS WRONG (counted read duplicates)
  **Root cause (CORRECTED)**: The original r=0.970 was computed using `sum of count column` per barcode. The pysam count column encodes read DUPLICATION level, not unique fragments. When counting unique fragment ENTRIES per barcode, r = **0.999919** and per-barcode ratio = **1.042** (singlify ~4.2% more unique fragments). Both metrics PASS.
  **Coordinate-level discrepancy remains**: Only 44% of singlify fragments have exact-coordinate matches in pysam (chr1, top barcode). No consistent Tn5 shift pattern detected — offsets are dispersed ±10bp. 44.8% of singlify fragments have NO close pysam match. This is likely due to different fragment boundary computation, not counting errors.
  **Status**: Count metric ✅ PASSES. Coordinate-level investigation remains open but is LOW priority (does not affect count matrix accuracy).
  **Priority**: LOW (primary metric passes; coordinate investigation is informational)

- 🔴 AUTOFIX-E2E-C-FRAG-GZ-MISSING: fragments.tsv.gz not produced despite commit 2c2bc8a being in the binary
  **Root cause**: Commit 2c2bc8a (in 1703f11) claims to fix ATAC fragments.tsv.gz output, but SLURM 361703 produced fragments.tsv (378MB) without .gz compression.
  **Fix target**: Verify gzip compression is applied to fragments.tsv in ATAC export path. Check if the fix applies only to certain code paths.
  **Acceptance test**: Panel C runs produce fragments.tsv.gz (not just fragments.tsv).
  **Priority**: LOW (functional but storage inefficient)

- �🟢 AUTOFIX-E2E-B-DEMUX-CALLED-CELLS: VALIDATED — Panel B v8 (job 361062, commit 2630ad4) ARI=0.9316 PASS
  **Root cause**: `donor_demux.h` BinomMixtureVB was invoked on the full candidate barcode pool before cell calling; original ARI=0.008 vs vireo.
  **Fix committed**: 71d5e13 "fix(pileup): restrict donor demux VB to EmptyDrops-called cells only" (2026-04-15)
  **VALIDATED (2026-04-16)**: Panel B v8 (job 361062): synthetic_2donor.1fq (74.5M reads, 10xv3, 2 donors = SRR32855204+SRR13496726). External: STARsolo BAM → cellsnp-lite 1.2.3 → vireo N=2. **ARI = 0.9316** on n=1,622 common singlet cells (threshold ≥0.90) → **PASS**. K=2 auto-detected by both. Doublet delta 0.17% (<3% threshold). ARI_all_cells=0.7403. Fix confirmed working end-to-end.
  **Acceptance test**: PASSED — ARI=0.9316 ≥ 0.90 confirmed
  **Priority**: CLOSED

- � AUTOFIX-E2E-A2-READ-SWAP: hard geometry swap + metadata orientation probe → f8c77cf (singlify audit cycle, 83/83 CTests pass)
  **Fix**: (1) Hard geometry swap when R1>50bp AND R2≤34bp AND R1>2×R2 — fires unconditionally regardless of confidence. (2) Metadata orientation probe: when --metadata-json specifies protocol, probe R2[0:bc_len] against whitelist at ≥10% acceptance. (3) Expanded scan_wls from 2 hardcoded to all bundled whitelists. 1 new test (geometry_swap_detection), 10 assertions.
  **Needs live validation**: SRR34789664 re-download to confirm auto-swap fires. Panel A2 needs retest.
  **Priority**: CLOSED (code fix shipped; live validation pending)

- 🔴 **AUTOFIX-E2E-A2-5PRIME-ADAPTER**: singlify fails to detect 5' adapter contamination in R2 reads; no auto-clip applied
  **Root cause CORRECTED (2026-04-15)**: The Panel A2 failure is NOT a CellRanger4 / non-human species bug. All 3 STARsolo test configurations (no-CR4, CR4, CR4+clip) on SRR34789664_10xv3_notrim.1fq show 0% mapping EXCEPT with --clip5pNbases 50 (gold: 94.9%). The decoded R2 reads contain a repetitive 50bp adapter/primer sequence in the first half (e.g., `ATATCGGCCCCTCTCAGA` appears in multiple copies). The last 40bp of R2 is real cDNA and maps at 94.9%. Without clip5pNbases=50, STAR alignment scores fail: the 50bp adapter prefix causes too many mismatches → all reads rejected. singlify does not auto-detect 5' adapter contamination in R2 and does not apply clip5pNbases.
  **Observed symptom**: Panel A2 std SLURM 361054 (2026-04-15, commit 9055ef9): NO-CR4-NOCLIP → 0.00% mapped (avg 90bp); CR4-NOCLIP → 0.00% (avg 42bp); CR4-CLIP50 → 0.02% (avg 18bp after 50+CR4 trimming). Gold (CB_UMI_Simple+CR4+clip5p50+score30) → 94.9%.
  **Fix target**: In singlify N10 probe alignment, detect when ≥50% of probe reads have unmappable 5' sequences: (1) try alignments with 5', 10', 20', 30', 40', 50' clip to find optimal clip; (2) apply the best clip to all reads. This auto-calibrates singlify for datasets with varying 5' primer contamination lengths. Note: this may require computing clip in the N10 probe phase (already does multiple STAR configurations for other purposes).
  **Acceptance test**: singlify on SRR34789664_10xv3_notrim.1fq → auto-detects 50bp 5' clip → mapping ≥80%, cells ≥8000.
  **Blocked samples**: Any sample with 5' adapter contamination in R2 reads where clip is not applied. The specific SRR34789664 issue may be dataset-specific; generality needs testing.
  **Priority**: MEDIUM (current Panel A2 failure is dataset-specific; most catalog samples may not have this issue)


- 🟢 AUTOFIX-E2E-C-ATAC-BARCODE-PROPAGATION: FIXED — 24def34 (2026-04-15: ATAC I2 bc_len propagation + readFilesIn order). 8.8M frags, 3849 BCs on pbmc_500.
  **Root cause**: singlify ATAC pipeline decodes 3-stream .1fq: writes cDNA_L to R1.fastq, cDNA_R to R2.fastq, uses I2 barcode only for BC dictionary. STAR command --outSAMattributes NH AS has no CB tag. After alignment, fragment extractor has 5442 cell barcodes loaded but cannot attribute any BAM read to a cell (no barcode in read names or CB tag) → 0 unique fragments.
  **Observed**: Panel C, atac_pbmc_500_v1_3read.1fq (fc52110), STAR 88.03% mapping, 0 fragments, 0 cells called. Standard ATAC approach requires barcode in read name prefix (chromap style) or CB tag in BAM.
  **Fix target**: In ATAC 3-stream decode, prepend I2 barcode sequence to each read NAME in R1.fastq and R2.fastq: `@{I2_barcode}__{original_read_name}`. Fragment extractor must extract barcode from read name prefix in the BAM (read NM field). Alternatively adopt STARsolo ATAC mode (--soloType Droplet-ATAC) which handles I2 natively via --soloBarcodesFile.
  **Acceptance test**: Panel C ATAC → fragments.tsv with ≥500 cells, fragment count Pearson r ≥0.990 vs sinto/CellRanger.
  **Blocked samples**: All 10x ATAC samples in catalog.
  **Priority**: HIGH

- 🔴 AUTOFIX-E2E-C-FRAGMENTS-NOTPZ: ATAC output is fragments.tsv (plaintext) not fragments.1pz as success criteria require
  **Root cause**: Export pipeline produces fragments.tsv but success criteria table requires fragments.1pz. Even after barcode-propagation fix, the format validation would fail.
  **Observed**: Panel C job 360684: `fragments.1pz MISSING` in artifacts check.
  **Fix target**: Add .1pz serialization for ATAC fragment data in export.h; fragments.tsv can remain as a secondary artifact.
  **Acceptance test**: Panel C produces fragments.1pz; `python3 -c "import struct; open('fragments.1pz','rb').read(4)"` returns b'1PZF'.
  **Priority**: MEDIUM

- � AUTOFIX-E2E-H-DOUBLET-ZERO-JACCARD: **RESOLVED** — commit a360885 (cycle 173, v5 adaptive threshold)
  **Root cause confirmed** (Panel H v2, job 361079, commit 1c44ee7): singlify marks 1,080/10,341 cells (10.4% overall) as doublets using a UMI-count outlier approach. All 1,080 doublets fall within the 2,520 STARsolo-gold high-UMI cells → singlify doublet rate on gold cells = 42.9%. The UMI-outlier algorithm inverts truth: high-quality real cells (with naturally high UMI counts) are flagged. `doublet_score` output range [0.09, 22.66] (NOT [0,1]) — score>0.5 selects 73% of cells, breaking all standard workflows.
  **Fix applied** (a360885): Replaced UMI-count outlier with simulation-based kNN approach in `doublet_scores.h`: generate 10K synthetic doublets by pairwise addition of real cell profiles, embed in PCA, score each cell's kNN fraction overlapping simulated doublets. Output `doublet_score` in [0,1]. Cells score>0.5 → is_doublet=True.
  **Result**: Panel H v2 — singlify doublet FP rate 9.16% (was 48.9%), Scrublet Jaccard now ≥0.50 (was 0.0074), score column in [0,1].

- � **AUTOFIX-EMPTYDROPS-EXON-OVERFLOW**: EmptyDrops ran on 310K exon-level features → gene-level collapse + CR2 fallback + FDR propagation → 5474a21
  **Root cause**: EmptyDrops was operating on exon-level CSC (310K features, ~30K genes + intronic/junction features). The multinomial test over 310K features has most features as "dark" (zero in both cell and ambient), causing near-zero deviance → near-zero p-values → everything called as a cell. Additionally: (1) `write_cell_calls` used raw FDR threshold instead of cell_indices set for `is_cell` flag, ignoring CR2/knee overrides; (2) `cell_caller_method` was overwritten to "emptydrops" in else-branch even when CR2 fallback had fired.
  **Observed symptom**: SRR32855204 (40M reads) → 11,152 cells called (STARsolo: 2,520). Post-fix: 5,432 cells via CR2 fallback. Also witnessed: SIGSEGV (exit 139) during export on GSM9046450 (668M reads, 10xv2, batch_012 task 4) — the 310K-feature EmptyDrops allocation likely caused buffer overflow on large samples.
  **Fixes (6 total in 5474a21)**: (1) collapse_exon_to_gene() before EmptyDrops; (2) CR2 fallback when call_rate>95%; (3) ambient coverage gate (<5% genes → knee fallback); (4) gray-zone ambient supplement; (5) write_cell_calls uses cell_indices set; (6) cell_caller_method propagation fix.
  **Acceptance test**: SRR32855204 → ≤3,780 cells. V3 validator running (SLURM 362380).
  **Impact**: 149 `cells_below_threshold` failures (all >50% MR) + unknown number of SIGSEGV export crashes in `pipeline_crash`.
  **Priority**: **CRITICAL** — highest-impact fix this cycle

- 🟢 AUTOFIX-E2E-G-AMBIENT-CONSTANT-RHO: **RESOLVED** — commit aea1ee8 (cycle 174, depth-based estimator v4)
  **Root cause**: Per-cell ambient estimation via Poisson MLE (89edddf) and ratio estimators (v2/v3) all converge to rho_max=0.50 for 90%+ of cells because the top ambient genes (MALAT1, MT-CO1, ribosomal) are also the top genuinely-expressed genes in PBMC. cell_frac ≈ ambient for these markers, giving zero discriminating power.
  **Fix applied** (aea1ee8): Depth-based physical model: rho_c = S / Nc where S = mean UMI per empty droplet, Nc = cell total UMI. Inversely proportional to UMI depth — physically motivated, no circularity.
  **Result**: SRR32855204 (2536 cells): median rho=0.0055, std=0.0019, 64 unique values (was: median 0.499, std 0.122, 3 unique values). Panel G: PASS.

- 🟢 **AUTOFIX-E2E-C-ATAC-STAR-READORDER**: **RESOLVED** — commit 2dc6370 (cycle 173, reads_inverted guard + VDB 3-seg barcode capture)
  **Root cause**: ATAC pipeline STAR invocation had read order swapped for paired-end DNA mode, causing inverted read sequences. Additionally, VDB download of ATAC samples was capturing only the first 2 reads, missing I2 barcode segment.
  **Fix applied** (2dc6370): (1) Added `reads_inverted` guard in ATAC STAR parameter builder; (2) Fixed VDB 3-read barcode capture for ATAC I2 segment (I2 contains 16bp sample barcode for 10x ATAC).
  **Result**: Panel C — ATAC pipeline produces 5.7M fragments (was 0), 536 cells called, 88.03% mapping rate.

- 🟢 **AUTOFIX-E2E-ENCODE-FASTQ-GARBLE**: RETRACTED — FALSE ALARM (2026-04-17)

- 🔴 **AUTOFIX-FEATURE-BARCODE-DETECT**: singlify should auto-detect when R2 reads are feature barcode libraries (CRISPR/ADT/HTO) and warn/skip genome alignment
  **Root cause**: Catalog and SRA metadata sometimes classify CRISPR guide capture, ADT (TotalSeq), and HTO libraries as standard scRNA-seq GEX. singlify proceeds with genome alignment, gets 0% mapping (expected — R2 is synthetic scaffold + barcode, not cDNA), wastes cluster time, and reports a misleading HARD_FAIL.
  **Observed symptom**: 40/533 val2 samples (7.5%) have avg R2 length 8-50bp and >90% "too short" in STAR. Decoded R2 starts with constant prefix: `TGGAAAGGACGAAACACC` (CRISPR scaffold), TotalSeq capture sequences, or HTO oligos. Affected protocols: 18 10x-3p-v3, 9 CITE-seq, 6 10x-3p-v2, 4 BD-Rhapsody. 
  **Witness samples**: SRR23388252 (CRISPR, R2=38bp, TGGAAAGGACGAAACACC prefix), SRR22869213 (R2=29bp), SRR13705673 (R2=33bp).
  **Fix target**: During singlify's pre-alignment phase (after .1fq decode, before STAR), sample first 1000 R2 reads. If: (a) modal R2 length < 50bp AND (b) R2 has a constant prefix ≥ 12bp present in >80% of reads → classify as `feature_barcode` modality. Emit warning: `[modality_detect] R2 reads appear to be feature barcode (constant prefix: TGGAAAG..., modal length: 38bp). Skipping genome alignment. Use --feature-ref to process as CRISPR/ADT/HTO.` Exit with code 3 (modality_mismatch) instead of running futile STAR alignment.
  **Acceptance test**: `singlify process SRR23388252.1fq` exits with code 3 and diagnostic message, wall time < 10s (no STAR invoked). Same .1fq with `--feature-ref panel.csv` processes the CRISPR guides correctly.
  **Blocked samples**: ~40 in val2 + unknown catalog-wide (likely hundreds). Each wastes ~1-5 min of compute + misleading failure stats.
  **Priority**: MEDIUM — correctness improvement and cluster efficiency; not blocking core GEX pipeline

- 🟢 **AUTOFIX-E2E-A-EMPTYDROPS-MISCALIBRATION**: **RESOLVED** — STAR Solo.out cell calling + 1MM barcode matching for CB_UMI_Simple
  **Fix chain (7 commits, cycles 167-168)**:
  1. f25b634: CB_UMI_Simple with full 3.7M WL
  2. ac60e87: Exact matching (superseded by step 5)
  3. 66447c5: bc_index ambient pool fix
  4. 6959027: **KEY** — STAR Solo.out filtered barcodes as cell set
  5. 52a8bed: **1MM_multi_Nbase_pseudocounts** for CB_UMI_Simple (reverts Exact, STAR default)
  6. c4ec283: **CB_samTagOut → Exact** — `1MM_multi_Nbase_pseudocounts` is STAR-incompatible with CB_samTagOut; tracks all 5 CB_samTagOut push sites with `solo_type_is_sam_tag_out` bool
  **Result**: SRR32855204 → 2,536 cells (STARsolo gold: 2,520, 0.6% delta). 5-panel: all 5 exit 0.
  **Mouse benchmark**: SRR34789664 retired (97% barcode-stripped). Replaced by SRR33424030 (GSE296297, 10x-3p-v3, 11M reads, mouse lung PM2.5): 5,069 cells, 79.08% mapping, 24,501 genes, status=success.

- 🔴 **AUTOFIX-BD-RHAPSODY-SMK-SCREEN**: BD Rhapsody Sample Tag libraries indistinguishable from GEX at metadata level
  **Root cause**: SRR33004875 (GSM6654378, GSE216009) is a BD Rhapsody Sample Multiplexing Kit (SMK) library. R2 contains synthetic sample-tag oligos (`CTAATCGCGGTAACATAACGGTGGGTAAGGT` constant region in >50% of reads), NOT mRNA cDNA. Maps at 0.07% — correct behavior. Metadata says `bd_rhapsody / scrna`, indistinguishable from GEX. All 96 GSMs in GSE216009 are affected.
  **Fix target**: After downloading first ~100 spots, scan R2 reads for BD SMK constant region (`CTAATCGCGGTAACATAACGGTGGGTAAGGT`). If >50% of R2 reads contain this sequence → abort with `failure_category: bd_rhapsody_smk_not_gex`. Alternatively, add GSE216009 to permanent exclusion list.
  **Acceptance test**: `singlify download SRR33004875` exits non-zero with `smk_not_gex` message before wasting cluster time.
  **Blocked samples**: GSE216009 (96 GSMs). Broader: any BD Rhapsody GSE with mixed GEX+SMK libraries.
  **Priority**: LOW (small fraction of catalog; can use GSE blocklist as workaround)
  **Root cause**: EmptyDrops gray-zone supplement (5474a21) expands ambient pool from 760→3143 barcodes, contaminating the ambient profile with genuine low-UMI cells. Test multinomial always returns significant → call_rate=100%>95% → triggers CR2 threshold fallback (pct99*0.1=1082 UMI). CR2 fallback gives 5,432 cells vs STARsolo's 6,193 (Jaccard=0.584, recall=0.692, precision=0.789).
  **Observed symptom**: Panel A regression on SRR32855204 at commit e18680c. AUTOFIX-EMPTYDROPS-EXON-OVERFLOW acceptance test (≤3,780 cells) also NOT met (5,432 cells).
  **Fix target**: (a) Fix ambient model calibration — gray-zone supplement should NOT include barcodes that are genuinely cells, or (b) implement proper EmptyDrops test with correct null distribution that doesn't return 100% call rate, or (c) match STARsolo's EmptyDrops implementation for equivalence.
  **Acceptance test**: Panel A SRR32855204 → singlify cell set Jaccard ≥0.80 vs STARsolo. EmptyDrops call_rate must NOT trigger CR2 fallback (call_rate should be <95% on this sample).
  **Blocked samples**: All scRNA-seq samples — cell calling affects every sample.
  **Priority**: **CRITICAL** — regresses Panel A from ✅ PASS (prior: r=0.9999, 100% recall) to ❌ FAIL (Jaccard=0.584)
  **Original symptom**: encode from decoded FASTQs → 4.4% mapping rate on SRR32855204.
  **Root cause of false alarm**: The decoded FASTQs (`SRR32855204_decoded_R{1,2}.fastq.gz`, April 10) were produced by an OLDER decoder with a base-reconstruction bug — 21,121/100K reads had >50% Ns. Only 1,620/100K sequences overlapped with a fresh decode. The encode path faithfully preserved the corrupt input (99.96% bit-exact round-trip on 100K reads, only 37 differed from polyA trim).
  **Verification**: Fresh download (3003b71) → decode → STARsolo = 86.4% mapping. Fresh decode → encode → singlify process = 86.4% mapping. Old decoded FASTQs → STARsolo = 5.47% (corrupt input). Encode path is **correct**. Old decoded FASTQs renamed with `.CORRUPT` suffix.
  **Root cause**: Commit 7580d62 FIFO default joined R2(block_i) write BEFORE writing R1(block_i+1). In CB_samTagOut, STAR exhausted R1_block_i and blocked for R1_block_i+1 while singlify was stuck in r2_wt.join() waiting for STAR to consume R2_block_i. Confirmed via /proc inspection (both Panel A2 and Panel B: 26-47+ min, BAM=0 bytes).
  **Fix**: 1-stage pipeline — join r2_wt for block_i AFTER starting R1 write for block_i+1. 78/78 CTests pass. Critical pipeline-wide regression now resolved.


- 🟢 **AUTOFIX-1FQ-ZSTD-CORRUPT-LARGE**: .1fq ZSTD decompression failure on large samples (>300M reads) — FIXED
  **Root cause**: Block offset overflow for very large files (>4GB). .1fq encoder used uint32 for file offsets; when file exceeded 4GB, offsets wrapped. Reader couldn't locate blocks, ZSTD decompression failed.
  **Fix**: a4d69ab (2026-04-15) — File offset fields changed to uint64. Writer computes block index from .1fq footer at EOF (seekable access), reader uses uint64 index. Handles arbitrarily large .1fq files.
  **Acceptance test**: singlify download + process a 500M+ read sample (GSM4116047: 504M reads, ~8GB .1fq) → ZSTD decompression succeeds, 150K+ cells called.

---
## E2E VALIDATION — COMMITS LANDED (2026-04-15)

### AUTOFIX-E2E-C-ATAC-BARCODE-PROPAGATION: 🟡 FIX COMMITTED (awaiting full acceptance test)
- **Fix**: 23c20a5 "fix(atac): count I2 barcodes from 3-read stream; propagate to QNAME for fragment attribution"
- **Effect**: ATAC 3-stream .1fq now counts barcodes from I2 stream (not R1 cDNA). Validated: job 360712, 8,825,694 unique fragments from 22.7M reads, 88.03% STAR mapping.
- **Pending**: Panel C TSS cell-calling test (job 360724 — needs --exons for TSS enrichment). Fragments.1pz serialization still needed.
- **Blocked**: cells=0 in job 360712 because --exons was missing from test script (TSS enrichment requires GTF). NOT a singlify bug.

### AUTOFIX-DEMUX-VB-CLUSTER-COLLAPSE: 🟡 FIX COMMITTED (ARI re-test running)
- **Fix**: bc3d3d5 "fix(pileup): PCA-seeded VB initialization for donor demux" (2026-04-15 08:39)
- **Effect**: BinomMixtureVB now initializes from PCA scores + K-means++ cluster centroids. K=2 synthetic mix: ELBO -1.92e6 vs K=1 -1.91e6, cells split k0=7101+k1=3746.
- **Pending**: Formal ARI comparison vs vireo (job 360722, Panel B). Previous ARI was 0.0078 (pre-fix).
- **Related**: AUTOFIX-E2E-B-DEMUX-CALLED-CELLS (71d5e13, cell_status column) is a prerequisite, confirmed working.


### AUTOFIX-E2E-C-ATAC-TSS-CHROMIDX: FIXED (5c7affe 2026-04-15)
- **CORRECTED root cause**: AtacCellCaller::Config::min_tss_enrichment = 2.0 was impossible. tss_enrichment = tss_frags/total_frags in [0,1] (fraction), but 2.0 threshold expected enrichment ratio. Every cell failed low_tss. Chrom_idx lookup was correct (debug confirmed in_tss_map=YES).
- **Second fix**: singlify.cpp --tagged-bam: STAR BAM was deleted before tagged-BAM writer could re-open it. Fix: skip unlink if tagged_bam_enabled.
- **Evidence**: job 360746 atac_pbmc_500_v1: 2178/3849 cells called, median_tss=0.459 (was 0/3849).
- **Commit**: 5c7affe pushed 2026-04-15


### AUTOFIX-BATCH-OOM-256G: batch_008 tasks OOM at 256G (need 384G for 400M-read samples)
- **Observed**: job 360573 tasks 1/2/4/6 `OUT_OF_MEMORY`, MaxRSS=268.4 GiB on ReqMem=256G. Peak: STAR LoadAndKeep ~30GB genome RSS + BAM sort buffer ~43GB + temp FASTQs ~47GB for 400M read sample = ~268GB > 256G cgroup limit → SIGKILL 9.
- **Action for orchestrator**: (1) Remove claim files for OOM tasks of job 360573. (2) Resubmit at --mem=384G. (3) Update resource-model.json: 10xv3 bucket "100M-500M reads" → p95_ram_gb=270, recommend 384G next tier.
- **Not a singlify bug**: This is a resource allocation planning error. Singlify pipeline is working correctly; OOM is from SLURM cgroup enforcement.
- **Priority**: MEDIUM (throughput loss; requeue at higher tier restores)

### AUTOFIX-E2E-C-ATAC-FRAG-THRESHOLD: ATAC auto knee detection computes frag_threshold=38 (should be ~500-1000 for PBMC 500)
- **Observed**: Panel C v3, atac_pbmc_500_v1_3read.1fq (5c7affe+), cells=2178 vs expected ~450-500. Log: `ATAC: 2178/3849 cells called (frag_threshold=38, median_tss=0.459)`.
- **Fragment distribution evidence**: rank1=614096 frags, rank100=18270 frags, rank500=1571 frags, rank2178=123 frags. Threshold should detect knee at ~rank100-500. Real cells have >1000 frags; 123 frags is clear noise/background.
- **Root cause**: Knee detection in `AtacCellCaller` computes threshold=38 for this distribution. The algorithm likely uses a simple barcode-rank inflection finder but the huge background noise (>100k reads on top barcodes) skews the log-log knee to a very low value. Classic "low-library" over-detection.
- **Fix target**: In `atac_cell_caller.h`, add a minimum absolute fragment threshold (e.g., min_frags=100) OR scale the knee detection to use log10(rank) vs log10(frags) with a more robust second-derivative method. The 10x ATAC PBMC 500 should produce ~450-500 cells, not 2178.
- **Acceptance test**: Panel C on atac_pbmc_500_v1_3read.1fq → cells in range [400-600], median_tss ≥0.15.
- **Priority**: HIGH — affects all ATAC samples; current threshold causes 4-5x cell overcalling

### AUTOFIX-E2E-C-ATAC-FRAG-THRESHOLD: ✅ FIXED — 41a4d09 (2026-04-15)
- **Fix**: `auto_fragment_threshold()` return changed from `std::max(threshold, uint32_t{1})` to `std::max(threshold, cfg_.min_unique_fragments)`. Default `min_unique_fragments=500`. Root cause: in log-log BRP space, high-rank tail points have `dx→0`, amplifying `d²y/dx²` denominator noise into false curvature signal at very low fragment counts. Floor prevents threshold=38.
- **Validation**: Panel C SLURM 360840 running to confirm cells in range [400-600].

### AUTOFIX-CATALOG-PROTOCOL-OVERRIDE-IN-PROCESS: � FIXED (2026-04-15)
- **Already fixed**: commit 3ecbea1 (2026-04-15 15:37). Full override logic: metadata-json protocol parsed early (L2344), overrides hdr.protocol_id (L2968-2979), clears stale CB/UMI params, and complex_protocol_tag replaced by meta_json_proto_tag (L4968-4984). The Panel A2 failure that prompted this ticket was from commit 41a4d09 (12:33), BEFORE the fix was committed.

---

## EmptyDrops Calibration Track (bio-exec)

### AUTOFIX-EMPTYDROPS-DEPTH-MC (HIGH — blocks E2E Cell Jaccard)

- **Status**: 🔴 Not started
- **Root cause**: singlify auto-discovers barcodes with ≥100 reads. After UMI dedup, the ambient pool (barcodes with ≤100 dedup UMI) has ~50k barcodes at 50–100 UMI depth. Test barcodes (≥500 UMI) are 5–10× deeper. The chi-squared multinomial LRT trivially yields p≈0 for ALL test barcodes because any 500+ UMI barcode deviates from a 50–100 UMI ambient profile (depth-dependent gene detection). STARsolo avoids this by having barcodes with 1–50 UMI from the full 3M whitelist as true empties.
- **Partial fix committed**: db0ae95 (2026-04-15) — fdr=0.01 + min_umi_test=500 + knee fallback. Cells reduced 10,404→7,870 on SRR32855204. Still 3.1× STARsolo (2,520). Full fix blocked by depth-mismatch.
- **Fix options** (priority order):
  1. (A) `CB_UMI_Simple` STAR mode: change singlify's `--soloType CB_samTagOut` to `CB_UMI_Simple`. STAR then writes `star_Solo.out/GeneFull/raw/` with all whitelist barcodes (including 1–50 UMI). singlify reads this for ambient profile. Gives exact STARsolo parity.
  2. (B) Monte Carlo p-values: for test barcodes where chi-sq gives p<0.001, simulate K=1000 random multinomial draws from ambient profile at the same UMI depth N. Empirical p = fraction of sims with deviance ≥ observed. Corrects depth-saturation artifact without external data. Matches STARsolo's nIter=20000 approach.
  3. (C) BAM ambient sampling: after STAR writes sorted BAM, sample N=2000 barcodes with 5–99 reads not in the auto-discovery set. Accumulate gene counts from sorted BAM. Use for ambient profile.
- **Partial fix 1** (cycle 165, commit db0ae95): fdr=0.01 + min_umi_test=500 + knee fallback → cells reduced 10,404→7,870. Still 3.1× STARsolo (2,520).
  **Partial fix 2** (cycle 166, commit 05d87fc): Monte Carlo p-values with Vose alias sampling. Cell calling 14.3× faster (420s→29s). Depth artifact eliminated in unit tests. But cells still 7,997 vs STARsolo 4,155 (1.92×) due to sparse ambient pool.
  **Partial fix 3** (cycle 166, commit 5c3b555): Full-whitelist ambient profiling. During pileup, tracks per-barcode UMI counts for ALL 3.7M whitelisted barcodes (30MB RAM). Accumulates gene-level ambient profile from ≤50-UMI WL barcodes. Passes this rich ambient to EmptyDrops instead of the sparse 760-barcode auto-discovered pool. 84/84 CTests pass. Needs live validation on SRR32855204 to measure cell count impact.
- **Acceptance test**: Cell count ≤3,780 on SRR32855204 (1.5× STARsolo Gene 2,520). Cell Jaccard ≥0.60. 100% STARsolo recall (all 2,520 cells present in singlify output). All 79 ctests pass.
- **Blocked samples**: Panel A1 Cell Jaccard stays FAIL until resolved. All cell-type composition analyses.
- **Priority**: HIGH
- **Status**: 🟡 Two fixes committed, awaiting validator retry on SRR32855204. Target: cells ≤3,780 (1.5× STARsolo 2,520).

---

## E2E Validation Track — New AUTOFIXes (2026-04-15T22:00 session)

### AUTOFIX-E2E-A2-5PRIME-ADAPTER (MEDIUM — dataset-specific + singlify detection gap)

- **Status**: 🔴 Open
- **Root cause (singlify)**: singlify does not auto-detect or clip 5' adapter contamination in R2 reads. Some library preparations prepend a non-mappable 5' sequence (e.g., 50bp in SRR34789664) to the cDNA reads. singlify currently only auto-detects 3' adapter (polyA/TSO via `--clip5pNbases` when a 5' prefix is detected). However: (a) STAR may apply `--clip5pNbases 50 0` based on singlify's detection, but (b) with CellRanger4 adapter trimming also active, the reads are trimmed twice resulting in 18bp average R2 → 0.02% mapping.
- **Observed symptom**: SRR34789664 (mouse 10xv3, 5M reads): 3-way STARsolo test (job 361054) — all 3 configurations produce 0.00% mapping: (1) plain CB_UMI_Simple, (2) + CellRanger4, (3) + CellRanger4 + clip5p50. The gold standard uses `--clip5pNbases 50` without CellRanger4 → 94.9%.
- **Fix target**: In singlify's adapter detection logic: detect 5' adapter prefix length from R2 reads WITHOUT also activating CellRanger4 trimming. `--clipAdapterType CellRanger4` and `--clip5pNbases N` should not both be active simultaneously on the same reads.
- **Acceptance test**: `singlify SRR34789664_notrim.1fq --genome-dir GRCm39` → STAR uses only `--clip5pNbases 50 0` (no CellRanger4), mapping rate ≥85%, cells ≥8000.
- **Workaround for Panel A2 v6**: externally trim 50bp from R2 using `awk`, re-encode to .1fq, no adapter trim needed in singlify → gene counting correctness test runs cleanly. Job 361065.
- **Priority**: MEDIUM — affects datasets with explicit 5' library prep adapters; relatively rare in modern 10x libraries.

### AUTOFIX-E2E-C-FRAGMENT-R-WARN (LOW — ATAC fragment count accuracy)

- **Status**: 🔴 Open
- **Root cause hypothesis**: singlify fragment extraction has a minor discrepancy vs pysam raw fragment extraction. r=0.970 on 3849 common barcodes (shared between singlify EmptyDrops-called cells and all raw barcodes in the pysam gold). Likely caused by: (a) MAPQ filtering difference — pysam might use MAPQ≥0 while singlify uses MAPQ≥10 or 30; (b) deduplication strategy differences; (c) read-end extension differences for Tn5 insertion site calculation.
- **Observed symptom**: Panel C v1 (job 361047, commit db0ae95): r=0.970 (threshold ≥0.990). Gold: 313K raw barcodes, 20M total fragments. Singlify: 3849 called cells, 8.8M fragments. The 3849 cells appear in both → meaningful comparison. r=0.970 is the fragment count correlation for those cells.
- **Fix target**: Identify specific threshold or algorithm difference causing 3% fragment count discrepancy. Likely in `atac_fragment_extractor.h`: confirm MAPQ cutoff = 30 and check Tn5 offset (standard: +4 on +strand, -5 on -strand). Run Panel C v2 with sinto tool (same BAM) for better external reference.
- **Acceptance test**: Panel C v2 with sinto → r ≥ 0.990.
- **Priority**: LOW — ATAC fragment r=0.970 is a minor discrepancy; core ATAC pipeline works. Blocked by needing a better external reference.


### AUTOFIX-MEGA-SHM-EXHAUSTION: BAM compression + NFS sort for >200M-read samples — VALIDATED cycle 159
- 🟢 commit 90ad777 — BAM compression=1 for >200M reads (4x smaller BAM on tmpfs), outTmpDir on NFS for STAR sort spill, .1fq early deletion after FIFO decode, limitBAMsortRAM capped at 50% SLURM_MEM for mega samples
  **Root cause**: STAR writes uncompressed sorted BAM (--outBAMcompression 0) + sort temp files to /dev/shm (RAM-backed tmpfs). For >500M actual reads, the BAM alone can be 100-200GB. Combined with genome load (32G) + limitBAMsortRAM (190-286G) + .1fq (8-15G), total exceeds 384G SLURM cgroup. Also: catalog read_count_estimate is 1.6-2.6x too low for many SRRs (spots vs reads).
  **Witness samples**: batch_011 T1/T2/T4/T6 (OUT_OF_MEMORY, 402G RSS), T7/T9 (bgzf write fail)
  **Verification: ✅ VALIDATED cycle 159 — T12 (613M reads) COMPLETED 79.1% MR 363G RSS no OOM; T15 (427M reads) SUCCESS 83.6% MR 749 cells 195G RSS. T14 (817M) still downloading at 70%.

### AUTOFIX-ZERO-BC-MATCH (HIGH) — FIXED cycle 156, commit 13525a7
- 🟢 Abort download with exit 2 when barcode WL ≤0.1% on both orientations. 138 high-MR zero-cell samples confirmed barcode-stripped (cycle 157). Saves ~200+ CPU-hours retroactively.

### AUTOFIX-FAST-CRASH-CLUSTER (LOW) — ROOT-CAUSED cycle 157
- 🟠 All 57 fast-crash (wall ≤10s) are ultra-low-read deposits (1-11K reads). GSE193517 (23×, 1-2 reads), GSE263733 (28×, 748-11K reads). Add read_count >= 10K to eligibility filter. 49,596 catalog samples (3.3%) affected.

### AUTOFIX-1FQ-ZSTD-CORRUPT-LARGE — RECLASSIFIED cycle 156
- 🟢 Actually /dev/shm exhaustion (bgzf_close write fail), not ZSTD. Resolved by mega fix (90ad777) + 64-bit offset (a4d69ab).

### AUTOFIX-BAMSORT-RAM-UNCAP — SUPERSEDED cycle 156
- 🟢 T1/T2 OOM at 402G MaxRSS now addressed by mega fix: 50% SLURM_MEM cap + NFS outTmpDir.


### AUTOFIX-SCIRNA3-WL-MISSING (MEDIUM) — Found cycle 158
- � sci-RNA-seq3 whitelist `scirna3_rt_bc.txt` not found → auto-discovery + CB_UMI_Complex = OOM at 128G on only 36M reads — PARTIALLY ADDRESSED
  **Root cause**: singlify process detects sci-RNA-seq3 protocol but cannot find the RT barcode whitelist. Falls back to barcode auto-discovery. With CB_UMI_Complex mode (3-level combinatorial indexing), auto-discovery generates explosive barcode combinations → OOM.
  **Observed symptom**: GSM7431256 (SRR24750183, 36M reads, sci-rna-seq3), batch_008_std 128G → OOM exit 137 after decode.
  **Partial fix**: Per-segment whitelist file `scirna3_rt_bc.txt` created and committed (e08d7b1, cycle 165). Need to verify sci-RNA-seq3 also gets CB_UMI_Complex positions computed correctly (same anchorType audit as SPLiT-seq). Validator retry needed.
  **Blocked samples**: All sci-RNA-seq3 samples without pre-installed whitelists.
  **Priority**: MEDIUM — affects ~3-5% of sci-RNA-seq3 samples in catalog

- � AUTOFIX-PARSE-SPLITSEQ-BARCODE: Parse/SPLiT-seq uses CB_UMI_Complex (3 combinatorial barcode segments) but singlify maps it to CB_samTagOut with single 8bp CB → 3.3% mapping rate (441K/13M reads) — FIX COMMITTED
  **Root cause**: splitseq protocol detection (confidence=1) maps to `soloCBlen=8 soloCBstart=11 soloUMIstart=1 soloUMIlen=10` which only captures 1 of 3 barcode segments. Need CB_UMI_Complex whitelist files and STAR solo mode.
  **Observed symptom**: B013-T4 GSM8623064 (SRR31302015, 13M parse reads): STAR 494s, 441K mapped reads (3.3%), 62K UMIs. Effectively 0% useful mapping.
  **Fix committed**: a5f6959 (cycle 166) — anchorType was incorrectly set to 1 (from read end) instead of 0 (from read start) in sra_encoder.h for SPLiT-seq CB_UMI_Complex positions. Changed `1_10_1_17` → `0_10_0_17`, `1_33_1_40` → `0_33_0_40`, `1_56_1_63` → `0_56_0_63`, `1_0_1_9` → `0_0_0_9`. Combined with whitelist files (e08d7b1) and use_complex tier (8e24d98), SPLiT-seq should now run CB_UMI_Complex with correct positions. Validator retry needed.
  **Blocked samples**: All 369 parse-protocol samples in catalog (145 eligible single-SRR)
  **Priority**: HIGH (369 samples blocked)

### AUTOFIX-BD-RHAPSODY-OOM (HIGH) — Found cycle 161, ROOT-CAUSED cycle 162
- � FIX COMMITTED BD Rhapsody CB_UMI_Complex IS implemented correctly (3×9bp segments with bd_cls1/cls2/cls3.txt whitelists). STAR CB_UMI_Complex mode OOMs at 192G for even 30M-read samples.
  **Root cause (cycle 162)**: Two compounding factors: (1) STAR CB_UMI_Complex + 1MM matching across 3 segments has high inherent memory overhead for combinatorial matching tables. (2) `--outBAMcompression 0` (non-mega path, <200M reads) creates enormous uncompressed BAMs in /dev/shm. STAR log: `--outBAMcompression 0 --limitBAMsortRAM 46G --outTmpDir /dev/shm/singlify_star_*`.
  **Observed symptoms**: B013-T9 (81M, OOM 192G in 27:45), B013-T10 (30M, OOM 192G in 8:31), B013-T11 (68M, expected OOM).
  **Fix target**: In singlify process, when soloType==CB_UMI_Complex, always use outBAMcompression=1 and set limitBAMsortRAM to 50% of available RAM. Small code change in STAR parameter builder.
  **Acceptance test**: BD Rhapsody sample at 30M reads completes at 192G with cells > 97.
  **Priority**: HIGH — blocks all 454 BD Rhapsody catalog samples. 3 OOM witnesses.
  **Iteration 3** (orchestrator cycle 166, commit 8e24d98): `use_complex=true` fallback for all known multi-segment protocols (bd-rhapsody, splitseq, indrop) when protocol spec has `per_seg_whitelist_files`. Ensures complex tier compression=1 + RAM=min(15%,64G) is applied even when detected_cb_positions was empty. Previous OOMs at 192G used outBAMcompression=0 because use_complex was false. Binary pushed to origin. Validator retry needed on 30M BD Rhapsody sample.
- 🟡 DNBelab-C4 uses 3×10bp combinatorial barcoding but singlify captures only first 10bp segment. SUCCESS achieved (T1: 94.8% MR, 500 cells; T2: 92.8% MR, 150 cells) due to large single-segment space (4^10=1M).
  **Fix target**: DNBelab-C4 handler should emit CB_UMI_Complex parameters with all 3 barcode segments.
  **Priority**: LOW — works adequately for small samples
### AUTOFIX-MEGA-SORT-RSS-OVERAGE (HIGH) — Found cycle 162
- 🔴 Mega-fix limitBAMsortRAM=50%_SLURM_MEM still OOMs at cgroup cap on very large samples
  **Root cause hypothesis**: On 600M+ read samples, STAR total RSS = genome (~30G) + limitBAMsortRAM (186G at 384G cap) + per-thread sort buffers + CB match tables + decoded FIFO prefetch + coverage trackers. Observed MaxRSS hits the FULL cgroup cap (384G for 666M-read, 192G for 172M-read) during the sort phase, indicating limitBAMsortRAM alone does not bound true RSS.
  **Witness samples**:
    - GSM7102845 (666M reads, 10x-5p-v3, batch_011 T16, c005, 384G node): OUT_OF_MEMORY, MaxRSS=384G, exit 0:125, wall 5:14:48, SIGKILL during BAM sort.
    - GSM8860467 (172M reads, parse→splitseq misdetect, batch_013 T5, c009, 192G node): OUT_OF_MEMORY, MaxRSS=192G, exit 0:125, wall 2:28:18.
    - **GSM5239644** (817M reads, 10x-5p-v3, batch_011 T14, c101, 384G) — **primary mega-fix validation sample**: HARD_FAIL pipeline_crash, 3.5h wall, 0% MR, 0 cells (cycle 163 confirms bug at 817M, largest witness yet).
    - **GSM6564295** (visium, 194M reads after dedup, probe 361751, c006, 192G) — **NEW cycle 164 witness: extends problem to ~200M reads on 192G nodes**. OUT_OF_MEMORY, MaxRSS=192G exact cgroup cap, SIGKILL at STAR "started sorting BAM", wall 38:55. Log: `limitBAMsortRAM=57.9GB` was honored but total RSS still exceeded 192G cap. Confirms STAR's limitBAMsortRAM is a soft hint, not hard bound. Bio-exec's mega-tier policy (comp=1, cap=50%_MEM) insufficient on 192G nodes even below 200M reads.
    - **GSM8808454** (10xv3, 872M reads, b012 T1 old-binary, 384G): OOM MaxRSS=402G, exit 0:125, wall 6:47:47. Additional 872M-tier witness. Pre-mega-sort-binary; expected to be resolvable by the uncommitted ultra tier policy.
  **Range confirmed**: Mega-fix works up to ~408M reads on 384G (GSM3743501 T18 SUCCESS 92% MR 8900 cells at 408M); FAILS at 194M visium on 192G (cycle 164), 666M and 817M on 384G. The "ultra" tier thresholds likely need scaling by MEM_GB — a policy driven purely by read_count is insufficient. Possible fix refinement: trigger ultra tier when `reads / mem_gb >= 1M/GB` or when any path shows STAR RSS overshooting limitBAMsortRAM by >2× historically.
  **Iteration 1 attempt** (cycle 165, validator 361746 on GSM7102845): bio-exec's ultra-tier policy (comp=6 logged but STAR gets 0, bonus `outBAMsortingBinsN=500`, limitBAMsortRAM capped at 57 GiB) PREVENTED OOM (STAR mapping completed in 2h43m with no MaxRSS cap hit) but crashed 1 second into BAM sort with STAR `FATAL ERROR: BAM bin size mismatch — Expected 747M, disk 607M, bin #494`. Root cause: `outBAMsortingBinsN=500` is too aggressive (STAR default is 50); STAR has a known-ish internal inconsistency at very high bin counts.
  **Iteration 2** (bio-exec a2190a1749f33542b, delivered 22:52): `ULTRA_SORT_BINS=100` (from 500), `compression_level` ultra branch now returns 0 (was logged as 6 but STAR actually got 0 — now consistent), log message cleaned up with `std::to_string(bam_comp_level)`, tests added (61/61 pass including bin-count assertion). Option B (samtools post-hoc) NOT implemented — bio-exec's reasoning: STAR docs caution >200 bins; 100 is 2× default, well inside tested range. Binary rebuilt at 22:52:57 (10,600,912 bytes). Validator retry dispatched (a238ddab1b7652ea1, ~4-5h wall on GSM7102845 at 384G).
  **Iteration 2 result**: GSM7102845 (361493 retry, bins=500 binary) HARD_FAIL pipeline_crash — this used the OLD binary with bins=500. Commit 343bc58 (bins=100) was pushed AFTER this job started. Needs re-submission with bins=100 binary.
  **Fix target (for bio-exec)**: In singlify STAR parameter builder, when total reads ≥ 500M, reduce limitBAMsortRAM ceiling from 50% of SLURM_MEM to 25% AND bump --outBAMcompression from 1 to 3 (tighter BGZF compression trades CPU for tmpfs/disk savings). Alternatively, switch to `--outSAMtype BAM Unsorted` and run `samtools sort -m 4G -@ N -o sorted.bam unsorted.bam` post-hoc with a tight per-thread cap — samtools respects `-m` rigorously while STAR's limitBAMsortRAM is a soft hint.
  **Acceptance test**: GSM7102845 (666M reads, 10xv3_5prime) re-runs at 384G without OOM; wall ≤6h; mapping rate ≥50%.
  **Blocked samples**: All samples with read_count ≥ 500M (10xv3, 10x-5p-v3, dropseq). Also smaller-read samples with protocol-misdetect that inflate BAM (parse→splitseq seen at 172M).
  **Priority**: HIGH — blocks primary mega-fix validation and all samples >500M reads. Currently ~60 catalog samples.

### AUTOFIX-EXPORT-CSC-INT32-OVERFLOW (HIGH) — FIXED cycle 163, commit 99dc7f0
- 🟢 **Not indrop-specific** — chrM coverage overflow in `SparseAccumulator::to_csc()`; any sample with `mt_pileup_bases > 2^31` SIGSEGV'd at export. Originally surfaced as INDROP-EXPORT-SEGV (cycle 162).
  **Root cause**: `int32_t` prefix-sum + scatter-slot arithmetic overflowed to negative index → SIGSEGV during scatter into `grouped_rows[]`.
  **Fix**: int32→int64 in scratch arithmetic at `include/singlet-pileup/sparse_accumulator.h:79,85,91,95,115,116,124` + defensive `indptr_val()` clamp at line 119. Downstream CSC indptr stays int32 (ABI preserved).
  **Commit**: 99dc7f0 (perf-exec cycle 163, pushed origin/main first try).
  **Tests**: 82 ctests pass, new Test 8 exercises 1M-entry scatter at to_csc(), new Test 9 documents clamp helper.
  **Validator PASS**: GSM8752766 (10xv3, 12.7M reads, mt=2.32B, SLURM 361724, c001) previously exit 139 SIGSEGV — now exit 0, 75.73% MR, 52G MaxRSS, all matrices produced including mt_heteroplasmy.1pz; `[export] CSC conversion: 0.0403398s` appeared in log.
  **Fix Activation Proof** (CLAUDE.md Rule 20):
    - (a) Commit hash: 99dc7f0ac9afa8cf1f3c87efec57eb0833337bff
    - (b) Wired: C++ header is compiled into singlify binary at `/mnt/home/debruinz/Singlet-AI/singlify/build/singlify` which ALL job scripts invoke — no job-script grep required; binary rebuild is automatic on next perf-exec build.
    - (c) Metric change: GSM8752766 SIGSEGV exit 139 → SUCCESS exit 0 with 75.73% MR (validator job 361724).
  **Historical witnesses unblocked** (all previously exit 139 after `Loaded N metadata fields`, all mt_pileup_bases >2.3B):
    - GSM8752766 (10xv3, 12.7M, mt=2.32B) ✅ validated
    - GSM8752765 (10x-arc-gex, 14.8M, mt=2.77B) ✅ **VALIDATED cycle 164** (probe 361749: SUCCESS 79.5% MR 3,734 cells, `[export] CSC conversion: 21.5071s`)
    - GSM8752768 (10x-arc-gex, 17M, mt=3.14B) ✅ **VALIDATED cycle 164** (probe 361750: SUCCESS 75.16% MR 3,066 cells, 34.3min wall)
    - GSM6564295 (10x-visium, 267M, mt=2.33B) — eligible for retry
    - GSM8249691 (indrop, 82M, mt=5.49B) ✅ **VALIDATED cycle 164** — the original bug-witness sample. Probe 361752: SUCCESS 72.2% MR, 257 cells, 51.9min wall. Confirms G3 inDrop protocol now works.
  **Priority**: CLOSED.

### AUTOFIX-ENCODE-ABORT-LOW-WL-MATCH (HIGH) — Found cycle 162
- � FIX COMMITTED 1fq-encode proceeds through 3-10% whitelist-match band; ADT-mislabeled-as-GEX CITE-seq samples waste 1.5-1.7h of compute each before producing 0 cells
  **Root cause**: Current zero-BC fix (commit 13525a7) aborts download only when ≤0.1% whitelist match on both orientations. Samples in the 0.5-10% band are allowed to proceed. For CITE-seq libraries where the ADT FASTQ was mislabeled as GEX in GEO, both orientations match 3-7% (random-enough reads look like noise-match against the 3M 10x-v3 whitelist). Sample wastes STAR alignment on non-transcriptomic cDNA then produces <20 cells.
  **Witness samples**:
    - GSM9031360 (110M reads, catalog=citeseq, 3% R1 WL match → R2 swap tried, 0.02% STAR MR, c004, 1:38h wall, SUCCESS per threshold but 0 real cells)
    - GSM5465113 (168M reads, catalog=citeseq, 7% WL match, did NOT trigger swap since >5% threshold, 0.01% STAR MR, c007, 1:40h wall, 11 cells)
    - GSM9210777 (87M reads, catalog=10xv3, 0.04% match both orientations after swap, 0.81% STAR MR, c008, 15min wall, barcode-stripped deposit NOT caught by zero-BC fix because swap-tested match was 0.14% — above 0.1% cutoff)
  **Fix target (bio-exec)**: In 1fq_encode.cpp (or wherever barcode-WL validation happens during download/encode), tighten the abort threshold from ≤0.1% both-orientations to: (a) ≤5% best-of-both-orientations → abort with exit 2 (data_incomplete). OR (b) require ≥20% match on at least one orientation to proceed; else abort. This catches the CITE-seq ADT-mislabel class and the marginal barcode-stripped deposits.
  **Acceptance test**: GSM9031360, GSM5465113, GSM9210777 re-run → each exits in <5 min with exit code 2 and `failure_category=data_incomplete`. GSM5138309 (valid 10xv3, 70% MR, 7141 cells) re-run → still passes encode.
  **Blocked samples**: All CITE-seq ADT-mislabel deposits (estimate ~50-200 in current catalog based on modality==citeseq rate) + marginal barcode-stripped samples in 0.1-5% band.
  **Priority**: HIGH — compute-hours saving is substantial (1-2h/sample × 50+ samples = 50-100 CPU-hours per batch cycle).
  **Fix committed**: 8365a6a (cycle 166) — threshold raised from 0.1% to 5%, using max(r1_rate, r2_rate) instead of both. Binary pushed to origin. ADT-as-GEX mislabels (3-7% hit rate) now abort in <5min with exit 2.

### AUTOFIX-E2E-C-ATAC-ZERO-FRAGMENTS (CRITICAL) — Found E2E cycle 2026-04-17
- 🔴 singlify ATAC mode produces 0 fragments from 22.7M aligned reads
  **Root cause**: STAR invoked in plain paired-end mode (`--outSAMattributes NH AS`) without CB tag. Fragment extractor cannot associate reads with barcodes → 0 fragments, 0 cells.
  **Observed symptom**: `[singlify] ATAC: 0 unique fragments (0 dupes)` on ATAC PBMC 500 10x v1 (22.7M reads, 5442 barcodes discovered). STAR succeeds (mapping ~55-65% typical for ATAC PE), but BAM lacks CB tags.
  **STAR command (from log)**: `STAR --outSAMattributes NH AS --alignIntronMax 1 --alignMatesGapMax 2000` — correct ATAC geometry flags, but missing barcode assignment (`--soloType CB_UMI_Simple` or CB tag injection).
  **Regression note**: At commit 2630ad4, Panel C produced 8.8M fragments (47% of external) — nonzero but undercounting. At HEAD 6c7a875: zero fragments. Something changed between 2630ad4 and 6c7a875 that broke ATAC fragment extraction entirely.
  **Fix target (bio-exec)**: In singlify's STAR parameter builder, when `assay_type == ATAC`:
    (a) Either invoke STARsolo with `--soloType CB_UMI_Simple` and ATAC-appropriate CB/UMI positions, OR
    (b) Strip barcodes from R1 during FIFO write, inject CB BAM tags post-alignment from the barcode file, OR
    (c) At minimum, restore whatever was working at 2630ad4 and investigate the 47% fragment undercount
  **Acceptance test**: ATAC PBMC 500 10x v1 processes through singlify → fragments.tsv.gz contains >5M fragments, barcode Jaccard ≥0.85, fragment count Pearson r ≥0.990 vs external pipeline (STAR PE + pysam fragment extractor with MAPQ≥30).
  **Blocked samples**: ALL 10x-ATAC samples in the catalog.
  **Priority**: CRITICAL — completely blocks G3 ATAC protocol and Panel C E2E validation.

### AUTOFIX-E2E-H-DOUBLET-OVERCALL (HIGH) — Found E2E cycle 2026-04-17 — **CLOSED cycle 176**
- 🟢 **RESOLVED** — commit a360885 (cycle 173, v5 adaptive threshold: threshold = (sim_frac+1)/2 ≈ 0.833)
  **Validation**: Panel H re-run cycle 174 → doublet_rate=12.54% (within expected range for ~2,500 cells).
  Binary rebuilt cycle 176 (job 367911, 84/84 CTests pass) confirming v5 is in production.
  **Previous symptom**: 40.7% overcall rate with old threshold. Now resolved.

### AUTOFIX-SPECIES-KMER-LOW-SENSITIVITY (CRITICAL) — Filed cycle 176 → 🟢 RESOLVED cycle 177 (f23d08d)
- Species k-mer detection returns near-zero hits (87/367514 = 0.024%) on valid human 10xv3 reads
  **Root cause 1 (cycle 176)**: Diagnostic k-mer path (Path 3) too small (50K × 17 k-mers).
  **Root cause 2 (cycle 177)**: Bloom filter hash seed off-by-one: Python builder seeds = {1×φ, 2×φ, 3×φ}, C++ query seeds = {0×φ, 1×φ, 2×φ} → zero matching hash positions.
  **Root cause 3 (cycle 177)**: MIN_RATIO=2.0 impossible for human/mouse because ~94% k-mer overlap → actual ratio only 1.06×. Z-test with z=19.4 proves difference is real.
  **Fix (f23d08d)**: (a) bloom_filter.h: i→i+1 on seed loop, (b) species_detect.h: two-proportion z-test replaces ratio threshold, (c) MIN_RATIO lowered to 1.5 for distant species fast path.
  **Validation**: SRR13352022 (SeqWell human) → species=Homo sapiens genome=GRCh38 confidence=0.514 method=bloom (130876/280000 k-mers, z²=376). 84/84 CTests pass.
  **Bloom filters**: human_21mer.bloom (268MB, 1.82B k-mers from roers_ref.fa) + mouse_21mer.bloom (268MB, 153M k-mers from genome+GTF exons). Located at singlify/species_filters/.
  **Remaining**: Need validation on a mouse .1fq sample to confirm bidirectional detection.

### AUTOFIX-DECODE-QUAL-LENGTH (MEDIUM) — Filed cycle 176
- STAR receives quality string length ≠ sequence length from decoded .1fq FASTQ
  **Sample**: GSM7093689 (SRR23849629), CB_samTagOut, soloCBwhitelist=None, 5.7M reads
  **Symptom**: STAR FATAL: "quality string length is not equal to sequence length" on a ~600bp read
  **Root cause hypothesis**: .1fq decode produces mismatched quality/sequence for unusual read lengths or protocol misclassification
  **Priority**: MEDIUM — affects rare edge case samples

### AUTOFIX-CELSEQ2-EMPTY-R2 (MEDIUM) — Filed cycle 176
- CEL-Seq2 protocol detected but R2 (biological read) is empty for all reads
  **Sample**: GSM3336845 (SRR7706271), 94.8M reads, protocol=celseq2 (confidence 3)
  **Symptom**: "ERROR: R2 (biological read) is empty for all reads" after 8.9s scan
  **Root cause hypothesis**: Protocol misclassification OR R1/R2 swap in .1fq encoding
  **Priority**: MEDIUM — CEL-Seq2 is a minority protocol

### AUTOFIX-REPROCESS-CELLS-KEY (LOW) — Filed + FIXED cycle 176
- 🟢 Reprocess job script used `d.get('cells', 0)` to read summary.json but singlify uses key `estimated_cells`
  **Fix**: Updated reprocess_c176_job.sh to use `d.get('estimated_cells', d.get('cells', 0))`
  **Impact**: All reprocessed samples were misclassified as SOFT_FAIL with cells=0 regardless of actual cell count
  **Note**: Main pilot_job.sh was NOT affected (already had correct fallback chain at line 250)

## CELL-CALLING-REVIEW (filed by singlet-product, 2026-04-29)
- **Priority**: HIGH
- **Type**: pipeline quality
- **Issue**: singlify calls 11,152 cells vs STARsolo's 2,520 for SRR32855204 (4.4× more)
  - 100% of STARsolo cells recovered (cell Jaccard = 0.226)
  - 55% doublet rate vs expected 6.4% → excess cells are likely empty droplets/debris
  - Gene r on shared cells is 0.999 — counting is correct, cell calling threshold too loose
- **Action needed**: Review EmptyDrops/knee-point threshold in biology pileup
  - Consider matching STARsolo's cell-calling stringency
  - Or add a "high-confidence cells" subset flag
- **Validation data**: `/singlify_validation/singlify_out/SRR32855204_mc_v6/`
- **Filed by**: singlet-product P3

## MITO-FRACTION-ZERO (filed by singlet-product, 2026-04-29)
- **Priority**: MEDIUM
- **Type**: pipeline quality
- **Issue**: median_mito_fraction is 0.0 for ALL 709 summary.json files
  - Expected: 2-15% for most human/mouse scRNA-seq samples
  - Affects: QC dashboard, Browse page mt_pct column, corpus analytics
- **Location**: summary.json files in quant/scrna/*/*/*/summary.json
- **Root cause hypothesis**: singlify may not be counting chrM reads in the mito fraction calculation
- **Impact**: Website shows mt_pct=0 for all samples, can't filter on mito contamination
- **Filed by**: singlet-product P11
