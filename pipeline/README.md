# singlify

A self-contained monorepo for processing single-cell sequencing data. Four stages, one binary, zero intermediate files:

```
.1fq  →  align  →  pileup  →  .1pz
```

Every stage is purpose-built for the stages adjacent to it. The `.1fq` format stores reads as 2-bit packed, barcode-sorted, protocol-annotated blocks that the aligner consumes directly — no ASCII FASTQ round-trip. The aligner streams unsorted BAM through an in-process pipe to the pileup engine, which extracts counts, SNPs, and splicing in a single pass and writes `.1pz` matrices. The pileup never runs independently — it is always driven by the aligner.

### Current Status (April 2026)

| Metric | Value |
|--------|-------|
| Pipeline wall time (40M reads, 16T) | **102–137s** (protocol-dependent) |
| Gene count correlation vs STARsolo | **r = 0.9998** |
| Protocol families validated | **8+** (10x v2/v3/v4/5', Drop-seq, inDrop, sci-RNA-seq3, BD Rhapsody) |
| Species validated | Human (GRCh38) + Mouse (GRCm39) |
| M4.4 batch test (98 random SRA samples) | **87% pipeline robustness, 0 crashes** |
| SRA encode speed (40M reads) | **19.6s** (5.1× faster than original) |
| .1fq decode speed (40M reads, 8T) | **4.6s** (8.1× faster than original) |
| .1fq compression (vs raw FASTQ) | **~24% smaller** with BINNED4 + zstd-4 |

## Why a monorepo

Generic tools (STAR, CellRanger, STARsolo, alevin-fry) treat the file format, aligner, and quantifier as independent components with FASTQ as the interchange format. This forces redundant parsing (ASCII ↔ binary ↔ ASCII), prevents cross-stage optimization, and makes the pipeline sensitive to version mismatches between tools.

singlify eliminates these boundaries:
- **.1fq ↔ aligner**: The aligner knows the .1fq block layout. It reads 2-bit packed sequences and pre-parsed barcodes/UMIs directly from .1fq blocks without decoding to FASTQ text. Format changes and aligner changes are co-evolved in the same commit.
- **aligner ↔ pileup**: The pileup engine receives BAM records through an in-process pipe. No intermediate BAM file is ever written to disk. The aligner and pileup are compiled into a single binary — they share memory layout assumptions and are always version-consistent.
- **pileup ↔ .1pz**: Export writes `.1pz` natively — no MTX intermediate, no Python post-processing.

This co-design enables optimizations that are impossible across tool boundaries: the aligner can skip re-aligning deduplicated reads that .1fq has already collapsed, barcode correction can use the .1fq whitelist dictionary, and pileup tuning can inform aligner output flags.

## Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ singlify                                                    │
│                                                             │
│  ┌──────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ .1fq     │ ───→ │ aligner      │ ───→ │ pileup +     │  │
│  │ reader   │ 2-bit│ (singlet-    │ BAM  │ export       │  │
│  │          │ block│  lite core)  │ pipe │              │  │
│  │ protocol │      │ SA search    │      │ exon/intron  │  │
│  │ auto-    │      │ 14-mer sort  │      │ splice jxn   │  │
│  │ detect   │      │ R2 dedup     │      │ SNP AD/DP    │  │
│  │ BC/UMI   │      │ CB hash      │      │ chrM pileup  │  │
│  │ parse    │      │ NUMA aware   │      │ UMI dedup    │  │
│  └──────────┘      └──────────────┘      │ donor demux  │  │
│                                           │     ↓        │  │
│                                           │ .1pz export  │  │
│                                           └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Input**: `.1fq` files only. Raw data (SRA accessions, FASTQ pairs) is first encoded to `.1fq` via `singlify download` or `singlify encode`, then processed. This two-step design means every sample is archived before processing — if the aligner or pileup improves, re-processing reads from `.1fq` without re-downloading.

**Output**: `.1pz` sparse matrices (readable by [singlepress](https://github.com/Singlet-Bio/singlepress)) + stats JSON.

## What the aligner extracts (via pileup)

- Per-exon and per-intron UMI counts (spliced/unspliced/ambiguous)
- Splice junction counts
- SNP genotypes (AD/DP) at ~7.4M common positions — used for donor demux, then exported as per-donor VCF + coverage maps (pipeline mode) or raw matrices (non-pipeline)
- chrM reads for heteroplasmy analysis
- Donor demultiplexing (Variational Bayes binomial mixture)

## Repository structure

```
singlify/
├── CMakeLists.txt
├── LICENSE                         # GPLv3
├── README.md
│
├── include/
│   ├── star_api.h                  # Aligner entry point: star_main_impl()
│   ├── lib1fq/                     # .1fq format codec (header-only)
│   │   ├── types.h                 #   Format spec: headers, blocks, enums, protocols
│   │   ├── packing.h               #   2-bit sequence packing, varint encoding
│   │   ├── compress.h              #   zstd/lz4 compression
│   │   ├── protocol.h              #   Protocol registry (24 chemistries)
│   │   ├── writer.h                #   Block-oriented writer
│   │   ├── reader.h                #   Block reader → DecodedBlock (byte-numeric)
│   │   ├── sra_encoder.h           #   SRA → .1fq streaming encoder
│   │   ├── fastq_encoder.h         #   FASTQ pair → .1fq encoder
│   │   ├── dedup.h                 #   H1 UMI deduplication
│   │   └── lib1fq.h                #   Single-include aggregator
│   └── singlet-pileup/             # Streaming pileup engine (header-only)
│       ├── pileup_engine.h         #   Single-pass BAM scan, multi-mapper resolution
│       ├── gene_model.h            #   GTF/BED parsing, interval tree
│       ├── sparse_accumulator.h    #   COO → CSC matrix accumulator
│       ├── umi_dedup.h             #   Barcode/gene/UMI deduplication
│       ├── pz_writer.h             #   Native .1pz output
│       ├── donor_demux.h           #   Genotype-based demultiplexing
│       └── export.h                #   Export orchestration
│
├── src/
│   ├── singlify.cpp                # Pipeline orchestrator + CLI
│   ├── 1fq.cpp                     # .1fq CLI: encode/decode/dedup/inspect
│   ├── star/                       # Aligner source (singlet-lite, derived from STAR 2.7.11b)
│   │   ├── STAR.cpp                #   Entry point: star_main_impl()
│   │   └── ...                     #   ~200 source files (SA search, stitching, solo BC)
│   └── test_*.cpp                  # Unit tests
│
├── whitelists/                     # Barcode whitelists per protocol
├── docs/                           # Format specs, optimization logs
├── scripts/                        # Smoke tests, validation, profiling
└── bench/                          # A/B benchmarking harness
```

## Milestone Metrics — Pipeline Readiness

Before initiating large-scale SRA processing (100K+ samples on NSF ACCESS / Anvil), the singlify pipeline must meet every metric below. Service units are finite and non-renewable — we cannot afford to reprocess due to format changes or pipeline bugs.

### M1: .1fq Format Stability (gate for archival)

The .1fq format (currently v4) must be **frozen** before archiving any data at scale. Criteria:

| # | Metric | Target | How to verify |
|---|--------|--------|---------------|
| M1.1 | **Protocol coverage** | Encode + round-trip decode for all 24 protocols in `protocol.h` | `singlify download` + `1fq decode --verify` on ≥1 real SRA file per protocol family |
| M1.2 | **Multi-stream support** | Files with 1–4 read streams (R1/R2/I1/I2) encode/decode correctly | Validated on 10x-arc-gex (3-stream) and a 4-stream library |
| M1.3 | **Segment descriptors** | `StreamDesc` segments correctly partition BC, UMI, cDNA, linker for each protocol | `1fq inspect` displays correct segment map; round-trip preserves segments |
| M1.4 | **Whitelist round-trip** | Barcode dictionary (BC_DICT flag) encodes/decodes for all whitelisted protocols | Verified for 3M-february-2018, 737K-august-2016, 737K-april-2014, gex_737K-arc-v1 |
| M1.5 | **Forward compatibility** | Reserved header bytes (34 bytes) sufficient for planned extensions; reader ignores unknown fields | New reader reads old files; old reader reads new files with unknown fields gracefully |
| M1.6 | **Quality fidelity** | BINNED4 quality preserves variant-calling accuracy (≤1% FDR regression vs FULL) | Compare SNP calls from BINNED4 vs FULL quality on ≥3 samples |
| M1.7 | **Dedup correctness** | `1fq dedup` produces identical alignment output to non-deduped input | `diff SJ.out.tab` between deduped and non-deduped runs on ≥3 samples |

### M2: Aligner Performance (gate for cost efficiency)

Each service unit on NSF ACCESS is ~1 core-hour. At 40M reads/sample and ~70K actionable samples, aligner speed directly determines whether we finish within budget. **The target is not a fixed number — it is the Pareto frontier**: match or beat pseudoaligners (alevin-fry, kallisto|bustools) in wall-clock time while retaining full genomic alignment.

| # | Metric | Target | Current | How to verify |
|---|--------|--------|---------|---------------|
| M2.1 | **Wall time (40M reads, 16T)** | **Beat alevin-fry** on same hardware | **102–137s** | End-to-end singlify on SRR32855204, 16–20 threads; measure alevin-fry on same data |
| M2.2 | **Correctness** | Bit-identical SJ.out.tab vs stock STAR 2.7.11b | ✅ Passing | Correctness test protocol |
| M2.3 | **Thread efficiency at 16T** | ≥40% (user/wall÷threads) | ✅ 44% | Thread scaling benchmark |
| M2.4 | **Species-agnostic** | Same binary works for GRCh38, GRCm39, any Ensembl genome | ✅ Tested on GRCm39 | Run on GRCm39 index |
| M2.5 | **No hard-coded constants** | Zero human-specific constants in hot path | ✅ Verified | Code review |
| M2.6 | **Output richness advantage** | Document capabilities absent in pseudoaligners | Documented | Competitive comparison: SJs, SNPs, intronic reads, donor demux, chrM |

### M3: Pileup Accuracy (gate for data quality)

| # | Metric | Target | How to verify |
|---|--------|--------|---------------|
| M3.1 | **Exon counts correlation** | Pearson r ≥ 0.995 vs STARsolo Gene matrix | `validate_e2e.py` on ≥3 samples |
| M3.2 | **Intron counts non-zero** | ≥70% of expressed genes have intron signal | Inspect intron_counts.1pz |
| M3.3 | **SNP genotyping concordance** | ≥98% agreement with CellSNP-lite at covered sites | `compare_cellsnp.py` |
| M3.4 | **Donor demux accuracy** | ≥95% agreement with Vireo on pooled samples | Validated on ≥2 multiplexed samples |
| M3.5 | **Native .1pz output** | All matrices written as .1pz (no MTX intermediate) | End-to-end run produces only .1pz files |

### M4: End-to-End Smoke Tests (gate for scale-up)

| # | Metric | Target | How to verify |
|---|--------|--------|---------------|
| M4.1 | **Protocol diversity** | End-to-end on ≥8 protocol families | singlify completes with exit 0 and non-empty .1pz for: 10x-v2, 10x-v3, 10x-v4, 10x-5p, Drop-seq, inDrop, sci-RNA-seq3, BD Rhapsody |
| M4.2 | **Chemistry auto-detect** | Correct protocol identified from SRA without manual hints | Auto-detects protocol matching catalog metadata on ≥50 samples |
| M4.3 | **Failure graceful** | Bad/corrupt/empty SRA produces clear error, non-zero exit, no crash | Tested on ≥5 known-bad accessions |
| M4.4 | **100-sample batch** | singlify completes on 100 randomly selected human SRA accessions | ≥90% success rate, failures diagnosable from logs |
| M4.5 | **Resource budget** | ≤10 CPU-hours per sample (40M reads average) | Measured across the 100-sample batch |

### M5: Automatic Species Detection (gate for zero-configuration)

| # | Metric | Target | How to verify |
|---|--------|--------|---------------|
| M5.1 | **Species detection accuracy** | ≥99% on 200+ commonly used GEO species | Test on 500+ SRA accessions with known species |
| M5.2 | **Detection speed** | <5s from first 100K reads | Benchmark on diverse SRA accessions |
| M5.3 | **Non-host identification** | Detect and report non-host contamination fraction | Validate on known mixed-species datasets |
| M5.4 | **Marker database** | Compact index (≤10 MB) covering 200+ species | Size check + species coverage audit |

### M6: Automatic Protocol Detection (gate for zero-configuration)

| # | Metric | Target | How to verify |
|---|--------|--------|---------------|
| M6.1 | **Protocol detection accuracy** | ≥95% without any metadata | Test on 100+ SRA accessions, compare to catalog ground truth |
| M6.2 | **Expandable signature table** | New assays addable by table row, not code change | Add a new protocol and verify detection works |
| M6.3 | **Modality detection** | Correctly identify: scRNA, ATAC, multiome, CITE-seq, Visium, bulk | Test on ≥3 examples of each modality |
| M6.4 | **Chemistry identification** | Distinguish: 10x v2/v3/v4/5’/3’, Drop-seq, inDrop, sci-RNA-seq3, BD Rhapsody, Parse, Smart-seq2/3 | Protocol-specific test samples |

### M7: Quality Control & Preprocessing (gate for production quality)

| # | Metric | Target | How to verify |
|---|--------|--------|---------------|
| M7.1 | **Model-based cell calling (EmptyDrops++)** | ≥95% concordance with CellRanger EmptyDrops, AND recover ≥5% more real cells | Validate on cell-hashing ground truth datasets |
| M7.2 | **Ambient RNA correction** | SoupX/CellBender-lite built into pipeline | Correlation with standalone SoupX on ≥3 samples |
| M7.3 | **Doublet detection** | ≥85% concordance with Scrublet/scDblFinder | Benchmark on known doublet-enriched datasets |
| M7.4 | **Per-cell QC metrics** | MT%, ribo%, intronic%, gene count, UMI count, complexity, mapping rate, saturation | Automated in pipeline JSON manifest |
| M7.5 | **Sequencing saturation** | Per-cell + aggregate saturation curves + complexity estimation | Compare with CellRanger saturation output |
| M7.6 | **CellRanger feature audit** | All cost-effective CellRanger features incorporated | Documented comparison matrix |
| M7.7 | **Empty droplet pre-filter** | ≥80% of empty-droplet BCs removed before alignment | Compare with CellRanger cell count; measure compute savings |

### M8: Donor-Level Annotations (gate for population-scale analysis)

| # | Metric | Target | How to verify |
|---|--------|--------|---------------|
| M8.1 | **Ancestry classification** | 5-superpopulation accuracy ≥90% using 1000G/gnomAD AIMs | Validate on HapMap/1000G samples with known ancestry |
| M8.2 | **Sex calling** | ≥99% concordance with metadata | chrY fraction, XIST, X:autosome ratio on ≥50 samples |
| M8.3 | **Karyotype inference** | Detect trisomy 21, sex aneuploidies | Validate on known aneuploid samples |
| M8.4 | **Per-donor reporting** | All M8 annotations reported per-donor when demux active | Multi-donor pooled samples |
| M8.5 | **Allele-specific expression** | Per-gene ASE ratio from phased VCF het SNPs | Validate on known imprinted genes + ASE benchmarks |

### M9: Modality Coverage (gate for universal sequencing support)

| # | Metric | Target | How to verify |
|---|--------|--------|---------------|
| M9.1 | **scRNA droplet** | 10x v2/v3/v4, Drop-seq, inDrop, sci-RNA-seq3, BD Rhapsody, Parse | End-to-end on ≥1 sample each |
| M9.2 | **scRNA plate-based** | Smart-seq2, Smart-seq3 (no barcode, full gene body coverage) | End-to-end with multi-junction gene counting |
| M9.3 | **ATAC-seq** | 10x scATAC, 10x multiome ATAC component | Fragment file + peak calling |
| M9.4 | **Multiome (GEX+ATAC)** | Joint processing, matched barcodes | 10x multiome end-to-end |
| M9.5 | **CITE-seq** | ADT counting from HTO/ADT reads | Feature barcode counting |
| M9.6 | **Visium** | Spatial barcode mapping + gene counts | Spatial coordinate output |
| M9.7 | **Bulk RNA-seq** | No-barcode mode, standard gene counting | End-to-end on bulk samples |
| M9.8 | **V(D)J / Immune repertoire** | TCR/BCR contig assembly, IMGT alignment, CDR3 extraction, clonotype clustering | End-to-end on 10x 5' V(D)J samples |
| M9.9 | **CRISPR guide capture** | Auto-detect guide libraries, count guides per cell | Guide matrix output on Perturb-seq data |
| M9.10 | **Long-read (future)** | ONT/PacBio single-cell protocols (MAS-seq) | Deferred — architectural consideration |

### M10: Deep Archive Format (gate for long-term storage)

| # | Metric | Target | How to verify |
|---|--------|--------|---------------|
| M10.1 | **Quality stripping** | `singlify archive --strip-quality` produces valid .1fq without quality | Round-trip: encode → pileup → strip → verify metadata preserved |
| M10.2 | **Size reduction** | ≥15% smaller than quality-retaining .1fq | Benchmark on 10+ samples |
| M10.3 | **Prerequisite gate** | Quality stripping only after: donor demux, mtDNA variants, expressed genome variants, mapping rate computed | Pipeline enforces prerequisite checks |

### M11: Zero-Argument Invocation (gate for autonomous processing)

| # | Metric | Target | How to verify |
|---|--------|--------|---------------|
| M11.1 | **`singlify SRR…` with no flags** | Complete pipeline from SRA accession alone | Test on 100+ diverse SRA accessions |
| M11.2 | **Species auto-detect (Bloom filter)** | ≥99% accuracy on 200+ species, <5s | Validate on 500+ SRA accessions with known species |
| M11.3 | **Protocol auto-detect (no metadata)** | ≥95% accuracy without SOFT metadata | Compare to catalog ground truth on 100+ samples |
| M11.4 | **Reference index registry** | `singlify index list/fetch/add` functional | Fetch + pipeline for ≥5 species |
| M11.5 | **Auto whitelist resolution** | Protocol → built-in whitelist, no `--whitelist` | All 24 protocols resolve correctly |
| M11.6 | **Auto thread detection** | nproc or $SLURM_CPUS_PER_TASK | Run on different core counts |
| M11.7 | **Genome LoadAndKeep** | `singlify genome load/unload` functional | Verify shared memory across multiple runs |
| M11.8 | **Adapter auto-selection** | Protocol → adapter type from signature table | All protocols trim correctly without manual flags |
| M11.9 | **Instrument detection** | NovaSeq/HiSeq/NextSeq from quality distribution | Correct detection on ≥50 samples |

### M12: UMI Error Correction (gate for counting accuracy)

| # | Metric | Target | How to verify |
|---|--------|--------|---------------|
| M12.1 | **Directional 1-Hamming merging** | 2–5% reduction in overcounted molecules | Compare unique UMI counts with/without correction on ≥5 samples |
| M12.2 | **Concordance with UMI-tools** | ≥99% agreement on per-gene UMI counts | Side-by-side on ≥3 samples |
| M12.3 | **Performance** | <5% additional wall-clock time | Benchmark before/after on 40M read sample |

### M13: Pipeline Provenance (gate for reproducibility)

| # | Metric | Target | How to verify |
|---|--------|--------|---------------|
| M13.1 | **JSON manifest per run** | Contains: version, args, auto-detected params, checksums, QC metrics | Validate schema on 100+ runs |
| M13.2 | **Reproduce from manifest** | Re-run with manifest params produces identical output | Bit-exact .1pz on ≥3 samples |
| M13.3 | **Embedded in .1pz** | Summary provenance in .1pz metadata section | Read back via singlepress |

### Milestone Sequence

```
M1 (format) → M2 (aligner) → M3 (pileup) → M12 (UMI correction) → M4 (smoke tests)
  → M11 (zero-arg) → M5/M6 (auto-detect) → M13 (provenance) → M7 (QC)
  → M8 (annotations) → M9 (modalities) → M10 (deep archive) → scale-up
```

M1 and M2 can progress in parallel. M12 (UMI correction) should ship with M3. M11 (zero-arg) is the enabler for autonomous operation — it depends on M5/M6 but can be developed in parallel. M13 (provenance) is low-effort and should ship early. M7/M8 add production QC and biological annotations. M9 broadens modality coverage (including V(D)J, CRISPR). M10 enables long-term storage cost reduction.

---

## Quick start

```bash
git clone https://github.com/Singlet-Bio/singlify.git
cd singlify && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)   # produces: singlify, 1fq
ctest              # run tests
```

**Requirements**: GCC ≥ 7 (13 recommended), htslib ≥ 1.10, zstd, libncbi-vdb, zlib, pthreads, OpenMP. Optional: lz4.

### Step 1: Archive raw data to .1fq

```bash
# From SRA (streams from NCBI S3, no .sra file on disk)
singlify download SRR30891714 -o SRR30891714.1fq

# From FASTQ pair
1fq encode --reads R1.fastq.gz R2.fastq.gz -o sample.1fq
```

### Step 2: Process .1fq → .1pz

```bash
singlify SRR30891714.1fq \
  --genome-dir /path/to/star_index \
  --whitelist 10x_whitelist.txt \
  --exons genes.gtf \
  --snps snps.vcf.gz \
  --out-prefix results/ \
  --threads 16 \
  --pipeline
```

### .1fq utilities

```bash
1fq inspect sample.1fq           # Show header, protocol, block count
1fq decode sample.1fq -o R1.fq.gz R2.fq.gz   # Decode to FASTQ (interop)
1fq dedup sample.1fq -o dedup.1fq             # H1 UMI collapse
```

## Output

### Standard (always)

```
results/
├── exon_counts.1pz            # Per-exon UMI counts (genes × cells)
├── intron_counts.1pz          # Per-intron UMI counts
├── sj_counts.1pz              # Splice junction counts
├── snp_ad.1pz                 # SNP allele depth          ← only without --pipeline
├── snp_dp.1pz                 # SNP total depth           ← only without --pipeline
├── pileup_stats.json          # Processing statistics
└── star_Log.final.out         # Alignment summary
```

### Pipeline mode (`--pipeline`)

SNP matrices are consumed internally for donor demultiplexing — they are **not** written to disk. Instead:

```
results/
├── donor_assignments.tsv      # Per-cell donor assignment + posterior probability
├── donor0.vcf                 # Per-donor VCF — genotype calls from VB posterior AF
├── donor1.vcf                 #   GT:AF:AD:DP per SNP with coverage in this donor
├── ...
├── donor0_coverage.tsv        # Per-donor coverage map — dp, ad, af_vb, gt per covered SNP
├── donor1_coverage.tsv
├── ...
├── mt_variants.tsv            # Mitochondrial heteroplasmy variants
└── mt_heteroplasmy.1pz        # Per-cell mitochondrial VAF matrix
```

The per-donor VCF contains one sample column per file with `FORMAT: GT:AF:AD:DP`. Genotypes are called from the VB posterior allele frequency (`beta_mu`): AF < 0.1 → `0/0`, AF > 0.9 → `1/1`, otherwise `0/1`. Only SNP sites with read depth > 0 for that donor are emitted.

## Performance

| Stage | Time | Notes |
|-------|------|-------|
| SRA → .1fq encode | **19.6s** (40M reads, 4T) | 5.1× vs fasterq-dump |
| .1fq → align + pileup + export | **≈ alignment time** | Pileup fully overlapped |
| Alignment (5M reads, 8T) | **32.4s** | 48% faster than stock STAR |
| Alignment (5M reads, 16T) | **24.3s** | Thread scaling plateau |

Alignment is 92% of wall time. The [singlify-perf agent](.github/agents/singlify-perf.agent.md) tracks the optimization frontier.

## Format specifications

| Format | Spec | Description |
|--------|------|-------------|
| `.1fq` | [docs/1FQ_FORMAT_SPEC.md](docs/1FQ_FORMAT_SPEC.md) | Column-oriented, block-compressed sequencing archive |
| `.1pz` | via [singlepress](https://github.com/Singlet-Bio/singlepress) | VOCSC compressed sparse column matrix |

## Aligner

The aligner is derived from [STAR 2.7.11b](https://github.com/alexdobin/STAR) with eight singlet-lite optimizations that together provide ~48% wall-clock speedup vs stock: hash barcode lookup, 14-mer chunk sort, consecutive R2 dedup, `-march=native` + NUMA interleave, SA boundary prefetch, lazy winBin reset, and PGO+LTO. The full source lives in `src/star/` and is compiled as a CMake OBJECT library directly into the `singlify` binary. See [PERFORMANCE_SUMMARY.md](PERFORMANCE_SUMMARY.md) for benchmarks and dead ends.

## License

GPLv3. See [LICENSE](LICENSE).
