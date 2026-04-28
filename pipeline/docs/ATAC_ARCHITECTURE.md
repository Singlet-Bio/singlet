# ATAC Fragment Extraction Architecture

> **Status**: Design spec (A1–A7 pre-implementation)  
> **Date**: 2026-04-12  
> **Author**: bio-exec agent, cycle 74  
> **DAG tasks covered**: A1 (Fragment Extraction), A2 (Bin Matrix), A3 (QC), A6 (E2E), A7 (Cell Calling)  
> **Blocking A4/A5 (Donor Demux, Ancestry)**: zero new infrastructure needed — donor_demux.h and ancestry.h accept any genomic BAM SNP pileup

---

## 1. Pipeline Flow

```
.1fq (10x-atac)
     │
     ▼ lib1fq::Reader — auto-detects SC_ATAC, bc_len=16 in I2
     │
     ├── R1.fastq (genomic, 50bp)  ──┐
     ├── R3.fastq (genomic, 50bp)  ──┤  STAR PE-DNA mode
     └── barcode injected to QNAME──┘  (--alignIntronMax 1)
                │
                ▼
     BAM (SortedByCoordinate, uncompressed, QNAME has "BC:AACCGG_readname")
                │
                ▼ atac_fragment_extractor.h
     ┌──────────────────────────────────────┐
     │  For each properly paired read pair: │
     │   fragment = (chrom, start, end, bc) │
     │   Tn5 shift: start+=4, end-=5        │
     │   Dedup by (chrom, start, end, bc)   │
     └──────────────────────────────────────┘
                │
         fragments.tsv.gz (sorted by chrom, start)
                │
         ┌──────┴──────────┐
         ▼                 ▼
  atac_bin_matrix.h    atac_qc.h
  (bins × cells .1pz)  (TSS enrich, frag size dist,
                         FRIP, per-cell complexity)
                │
                ▼
  atac_cell_calling.h (A7)
  (TSS enrichment + unique fragments threshold → filtered barcodes.tsv)
```

---

## 2. Architecture Decisions

### Q1: Alignment Strategy — STAR in PE-DNA Mode (not bwa-mem2)

**Decision: STAR with splice-disabling flags.**

Rationale:
- STAR is already bundled; no new binary needed for A1/A6
- Shares the existing GRCh38/GRCm39 genome index with scRNA — no separate ATAC index
- Splice-aware behavior disabled via `--alignIntronMax 1 --alignMatesGapMax 2000 --alignSJDBoverhangMin 999`
- Fragment accuracy validated in literature: STAR ATAC achieves ~95–97% overlap with CellRanger ATAC on 10x datasets (Granja et al. 2021 benchmarks)
- `--soloType` OMITTED — STAR runs as a plain PE aligner, no CB/UMI handling by STAR itself
- Output: `BAM SortedByCoordinate`, uncompressed (same as parallel pileup mode)

**Future**: bwa-mem2 as bundled optional aligner (S4-class effort, ~1 sprint). Better for ATAC because no splice scoring overhead and lower false-positive multimapping near pericentromeric repeats. Flag as `--atac-aligner bwa-mem2` when available.

**STAR flags for ATAC mode:**
```
--alignIntronMax 1              # disable splicing
--alignMatesGapMax 2000         # max fragment size (ATAC typically <1000bp for accessible)
--alignSJDBoverhangMin 999      # prevent SJ reporting
--outSAMtype BAM SortedByCoordinate
--outBAMcompression 0
--outSAMattributes NH AS        # minimal SAM tags; CB not used (barcode in QNAME)
--outSAMunmapped None
--runMode alignReads
# NO --soloType, NO --soloCBwhitelist, NO --soloUMIlen
```

---

### Q2: Barcode Handling — QNAME Prefix Injection

**Decision: Embed barcode in QNAME during .1fq decode.**

10x ATAC layout:
- R1 = genomic (50bp) → R1_fifo
- R2 = barcode (16bp, I2 index read) → not a genomic read, stored as barcode in .1fq
- R3 = genomic (50bp) → R2_fifo (STAR R2)

In .1fq encoding: `lib1fq` stores R1 and R3 as stream[0]/stream[1] (both GENOMIC type), and the I2 barcode in the per-read barcode column. The CandidateSpec for `10x-atac` has `bc_len=0` because the geometry cannot be detected from R1 alone — the encoder must receive R2 (I2) as a separate stream.

During decode, the QNAME prefix injection approach:
```
Original:  @READNAME
Injected:  @BC:AACCGGTTAACC_READNAME
```
The `atac_fragment_extractor.h` extracts the barcode from QNAME by scanning for the `BC:` prefix and `_` delimiter. This avoids needing a SAM CB tag from STAR and works regardless of barcode whitelist matching.

**Fallback for SRA data**: When I2 is absent (SRA often strips index reads), the barcode field in .1fq will be empty (16 `N`s). The fragment extractor skips reads with N-only barcodes. Cell calling (A7) discovers the real barcodes from the frequency distribution.

---

### Q3: Fragment Extraction Core

From BAM (sorted by coordinate), for each read:
- Skip if `!BAM_FPROPER_PAIR` (flag 0x2 must be set)
- Skip secondary/supplementary/unmapped
- Extract barcode from QNAME prefix `BC:XXXX_`
- Retain only the **read1** of each pair (flag `BAM_FREAD1`): prevents double-counting
- Fragment boundaries from the read1 record:
  - `start = b->core.pos` (read1 5' end, 0-based)
  - `end = b->core.pos + abs(b->core.isize)` (TLEN field gives insert size including both reads)
- Tn5 correction: `start += 4`, `end -= 5`
- Output fragment struct: `{tid, start, end, barcode_packed}` (20 bytes)

Fragment output format (fragments.tsv.gz):
```
chr1    10450    10600    AACCGGTTAACCTTGG    1
```
Columns: chrom, start (0-based), end (exclusive), barcode, duplicate_count

---

### Q4: Deduplication — In-Memory Hash Set

**Decision: `std::unordered_set<FragmentKey>` with packed 128-bit key.**

Fragment key (16 bytes):
```cpp
struct FragmentKey {
    int32_t  tid;       // 4 bytes
    int32_t  start;     // 4 bytes (post-Tn5)
    int32_t  end;       // 4 bytes (post-Tn5)
    uint32_t bc_lo;     // lower 32 bits of packed barcode
    uint64_t bc_hi;     // upper 64 bits — total barcode = 12 bytes for 24 base barcodes
};
// For 16bp ATAC barcode (4-bit encoding): 8 bytes. Key = 20 bytes total with padding.
```

**Memory estimate for 200M PE reads (100M fragments)**:
- Raw fragments in flight: 100M × 20 bytes = 2.0 GB
- Hash set (load factor 0.7): 100M / 0.7 × 32 bytes per slot = ~4.6 GB peak
- Practical estimate after Tn5 correction and 80% unique rate: ~80M unique × 32 bytes = 2.6 GB
- **Recommendation**: Process per-chromosome (parallel_pileup model). Each chromosome fits in ~100–300 MB. No global dedup required since (chrom, start, end, bc) is chromosome-scoped.

**Per-chromosome dedup with sorted BAM**: The BAM is sorted by coordinate. Walk chromosome by chromosome. Each chr has ≤15M fragments. Hash set per chromosome: 15M × 32 = 480 MB peak. This is the same architecture as the parallel pileup workers.

**Alternative** (sort-based): merge-sort 100M × 20 bytes = 2 GB, scan for duplicates. Simpler but 3–4× slower. Reject.

---

### Q5: Bin Matrix

**Genome tiling**:
- GRCh38 (3.1 Gbp) / 500 bp = 6.2M bins total
- Variable-resolution support: bin_size configurable (default 500 bp; 1 kbp, 5 kbp also useful)
- Bin ID = per-chrom offset table (precomputed from genome header) + pos / bin_size
- `bin_offsets[tid]` array: sum of bins on all chromosomes with lower tid index

**Sparse accumulator reuse**:
- `SparseAccumulator<uint16_t>` from `sparse_accumulator.h` — direct reuse, no changes
- Feature index = bin_id (int32), cell index = barcode_index (int32), value = fragment count
- 6.2M bins × 5K cells = dense memory 31B entries → NOT dense; sparse OK
- Typical nnz: 5K cells × 5K unique fragments each = 25M entries (1 per fragment-bin hit)
- .1pz matrix: rows=bins, cols=cells, nnz≈25M — ~200 MB on disk

**Feature name generation**: `chr1:0-500`, `chr1:500-1000`, ... written to feature names file. For 6.2M bins this is a ~100 MB text file. Consider compact format: `tid:bin_idx` integers with chrom name lookup.

---

### Q6: QC Metrics (A3)

| Metric | Method | Cost |
|--------|--------|------|
| Fragment size distribution | Histogram of `(end-start)` in 10bp bins (0–2000bp) | O(n) in-pass |
| TSS enrichment | Count fragments overlapping TSS±2000bp, normalize by background (1900–2000bp flanks). Need TSS BED from GTF. | O(n log t) interval overlap |
| NFR fraction | Fraction fragments with (end-start) < 150 bp (nucleosome-free) | derived from hist |
| Mononucleosome fraction | Fraction 150–300 bp | derived from hist |
| Per-cell unique fragments | barcode → count map | O(n) |
| FRIP proxy | fraction of fragments in "signal bins" (bins with ≥2 cells) | post-bin-matrix pass |

TSS annotation: extract from GTF (already loaded for scRNA pipeline) — reuse `gene_model.h` TSS coordinates + 4 bp Tn5 correction. TSS enrichment score = mean_signal_in_TSS_window / mean_signal_in_background.

---

### Q7: Integration with Existing Pipeline

**Reuse map**:

| Module | Reuse | Notes |
|--------|-------|-------|
| `lib1fq` protocol detection | ✅ as-is | SC_ATAC already detected |
| `.1fq → FIFOs decode` | ✅ adapted | Inject BC tag in QNAME; both streams are genomic |
| STAR alignment | ✅ adapted | New flag set: PE-DNA mode, no soloType |
| `pileup_engine.h` BAM I/O layer | ✅ adapted | Reuse htslib open/parallel-region scan; replace processing core |
| `SparseAccumulator` | ✅ as-is | Bin matrix accumulation |
| `pz_writer.h` | ✅ as-is | Output bin matrix as .1pz |
| `gene_model.h` | ✅ adapted | Extract TSS positions for QC enrichment |
| `export.h` | ✅ as-is | Provenance and metadata |
| `donor_demux.h` | ✅ as-is | SNP pileup on ATAC fragments works identically to scRNA |
| `ancestry.h`, `sex_calling.h` | ✅ as-is | Consume SNP matrix regardless of modality |
| `cell_calling.h` (EmptyDrops++) | ⚠️ not applicable | ATAC cell calling uses TSS enrichment + unique fragment count |
| `umi_dedup.h` | ❌ not used | ATAC uses position-based dedup, not UMI |
| `pileup_engine.h` exon counting | ❌ not used | Replaced by fragment extraction |

**Divergence point**: After STAR alignment produces sorted BAM, the pipeline branches:
- scRNA path → existing `pileup_engine.h`
- ATAC path → new `atac_fragment_extractor.h`

The branch is controlled by `AssayType::SC_ATAC` detected from `.1fq` header.

---

## 3. New Headers Required

### `include/singlet-pileup/atac_fragment_extractor.h` (~300 LOC)
```cpp
// Core A1 data structures and extraction logic

struct AtacFragment {
    int32_t  tid;
    int32_t  start;   // 0-based, post-Tn5 (+4)
    int32_t  end;     // exclusive, post-Tn5 (-5)
    uint64_t barcode; // 4-bit packed 16bp barcode
};

// Extract fragments from sorted BAM with barcode in QNAME prefix "BC:XXXX_"
// Returns deduplicated fragments sorted by (tid, start, end)
// Tn5 shifts applied; only BAM_FPROPER_PAIR read1 records processed
std::vector<AtacFragment> extract_fragments(
    const std::string& bam_path,
    const BarcodeIndex& barcode_index,
    int tn5_shift_pos = 4,
    int tn5_shift_neg = 5,
    int n_workers = 1
);

// Write fragments to fragments.tsv.gz (cellranger-atac compatible format)
void write_fragments_tsv(
    const std::vector<AtacFragment>& frags,
    const sam_hdr_t* hdr,
    const BarcodeIndex& bc_index,
    const std::string& out_path
);
```

### `include/singlet-pileup/atac_bin_matrix.h` (~200 LOC)
```cpp
// A2: Genome tiling and fragment → bin overlap counting

struct BinGenome {
    std::vector<int32_t> chrom_bin_offsets; // cumulative bin counts per chrom
    int32_t bin_size;                        // default 500
    int64_t total_bins;
};

BinGenome tile_genome(const sam_hdr_t* hdr, int32_t bin_size = 500);

// Ingest fragment list → SparseAccumulator<uint16_t>
// Feature index = bin_id, cell index = barcode_index
void count_fragments_in_bins(
    const std::vector<AtacFragment>& frags,
    const BinGenome& bins,
    SparseAccumulator<uint16_t>& acc
);

// Generate feature name strings: "chr1:0-500", "chr1:500-1000", ...
std::vector<std::string> bin_feature_names(const sam_hdr_t* hdr, const BinGenome& bins);
```

### `include/singlet-pileup/atac_qc.h` (~250 LOC)
```cpp
// A3: Fragment size distribution, TSS enrichment, per-cell metrics

struct AtacQCResult {
    std::vector<int32_t> frag_size_hist;    // index = size/10, count
    double tss_enrichment_score;            // signal peak / background
    double nfr_fraction;                    // <150bp / total
    double mono_fraction;                   // 150-300bp / total
    std::unordered_map<uint64_t, int32_t> per_cell_unique_frags;
    double frip;                             // fraction in signal bins
};

AtacQCResult compute_atac_qc(
    const std::vector<AtacFragment>& frags,
    const GeneModel& gene_model,            // for TSS extraction
    const SparseAccumulator<uint16_t>& bin_matrix,
    int min_signal_cells = 2                // FRIP: bins with >=N cells
);
```

### `include/singlet-pileup/atac_cell_calling.h` (~150 LOC)
```cpp
// A7: Barcode filtering for ATAC (no UMI knee — use TSS enrichment + unique frag count)

struct AtacCellFilter {
    int32_t min_unique_frags;    // default 1000
    double  min_tss_enrichment;  // default 4.0
};

// Returns set of barcode indices passing filters
std::vector<int32_t> call_atac_cells(
    const AtacQCResult& qc,
    const AtacCellFilter& filter = {}
);
```

---

## 4. Memory and Compute Estimates (200M PE reads, 5K cells, GRCh38)

| Stage | Memory | Wall time (20 threads) |
|-------|--------|----------------------|
| `.1fq` decode → STAR FIFOs | ~500 MB | ~45s (decode) |
| STAR PE alignment | ~30 GB (genome in shm) + 4 GB RAM | ~250s (alignment bottleneck) |
| BAM → fragment extraction (per-chrom dedup) | 300 MB peak per worker | ~30s after sort |
| `fragments.tsv.gz` write | ~3 GB on disk | ~10s |
| Bin matrix accumulation (6.2M × 5K) | ~800 MB sparse acc | ~15s |
| QC metrics | ~200 MB | ~10s |
| **Total** | **~35 GB peak** | **~360s (~6 min)** |

Comparison: CellRanger ATAC on same data ≈ 2–4 hrs (sequential, no shared memory genome).  
singlify ATAC target: **≤10 min end-to-end** (primarily STAR-bound).

---

## 5. Reuse Map Summary

```
REUSE AS-IS:     lib1fq detection, SparseAccumulator, pz_writer, donor_demux, ancestry, sex_calling, export
REUSE ADAPTED:   .1fq decoder (QNAME injection), STAR invocation (PE-DNA flags), gene_model (TSS extraction)
NEW HEADERS:     atac_fragment_extractor.h, atac_bin_matrix.h, atac_qc.h, atac_cell_calling.h
NEW SRC:         test_atac_fragment.cpp, test_atac_bins.cpp
INTEGRATION:     singlify.cpp ATAC branch (~150 LOC) — documented in INTEGRATION_NOTES.md
```

---

## 6. Implementation Order (Minimizing Dependencies)

| Order | Task | DAG | Dependency | LOC |
|-------|------|-----|------------|-----|
| 1 | `atac_fragment_extractor.h` + unit tests | A1 | htslib only | ~300 |
| 2 | Singlify ATAC branch in `singlify.cpp` | A1 | Step 1 | ~150 (integration notes) |
| 3 | E2E validation vs CellRanger ATAC | A6 | Steps 1–2 | ~0 (scripts) |
| 4 | `atac_bin_matrix.h` + unit tests | A2 | A1 frags | ~200 |
| 5 | `atac_qc.h` + TSS enrichment | A3 | A1 frags, gene_model | ~250 |
| 6 | `atac_cell_calling.h` | A7 | A3 QC | ~150 |
| 7 | A4/A5 donor demux + ancestry | A4, A5 | A1 fragments (donor_demux.h unchanged) | ~50 glue |

**Total new code**: ~1,100 LOC + ~400 test LOC = ~1,500 LOC

---

## 7. LOC Estimates per Component

| File | Type | LOC |
|------|------|-----|
| `include/singlet-pileup/atac_fragment_extractor.h` | Header | 300 |
| `include/singlet-pileup/atac_bin_matrix.h` | Header | 200 |
| `include/singlet-pileup/atac_qc.h` | Header | 250 |
| `include/singlet-pileup/atac_cell_calling.h` | Header | 150 |
| `src/test_atac_fragment.cpp` | Test | 250 |
| `src/test_atac_bins.cpp` | Test | 150 |
| `singlify.cpp` ATAC branch (integration) | Integration | 150 |
| **Total** | | **~1,450** |

---

## 8. Open Questions / Risks

1. **I2 barcode in SRA data**: SRA `prefetch` often includes I2 if `--include-technical` was used at encoding time. The lib1fq encoder must handle 3-stream ATAC (R1, R2=barcode, R3). Verify corpus sample SRR32855204 has I2 before implementation.

2. **STAR splice scoring for ATAC**: Even with `--alignIntronMax 1`, STAR still loads the splice junction DB and scores reads against it. Benchmark whether STAR PE-DNA mode alignment rate matches bwa-mem2 on a test ATAC sample. If >5% rate penalty, prioritize bwa-mem2 bundling.

3. **Fragment size cutoff for Tn5**: Standard +4/-5 is for forward/reverse strand respectively. The pileup engine already tracks `BAM_FREVERSE` — use strand-aware Tn5 correction: if read1 is forward: start+=4; if read1 is reverse: end-=5. Emit nucleosome-free fragments as-is.

4. **737K-cratac-v1 whitelist**: The file is referenced in `CandidateSpec` but not confirmed bundled. Check `singlify/whitelists/` — currently only RNA whitelists present. Must obtain or generate ATAC-specific whitelist before whitelist-based cell calling (A7 fallback).

5. **Multiome ATAC (`SC_MULTIOME_ATAC`)**: Same fragment extraction logic; barcode whitelist is `gex_737K-arc-v1.txt` (already bundled). The GEX and ATAC components share barcodes — enables joint cell calling. Out of scope for A1–A7 but architecture supports it via AssayType branching.
