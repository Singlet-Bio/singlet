# singlet-pileup Development Roadmap

> **Date**: 2026-04-08  
> **Binary version at time of writing**: v0.2.0 (`build_v2/singlet-pileup`)  
> **Test sample**: GSM8313394 — 38M reads, 2,013 cells, GRCh38, 10x Chromium v3  
> **Status convention**: ✅ Complete · 🔶 Partial · ❌ Not started

---

## 0. Why This Document Exists

singlet-pileup's Phase 1 core (exon/intron/SJ/SNP/chrM in one BAM pass) is substantially
complete and functionally validated on one sample. But three important categories of work
remain before the binary is production-ready:

1. **Counting accuracy gaps** — TE fractional counting, 1/NH activation, multi-gene boundary
2. **Output format** — MTX is inefficient; native .1pz is required
3. **Downstream steps** — Vireo (donor demux) and mgatk (mt heteroplasmy) are Python tools
   with external dependencies; both can be re-implemented as ~500-800 LOC C++ extensions
   inside singlet-pileup with better performance and no extra runtime dependencies

---

## 1. Outstanding Work: Phase 1 Completion

### 1.1 Native `.1pz` Output (Replace MTX Entirely)

**Current state**: The `SparseAccumulator<T>` produces a `CSCMatrix` struct
`{indptr, indices, data, nrows, ncols}`. Then `mtx_writer.h` dumps this to `.mtx` + TSV
companion files. Downstream, a separate Python script converts MTX → `.1pz`.

**Why `.mtx` is the wrong format**:
- MTX writes every non-zero as `row col val\n` — three 64-bit text conversions per entry
- At 50M reads × ~2 exon hits/read → ~100M entries → 100M `fprintf` calls → ~20-40 seconds
- MTX + TSV requires multiple file handles, multiple filesystem operations
- Python conversion script adds another 30-60 seconds
- Total overhead: 50-100 seconds per sample vs. zero if written natively

**Why native `.1pz` is straightforward**:  
The `.1pz` format (`pz_codec.cpp`) is a VOCSC (value-optimized CSC) format with the
following C-compatible write path:

```
PZHeader   (96 bytes, fixed)
  ├── magic = 0x5A315054 ("TP1Z")
  ├── version = 1
  ├── vt_code: 1=uint8, 2=uint16, 3=uint32
  ├── m = n_features (rows)
  ├── n = n_barcodes (cols)
  ├── nnz
  ├── num_chunks, chunk_cols (typically 64 cols/chunk)
  └── metadata_offset, colsums_z_sz, ...
[perm] — zstd-compressed sorted row-index permutation
[indptr chunks] — zstd-compressed column pointer array
[data chunks × num_chunks]
  ├── [byte-split(values)] — adaptive byte plane separation
  └── zstd-3 per chunk
[metadata section]  — zstd[TAG_ROWNAMES + TAG_COLNAMES + optional KV]
[colsums] — zstd[double array, one per column]
PZFooter  (16 bytes, CRC32 + magic)
```

**Implementation plan** — new file `include/singlet-pileup/pz_writer.h`:

```cpp
namespace singlet {

// Single-function interface identical to SparseAccumulator::CSCMatrix contract
template<typename ValT>
bool write_1pz(
    const std::string& path,
    uint32_t n_features,
    uint32_t n_barcodes,
    const std::vector<int32_t>& indptr,   // CSC column pointers (size n_barcodes+1)
    const std::vector<int32_t>& indices,  // CSC row indices    (size nnz)
    const std::vector<ValT>& data,        // CSC values         (size nnz)
    const std::vector<std::string>& rownames,  // feature IDs
    const std::vector<std::string>& colnames,  // barcode strings
    int zstd_level = 3
);
```

**Data flow for encoding**:
1. Compute column sums (one pass over CSC data, store as `double[]`)
2. Build VOCSC: sort indices within each column (already guaranteed from `to_csc()`),
   write row indices as byte-split across chunks of `chunk_cols` columns
3. zstd-compress each chunk at level 3
4. Write metadata: serialize rownames/colnames as null-terminated UTF-8 blobs,
   zstd-compress the blob, write with TAG_ROWNAMES/TAG_COLNAMES framing
5. Write header, chunks, metadata section, colsums, footer in one sequential file write

**Byte-split algorithm** (the core compression trick from pz_codec.cpp):
```
For each chunk of C columns containing K values:
  value_bytes[k] = ValT is uint16 → split into low byte[] and high byte[]
  compressed = zstd(low_bytes || high_bytes)
  → ~40% better ratio than zstd on interleaved bytes
```

This is ~400 LOC of C++ in a header-only file, no Python required, no external Python
interpreter call. `zstd.h` is already a CMake dependency.

**Remove from codebase**:
- `include/singlet-pileup/mtx_writer.h`
- `python/mtx_to_1pz.py`
- All `write_mtx()` calls in `src/main.cpp`

**Success metric**: `md5sum output.1pz == md5sum python_converted.1pz` on the same CSC data
(bit-exact round-trip). Validated on GSM8313394 exon_counts. Write speed target: >500 MB/s
(below zstd throughput floor); total write time for all matrices <5 seconds.

**Testing framework**:
```bash
# Unit test: write random 10K×2K sparse matrix, read back, compare
tests/test_pz_writer.cpp — generate random CSC, write_1pz, pz_read (via Python), assert exact match
# Integration test on real data
scripts/validate_1pz.sh: run pileup → .1pz, then python pz_read → compare vs. old MTX output
```

---

### 1.2 TE Subfamily Quantification via Augmented GTF

**Current state**: GTF loader (`gene_model.h`) parses only standard Ensembl features
(gene/exon/transcript). RepeatMasker TE annotations are a separate data source not yet
loaded.

**What the proposal requires**: TEs (L1HS, AluY, HERVH, SVA_A, ~1,200 subfamilies) appear
as `gene` entries in an augmented GTF where RepeatMasker loci are grouped by subfamily.
Every exon in the exon interval tree is either a canonical gene exon or a TE-subfamily
"exon" (the consensus TE body). Multi-mapped reads (NH > 1) hitting TE elements receive
1/NH fractional credit toward the TE subfamily row in `exon_counts`.

**Data structures**:
```
GeneRecord already handles arbitrary gene_id/gene_name.
TE subfamilies are simply genes with biotype="transposable_element".
No structural change to interval tree or accumulator required.

New flag in GeneRecord:
  bool is_te = false;  // sourced from gene_biotype == "transposable_element"

New field in exon feature metadata:
  uint8_t biotype;  // 0=protein_coding, 1=lncRNA, ..., 15=TE
```

**1/NH fractional counting — activate the existing flag**:

The `PileupConfig::multimapper_frac` flag exists but is not yet wired into the counting
loop. The current multi-mapper logic (gene-unique buffer) is correct for unique-gene
multi-mappers but wrong for TEs, where every alignment should contribute 1/NH.

```cpp
// In the main read loop (pileup_engine.h), after gene lookup:
if (is_te_gene && nh > 1) {
    // Fractional: accumulate 1/NH credit
    // Since SparseAccumulator<uint16_t> is integer, scale: store as fixed-point
    // OR: maintain a separate SparseAccumulator<float> for TE fractional counts
    te_frac_acc_.increment_frac(exon_idx, bc_idx, 1.0f / nh);
} else {
    // Gene-unique path (existing logic unchanged)
}
```

**Implementation note — float accumulator**: The current `SparseAccumulator<uint16_t>` is
insufficient for fractional values. Two options:
- (A) `SparseAccumulator<float>` — straightforward but 4× memory per entry vs. uint16
- (B) Store TE counts as uint16 scaled: 1/NH × 65535, clamp — acceptable for subfamily
  quantification where precision <1% is irrelevant

Option B is recommended for memory efficiency. Add `increment_frac(feat, bc, w)` method
that accumulates `(uint16_t)(w * 65535)` and performs integer summation.

**Augmented GTF construction** — not in C++ binary, done once at index-build time:
```bash
# Merge Ensembl 113 GTF + RepeatMasker BED → augmented_genes.gtf.gz
# Script: scripts/build_augmented_gtf.py
# Output: /cellarium/reference/GRCh38-2024-A/genes/augmented_genes.gtf.gz
# Size: ~80K features (78K genes + 1.2K TE subfamilies)
```

**Success metric**: TE subfamily totals from singlet-pileup agree with TEtranscripts
(Python ground-truth tool) to within 10% at subfamily level on GSM8313394.
Pearson r of TE counts vs. TEtranscripts: r > 0.9.

---

### 1.3 Production Streaming Mode (`CB_samTagOut`)

**Current state**: `scripts/stream_pipeline.sh` uses `--soloType CB_UMI_Simple` + 
`--soloFeatures Gene` for validation comparison. Production mode uses `CB_samTagOut`
(no counting by STAR, faster).

**What needs changing**:
1. `stream_pipeline.sh`: add `--soloType CB_samTagOut` path that skips STARsolo matrices
2. STAR command in `stream_pipeline.sh` produces BAM via `--outStd BAM_Unsorted` piped
   through `tee` to both singlet-pileup and `samtools sort`
3. Validation scripts need to handle the case where no STARsolo Gene matrix exists
   (production mode) — validate against pre-computed reference instead

**Timing target**: STAR `CB_samTagOut` saves ~2-9 min per sample vs. `CB_UMI_Simple`
(no Solo matrix allocation, no per-gene feature counting). Measure before/after on 5
samples to confirm.

---

### 1.4 Multi-Sample Validation Suite (100 Samples)

**Current state**: Validated on GSM8313394 only. One sample is not production evidence.

**Required**: Run `validate_full.py` on 100 heterogeneous samples before deploying Pipeline V2:

| Stratum | Count | Criteria |
|---|---|---|
| 10x 3' v3, human, >50M reads | 25 | Primary use case |
| 10x 3' v3, human, <10M reads (small) | 15 | Low-read edge cases |
| 10x 3' v2, human | 15 | Older chemistry (UMI len 10 vs 12) |
| 10x 3' v3, mouse | 15 | Non-human species |
| Multiplexed (multiple donors) | 10 | Vireo non-trivial |
| High TE expression (cancer/ESC) | 10 | TE quantification stress |
| High mt fraction (>20% chrM) | 10 | mgatk stress |

**Batch validation script**: `scripts/validate_batch.sh` — takes a file of GSM IDs,
submits N parallel SLURM jobs, collects JSON stat files, runs `scripts/aggregate_validation.py`
to compute population-level correlation distributions.

**Pass criteria across 100 samples**:
- Median gene Pearson r > 0.999
- P5 gene Pearson r > 0.995 (5th percentile — bad samples still acceptable)
- Median cell-level r > 0.9999
- Max reconstruction error (exon→gene) = 0% across all samples
- chrM FPR < 0.01% in all samples
- No output completeness failures

---

## 2. Phase 2 — Internal Vireo Replacement (`singlet-demux`)

### 2.1 Why Internalize Vireo

**vireoSNP performance constraints**:
- Python + numpy + scipy for sparse matrix ops → slow startup (~15s import), GIL-bound
- Single-threaded VB iteration; `nproc` parallelizes only across random restarts, not
  within a single optimization
- External command-line invocation means pipe-breaking: need to write AD/DP to disk,
  call vireo, read donor_assignments.tsv back
- No C++ ABI — we can't call vireo inline from singlet-pileup's main loop

**What we gain by internalizing**:
- AD/DP matrices never touch disk (they live as `SparseAccumulator<uint8_t>` in memory
  at the end of the BAM stream)
- Vireo runs immediately after streaming completes, on the in-memory CSC matrices
- Total pipeline: STAR → singlet-pileup → [demux inline] → write .1pz + donor_assignments.tsv
- No subprocess, no serialization overhead, ~5 min wall-clock for single-donor → <30s

### 2.2 Algorithm (Variational Bayes Binomial Mixture)

The core Vireo algorithm is a variational Bayesian inference over a binomial mixture:

**Model**:
```
cell c has genotype assignment z_c ∈ {0, ..., K-1, doublet}
at SNP j, donor k has allele frequency θ_jk ~ Beta(α, β)
reads: AD[j,c] | DP[j,c], θ_j,z_c ~ Binomial(DP[j,c], θ_j,z_c)
```

**VB update equations** (coordinate ascent):

*E-step* — update assignment probabilities q(z_c):
```
log q(z_c = k) ∝ Σ_j [ AD[j,c] · E[log θ_jk] + 
                         (DP[j,c] - AD[j,c]) · E[log(1-θ_jk)] ]
where E[log θ_jk] = ψ(α_jk) - ψ(α_jk + β_jk)  (digamma function)
```

*M-step* — update genotype parameters q(θ_jk) = Beta(α_jk, β_jk):
```
α_jk = α_prior + Σ_c [ q(z_c=k) · AD[j,c] ]
β_jk = β_prior + Σ_c [ q(z_c=k) · (DP[j,c] - AD[j,c]) ]
```

*ELBO* for convergence check:
```
L = Σ_c Σ_j [ AD[j,c]·E[log θ_jk(c)] + (DP-AD)[j,c]·E[log(1-θ_jk(c))] ]
    - KL[q(θ) || p(θ)] - KL[q(z) || p(z)]
```

Converges in ~50-100 iterations. Run 20-50 random restarts (different K_init), keep best ELBO.

### 2.3 C++ Implementation Plan

**New file**: `include/singlet-pileup/donor_demux.h`

```cpp
namespace singlet {

struct DonorDemuxConfig {
    int n_donors = -1;          // -1 = auto-detect (1 to max_donors)
    int max_donors = 8;
    int max_iter = 200;
    int n_inits = 30;           // random restarts
    float alpha_prior = 0.3f;  // Beta prior α
    float beta_prior = 0.5f;   // Beta prior β
    float eps_conv = 1e-4f;    // ELBO convergence threshold
    int threads = 4;
    uint64_t seed = 42;
};

struct DonorAssignment {
    int donor_id;           // 0..K-1, or K=doublet, K+1=unassigned
    float prob_max;         // posterior P(z_c = donor_id)
    float prob_doublet;
};

// Input: CSC AD and DP matrices (SNP × cell), both uint8 (saturated at 255 is fine)
// Returns per-cell assignments
std::vector<DonorAssignment> run_vireo(
    const SparseAccumulator<uint8_t>::CSCMatrix& ad_csc,
    const SparseAccumulator<uint8_t>::CSCMatrix& dp_csc,
    const DonorDemuxConfig& cfg
);
```

**Key C++ design decisions**:

1. **Dense float scratch buffers** for theta params and assignment probs (not sparse — 
   at 7.4M SNPs × K donors, we only work with the ~2-10K SNPs that have coverage):
   ```cpp
   // Covered SNPs: those with DP.nnz > 0
   // Extract indices of covered SNPs once, work in reduced dense subspace
   std::vector<int> covered_snps;  // indices of SNPs with any coverage
   int n_cov = covered_snps.size();  // typically 2K-10K even for 7.4M panel
   
   // Dense matrices in covered-SNP subspace:
   float theta[n_cov][K];    // genotype allele freqs
   float alpha[n_cov][K];    // Beta α per SNP per donor
   float beta[n_cov][K];     // Beta β per SNP per donor
   float probs[n_cells][K];  // assignment probabilities (log-domain)
   ```

2. **digamma approximation**: Use the standard asymptotic series
   `ψ(x) ≈ log(x) - 1/(2x) - 1/(12x²)` — accurate to 1e-5 for x > 1, sufficient for VB.

3. **Parallelism**: OpenMP `#pragma omp parallel for` over the n_cells E-step loop.
   Each cell is independent given current theta → embarrassingly parallel.

4. **Auto-detect n_donors**: Run for K = 1, 2, ..., max_donors, pick highest ELBO.
   This is the primary use of the n_inits budget.

**Complexity**: O(n_cov × n_cells × K × n_iter × n_inits)
- n_cov ≈ 5,000 (covered SNPs), n_cells ≈ 5,000, K = 1-4, n_iter = 100, n_inits = 20
- ≈ 5K × 5K × 4 × 100 × 20 = 2 × 10^10 ops → ~2s at 10^10 FLOPS/core

### 2.4 Ground-Truth Validation

**External ground truth**: Run vireoSNP (Python) on the same AD/DP matrices saved to disk.
Compare assignments:

```
# Success criteria:
# - Donor assignment NMI > 0.99 (adjusted mutual information vs. vireoSNP)
# - Doublet probability Pearson r > 0.99
# - n_donors detected matches vireoSNP's best-ELBO K
# - Runtime: <30s for 5K cells × 1 donor, <120s for 10K cells × 4 donors
```

**Synthetic validation**:
```python
# Generate: 4 donors × 1K cells, 3K SNPs, known ground-truth assignments
# Add noise: 10% allelic dropout, 5% doublets
# Run both vireoSNP and singlet-demux, compare to known labels
# Metric: F1 score on donor assignment, AUROC for doublet detection
```

**Test harness**: `tests/test_donor_demux.cpp` — synthetic sparse AD/DP, run C++ VB loop,
compare to known assignments loaded from a pre-computed reference file.

---

## 3. Phase 2 — Internal mgatk Replacement (`singlet-mt`)

### 3.1 What mgatk Actually Does (and Why We Can Do Better)

mgatk's core computation is conceptually simple:
1. Extract all reads aligned to chrM (already done by singlet-pileup's chrM accumulator)
2. For each read, for each base position, count ref vs. alt alleles
3. Create `cells × mt_positions` matrix of allele fractions

The complexity in mgatk is almost entirely **I/O orchestration** — Snakemake workflow,
Java runtime, per-cell BAM splitting, file coordination. The actual math is:
```
VAF[cell, pos] = alt_count[cell, pos] / (alt_count[cell, pos] + ref_count[cell, pos])
```

**Problem with mgatk's approach**: It splits the chrM.bam by barcode into per-cell mini-BAMs,
then piles up each mini-BAM separately. This is O(n_cells × n_mt_reads) in I/O. For 5K cells
with 16,569 mt positions, that's 5K file open/close operations + 5K streaming passes over
mt reads.

**Our approach**: Single streaming pass over chrM reads (the existing `chrm_buffer`).
This is already how singlet-pileup's main loop works — extend the chrM accumulation:

```
chrM buffer → instead of just writing a BAM, accumulate:
  AT[cell, pos] = ref allele count (A/T/C/G)
  For mt, there are only 16,569 positions × 4 bases = 66,276 feature dimensions
  = trivially small compared to exon/intron/SNP accumulators
```

### 3.2 Data Structures

**New accumulator** (added to pileup_engine.h):

```cpp
// Mitochondrial allele counts: 4 bases × 16,569 positions = 66,276 "features"
// Per-cell: use SparseAccumulator<uint16_t> with feature = pos*4 + base_idx
// uint16 can hold up to 65,535 reads per cell at each mt position — sufficient
// (typical chrM coverage per cell: 100-10,000x; position-specific 100-500x)

SparseAccumulator<uint16_t> mt_acc_;
// feature index: pos * 4 + {A=0, C=1, G=2, T=3}
// n_features = 16569 * 4 = 66276
```

During the main BAM loop, when `chrom_name == "chrM"` or `chrom_name == "MT"`:
```cpp
// Instead of buffering raw reads to chrM.bam:
// Pileup inline: for each base in the read CIGAR walk, 
// extract base at each mt reference position
for (int qi = 0, ri = ref_start; qi < seq_len; ++qi, ++ri) {
    if (ri < 0 || ri >= 16569) continue;
    int base = bam_seqi(seq, qi);  // 1=A, 2=C, 4=G, 8=T, 15=N
    if (base == 15) continue;      // skip N
    int base_idx = base_to_idx[base];  // 0-3
    uint32_t feat = ri * 4 + base_idx;
    mt_acc_.increment(feat, bc_idx);
}
```

This eliminates the chrM.bam file entirely (or keeps it only if mgatk comparison is needed).
Memory: 66,276 positions × 5K cells × ~sparse factor 0.1 = ~3.3M entries × 2 bytes = ~7MB.

### 3.3 Heteroplasmy Matrix Computation

After streaming is complete, compute the per-cell heteroplasmy matrix:

```cpp
struct MtHetResult {
    // variable_positions: positions where any cell shows heteroplasmy
    std::vector<int32_t> variable_positions;   // position IDs
    std::vector<char> ref_alleles;             // ref allele at each position
    std::vector<char> alt_alleles;             // alt allele at each position
    
    // Donor consensus: for each donor, the plurality allele at each position
    // (used to subtract germline background)
    std::vector<std::vector<char>> donor_consensus;  // [donor][position]
    
    // Per-cell × variable_position sparse matrix of alt allele fractions
    SparseAccumulator<float> heteroplasmy;
};

MtHetResult compute_mt_heteroplasmy(
    const SparseAccumulator<uint16_t>::CSCMatrix& mt_csc,
    const std::vector<DonorAssignment>& assignments,
    float min_af = 0.01f,      // min heteroplasmy to report
    float max_af = 0.99f,      // max (homoplasmic positions excluded)
    int min_coverage = 5       // min reads per cell at position
);
```

**Algorithm**:
1. Convert `mt_csc` to per-position, per-cell ref/alt counts
   (given known rCRS reference allele at each position, loaded from a 16,569-byte file)
2. For each donor: across all assigned cells, aggregate → find plurality allele at each
   position → this is the donor consensus
3. For each cell: for each position, compute VAF relative to donor consensus
4. Threshold: only emit positions where >0 cells show 0.01 < VAF < 0.99

**Reference allele source**: A simple 16,569-byte rCRS reference string
(`reference/rCRS.txt`). This is smaller than any library.

### 3.4 Ground-Truth Validation

**External**: Run mgatk on same chrM.bam, compare output:
```
# Success criteria:
# - VAF Pearson r > 0.99 at covered positions (cells × positions)
# - Same set of "variable positions" detected (Jaccard similarity > 0.9)
# - Runtime: <30s for typical 5K-cell sample (vs. mgatk's 30-90 min)
# - Memory: <100 MB additional (vs. mgatk's 2-4 GB for BAM splitting)
```

**Synthetic validation**:
```python
# Generate: 2 haplotypes at known 40/60 ratio, 3 heteroplasmic positions
# Simulate per-cell reads with Binomial(coverage, VAF) sampling
# Verify: detected VAF ≈ 0.4 ± 0.02 across cells
```

---

## 4. Phase 3 — Orchestration and Integration

### 4.1 New `--pipeline` Mode

Once demux and mt heteroplasmy are internalized, singlet-pileup can run the full
pipeline in a single invocation:

```bash
singlet-pileup \
  --pipeline \                          # enables demux + mt inline
  --barcodes barcodes.tsv \
  --exons augmented_genes.gtf.gz \
  --snps common_snps_hg38.tsv.gz \
  --out-prefix ./output/ \
  --threads 12 \
  --n-donors auto \
  starsolo_output.bam
```

**Output** (all native .1pz, no intermediate files):
```
output/
  exon_counts.1pz         (features × cells, uint16)
  intron_counts.1pz       (features × cells, uint16)
  sj_counts.1pz           (junctions × cells, uint16)
  donor_assignments.tsv
  mt_heteroplasmy.1pz     (cells × variable_mt_positions, float32)
  pileup_stats.json
  # snp_ad/dp NOT written (only used internally for demux)
```

The sorted BAM for VCF calling remains a separate downstream step driven by the
orchestrator script (not by singlet-pileup itself — variant calling needs a dedicated
bcftools/GATK call anyway).

### 4.2 Pipeline Script (`scripts/run_pileup_v2.sh`)

Full streaming pipeline script. Key improvement over current `stream_pipeline.sh`:
uses `CB_samTagOut` mode and `tee` process substitution for zero-disk BAM:

```bash
STAR ... --soloType CB_samTagOut \
         --outStd BAM_Unsorted \
         --runThreadN 8 \
| tee >(singlet-pileup --pipeline --barcodes ... --exons ... --snps ... \
          --threads 4 --out-prefix $OUT_DIR -) \
| samtools sort -@ 4 -m 4G -o $OUT_DIR/sorted.bam

# After pileup completes: VCF calling using donor assignments
python scripts/call_donor_vcf.py \
  --bam $OUT_DIR/sorted.bam \
  --donors $OUT_DIR/donor_assignments.tsv \
  --out-dir $OUT_DIR
# Then delete sorted.bam
```

---

## 5. The `.mtx` Abolition Plan

> **Short answer to the question**: Yes, absolutely. Writing `.mtx` + TSV is not just
> inefficient — it's the wrong abstraction for a C++ streaming pipeline.

### 5.1 What `.mtx` Costs

For a typical 5K-cell, 50M-read sample:
- exon_counts.mtx: ~100M non-zeros → `fprintf("%d %u %d\n", ...)` → 100M syscalls
- Buffered: ~20-40 seconds of printf-to-disk
- File sizes: ~500MB→compressed to ~15MB by Python later

vs. native .1pz write:
- Same CSC data → `fwrite(compressed_chunk, ...)` → 8-16 writes total
- ~1-3 seconds total write time
- Output already compressed: ~15MB directly

**The MTX → .1pz conversion Python script** is a complete workaround for a missing C++
capability. Once `pz_writer.h` exists, it should be deleted.

### 5.2 Migration Plan

**Stage A** (immediate): Add `pz_writer.h` alongside `mtx_writer.h`. Both exist.
Add `--output-format {mtx,1pz}` CLI flag (default `mtx` for validation continuity).

**Stage B** (after 100-sample validation): Flip default to `1pz`. Keep `--output-format mtx`
as a debugging/comparison flag.

**Stage C** (production): Remove `mtx_writer.h` entirely. Remove `python/mtx_to_1pz.py`.

### 5.3 pz_writer.h Implementation Notes

The `.1pz` format writes CSC directly. The `SparseAccumulator<T>::CSCMatrix` struct matches
the format's requirements exactly:
- `indptr` → `p` (column pointers)
- `indices` → `i` (row indices, sorted within each column — guaranteed by `to_csc()`)
- `data` → `x` (values)

The only extra work is:
1. Byte-split transform for each chunk of `chunk_cols` (64) columns
2. zstd compression of each byte-plane pair
3. Metadata serialization (rownames + colnames → null-delimited byte blob → zstd)
4. CRC32 accumulation over all written bytes

All of these are standard C operations. The `pz_codec.cpp` source is the reference
implementation — extract the write path (~600 lines) into a header-only template, removing
the Python/pybind11 bindings and replacing with a pure C++ file-writing interface.

**One important decision**: The `.1pz` file needs to be stable-format-compatible with
what singlepress Python reads. The `PZHeader` struct is fixed at 96 bytes with all offsets
specified. As long as the C++ writer produces exactly the same byte layout as `pz_codec.cpp`'s
write path (which it must — it's a re-implementation of the same format), all existing
singlepress readers will work without modification.

**Column-sum embedding**: `.1pz` stores per-column sums in a compressed section
(used for fast normalization in `pz_colsums()`). For exon_counts, this is the total
UMI count per cell — a key QC metric. The C++ writer should compute and embed these
at write time rather than requiring a second pass.

---

## 6. Test and Debug Harness

### 6.1 Unit Test Coverage (current + planned)

| Test | File | Status | What it tests |
|---|---|---|---|
| Interval tree | `test_interval_tree.cpp` | ✅ 6 tests | Overlap queries, edge cases |
| Sparse accumulator | `test_sparse_accumulator.cpp` | ✅ 7 tests | Dedup, saturation, CSC structure |
| Pileup integration | `test_pileup_integration.cpp` | ✅ 10 checks | SAM→pileup→MTX round-trip |
| .1pz writer | `test_pz_writer.cpp` | ❌ not started | CSC→.1pz bit-exact round-trip |
| Donor demux VB | `test_donor_demux.cpp` | ❌ not started | Synthetic AD/DP → known assignments |
| mt heteroplasmy | `test_mt_heteroplasmy.cpp` | ❌ not started | Synthetic mt reads → known VAF |

### 6.2 Integration Validation (`validate_full.py`)

The existing `validate_full.py` covers 7 checks on one sample. Extend to:

```python
# Check 8: .1pz round-trip
P_1pz = singlepress.read("exon_counts.1pz")
P_mtx = scipy.io.mmread("exon_counts.mtx")
assert np.allclose(P_1pz["values"], P_mtx.data)  # bit-exact float comparison

# Check 9: Donor demux agreement with vireoSNP
vireo_out = pd.read_csv("vireo_reference/donor_assignments.tsv", sep="\t")
pileup_out = pd.read_csv("donor_assignments.tsv", sep="\t")
common = set(vireo_out.barcode) & set(pileup_out.barcode)
nmi = sklearn.metrics.adjusted_mutual_info_score(
    vireo_out.loc[vireo_out.barcode.isin(common), 'donor_id'],
    pileup_out.loc[pileup_out.barcode.isin(common), 'donor_id']
)
assert nmi > 0.99, f"Donor demux NMI = {nmi:.4f}"

# Check 10: mt heteroplasmy agreement with mgatk
# Load mgatk reference output from pre-computed run
# Compare variable positions (Jaccard), VAF correlation
```

### 6.3 Diverse Sample Debugging Protocol

**For each new feature (1/NH TE, donor demux, mt heteroplasmy), debug on a panel of 5 samples**:

```
Sample tier 1 (easy):
  GSM8313394 — 38M reads, 2K cells, human, clean PBMC (existing smoke test)
  
Sample tier 2 (stress):
  High-TE: ESC or cancer line sample (search GEO: "embryonic stem cells scRNA-seq")
  Multi-donor: pooled PBMC sample with ≥3 known donors (search GEO: "pooled donors scRNA-seq")
  High-mt: apoptotic/damaged sample with >15% chrM reads
  
Sample tier 3 (edge cases):
  Tiny (<1M reads): check empty-matrix handling, no OOB access
  Huge (>200M reads): check 64-bit counter overflow, memory ceiling
  Mouse (mm10): check chromosome name normalization, different GTF structure
```

**O(minutes) debug loop**:
```bash
# On already-downloaded sample (existing BAM on disk):
/usr/bin/time -v singlet-pileup [opts] sample.bam 2>&1 | tee debug.log
python validate_full.py output/ starsolo/ 2>&1 | tee validation.log

# Diff comparison vs. reference:
python scripts/gene_diff.py output/ > gene_differences.txt
head -30 gene_differences.txt  # see which genes diverge
```

### 6.4 Performance Profiling (`scripts/profile_pileup.sh`)

Existing script already covers 5 test scenarios (SNP-only, SNP+exon, 8-thread, streaming,
perf stat). Add after each new feature:

```bash
# Regression check: ensure each new feature adds ≤20% wall time
# Baseline (SNP+exon, 8 threads, GSM8313394): record in DEVLOG
Time_baseline=XX.Xs

# After TE counting addition:
Time_te=XX.Xs
assert Time_te < Time_baseline * 1.2  # ≤20% overhead for TE layer

# After inline demux addition:
Time_demux=XX.Xs  # should add ~30s (VB inference is post-streaming)
```

---

## 7. Implementation Order and Priority

```
Priority 1 (unblocks production):
  1.1 pz_writer.h — native .1pz output              (~400 LOC, 3-5 days)
  1.3 CB_samTagOut streaming mode                    (~20 LOC script change, 1 day)

Priority 2 (completes Phase 1 spec):
  1.2 TE augmented GTF + 1/NH fractional counting    (~200 LOC, 2-3 days)
  1.4 100-sample validation batch                    (~2 days setup, runs overnight)

Priority 3 (Phase 2 — high value, medium effort):
  2.x donor_demux.h (Vireo replacement)              (~600 LOC, 5-7 days)
  3.x mt_heteroplasmy extension                      (~400 LOC, 3-4 days)
  
Priority 4 (integration + cleanup):
  4.1 --pipeline mode + run_pileup_v2.sh            (~100 LOC, 1-2 days)
  Remove mtx_writer.h once .1pz default             (delete code)
```

**Total effort estimate** (Phase 1 + 2 completion):
- Coding: ~10-15 days of focused C++ development
- Testing and validation: ~3-5 days (primarily wait time for batch jobs)
- Net timeline: 3-4 weeks from now to production-ready binary

---

## 8. Open Questions and Design Decisions

| Question | Options | Recommendation |
|---|---|---|
| chrM BAM: keep or eliminate? | (A) Eliminate, use inline mt pileup only; (B) Keep as optional debug output | Keep with `--out-chrm` flag for mgatk comparison phase; eliminate after mt validated |
| TE float accumulator: float vs fixed-point uint16? | (A) `SparseAccumulator<float>` 4× memory; (B) scaled uint16 | uint16 scaled (B): precision <1% error acceptable for subfamily-level quantification |
| Vireo n_donors auto-detect ceiling | max_donors = 4, 6, or 8? | 8 (handles most GEO pooled experiments; beyond 8 is rare and computation increases linearly) |
| Donor VCF generation: in singlet-pileup or external? | (A) bcftools call from C++; (B) external script | External: keeps binary focused; bcftools is already in pipeline toolchain |
| 1MM UMI correction | Phase 2 or never? | Phase 2 — exact dedup captures 96-98% of molecules; add when gene-level correlation plateaus |

---

## 9. Success Criteria Summary

| Milestone | Metric | Target |
|---|---|---|
| Phase 1: .1pz output | Write speed | >500 MB/s, <5s total per sample |
| Phase 1: .1pz output | Round-trip fidelity | Bit-exact vs. Python MTX→.1pz conversion |
| Phase 1: TE quantification | TE subfamily r vs. TEtranscripts | r > 0.9 |
| Phase 1: 100-sample validation | Median gene r | >0.999 |
| Phase 1: 100-sample validation | P5 gene r | >0.995 |
| Phase 2: Donor demux | NMI vs. vireoSNP | >0.99 |
| Phase 2: Donor demux | Runtime (5K cells, 1 donor) | <30s vs. vireoSNP ~2 min |
| Phase 2: mt heteroplasmy | VAF r vs. mgatk | >0.99 |
| Phase 2: mt heteroplasmy | Variable positions (Jaccard) | >0.90 |
| Phase 2: mt heteroplasmy | Runtime vs. mgatk | 10× faster (seconds vs. minutes) |
| Overall pipeline | Total wall clock (50M read sample) | <45 min (STAR-dominated) |
| Overall pipeline | Memory peak | <34 GB (32 GB STAR + <2 GB pileup) |

---

## 10. New Feature Roadmap (April 2026+)

### 10.1 Automatic Species Detection (M5)

**Goal**: Download any SRA accession and automatically identify the host species (and non-host contamination) without user input. Support 200+ commonly used GEO species.

**Approach**:
- For each supported species, curate 50–100 transcripts that are (a) highly and consistently expressed across tissue types, and (b) have low cross-species homology
- Build a compact species marker index: Bloom filter or minimizer-based (≤10 MB total for all species)
- Streaming classifier: hash the first ~100K reads against all species markers, rank by hit count
- Report: primary host species (confidence %), secondary species (contamination %), recommended genome index
- After host detection, auto-select the correct STAR genome index and map; optionally map unmapped reads against candidate non-host genomes

**Implementation**: New header `include/lib1fq/species_detect.h` + species marker database in `data/species_markers/`

### 10.2 Automatic Protocol Detection (M6)

**Goal**: The user should never need to specify protocol, chemistry, modality, or barcode structure. `singlify SRR12345678` should auto-detect everything.

**Approach**:
- Examine the first ~10K reads to identify barcode/UMI/cDNA segment structure
- Build an expandable **protocol signature table** in `protocol.h` encoding: R1/R2/I1/I2 lengths, barcode position + length, UMI position + length, whitelist file, linker sequences, chemistry-specific markers, **adapter sequences for trimming**
- The protocol table must include adapter fields: `adapter_type` (poly-A, TSO, Nextera ME, Illumina universal, poly-G), `adapter_sequences`, `trim_position` so that adapter trimming is protocol-auto-selected
- Key distinguishing features: R1 length (26bp=10xv2, 28bp=10xv3, 16bp+10bp=inDrop), barcode whitelist matching, linker sequence detection, poly-T detection for 3' vs 5'
- Support: 10x v2/v3/v4/5'/3', Drop-seq (variable BC), inDrop (split BC with linker), sci-RNA-seq3 (combinatorial), BD Rhapsody, Parse Biosciences, CITE-seq (ADT reads), 10x multiome (GEX vs ATAC), Visium (spatial BC), Smart-seq2/3 (no BC), V(D)J (immune repertoire), CRISPR guide capture, bulk RNA-seq
- For development: use SOFT metadata as a hint. For production: work with raw SRA bytes only

### 10.3 UMI Error Correction (M12)

**Goal**: Correct UMI sequencing errors using directional 1-Hamming-distance merging.

**Algorithm** (same as UMI-tools 'directional'):
1. For each (barcode, gene) group, collect all UMI sequences and their read counts
2. Build a directed graph: edge from UMI_A → UMI_B if Hamming(A,B) = 1 AND count(A) ≥ 2×count(B) − 1
3. Identify connected components; collapse each component to its highest-count representative
4. The merged count = sum of all UMIs in the component

**Implementation**: ~200 LOC in `include/singlet-pileup/umi_dedup.h`. Operates during the existing dedup pass — minimal additional overhead. Build adjacency graph per-group using XOR-based Hamming distance check (fast for 12-bp UMIs = 24-bit representation).

**Expected impact**: 2–5% reduction in overcounted unique molecules, particularly for deeply sequenced samples. Brings singlify into parity with CellRanger and UMI-tools on counting accuracy.

**Success criteria**: ≥99% concordance with UMI-tools 'directional' on per-gene UMI counts. <5% additional wall-clock overhead.

### 10.4 Sequencing Saturation & Complexity Estimation (M7)

**Goal**: Report sequencing depth adequacy and library complexity.

- **Per-cell saturation**: Track total vs unique (barcode, gene, UMI) tuples during dedup. Saturation = 1 − (unique_molecules / total_reads). A saturation of 0.9 means 90% of reads are duplicates.
- **Aggregate saturation**: Pooled across all cells.
- **Complexity estimation**: Subsampling extrapolation (Lander-Waterman model or preseq-style). Estimate how many additional unique molecules would be captured at 2×, 5×, 10× current depth.
- **Output**: Per-cell saturation in QC metrics JSON + aggregate saturation curve data for plotting.
- **Implementation**: ~100 LOC. Counters added to the existing UMI dedup loop (near-zero overhead).

### 10.5 Model-Based Cell Calling (EmptyDrops++) (M7)

**Goal**: Accurately distinguish real cells from empty droplets using a generative model.

**Approach**: Go beyond simple knee-point heuristics. Fit a model that captures:
1. **Ambient RNA profile**: Gene expression distribution in empty droplets (bottom 1% of barcodes). This is the null model.
2. **Cell profile**: Diverse gene expression deviating from ambient. Each real cell's expression is drawn from a different distribution than the ambient.
3. **Statistical test**: For each barcode, compute P(data | ambient) using a Dirichlet-Multinomial model. Barcodes with low P are likely real cells.
4. **Posterior probabilities**: Output per-barcode probability of being a real cell, not just binary calls. Users choose their FDR threshold.

**Key research questions** (resolve before implementing):
- Is Dirichlet-Multinomial sufficient, or should we use a mixture model (ambient + K cell-type components)?
- How to handle barcodes near the decision boundary (damaged cells, doublets, cell fragments)?
- Can we use mitochondrial fraction, gene diversity, and SNP heterogeneity as additional features?
- How to define success rigorously?

**Success criteria**:
- ≥95% concordance with CellRanger EmptyDrops on true-positive cells
- Recover ≥5% more real cells that CellRanger misses (validated on datasets with cell-hashing ground truth, e.g., MULTI-seq or 10x cell multiplexing)
- FPR ≤1% (false-positive empty droplets called as cells)
- Processing time: <10s for 1M barcodes

### 10.6 Per-Cell QC Metrics (M7)

**Goal**: Output standard field QC metrics for every cell.

| Metric | Computation | Standard threshold |
|--------|-------------|-------------------|
| MT fraction | chrM reads / total reads per cell | >20% → flag |
| Ribosomal RNA fraction | RPL*/RPS*/Mt-Rnr* gene UMIs / total UMIs | >50% → flag |
| Intronic read fraction | intron hits / (exon + intron) hits | Informative (higher = nuclear enrichment) |
| Gene count | Number of genes with ≥1 UMI | <200 → flag |
| UMI count (nCount_RNA) | Total UMI molecules per cell | <500 → flag |
| Complexity ratio | genes / UMIs | <0.1 → flag (low diversity) |
| Mapping rate | mapped reads / total reads per cell | <30% → flag |
| Saturation | 1 − (unique / total) per cell | Informative |

**Output**: Include in pipeline `qc_metrics.json` (aggregate) and as a per-cell QC layer in .1pz metadata. These metrics are exactly what Scanpy/Seurat workflows compute in their `pp.calculate_qc_metrics()` step.

### 10.7 Pipeline Provenance Manifest (M13)

**Goal**: Full machine-readable provenance for every pipeline run.

**Output file**: `pipeline_manifest.json` alongside .1pz output files.

**Contents**:
```json
{
  "singlify_version": "0.5.0",
  "singlify_git_hash": "abc1234",
  "timestamp_start": "2026-04-11T10:00:00Z",
  "timestamp_end": "2026-04-11T10:02:15Z",
  "input": {
    "accession": "SRR32855204",
    "1fq_path": "/path/to/SRR32855204.1fq",
    "1fq_crc32": "0xDEADBEEF",
    "total_reads": 40400000
  },
  "auto_detected": {
    "protocol": {"tag": "10x-arc-gex", "confidence": "HIGH", "wl_match_rate": 0.89},
    "species": {"key": "human_GRCh38", "confidence": 0.997, "contamination": {"mouse": 0.021}},
    "instrument": {"model": "NovaSeq 6000", "confidence": 0.85},
    "modality": "scRNA",
    "strand": "forward"
  },
  "references": {
    "genome_dir": "/ref/GRCh38-2024-A/star_2.7.11b",
    "gtf": "/ref/GRCh38-2024-A/genes/genes.gtf",
    "snp_vcf": "/ref/GRCh38-2024-A/snps/common_all.vcf",
    "whitelist": "3M-february-2018.txt"
  },
  "parameters": {
    "threads": 20,
    "min_mapq": 20,
    "umi_correction": "directional_1hamming",
    "cell_calling": "emptydrops_pp"
  },
  "qc_summary": {
    "total_reads": 40400000,
    "mapped_reads": 34905600,
    "mapping_rate": 0.864,
    "cells_called": 12089,
    "median_genes_per_cell": 1847,
    "median_umis_per_cell": 4923,
    "sequencing_saturation": 0.72,
    "median_mt_fraction": 0.042
  },
  "outputs": [
    {"file": "exon_counts.1pz", "md5": "..."},
    {"file": "intron_counts.1pz", "md5": "..."},
    {"file": "sj_counts.1pz", "md5": "..."}
  ]
}
```

Also embed a summary of this manifest in each .1pz metadata section (so .1pz files are self-documenting).

### 10.8 CellRanger-Inspired Preprocessing (M7)

**Goal**: Incorporate CellRanger capabilities that provide cost-effective preprocessing and quality control.

**Features**:
- **Ambient RNA background removal**: Lightweight SoupX/CellBender-like correction in C++, applied to count matrices at export time
- **Doublet detection**: C++ implementation inspired by Scrublet/scDblFinder. Signals: simulated doublets from observed data, mitochondrial DNA fraction as hint, observed-to-expected UMI ratio, genotype-based detection (when SNP pileup data available)
- **Sequence-informed QC**: Pre-mapping signals (barcode quality, poly-A content, adapter contamination) to flag low-quality cells
- **CellRanger feature parity audit**: Systematic evaluation of all Cell Ranger features vs singlify, identify low-cost preprocessing wins

### 10.9 Adapter Trimming Completeness (M1)

**Goal**: Protocol-specific adapter trimming without manual flags.

Add adapter sequences as a key field in the protocol signature table:

| Protocol family | Adapter type | Sequence/pattern | Trim position |
|----------------|-------------|-----------------|---------------|
| 10x 3' (v2/v3/v4) | poly-A tail | AAAAAAA… | R2 3' end |
| 10x 5' (v2/v3/v4) | TSO (template switching oligo) | AAGCAGTGGTATCAACGCAGAGTAC | R2 5' end |
| 10x scATAC | Nextera ME | CTGTCTCTTATACACATCT | R1/R2 3' end |
| Smart-seq2/3 | Nextera ME + poly-A | CTGTCTCTTATACACATCT + AAAAAAA… | R1/R2 3' end |
| Drop-seq | poly-A tail | AAAAAAA… | R2 3' end |
| NovaSeq (all protocols) | poly-G artifact | GGGGGGG… | R2 3' end |
| Illumina universal | Illumina adapter | AGATCGGAAGAGC | R1/R2 3' end |

Protocol auto-detection → adapter auto-selection. No `--clip` or `--trim` flags needed. The adapter field is part of the protocol signature table in `protocol.h`.

### 10.10 Ancestry, Sex, Karyotype, and ASE (M8)

**Goal**: Extract donor-level biological annotations from pileup data.

- **Ancestry**: Classify using 1000 Genomes Project / gnomAD ancestry-informative markers (~2000 AIMs for 5-superpopulation classification). Score from pileup AD/DP calls. Bundled AIMs VCF per species in reference index package.
- **Sex**: chrY read fraction, XIST expression level, X:autosome coverage ratio
- **Karyotype**: Per-chromosome read depth normalization to detect copy number changes — trisomy (chr13, 18, 21), sex aneuploidies (XXX, XXY, XYY, XO), large-scale deletions/duplications
- **Allele-specific expression (ASE)**: With phased VCF input (bundled 1000G phased panel per species), aggregate allelic depth (AD) across all heterozygous SNPs per gene per cell → per-gene ASE ratio (reference allele fraction). Also compute per-cell het SNP count as a donor fingerprint metric. ~300 LOC, essentially free given existing per-SNP AD/DP pileup.
- All annotations reported per-donor when donor demultiplexing is active

### 10.11 Broad Modality Support (M9)

**Goal**: Support all major single-cell and spatial genomics assay types.

| Modality | Key implementation | Status |
|----------|-------------------|--------|
| scRNA droplet (10x, Drop-seq, inDrop, sci-RNA-seq3, BD, Parse) | Existing pipeline | Partial |
| scRNA plate-based (Smart-seq2, Smart-seq3) | No-barcode mode, full gene body coverage, robust multi-junction gene counting | Not started |
| scATAC-seq | Fragment file generation, peak calling | Not started |
| 10x Multiome (GEX+ATAC) | Joint processing, matched barcodes, 3-stream .1fq | Partial (.1fq only) |
| CITE-seq | ADT/HTO feature barcode counting | Not started |
| Visium | Spatial barcode mapping + gene counts + coordinate output | Not started |
| Bulk RNA-seq | No-barcode mode, standard gene counting | Not started |
| V(D)J / Immune repertoire | Contig assembly, IMGT alignment, CDR3 extraction, clonotype clustering | Not started |
| CRISPR guide capture | Auto-detect, count guides per cell, guide_counts.1pz | Not started |
| Long-read (ONT/PacBio, MAS-seq) | Future — different aligner and error model required | Deferred |

### 10.12 V(D)J / Immune Repertoire Support (M9)

**Goal**: Full pipeline support for TCR/BCR from 10x 5' V(D)J chemistry.

**Pipeline stages**:
1. **Auto-detection**: V(D)J libraries detected from protocol signature (distinct from GEX — typically shorter R2, enriched V/J gene segments)
2. **.1fq encoding**: Barcode + UMI extraction same as GEX; R2 is V(D)J amplicon sequence
3. **Contig assembly**: Per-barcode overlap-consensus assembly from short reads. Group reads by barcode, use overlap graph to assemble full-length V(D)J contigs
4. **V/D/J gene alignment**: Align assembled contigs against IMGT reference database (bundled per species in reference index package)
5. **CDR3 extraction**: Identify CDR3 region from contig alignment — conserved Cys and Phe/Trp anchor residues
6. **Clonotype clustering**: Group cells with identical or near-identical CDR3 sequences
7. **Output**: Per-cell clonotype assignments, CDR3 sequences, V/J gene usage matrix (as .1pz), detailed per-contig annotations in manifest JSON

**Effort**: ~2000+ LOC. This is a significant feature but essential for autonomous processing of all single-cell data.

### 10.13 CRISPR Guide Capture (M9)

**Goal**: Auto-detect and count CRISPR guide capture libraries.

- Detect feature barcode libraries from read structure (short R2 matching guide reference sequences)
- Count guide reads per cell using feature barcode matching (Hamming distance ≤1)
- Output `guide_counts.1pz` with guide-by-cell matrix
- Include guide capture metrics in pipeline manifest (total guide reads, assignment rate, guides-per-cell distribution)
- Auto-detection: if a substantial fraction of R2 reads match known guide sequences from a user-provided guide library CSV (or auto-detected from Addgene/published libraries), activate guide counting mode

### 10.14 Multi-Junction Gene Counting (M3)

**Goal**: Robustly handle reads spanning multiple splice junctions for per-gene expression.

Critical for plate-based assays (Smart-seq2/3) where reads span the full gene body. Current exon/intron/SJ counting works per-feature; gene-level aggregation must handle:
- Reads spanning 2+ splice junctions that confirm a specific transcript isoform
- Reads mapping to overlapping genes at boundaries
- Ambiguous multi-gene assignments resolved by junction evidence
- Consistent counting between droplet (short reads, 1-2 exons) and plate-based (long reads, many exons) assays

**Note**: RNA velocity matrices (S/U/A) are NOT needed as a separate feature — they can be computed trivially and quickly from our existing per-exon, per-intron, and per-SJ count matrices by downstream tools.

### 10.15 Zero-Arg Automation Architecture (M11)

**Goal**: `singlify SRR12345678` — zero flags, fully autonomous.

**Components**:

1. **K-mer species classifier (Bloom filter)**: Build a species marker Bloom filter index (≤10 MB for 200+ species). Stream first 100K R2 reads, classify host in <5s. Species key → auto-select genome index, GTF, SNP VCF.

2. **Pre-built reference index registry**: JSON manifest at `~/.singlify/references.json` mapping species keys to local paths. Commands: `singlify index list/fetch/add`. Each index package includes: STAR genome index, GTF annotation, common SNP VCF (gnomAD/dbSNP), phased SNP VCF (1000G, for ASE), ancestry-informative marker VCF.

3. **Internalized whitelist resolution**: All 24 supported protocol whitelists bundled in `whitelists/` directory. Protocol detection → whitelist auto-resolved. No `--whitelist` flag needed.

4. **Auto thread detection**: `min(nproc, 20)` or `$SLURM_CPUS_PER_TASK`. ~10 LOC.

5. **Shared-memory genome index**: `singlify genome load <species_key>` → `--genomeLoad LoadAndKeep`. Saves ~40s/sample × 70K samples.

6. **Model-based cell calling**: EmptyDrops++ (see §10.5) replaces the `--barcodes` requirement.

7. **Instrument detection**: Quality score distribution + poly-G artifact detection → NovaSeq/HiSeq/NextSeq classification. Embedded in provenance manifest.

8. **Adapter auto-selection**: Protocol → adapter type from signature table (see §10.9). No manual `--clip` flags.

### 10.16 Deep Archive .1fq (M10)

**Goal**: After all downstream analysis is complete (donor demux, mtDNA variants, expressed genome variants, mapping rate preserved), strip quality scores from .1fq for long-term storage.

- `singlify archive --strip-quality INPUT.1fq -o ARCHIVE.1fq` — produces a valid .1fq without quality columns
- Prerequisite gate: pipeline enforces that pileup, donor demux, and variant calling are complete before allowing quality stripping
- Expected size reduction: 15–25% (quality is ~20% of .1fq file)
- Archived .1fq retains: sequences, barcodes, UMIs, protocol metadata, segment descriptors — sufficient for re-alignment if needed
