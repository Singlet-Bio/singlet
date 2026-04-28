# CITE-seq / ADT Processing Architecture

> Design spec for T1–T4. Written 2026-04-12.

---

## 1. Pipeline Flow

```
Input: .1fq bundle (n_streams=3, assay_type=CITE_SEQ_GEX)
           │
           ├─ Stream 0 (R1, StreamRole::R1)     → barcode + UMI (16+12 bp, 10xv3)
           ├─ Stream 1 (R2, StreamRole::R2)      → cDNA insert   → STAR → pileup engine
           └─ Stream 2 (R2, StreamRole::FEATURE) → ADT/HTO tag   → adt_matcher.h → adt_counter.h

                        ┌─────────────────────────────────────┐
                        │   singlify main loop (parallel)      │
                        │                                      │
    GEX path:           │   STAR align R2, pileup per cell     │  → exon_counts_matrix.1pz
    ADT path:           │   match R2 vs TagRef, UMI dedup      │  → adt_counts_matrix.1pz
    HTO (optional):     │   hto_demux() on ADT HTO subset      │  → cell_assignments.tsv
                        └─────────────────────────────────────┘
```

### Processing Order

1. **Decode phase**: lib1fq reader emits three streams per read-group: GEX (R1+R2) and ADT (R1+ADT_R2). Both share the same cell barcode (R1).
2. **Alignment phase**: GEX R2 fed to STAR. ADT stream bypasses STAR entirely — no alignment needed; tag matching is string-based.
3. **Pileup / counting phase**: GEX pileup runs as today. ADT counting runs in parallel in a separate worker; the barcode index the two paths share is identical (same whitelist).
4. **Export phase**: Both matrices exported to `{out_prefix}/adt_counts_matrix.1pz` alongside the existing GEX matrix.

---

## 2. Design Question Answers

### Q1 — Read Layout in SRA

CITE-seq data is deposited in SRA as **two separate accessions**:
- GEX accession: standard 10xv3 layout (R1=BC+UMI 28bp, R2=cDNA 90bp)
- ADT accession: same R1 layout, R2 = antibody tag read (usually 28–60bp)

Both accessions share the same cell barcodes, so during `singlify download`, the user provides both SRR IDs. Encoding stores them as a **single .1fq bundle** with `n_streams=3` (R1, GEX_R2, ADT_R2) and `assay_type=CITE_SEQ_GEX`. The `stream_roles[2]` field is set to `StreamRole::FEATURE`.

**Decision: single .1fq, 3-stream bundle** — consistent with how Multiome stores GEX+ATAC (`n_streams=4`). Dual-.1fq was rejected: requires two open file handles, breaks single-file provenance, and the `stream_roles` field already supports this.

### Q2 — Tag Matching Algorithm

Modeled on `crispr_guide.h` (N18). Key differences:
- 15bp TotalSeq-A/B/C barcodes (fixed length; no length variability in common kits)
- **Hamming-1 tolerance** on the 15bp barcode region (more permissive than exact-match for guides)
- Read structure: `R2 = [ADT_barcode:15bp][polyA_tail or constant_region]`
  - Scan first 20bp of R2 for the tag barcode (offset ≤ 5 to handle soft-clips)

Algorithm (in `adt_matcher.h`):
```
For each ADT R2 read:
  1. For offset = 0..5:
       candidate = R2[offset..offset+15]
       id = tag_ref_.lookup(candidate)       // hash exact match
       if id >= 0: return {id, offset}
       id = tag_ref_.lookup_hamming1(candidate)  // pre-expanded hash
       if id >= 0: return {id, offset}
  2. Return no-match
```

**Pre-expanded hash**: at TagRef load time, generate all 45 Hamming-1 neighbors of each 15bp barcode and insert them into a secondary `unordered_map<string,int>`. One-time cost at startup, ~200 tags × 45 × 16B = ~144KB. O(1) per read.

### Q3 — UMI Deduplication

Reuse `DirectionalUmiStore` from `umi_dedup.h` unchanged. The cell barcode index is the same integer index as the GEX whitelist. Per-cell × per-tag UMI dedup:
- Key: `(cell_idx, tag_id)` → `DirectionalUmiStore`
- UMI length: same as GEX (12bp for 10xv3, 10bp for 10xv2)

### Q4 — Output Format

- `adt_counts_matrix.1pz`: tags × cells sparse matrix (CSC, matching GEX barcode order)
- `adt_features.tsv`: tag names (one per row, matches matrix row order)
- `adt_barcodes.tsv`: symlink or copy of GEX barcodes (same order guaranteed)
- Small matrix: 50–500 tags × 1K–100K cells. Total size < 1MB. No chunking needed.
- Reuse `SparseAccumulator<uint32_t>` + `pz_writer.h` export path (same as ATAC/GEX).

### Q5 — HTO Demultiplexing (T3, optional)

Two algorithms implemented in `hto_demux.h`:
1. **CLR + k-medoids** (HTODemux-style, default):
   - CLR-normalize per cell: `x_i = log(x_i / geometric_mean(x))`
   - k-medoids clustering with k = n_HTO + 1 (doublet cluster)
   - Assignment: argmax CLR value, singlet if confident gap
2. **Negative-binomial mixture** (demuxEM-style, `--hto-method nbinom`):
   - EM with two NB components per HTO (background vs signal)
   - More robust for very unequal HTO loading

Output: `hto_assignments.tsv` columns: `barcode, assignment, confidence, method`
- assignment ∈ {`sample_N`, `doublet`, `negative`}

### Q6 — Pipeline Integration / Mode Detection

**Auto-detection**: Protocol detector already sets `AssayType::CITE_SEQ_ADT` and `CITE_SEQ_GEX`. When `singlify` opens a .1fq and finds `n_streams=3` with `stream_roles[2]=FEATURE`, it auto-enters CITE-seq mode. No `--cite-seq` flag needed.

**Tag reference CSV**: Required arg `--tag-ref tags.csv` (path to feature barcoding library CSV, same format as CellRanger Feature Reference: `id,name,read,pattern,sequence,feature_type`). If absent and stream 2 is FEATURE, singlify exits with a helpful error.

**CLI flags**:
```
--tag-ref <path>          Feature barcode CSV (required for CITE-seq .1fq)
--hto-demux               Enable HTO demultiplexing (auto if HTO features present)
--hto-method clr|nbinom   Demux algorithm (default: clr)
--adt-hamming <0|1>       Barcode mismatch tolerance (default: 1)
```

---

## 3. New Headers Needed

| Header | LOC est. | Depends on | Description |
|--------|----------|------------|-------------|
| `include/singlet-pileup/adt_matcher.h` | ~150 | none | TagRef CSV loader + Hamming-1 hash lookup |
| `include/singlet-pileup/adt_counter.h` | ~120 | `adt_matcher.h`, `umi_dedup.h`, `sparse_accumulator.h` | Per-cell×per-tag UMI dedup + SparseAccumulator |
| `include/singlet-pileup/hto_demux.h`   | ~250 | `adt_counter.h` | CLR+k-medoids + NB mixture demultiplexing |

**Total new code**: ~520 LOC (headers only, no CMakeLists.txt changes).

**Integration shim in singlify.cpp**: ~80 LOC additional (stream dispatch, `--tag-ref` arg parse, ADT export call). Document in `INTEGRATION_NOTES.md`.

---

## 4. Integration with Existing Pipeline

```
singlify.cpp integration points:
  1. Arg parsing (line ~1280): add --tag-ref, --hto-demux, --hto-method, --adt-hamming
  2. .1fq open (line ~428): detect n_streams==3 && stream_roles[2]==FEATURE → set cite_seq_mode=true
  3. Read dispatch loop: when cite_seq_mode, route stream[2] reads to Adt_counter::count()
     instead of STAR. GEX path unchanged.
  4. Post-alignment (after pileup, line ~3008): call adt_counter.export_matrix(out_prefix)
  5. Optionally: call hto_demux() if --hto-demux or all tags are HTO type
```

**Parallel pileup compatibility**: ADT counting is independent of STAR/pileup, so it runs in the same parallel worker threads that decode reads. Each worker holds a thread-local `AgtCounter` shard; merge at finalize (same pattern as parallel pileup cross-worker merge, commit `3141e4a`).

---

## 5. Implementation Order

```
T1 → T2 → T3 → T4
```

| Task | What | Acceptance Criteria |
|------|------|---------------------|
| T1: `adt_matcher.h` | TagRef CSV load + Hamming-1 lookup | Unit test: 200 tags, 1M reads, match rate ≥99% on synthetic data |
| T2: `adt_counter.h` | UMI dedup + SparseAccumulator export | Unit test: known UMI collisions deduped correctly; 1pz round-trip |
| T3: `hto_demux.h` | CLR+k-medoids + NB mixture | Unit test: 4-sample HTO sim → ≥95% assignment accuracy |
| T4: E2E wire-up | singlify.cpp integration + CLI + .1fq 3-stream encode | Validate vs CITE-seq-Count on ≥3 real datasets |

---

## 6. Validation Strategy

### Gold Standard: CITE-seq-Count v1.4 (primary)

```bash
CITE-seq-Count -R1 $ADT_R1 -R2 $ADT_R2 \
  -t tags.csv -cbf 1 -cbl 16 -umif 17 -umil 28 \
  -cells $N_CELLS -o $OUTDIR
```

Compare per-cell × per-tag count matrix (Pearson r, ≥3 datasets).

### Datasets for Validation

| ID | Description | SRR_GEX | SRR_ADT | Tags |
|----|-------------|---------|---------|------|
| CA1 | 10x PBMC (8k) with TotalSeq-A | SRR8315305 | SRR8315306 | 13 ADT |
| CA2 | 10x PBMC with HTO (4-plex) | public Seurat demo | — | 4 HTO |
| CA3 | 10x Multiome + CITE-seq | to be sourced | — | ~30 ADT |

### Acceptance Thresholds

- T1 tag matching: per-read concordance ≥ 99.0% vs CITE-seq-Count
- T2 count matrix: Pearson r ≥ 0.998 per tag (averaged across cells)
- T3 HTO demux: singlet assignment concordance ≥ 95% vs Seurat HTODemux
- T4 E2E: all above on ≥3 datasets, wall overhead vs GEX-only ≤ 5%

---

## 7. Single vs Dual .1fq — Decision Record

**Decision: single .1fq with `n_streams=3`**

Rationale:
- `.1fq` `Header.stream_roles[4]` already reserves bytes 58–61 for up to 4 stream roles; `StreamRole::FEATURE = 6` already defined in `types.h`.
- `Header.n_streams` field already used for Multiome (GEX+ATAC = 4 streams). Three-stream CITE-seq is a natural extension.
- Single file preserves provenance (one manifest entry, one file hash).
- `singlify download` already accepts multiple SRR IDs; the encoder interleaves matched read-pairs by read name and packs into one bundle.
- Dual-.1fq alternative would require: two open file handles, read-name synchronization at decode time, two separate provenance manifests — net complexity increase with no benefit.

**Rejected alternative: separate `_gex.1fq` + `_adt.1fq`**
- Would require user to pass two files on command line
- Breaks single-file streaming assumption used by `--parallel-pileup`
- More complex than 10x Multiome which already uses the multi-stream path

---

## 8. LOC Summary

| Component | LOC |
|-----------|-----|
| `adt_matcher.h` (T1) | ~150 |
| `adt_counter.h` (T2) | ~120 |
| `hto_demux.h` (T3) | ~250 |
| `singlify.cpp` shim (T4) | ~80 |
| Unit tests (3 × test_*.cpp) | ~300 |
| **Total** | **~900 LOC** |

Reused verbatim: `umi_dedup.h`, `sparse_accumulator.h`, `pz_writer.h`, `export.h` — zero modification.
