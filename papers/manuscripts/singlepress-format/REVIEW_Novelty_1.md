# Novelty Review — SinglePress .1pz Format Paper

**Reviewer persona**: SOTA positioning and claim validity  
**Date**: 2026-04-13  
**Confidence**: 0.88 (high — claims are well-substantiated overall; issues are addressable)

---

## Summary

The manuscript presents a compelling format with strong empirical evidence supporting its core claims. The 3,253-dataset evaluation and systematic codec exploration are clearly above the bar for a bioRxiv preprint. The main novelty concerns center on: (1) incomplete characterization of BPCells' value proposition, (2) internal inconsistencies between main text and supplementary zstd level references, and (3) a protocol count mismatch.

---

## Critical Issues (0)

None. No core claims are invalid or unsupported.

---

## Major Issues (4)

### M1. BPCells' disk-backed lazy operations omitted from comparison

**Location**: Table 5 (feature comparison), Discussion "Ecosystem positioning"

BPCells' primary value proposition is not just compression — it's a disk-backed *computation engine* that supports lazy matrix operations (multiply, transpose, arithmetic, subsetting) without loading the full matrix into RAM. This is what enables "44M cells on a laptop" as reported in Parks & Greenleaf 2025. The paper frames the BPCells comparison entirely on compression ratio and read speed, where SinglePress wins — but omits BPCells' distinguishing capability that SinglePress does not replicate. This is the most significant fairness gap in the manuscript.

**Fix**: Add a "Disk-backed operations" or "Lazy evaluation" row to Table 5 (BPCells: ✓, SinglePress: ---), and add one sentence to the Discussion acknowledging this design difference.

### M2. Supplementary zstd level inconsistency (zstd-1 vs. zstd-3)

**Location**: S4 title, Algorithm 1–2, Tables 8/10/12 vs. main text + S12

The main text and S12 consistently say **zstd-3** is the production backend. S4's title says "The production codec: bit-planes + bitmap + **zstd-1**", Algorithm 1–2 specify `zstd-1(B')`, and Tables 8/10/12 label the champion as "zstd-1." This creates confusion about what the actual production codec is. A real reviewer will flag this as a credibility issue.

**Fix**: Harmonize all supplementary references to match the actual production level (zstd-3).

### M3. Protocol count mismatch: "nine protocols" → actually 10

**Location**: Abstract line 3, Methods §Dataset selection

The abstract and methods say "nine protocols" but the supplementary table explicitly enumerates 10 distinct protocols: 10x v2, v3, v4, 5', Multiome, Drop-seq, Seq-Well, DNBelab C4, BD Rhapsody, inDrop. The sum of per-protocol n's in the supplementary table is also consistent with 10 groups.

**Fix**: Change "nine protocols" → "ten protocols" in abstract and methods.

### M4. BPCells bib entry year mismatch

**Location**: refs.bib `parks2024bpcells`

The bib entry says `year={2024}` but the DOI is `10.1101/2025.03.27.645853` (March 27, 2025 bioRxiv). Should be `year={2025}`.

**Fix**: Change `year={2024}` → `year={2025}` in bib entry.

---

## Minor Issues (5)

### m1. H5AD LZF compression variant not discussed

H5AD supports LZF in addition to gzip. LZF is faster to decompress at the cost of lower compression. Mentioning this briefly (even just "H5AD with gzip, its default and highest-ratio backend") would preempt reviewer pushback.

### m2. Parquet listed in FEATURE_BRIEF.json but absent from benchmarks

Parquet is a SOTA competitor in the feature brief but not benchmarked or discussed in the paper. Either remove from the brief or add a brief mention in Discussion for why it's excluded (not designed for sparse data).

### m3. Missing "multi-layer" row in feature table

AnnData/H5AD supports multiple matrix layers (raw, normalized, etc.) in a single file. SinglePress stores exactly one count matrix. This design trade-off should appear in Table 5 as a "Multi-layer support" row.

### m4. Abstract "868 MB/s" could be clearer

The 868 MB/s claim should specify it's the median across the full benchmark. A reader may assume it's a peak or worst-case number.

### m5. Feature table footnote $^a$ undersells BPCells

Footnote says "BP-128 bitpacking uses delta encoding on sorted indices, but does not exploit value frequency distribution." BPCells actually does bitpack the values too, not just the indices. The claim is correct that it doesn't exploit *value frequency*, but a hostile reviewer might read this as a straw-man.

---

## Feature Comparison Table Accuracy (Table 5)

| Row | Assessment |
|-----|-----------|
| Sparse-native storage | ✓ Accurate |
| Domain-aware compression | ✓ Accurate; BPCells "partial" is fair |
| Embedded gene/cell names | ✓ BPCells "sidecar" is accurate |
| Embedded obs/var DataFrames | ✓ Accurate |
| Column-range random access | ✓ H5AD "partial" is fair (backed mode) |
| Row-range random access | ✓ Accurate |
| Stored column sums | ✓ Unique to SinglePress |
| On-the-fly normalization | ✓ Unique to SinglePress |
| CRC32 integrity | ✓ Accurate |
| Single-file | ✓ Accurate |
| Python API | ✓ BPCells "---" is correct |
| R API | ✓ Accurate |
| PyTorch DataLoader | ✓ TileDB-SOMA "partial" is fair |
| **Missing: Disk-backed lazy ops** | **BPCells ✓, others ---** |
| **Missing: Multi-layer support** | **H5AD ✓, .1pz ---** |

---

## Claim Verification Summary

| Claim | Source | Verdict |
|-------|--------|---------|
| 9.5× median compression | Table 1, 3198 datasets | ✓ Backed |
| 868 MB/s decode | Table 1, 200-dataset sample | ✓ Backed |
| 2–4× smaller than H5AD | Table 1: H5AD 2.7× vs .1pz 9.5× → ~3.5× | ✓ Backed |
| 2.5× smaller than BPCells | Table 2: BPCells 3.8× vs .1pz 9.5× → 2.5× | ✓ Backed |
| Column-range reads 2–20× faster | §Column subsetting: median 8×, range 2–20× | ✓ Backed |
| GPU utilization 2.5–3× higher | §GPU training: 43% vs 15% = 2.87× | ✓ Backed |
| scATAC 4.7× compression | §scATAC: median 4.7× | ✓ Backed |
| Write 6.4× faster than H5AD | §Write speed: median 441 vs 63 MB/s = 7.0× | ⚠ Slightly undersold (7.0×, not 6.4×) |
| Near Shannon entropy (0.87×) | §Frontier: median 0.87× | ✓ Backed |

---

## Fixes Applied

1. **M3**: Changed "nine protocols" → "ten protocols" in abstract and methods
2. **M4**: Fixed BPCells bib year 2024 → 2025
3. **M1**: Added "Disk-backed lazy operations" row to feature comparison table; added sentence to Discussion
4. **M2**: Harmonized S4 zstd-1 → zstd-3 throughout supplementary

---

## Verdict

**PASS with Major revisions applied.** The core claims are valid and well-supported by data. The BPCells comparison is the paper's biggest vulnerability — the addition of the disk-backed operations acknowledgment substantially strengthens the paper's fairness. The zstd level inconsistency would be caught by any careful reviewer and damages credibility if left unfixed.
