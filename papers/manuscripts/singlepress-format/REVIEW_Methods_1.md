# REVIEW — Methods — Pass 1

## Confidence: 0.85

## Critical Issues (must fix)

1. **Peak memory not reported anywhere** (Methods §Benchmark hardware). No peak RSS is reported for any benchmark — not for read, write, or GPU training. Peak memory is a standard requirement for format papers (BPCells reports it; SOTA_TABLE.md lists it as an acceptance metric with ≤2 GB target). Reviewers will request this. Need to add at minimum: peak RSS for full-file read at several dataset sizes, and ideally for write and column-range read.

2. **Hardware specification too vague** (Methods §Benchmark hardware, line ~785). "AMD EPYC, 128 GB RAM" is insufficient — EPYC spans 4 generations with 2-4× performance variation. Need: exact model (e.g., EPYC 7763), core count, clock speed, L3 cache size, memory configuration (channels, bandwidth). "Shared parallel filesystem" needs a type (Lustre, GPFS, BeeGFS) and approximate bandwidth, since I/O benchmarks depend critically on this. The SLURM outputs show benchmarks ran on multiple distinct nodes (g003, g008, g051) — if different CPU benchmarks ran on different nodes, this must be disclosed and controlled for.

3. **Methods claims "median of 5 replicates" but benchmark code uses 3 trials** (Methods §Benchmark hardware, line ~786). The `benchmarks_v3.py` script sets `N_TRIALS = 3` with `WARMUP = 1`. The Methods text says "median of 5 replicates with a warm-up run discarded." This is a factual discrepancy that must be corrected to match the actual code. The threading benchmark correctly uses 100 reps, which exceeds the claim. Fix the Methods to state the actual trial counts.

## Major Issues (should fix)

1. **Ablation study claims "seven datasets" but table shows six** (Results §Read performance, ~line 272 vs Table 3). Text: "An ablation study across seven datasets (4.6M–113M nonzeros)" — Table 3 (tab:ablation) has 6 data rows. Either a dataset was dropped from the table or the text count is wrong. Fix to match.

2. **GPU hidden-dimension range inconsistent between Results and Methods** (Results ~line 350 vs Methods ~line 805). Results: "h = 32 to h = 2,048, 8 GFLOP–0.5 TFLOP". Methods: "h = 32 to h = 4,096, 8 GFLOP–1 TFLOP". These must agree.

3. **Table 2 compares medians across unmatched dataset pools** (Results §Format comparison, Table 2). `.1pz` median is from n=3,198 datasets while other formats use n=89 or n=20. Comparing medians across different-sized, potentially non-overlapping samples can introduce selection bias. The paper should explicitly note that the `.1pz` median spans a broader dataset pool, or report the matched-subset medians (n=89 or n=20 shared datasets) alongside the full-pool median.

4. **Software versions incomplete** (Methods, Declarations §Availability). Missing: h5py version, scipy version, anndata/scanpy version, BPCells version (and R version used for BPCells benchmarks), zstd library version, compiler (GCC/Clang version), OpenMP version, simpleaf version for quantification. Pin all software versions used.

5. **Cache state not specified for main benchmarks** (Methods §Benchmark hardware). The zstd-level sweep explicitly uses tmpfs to eliminate I/O variability, but the main 200-dataset decode throughput and 25-dataset cross-format benchmarks don't state whether files were in page cache (warm) or read from disk (cold). On a shared filesystem, this can cause 2-10× variation. State the cache policy.

6. **scATAC generalization claim based on 3/4 synthetic datasets** (Results §scATAC, ~line 421). Three of four scATAC datasets are synthetic. The text claims VOCSC "generalizes beyond scRNA-seq" but the evidence is thin. Acknowledge the limitation more explicitly or add more real scATAC datasets.

7. **BPCells decode throughput omitted from Table 2** (Table 2, line ~196). BPCells row shows "---" for decode throughput. If BPCells R read speed was measured (103 MB/s per §BPCells head-to-head), it should appear in the table. Omitting it while reporting other formats' speeds creates an asymmetry.

## Minor Issues (nice to fix)

1. **Census comparison acknowledged as asymmetric but still prominent** (Results §Census, ~line 470). Comparing local .1pz reads (16ms) to TileDB-SOMA remote API (29s) mixes format efficiency with network latency. The text acknowledges this but still calls it "three orders of magnitude faster." Consider reframing as "three orders of magnitude lower latency for local access" to avoid implying a format-level comparison.

2. **scATAC "4.7× median" from 4 datasets** (Results §scATAC, ~line 422). A median of 4 data points has limited statistical meaning. Consider reporting mean ± range instead, or noting the small sample size.

3. **No confidence intervals or error bars on speedup claims**. The "3.1× faster" (vs H5AD), "2.5× smaller" (vs BPCells), and "6.4× faster write" claims have no uncertainty bounds. With 3 trials per measurement, even basic IQR ranges would strengthen credibility.

4. **"report median wall-clock time over 5 trials with warmup"** in Results (line ~281) contradicts the 3-trial code. Fix consistently with Critical Issue #3.

5. **Operating system not specified** (Methods). Linux distribution and kernel version can affect I/O scheduling and filesystem performance.

## Strengths

1. **Exceptional breadth**: 3,253 datasets across 9 species and 9 protocols is far beyond typical format paper benchmarks (usually 3-5 datasets). The cross-species and cross-protocol analysis convincingly demonstrates generalizability.

2. **Compression frontier analysis is rigorous**: Shannon entropy bounds, zstd level sweeps, 10 alternative codecs, 70+ codec variants, and honest permutation overhead accounting in sorting experiments. The supplementary codec evaluation (Sections S1-S12) is unusually thorough.

3. **Fair BPCells comparison**: The paper correctly identifies that BPCells' reported 7.4-8.6 GB/s is in-memory unpacking speed (not end-to-end) and provides the apples-to-apples wall-clock comparison. It also fairly notes that BPCells' R reader is marginally faster per-byte.

4. **Ablation study decomposes speed advantage cleanly**: Table 3 separates codec speedup from threading contribution, avoiding conflation of orthogonal effects.

5. **Limitations section is honest**: Acknowledges integer-only storage, single-matrix-per-file limitation, and GPU benchmark simplicity.

6. **Reproducibility of compression claims**: Since compression ratios are deterministic (same input → same output), the single-measurement approach is valid for size comparisons. The R² = 0.990 prediction model provides an independent cross-check.
