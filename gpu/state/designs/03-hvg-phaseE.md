---
feature: 3
module: preprocess/hvg.h
cycle: 59 or later (pending 57, 58)
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy pass
extends: 03-hvg.md (the original design doc)
status: draft
depends_on: 55c-fixes, 55c-novel-attempts-7-to-11
---

# Feature 3 — HVG Phase E benchmark + Gaussian-WLS LOWESS novel pursuit + autonomy

Feature 3 (`preprocess/hvg.h`) is **runtime-correct as of Cycle 55c**: all four HVG tests pass with jaccard=1.0 / spearman=1.0 / rank_rel_err ≤ 0.0015 against scanpy's SeuratV3 and PearsonResiduals flavors. This document specs the frontier push.

## 1. Success metrics (for frontier promotion)

On GSM4037629 (310,797 × 20,866, nnz=4.2M) at small-real scale:

- **Wall time (p50)**: ≤ rapids-singlecell `pp.highly_variable_genes`, OR ≤ scanpy × 20.
- **Peak device memory**: ≤ rapids-singlecell.
- **Correctness**: jaccard ≥ 0.99 on top-2000 HVG set vs scanpy, rank_rel_err ≤ 0.05 (tie-aware metric per Cycle 55c).

Promotion gate: dominance on wall OR memory, correctness match, autonomy delta ≤ 10%.

## 2. SOTA baselines

| Baseline | Variant | Notes |
|---|---|---|
| **rapids-singlecell `pp.highly_variable_genes(flavor='seurat_v3')`** | GPU cupy | Primary GPU competitor; uses cuml LOWESS |
| **scanpy `pp.highly_variable_genes(flavor='seurat_v3')`** | CPU | Uses scikit-misc LOWESS; the Cycle 55c reference |
| **scanpy `pp.highly_variable_genes(flavor='pearson_residuals')`** | CPU | Lause 2021 reference |
| **Seurat v3 `FindVariableFeatures(selection.method='vst')`** | R CPU | Source of truth for SeuratV3; gold standard users defer to |
| **scTransform v2 (`sctransform::vst`)** | R CPU | Regularized NB; the Hafemeister & Satija 2019 reference for PearsonResiduals |

## 3. Bench configurations (Cycle 59 dispatch)

Scales: tiny synthetic (500×200), small real (GSM4037629), medium concat (~100k cells).

Configurations per scale:

1. `ours_seurat_v3_manual`
2. `ours_seurat_v3_auto` (Rule 31 autotune of `n_top_genes`, `span`, `clip`)
3. `ours_pearson_residuals_manual`
4. `ours_pearson_residuals_auto`
5. **`ours_seurat_v3_gaussian_wls`** — Rule 30 novel variant (§4 below)
6. `rapids_seurat_v3`
7. `scanpy_seurat_v3`
8. `scanpy_pearson_residuals`
9. `seurat_v3_R` — R subprocess, only at small scale (likely too slow at medium)
10. `sctransform_v2_R` — R subprocess, small scale only

Metrics: wall p50 (5 iters + 2 warmup), peak dev/host mem, gene-set jaccard vs scanpy (top-2000), rank_rel_err, throughput (cells/sec).

## 4. Novel pursuit (Rule 30) — Gaussian-kernel WLS LOWESS

**From Cycle 55c novel-attempts.md**: the current cubic-WLS LOWESS (with tricube weights matching scikit-misc) is correct but potentially sub-optimal. Alternative: **Gaussian-kernel WLS in the sorted domain**.

### Hypothesis

The cubic-WLS LOWESS loop in `compute_loess_kernel` has two costs:

1. **Per-gene neighborhood scan**: O(span × n_genes) per gene × n_genes genes = O(n² × span). On 310k genes with span=0.3, this is ~29B point-pair operations.
2. **Tricube weight evaluation** + **3×3 normal-equations solve** per gene.

A Gaussian-kernel variant replaces the hard-windowed tricube kernel with a soft exponential decay `w(d) = exp(-d² / h²)` where `h` is chosen to give the same effective degrees of freedom. The sorted domain lets us truncate the kernel at `3h` (99.7% tail), giving per-gene O(6h × 1) = O(constant per gene) with amortized global complexity of **O(n)** instead of O(n² × span).

### Claim

On the HVG tiny and small-real fixtures, the Gaussian-WLS variant:

- Produces fitted `ve` (expected variance) values that differ from scikit-misc cubic LOWESS by ≤ 1% relative on the top-2000 genes.
- Produces HVG rankings (jaccard top-2000) ≥ 0.99 vs cubic LOWESS.
- Runs in ≤ 50% of the cubic kernel wall time on the small-real scale.
- Memory footprint ≤ cubic LOWESS (no additional workspace beyond the kernel's own smem).

### Success metric

If the three gates above are met AND the full HVG correctness test (`HvgTest.Tiny_*` + `HvgTest.Gsm4037629_*`) passes with the Gaussian variant as the primary LOWESS, it becomes the primary path in `hvg.h` and the cubic-tricube variant becomes `loess_kernel_legacy`. If not, append to `state/novel-attempts.md` as a closed negative result with the residual gate numbers so future attempts know what to beat.

### Additional novel idea: no-LOWESS Pearson variant

Lause 2021 PearsonResiduals does not actually need LOWESS at all — the NB residuals are computed directly from per-gene `μ` and `θ`. Our current PearsonResiduals flavor already bypasses LOWESS. Document in the design doc that this flavor is Rule-30-optimal by construction (closed form, no fitting), and the autonomy-pass should prefer it when the user has not specified a flavor AND the library size distribution is heavy-tailed (where Seurat v3 under-weights the high-depth tail).

### Additional novel idea: adaptive clip

Cycle 55b applied uniform `clip = √N_cells` per Lause 2021. A tighter bound is per-gene adaptive: `clip_gene = √(N × min(1, θ/(θ+μ)))`. In the low-count high-θ regime, this is strictly smaller than √N and reduces rank instability. Ship as opt-in first, promote to default if the benchmark shows tighter correctness on the medium-concat scale.

## 5. Autonomy pass (Rule 31)

Current `HvgConfig` likely exposes: `flavor`, `n_top_genes`, `span`, `clip_value`, `loess_kernel` (cubic vs Gaussian). All five auto-tune.

| Config field | Current default | Auto-tune strategy |
|---|---|---|
| `flavor` | user-specified | On-device library-size distribution test: if CV > 0.8 use PearsonResiduals (heavy-tailed); else SeuratV3. |
| `n_top_genes` | 2000 hard-coded | Variance-explained plateau detector: run SVD on the top-k curve for k ∈ {500, 1000, 2000, 5000} and pick the k where the 80%-var-explained flattens. Capped at 5000. |
| `span` | 0.3 hard-coded (scanpy default) | Auto = `max(0.2, min(0.5, 500/n_genes))` — Silverman-style bandwidth. |
| `clip_value` | `√N` (Lause uniform) | Adaptive per-gene `√(N × min(1, θ/(θ+μ)))` (see §4). |
| `loess_kernel` | cubic tricube | Auto-pick based on wall time budget: if `n_genes > 100k`, Gaussian; else cubic. After novel pursuit, Gaussian becomes default. |

No-args `hvg::select(matrix)` returns the top-k HVG set with all five parameters auto-chosen and decisions recorded in `HvgResult::metadata` under `_autotune_*` keys.

## 6. OOC streaming contract (Rule 14)

Billion-cell HVG: two-pass across `.1pz` shards.
- Pass 1: per-chunk gene sum + gene sum-of-squares, accumulated across chunks into global per-gene mean + variance.
- Pass 2: per-chunk gene LOWESS fit (using global per-gene mean + variance) + per-gene standardized residual variance, accumulated globally.
- Final: per-gene ranking on device, top-k extraction.

Chunk size: `min(free_dev_mem / 8, 500k cells)`. Per-pass reductions use cub::DeviceSegmentedReduce for determinism opt-in.

## 7. Determinism contract

HVG involves (1) gene-level reductions and (2) top-k ranking. Both can be non-deterministic in fp32.

- `deterministic=false` default: warp-shuffle reductions, unstable sort for top-k. Fastest.
- `deterministic=true` opt-in: cub::DeviceSegmentedReduce for sums, stable_sort for top-k. Slower but bit-exact across runs.

Per Cycle 55c lesson: tie-aware metric + score-tolerance gate is mandatory in the correctness test regardless of determinism setting.

## 8. Phase E dispatch spec (for Cycle 59 or 60)

Dispatch a `gpu-bench` worker with:
- Read this doc + `state/designs/03-hvg.md` + Cycle 55c novel-attempts 7–11.
- Run the 10-configuration bench table at three scales.
- Implement the Gaussian-WLS LOWESS variant as a prototype kernel in the bench driver (not yet in `hvg.h`). If the Rule 30 gates pass, a subsequent cycle integrates it into the header.
- Implement the adaptive-clip variant as a prototype. Same integration policy.
- Implement the library-size-CV flavor autotuner as a real change in `hvg.h` (it's a single reduction over `snp_dp` or `cell_qc total_umis`, trivial).
- Write to `state/benchmark-registry.md`, `state/pareto-frontier.md` (if promoted), `state/novel-attempts.md` (record gates hit/missed for Gaussian variant).

## 9. Open questions

- **Does rapids-singlecell use the same cubic-tricube LOWESS as scanpy?** Probably yes (they forked scanpy's code path). If they use a different LOWESS, our Gaussian variant might not beat theirs on correctness — verify during bench.
- **Seurat v3 R subprocess on 310k genes**: likely >30 min. Cap at a subsampled reference (e.g. 10k genes) for the R comparison.
- **Does the `n_top_genes` autotuner contradict Rule 31's "simple defaults" spirit?** No — the plateau detector adds ONE dial (`max_k`) to bound the search, with a sane default of 5000.

## 10. Links

- Original: `state/designs/03-hvg.md`
- Cycle 55c episode: `state/cycle-log.md` (final block)
- Novel attempts 7–11: `state/novel-attempts.md`
- Cycle 56 bench infrastructure: `bench/include/singlet_gpu/bench/harness.h`
