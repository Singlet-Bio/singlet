---
feature: 2
module: preprocess/lognorm.h
cycle: 58 (pending Cycle 57 return)
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy pass
extends: 02-lognorm.md (the original design doc)
status: draft
---

# Feature 2 — lognorm Phase E benchmark + novel pursuit + autonomy pass

Feature 2 (`preprocess/lognorm.h`) is already **runtime-correct** as of Cycle 55b — all 5 runnable Lognorm correctness tests pass on g001 (V100S sm_70). This document specs the work needed to promote it to `state/pareto-frontier.md` and to engage Rules 30 (beat the literature) and 31 (autonomy by default).

## 1. Success metrics

Promotion to the frontier requires dominance on at least one axis vs SOTA at the small-real scale (GSM4037629, 310,797×20,866, nnz=4.2M), with correctness within tolerance. The metrics:

- **Wall time (p50 of 5 timed iters)**: must be ≤ best SOTA on GPU OR ≤ 10× CPU SOTA.
- **Peak device memory**: must be ≤ best SOTA on GPU.
- **Throughput (cells/sec on normalize-and-log1p)**: report; not a promotion gate.
- **Correctness**: fp32-exact values vs the reference implementation for the chosen algorithm, with a score-tolerance gate per the Cycle 55c lesson (ULP noise excluded).

## 2. SOTA baselines to beat

| Baseline | Language | Algorithm variant | Notes |
|---|---|---|---|
| **rapids-singlecell `pp.normalize_total` + `pp.log1p`** | Python / cupy | naive total-count | Primary GPU competitor |
| **scanpy `pp.normalize_total` + `pp.log1p`** | Python / CPU | naive total-count | Primary CPU baseline |
| **scran deconvolution (`computeSumFactors`)** | R / CPU | pool-based deconvolution | Statistical gold standard; CPU-only |
| **CellRanger raw total-count** | — | naive total-count | Widely used; effectively identical to scanpy |

## 3. Bench configurations (Cycle 58 dispatch)

For each of the three scales (tiny / small-real / medium-concat), bench the following configurations with 5 timed iterations + 2 warmup:

1. `ours_total_count_manual` — `lognorm::normalize(..., flavor=total_count, target_sum=1e4)` with explicit config.
2. `ours_total_count_auto` — `lognorm::normalize(..., flavor=auto)` using Rule 31 auto-tune of target_sum (median library size).
3. `ours_deconvolution_manual` — our GPU-native size factor deconvolution (see §5) with explicit config.
4. `ours_deconvolution_auto` — auto-tuned pool-size deconvolution.
5. `rapids_total_count` — rapids-singlecell GPU.
6. `scanpy_total_count` — scanpy CPU.
7. `scran_deconvolution` — scran R subprocess CPU.

All correctness spot-checks compare our output against the matching reference (rapids-singlecell for GPU variants, scanpy for CPU total-count, scran for deconvolution). Tolerance: fp32-cast-exact for values, tie-aware rank-equivalent for gene ordering (per Cycle 55c lesson).

## 4. Novel pursuit (Rule 30) — closed-form deconvolution size factors

**Hypothesis**: scran deconvolution is the statistical gold standard for size factor estimation but is slow (R, CPU, pool-based). There is an opportunity to derive size factors in closed form from:

1. The library-size curve (`cell_qc_metrics.tsv`'s `total_umis` column — already computed by singlify for every sample).
2. The SNP pileup (`snp_dp.1pz` — sums to a per-cell "expressed-genome depth" that is a cleaner library-size proxy because it excludes gene-specific amplification bias).
3. The saturation curve (`saturation_metrics.tsv` — per-cell sequencing saturation, directly usable as a sampling-effort correction).

**Claim**: with these three inputs, we can derive a per-cell size factor that reproduces scran's output within a small residual and is computable in O(N) on device with zero iteration. The closed form:

```
s_i = total_umis_i × (1 - saturation_i) × (snp_dp_i / median(snp_dp)) × corr_factor
```

where `corr_factor` is a per-gene residual regression of scran size factors against the above composite for a held-out validation set. The residual — if it is small — is the novel contribution: a size factor that beats scran on wall time by 100×+ while matching its biological signal.

**Success metric for the novel variant**:
- Correlation with scran size factors: ≥ 0.98 Pearson on GSM4037629.
- Wall time for size factor computation: ≤ 1% of scran's wall time on the same sample.
- Downstream DE recovery (top-100 Wilcoxon markers in a reference cell-type pair): Jaccard ≥ 0.95 vs scran-normalized.

If these gates are hit, the novel variant becomes the primary deconvolution path in `lognorm.h` and scran becomes the `--legacy` fallback. If not, log the failed attempt in `state/novel-attempts.md` with the residual trajectory so a future cycle can try a different composite.

**Files to read before implementing**:
- `/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE127/GSE127918/GSM4037629/cell_qc_metrics.tsv` — verify columns present.
- `/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE127/GSE127918/GSM4037629/snp_dp.1pz` — load via Cycle 56's `singlet_gpu::io::load()` and confirm it is per-cell × per-SNP with the expected shape.
- `/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE127/GSE127918/GSM4037629/saturation_metrics.tsv` — confirm per-cell `saturation` column.

## 5. Autonomy pass (Rule 31)

Current `LognormConfig` likely exposes: `flavor`, `target_sum`, `pool_size`, `downsample_target`. All four should auto-tune.

| Config field | Current default | Auto-tune strategy |
|---|---|---|
| `flavor` | `total_count` (user must pick) | Compute coefficient of variation of library sizes on-device; if CV > 0.5 use `deconvolution`, else `total_count`. Zero host roundtrip. |
| `target_sum` | `1e4` (hardcoded) | On-device median of library sizes. If the user pins it, respect. |
| `pool_size` | — (required for deconvolution) | `min(100, n_cells / 10)` — scran's own default heuristic, implementable as a single integer expression. |
| `downsample_target` | none | If `flavor == downsample`, auto = 25th percentile of library sizes. |

The no-args `lognorm::normalize(matrix)` call must set all four automatically. The explicit `LognormConfig` overrides any subset; un-pinned fields stay auto. Record auto-tune decisions in the output metadata map under `_autotune_{field}` keys per the Cycle 56 pattern.

## 6. Out-of-core streaming contract (required by Rule 14)

Billion-cell lognorm: stream CSC chunks from `.1pz` via the Cycle 16 streaming driver (feature 16, deferred). Chunk size = `min(free_dev_mem / 4, 1M cells)`. Per-chunk: compute size factors from the chunk's library-size column only (for `total_count`) OR accumulate the global library-size distribution in a two-pass strategy (for `deconvolution` + `auto`). Log1p is element-wise and fully streamable.

This section is NOT implemented in Cycle 58. It is a contract for when feature 16 lands. Document here so it is not forgotten.

## 7. Determinism contract

Lognorm involves reductions (sum per cell for size factors). Reductions are non-deterministic in fp32 by default (atomicAdd + warp-shuffle ordering). Provide a `deterministic=true` path using segmented scan (cub::DeviceSegmentedReduce) that is slower but bit-exact across runs. Default = non-deterministic (fast). Users opt in.

## 8. Phase E dispatch spec (Cycle 58)

Dispatch a `gpu-bench` worker (general-purpose Sonnet) with:

- Read this doc + `state/designs/02-lognorm.md` (the original).
- Run the 7-configuration benchmark table on g001 sm_70 at three scales.
- Implement the novel closed-form deconvolution variant as a one-off prototype inside the bench driver (NOT in `lognorm.h` yet — if the Rule 30 gates pass, Cycle 59 will do the header integration).
- Write output to `state/benchmark-registry.md`, promotion decision + row to `state/pareto-frontier.md`.
- Return ≤45-line summary with per-config numbers, novel-variant gate results, autonomy-delta (auto vs manual % difference), and Cycle 59 recommendation.

## 9. Open questions / risks

- **Does scran deconvolution actually run at scale on 310k cells?** It may be unworkably slow even at the small-real scale. If scran takes >30 min, cap the comparison at 20k cells or a subsampled reference, and note the extrapolation.
- **Is the singlify `snp_dp.1pz` always produced?** Only for samples run with `--snps VCF`. If GSM4037629 lacks it (catalog says yes, but confirm at bench time), the novel-pursuit gate can't run on this sample alone — fall back to the 2-sample subset that does have snp_dp.
- **fp32 library size overflow** on very-high-depth cells. Use fp64 accumulator for the per-cell sum, then cast back to fp32 for the divide (single cell_qc row = negligible memory). Document in the header.

## 10. Links

- Original design: `state/designs/02-lognorm.md`
- Cycle 55b episode (runtime correctness landing): `state/cycle-log.md` near line 1121
- Cycle 57 bench infrastructure (reusable): `bench/include/singlet_gpu/bench/harness.h`
- Reference-data paths: `/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE127/GSE127918/GSM4037629/`
