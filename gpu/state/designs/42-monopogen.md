---
feature: monopogen
raceway_id: 42
roadmap_id: 42
module: include/singlet-gpu/variants/monopogen.h + python/singlet_gpu/variants/monopogen.py + r/R/variants.R
status: design
tolerance: germline call agreement ≥ 0.95 vs Monopogen Python on common SNPs; somatic call precision ≥ 0.90 on planted synthetic variants
target_perf: 100k cells × 1M candidate SNV sites × 5 ancestry LD panels ≤2 min on A100 (Monopogen Python is ~2-3 hours)
ooc_plan: per-chromosome tile loop; LD panel streamed per chromosome
---

## Why this exists

Cycle 41 lookahead: **Monopogen** (Dou et al. Nature Biotechnology 2023) is the leading somatic SNV caller from scRNA pileup data. **Entirely CPU** (Python + external variant callers). **Third cycle to exploit `snp_ad.1pz`/`snp_dp.1pz`** (after cycles 38 NEBULA + 40 DAESC). 17th "first GPU" candidate.

Distinct from cycles 38/40:
- NEBULA (38): treats SNPs as **genotypes** to predict expression → eQTL mapping.
- DAESC (40): treats SNPs as **allelic balance** per cell → imprinting / XCI / cis-regulation.
- Monopogen (42): treats SNPs as **variant candidates** → germline genotyping + somatic variant discovery.

Biological use cases: tumor clonal hierarchy from somatic SNVs, mosaic germline variant calling from scRNA without WGS, de novo variant discovery in developmental datasets.

## Algorithm — Monopogen

```
Inputs:
  snp_ad (n_cells × n_snps): alt-allele counts (from snp_ad.1pz)
  snp_dp (n_cells × n_snps): total depth (from snp_dp.1pz)
  ld_panel: LD correlation matrix from a reference population (HapMap / 1000G)
  (optional) germline_vcf: known-variant panel to bootstrap genotyping

Outputs:
  germline_genotypes (n_snps × 3): P(hom_ref, het, hom_alt) per SNP
  germline_calls (n_snps): argmax genotype
  somatic_candidates: (snp_idx, alt_frequency, ld_refined_score, p_value) list
  qual_scores (n_snps): variant quality score

Algorithm:
  1. Per-SNP pileup aggregation:
       a. Sum snp_ad across cells → AD[snp].
       b. Sum snp_dp across cells → DP[snp].
       c. alt_freq[snp] = AD[snp] / max(DP[snp], 1).
  2. Germline genotyping:
       a. Bayesian model with 3-state prior (0/0, 0/1, 1/1) + binomial likelihood.
       b. Genotype posterior: P(g | AD, DP) ∝ P(AD | DP, g) * P(g).
       c. Use hardy-weinberg equilibrium prior across SNPs.
  3. Somatic candidate discovery:
       a. Filter SNPs where alt_freq > 0.01 AND alt_freq < 0.30 (not common germline).
       b. For each candidate, compute LD-refined score:
          score[snp] = alt_freq * (1 - LD_correlation(snp, nearest_germline_het))
       c. High score = somatic (not explainable by germline LD).
  4. Statistical testing:
       a. Likelihood ratio test: germline-only model vs germline + somatic model.
       b. BH FDR correction.
  5. Quality scoring: Phred-scaled posterior.
```

## GPU implementation strategy

Native CUDA. The hot paths are per-SNP pileup aggregation (trivial cuSPARSE), germline Bayesian genotyping (elementwise kernel), and LD-refined scoring (sparse SpMM against LD panel).

### Kernels

1. **Per-SNP pileup aggregation** (`snp_pileup_kernel`): sum `snp_ad` and `snp_dp` across cells per SNP. Output dense `(n_snps)` vectors. `cub::DeviceSegmentedReduce::Sum` or simple row-sum since SNPs are columns.

2. **Germline Bayesian genotyping** (`germline_geno_kernel`): 1 thread per SNP, computes binomial log-likelihood for 3 genotypes + HWE prior, softmax to posterior, argmax to call. Trivially parallel.

3. **LD-refined somatic scoring** (`ld_refined_score_kernel`): sparse SpMV against the LD panel. LD panel is precomputed host-side as sparse CSR (few nnz per row — top-100 LD neighbors), uploaded once. For each candidate somatic SNP, compute `alt_freq - sum(LD[snp, j] * alt_freq[j])` to subtract the linkage-explained component.

4. **Likelihood ratio test** (`lrt_kernel`): elementwise per SNP.

5. **BH FDR correction** via `cub::DeviceSegmentedSort` + cummin (cycle 11/36/38 pattern).

## Numerical stability

- fp32 throughout. Binomial log-likelihood via `lgammaf` + log-space addition.
- HWE prior at p=allele_freq stable; clamp p to `[1e-6, 1-1e-6]`.
- LD correlation values ∈ `[-1, 1]`, no overflow.

## Memory layout

- Input: `snp_ad` CSC + `snp_dp` CSC via cycle 0 loader.
- Pileup aggregates: `n_snps × 8 bytes` (AD + DP). At 1M: 8 MB.
- Germline posteriors: `n_snps × 3 × 4 bytes`. At 1M: 12 MB.
- LD panel: sparse CSR `n_snps × n_snps`, ~100 nnz/row. At 1M × 100: 400 MB (int idx + float val).
- Somatic output: list + scores, ~16 MB.
- Total: ~500 MB workspace.

## Streams

One stream. Per-chromosome tile loop is the outer loop.

## Out-of-core

Per-chromosome tile: load LD panel for chromosome K, process candidates on that chromosome, free, move to chromosome K+1. Reuses cycle 35 Numbat per-chromosome pattern.

## Determinism

No stochasticity. Bit-identical.

## Correctness test spec

Test: `tests/variants_monopogen_correctness.cpp`.

Reference: Monopogen Python via subprocess (GitHub install). Fallback: pure-Python binomial genotyping + LD correction without the ML refinement step. Always produces a reference for the synthetic test.

5 test cases:
1. **`Monopogen_TinySynthetic_VsPython`**: 200 cells × 500 SNPs with 20 planted germline hets + 10 planted somatic. Germline agreement ≥ 0.95 vs Python; somatic precision ≥ 0.80.
2. **`Monopogen_GSM_RealData`**: load real sample with snp_ad/snp_dp. Confirm finite counts, germline call count non-zero.
3. **`Monopogen_HWE_Consistency`**: germline allele frequencies satisfy HWE approximately on common SNPs.
4. **`Monopogen_LdRefinement_ReducesFalsePositives`**: with LD panel, fewer somatic candidates are called than without.
5. **`Monopogen_Determinism_BitIdentical`**: bit-identical across two runs.

## Target performance

| Scale | Cells | SNPs | Wall (target) | Monopogen Python |
|---|---|---|---|---|
| tiny | 200 | 500 | <100ms | ~10s |
| 10k | 11,560 | 100k | <10s | ~20 min |
| 100k | ~120k | 1M | <2 min | ~3 hours |

## Implementation notes

- Header path: `include/singlet-gpu/variants/monopogen.h` (~900 LOC, under 1500 cap).
- New module path `singlet-gpu/variants/` (NEW).
- Python wrapper: `python/singlet_gpu/variants/monopogen.py` (~150 LOC).
- R wrapper: `r/R/variants.R` (~120 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuSPARSE + cub.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycles 38/40 (SNP pileup patterns).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU Monopogen-style somatic SNV calling from scRNA)` first comment.
- Use factornet field-access style.

## Risks

1. **LD panel input format**: ship a pre-computed CSR LD matrix as a TSV or binary fixture. Document format.
2. **Somatic calling is inherently noisy** at low scRNA depth. Report with conservative FDR.
3. **Monopogen Python install** is heavy (multiple upstream callers). Fall back cleanly.
4. **ML refinement step** (Monopogen's final filter using a trained classifier) deferred — ship rule-based filter as MVP. Document.
