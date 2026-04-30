---
feature: 11
module: gsea/fgsea.h, gsea/aucell.h, gsea/gsva.h, gsea/collapse_pathways.h
cycle: 64 or later
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy pass + Rule 28 accessory library
extends: 11-gsea.md
status: draft
mandate_v2_ref: §D (fgsea + gene-set accessory library); Rule 28 (gene-set .1pz)
---

# Feature 11 — GSEA + gene-set accessory library Phase E frontier push

Per Mandate v2 §D, feature 11 owns GPU-native fgsea, AUCell, GSVA, and the gene-set accessory library under `data/genesets/*.1pz`. This doc specs the frontier push and makes the three biggest novel-pursuit claims in the library: **closed-form ES p-value via characteristic function**, **collapse-pathways via device Jaccard bitset clustering**, and **gene-set data as `.1pz`** (Rule 28).

## 1. Success metrics

On a standard scRNA DE gene ranking (GSM4037629 cell-type pair vs all) against MSigDB Hallmark (~50 gene sets):

- **fgsea wall p50**: ≤ R fgsea wall × 0.01 (100× speedup headline).
- **ES p-value correlation vs R fgsea**: Pearson ≥ 0.995 on the `-log10(pval)` column across all gene sets (ties below 1e-6 p-value allowed).
- **AUCell wall p50** on 20k cells × 50 gene sets: ≤ R AUCell × 0.01.
- **Peak dev mem**: ≤ 500 MB for 20k cells × 1000 gene sets × full Hallmark+C2+C5.
- **Gene-set library load wall**: ≤ 100 ms for the full Hallmark+C2+C5+C7+C8 bundle (millions of gene-set memberships).

Frontier promotion gate: dominance on wall at the standard Hallmark workload, correlation ≥ 0.99 with R fgsea, Rule 31 autonomy delta ≤ 10%.

## 2. SOTA baselines

| Baseline | Path | Notes |
|---|---|---|
| **R `fgsea::fgsea`** | C++ / R / OpenMP | Primary reference; the algorithm we must reproduce statistically |
| **GSEApy** | Python | CPU Python reference for the same algorithm |
| **R `AUCell::AUCell_calcAUC`** | R | AUCell reference |
| **R `GSVA::gsva`** | R | GSVA reference |
| **Vision (Python)** | Python | Signature scoring reference |
| **clusterProfiler `GSEA()`** | R Bioconductor | Most-used R GSEA wrapper; measures how much of total workflow wall we save |
| **escape (R)** | R | Recent GPU-adjacent GSEA wrapper — still CPU at core |

## 3. Bench configurations

Scales (by gene-set count):
- **tiny**: 10 gene sets (smoke)
- **small**: 50 Hallmark sets (primary competitive scale)
- **medium**: ~6000 C2 sets (canonical "large Hallmark+C2" workload)
- **large**: ~15000 full MSigDB H+C2+C5+C7+C8 (stress)

Cell-count scales:
- small: 20k cells (GSM4037629)
- medium: 100k concat
- large: 1M synthetic

Configurations:

1. `ours_fgsea_permutation` — baseline port of fgsea algorithm (for correctness reference and sanity check)
2. **`ours_fgsea_closed_form_cf`** — Rule 30 characteristic-function closed-form p-value (§4a)
3. `ours_aucell`
4. `ours_gsva`
5. **`ours_collapse_pathways_jaccard_bitset`** — Rule 30 device Jaccard clustering (§4b)
6. `ours_auto` — Rule 31 path
7. `r_fgsea` — R subprocess reference (small + medium only; large will be too slow)
8. `gseapy_prerank`
9. `r_aucell`
10. `r_gsva`
11. `clusterprofiler_gsea`

Metrics: wall p50, peak dev mem, ES / NES / p-value correlation vs R fgsea, FDR correlation, leading-edge gene set jaccard (top-5 gene sets), throughput (gene sets × cells / sec).

## 4. Novel pursuit (Rule 30)

### 4a. Closed-form ES p-value via characteristic function

**The fgsea bottleneck.** fgsea computes the enrichment score (ES) p-value via **adaptive multilevel permutation**: generate many random gene rankings, compute ES for each, fit the null distribution. For 1e6 permutations × 50 gene sets this is ~5e7 ES evaluations — dominates fgsea wall time by 80%+.

**The closed-form alternative.** The ES statistic is `max_i |S_hit(i) - S_miss(i)|` where S_hit/S_miss are running cumulative sums of per-gene weights. For a random permutation under the null, this is the **max of a discrete random walk** whose increments have known per-step distribution (from the weights histogram). The characteristic function of the max is derivable from the characteristic function of the single-step increment, and the tail can be numerically inverted via Gil-Pelaez at any desired accuracy.

**Algorithm**:
```
Input: ranked gene weights W[n], gene-set membership G[n], gene-set size k
Output: ES, p-value

1. Compute observed ES = max_i |running_hit_sum(i, W, G) - running_miss_sum(i, W, G)|
2. Compute the increment distribution: p_hit = k/n, increment_step_hit = W[i]*(1-p_hit),
   increment_step_miss = -W[i]*p_hit. These define a discrete random walk Δ_i with
   known mean, variance, and higher cumulants.
3. The characteristic function of the running-max of the walk is:
     φ_max(t) = ∫ exp(i*t*s) f_max(s) ds
   where f_max is derivable in closed form via the Wiener-Hopf factorization for
   Gaussian random walks (Andersen 1953), and as an approximation via cumulant
   matching for non-Gaussian walks.
4. Invert φ_max at the observed ES using Gil-Pelaez:
     P(ES > s) = 1/2 - (1/π) ∫₀^∞ (1/t) Im[exp(-i*t*s) φ_max(t)] dt
5. That integral evaluates in ~100 quadrature points on device in microseconds.
   Total cost: O(n) for the observed ES, O(const) for the p-value, per gene set.
```

**Expected speedup**: fgsea does ~20000 permutations at default settings × 50 gene sets = 1e6 ES evaluations; we do 50 ES evaluations (one per gene set) + 50 × 100 = 5000 Gil-Pelaez quadrature evaluations. Total work reduction: ~200×. Add batched gene-set parallelism and the GPU wall advantage over R fgsea (which is OpenMP CPU) and the headline is **100× wall time** on the small-scale benchmark.

**Gate**: Pearson correlation of `-log10(p-value)` with R fgsea ≥ 0.995 on a reference ranking. ES values bit-exact within fp32 ULP. Wall ≤ R fgsea × 0.01.

**Risk**: the Wiener-Hopf factorization assumes the walk increments are light-tailed. For gene rankings with extreme outliers (rare), the cumulant approximation may deviate from the true tail. Mitigation: auto-fall-back to permutation for gene sets where the cumulant skew exceeds a threshold.

### 4b. Collapse-pathways via device Jaccard bitset clustering

fgsea's `collapsePathways` function takes the list of significant gene sets and clusters them by gene overlap (Jaccard), returning a non-redundant "headline" set. Published implementation uses pairwise Jaccard computation on R lists — O(n²) with high constant.

**GPU device variant**:
- Represent each gene set as a **bitset over the universe of genes** (n_genes ≤ 30000 → ≤ 1000 uint32 words per gene set). Each gene set = ~4 KB.
- Pairwise Jaccard = `popcount(A & B) / popcount(A | B)` — two `__popc` + two bit-op per uint32 pair. Fully coalesced in global memory.
- Cluster via device BFS: union-find on the Jaccard graph with threshold 0.5.

**Expected wall**: ~1 ms for 1000 significant sets on H100-class hardware vs ~10 sec for R fgsea's collapsePathways. **10000× headline.**

**Gate**: resulting headline set identical (up to cluster-label permutation) to R fgsea on a reference.

### 4c. AUCell via per-cell warp-sort + closed-form AUC

AUCell computes, for each cell and each gene set, the AUC of the gene-set genes in the cell's gene ranking. The standard algorithm sorts per cell O(n log n), then integrates.

**GPU variant**: use `cub::WarpSort` per cell (cells are independent) with the gene set represented as a bitset lookup. The AUC becomes `sum_{g in set} rank(g) / (k × (n - k))` where rank is the warp-local rank. Entire AUCell matrix computed in one kernel launch.

**Expected wall**: ~100 ms for 20k cells × 50 gene sets vs ~60 sec for R AUCell. **600× headline.**

### 4d. GSVA via bimodal step + device integration

GSVA's signature scores use a non-parametric bimodal step function convolved with the per-cell gene rank. Closed-form on device via a single warp-scan. ~100× R GSVA.

### 4e. fgsea multilevel permutation as the correctness reference

We still need ONE fgsea implementation on device to serve as the correctness ground-truth for 4a's closed-form variant in the edge cases where Wiener-Hopf assumptions break. Ship `ours_fgsea_permutation` as the validation fallback; use 4a's closed-form as the primary.

## 5. Gene-set accessory library (Rule 28 + Mandate v2 §D)

### The accessory data tree

`singlet-gpu/data/genesets/` fetched via CMake `FetchContent`:

| File | Source | Approx size |
|---|---|---|
| `go.bp.1pz` | Gene Ontology BP (from obo, latest) | ~25k sets × 30k genes |
| `go.mf.1pz` | Gene Ontology MF | ~4k sets |
| `go.cc.1pz` | Gene Ontology CC | ~4k sets |
| `reactome.1pz` | Reactome pathways | ~2.5k sets |
| `kegg.1pz` | KEGG (license permitting) | ~500 sets |
| `msigdb.h.1pz` | MSigDB Hallmark | 50 sets |
| `msigdb.c2.1pz` | MSigDB C2 | ~6k sets |
| `msigdb.c5.1pz` | MSigDB C5 | ~10k sets |
| `msigdb.c7.1pz` | MSigDB C7 (immunology) | ~5k sets |
| `msigdb.c8.1pz` | MSigDB C8 (cell-type signatures) | ~700 sets |
| `cellmarker.1pz` | CellMarker 2.0 | ~1.5k cell-type markers |
| `panglaodb.1pz` | PanglaoDB | ~200 cell types |
| `dorothea.1pz` | DoRothEA TF regulons | ~475 TFs × targets |
| `collectri.1pz` | CollecTRI | ~1200 TFs |

### Encoding

Each `.1pz` stores the **gene-set × gene boolean membership matrix** using the same VOCSC chunk encoder as expression matrices, with `vt_code=1` (uint8) and values always 0 or 1. A sparse CSC where `indices[i]` = gene index and `values[i]` = 1 (the values column is compressed to its minimum by the existing byte-split encoder — basically free).

TLV sidecar carries:
- Hierarchy edges for GO (`META_TAG_HIERARCHY_IS_A`, `META_TAG_HIERARCHY_PART_OF`).
- Parent/child links for Reactome.
- External ID cross-references (Ensembl, Entrez, HGNC, Uniprot) as flattened KV strings.
- Gene-set metadata: name, description, source URL, revision date.

### Loader path

`singlet_gpu::io::load("data/genesets/msigdb.h.1pz")` returns a `PzLoadResult` with `matrix` as a `SparseMatrixGPU<float>` — the GSEA kernels read gene-set memberships through **the exact same zero-copy path** as expression matrices. No special-case code. Per Rule 28.

Loading the full Hallmark+C2+C5 bundle (~21k gene sets × 30k genes = ~630M bits = ~80 MB after compression) target: 100 ms.

### Build-time tooling

Scaffold a separate tool `tools/geneset_packager.py` that converts:
- `.gmt` (MSigDB native) → `.1pz`
- `.obo` (GO OBO format) → `.1pz` with hierarchy sidecar
- Reactome bioconductor dumps → `.1pz`

Run once during library build; outputs committed to a separate versioned accessory repo (NOT vendored into the main source tree).

## 6. Autonomy pass (Rule 31)

No-args `gsea::fgsea(ranking)` returns enrichment results against the default gene-set library (MSigDB Hallmark) with:

| Config | Auto |
|---|---|
| `gene_sets` | Auto: `msigdb.h` loaded from the accessory library |
| `n_permutations` | Auto: 0 for closed-form CF path, 10000 for fallback permutation path |
| `min_size` | Auto: 15 |
| `max_size` | Auto: 500 |
| `weight_type` | Auto: 'classical' (McInnes ES) unless user pins |
| `p_method` | Auto: `closed_form_cf` if cumulant-skew check passes; else `permutation` |
| `multiple_testing` | Auto: BH FDR |
| `collapse_threshold` | Auto: 0.5 Jaccard |

## 7. Determinism contract

- Closed-form CF path is deterministic by construction.
- Permutation fallback is seeded via user-supplied `random_seed`.
- Collapse-pathways BFS is deterministic.
- AUCell / GSVA / Vision are deterministic.

## 8. Phase E dispatch spec (Cycle 64 or later)

Dispatch two workers in parallel:

**Worker A (gpu-kernel-dev)**: implement 4a (CF closed-form), 4b (Jaccard bitset), 4c (AUCell warp sort), 4d (GSVA) as prototype kernels.

**Worker B (gpu-bench)**: run the 11-config bench table at all scales, verify ES/NES/p correlation against R fgsea, bench the CF closed-form variant vs R fgsea wall, measure gene-set loader speed.

Both write results to benchmark-registry.md, pareto-frontier.md, novel-attempts.md.

## 9. Open questions

- **Does the Wiener-Hopf factorization have a tight enough error bound for the GSEA tail (1e-10 p-values)?** Andersen 1953 gives O(1/√n) convergence of the CF approximation to the true walk max distribution. Good enough for n ≥ 100 genes in the set (typical). For smaller sets, fall back to exact enumeration (O(2^k) — trivial for k ≤ 20).
- **Gene-set library licensing**: MSigDB requires attribution; KEGG's license is restrictive (we may only ship link to the user's own download). Document in each `.1pz` TLV.
- **Is there a GO DAG query we need inside the library?** Yes: "all descendants of GO:0008150" for propagation. Ship as a precomputed closure in the hierarchy sidecar.
- **`data/genesets/` FetchContent URL**: need to host the accessory bundle somewhere. Suggest: separate GitHub release artifact per singlet-gpu version tag.

## 10. Links

- Original: `state/designs/11-gsea.md`
- R fgsea: Sergushichev 2016 preprint; GitHub https://github.com/ctlab/fgsea
- MSigDB: http://www.gsea-msigdb.org/gsea/msigdb
- Gil-Pelaez 1951 characteristic function inversion
- Andersen 1953 Wiener-Hopf factorization for random walks
- Mandate v2 §D: `CLAUDE.md` search for "fgsea + gene-set accessory library"
- Rule 28: `CLAUDE.md` search for "Gene-set databases live in `.1pz`"
