---
feature: 14
module: anno/mt_lineage.h, anno/clonal_tree.h
cycle: 68 or later
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy
extends: 14-mt-lineage.md
status: draft
exploit: singlify-unique (mt_alleles.1pz)
---

# Feature 14 — MT heteroplasmy lineage tracing Phase E (unique-to-us)

**Second signature unique-to-us kernel.** singlify's `mt_alleles.1pz` stores per-cell × per-MT-site allele counts (produced automatically when `--pipeline` is enabled). mgatk, MQuad, and ATAC-MT approaches all exist but require a separate pileup pass. We load the matrix in ~10 ms from `.1pz` via Cycle 56's loader and run lineage inference entirely on device.

## 1. Success metrics

On a multi-donor test sample with known donor labels (uses donor_assignments.tsv for ground-truth lineage mixing):

- **End-to-end wall (variant calling → clone assignment → tree inference)**: ≤ `mgatk` wall × 0.01 (100×).
- **Clone-assignment ARI vs mgatk**: ≥ 0.95 on matched variants.
- **Tree topology distance (Robinson-Foulds)**: ≤ mgatk tree + 2 edges.
- **Peak dev mem**: ≤ 200 MB for 20k cells × 16569 MT positions.

Frontier gate: dominance on wall AND ARI ≥ 0.95 AND R-F distance ≤ mgatk + 2. Rule 31 autotune delta ≤ 10%.

## 2. SOTA baselines

| Baseline | Path | Notes |
|---|---|---|
| **mgatk** | Python / R | The canonical MT lineage tool; slow (CPU, per-cell Python loop) |
| **MQuad** | Python | Variant quality filter + clone calling |
| **SCITE** | C++ | Probabilistic mutation tree inference (for DNA, adapted for MT) |
| **vireoSNP** | Python | Donor demultiplexing (not lineage but shares the variant-calling stage) |

## 3. Bench configurations

Scales:
- **small**: GSM4037629 subset with MT outputs
- **medium**: 5-sample concat with mt_alleles.1pz from each
- **large**: 1M synthetic with embedded ground-truth clonal hierarchy

Configs:

1. `ours_mt_variant_calling` — fused per-site heteroplasmy fraction + confidence
2. `ours_clone_jaccard_bitset` — Rule 30 §4a
3. `ours_tree_scite_approx` — Rule 30 §4b
4. `ours_auto`
5. `mgatk_r`
6. `mquad_python`
7. `scite_cpp` (for dense reference trees)

Metrics: wall, peak mem, variant-call agreement, clone ARI, R-F tree distance.

## 4. Novel pursuit (Rule 30)

### 4a. Clone assignment via device Jaccard bitset

mgatk's clone assignment is hierarchical clustering on a binary variant-presence matrix (cell × variant → {0, 1, NA}). Standard implementation: pairwise Jaccard distance in a double loop, O(n²) in Python.

**GPU variant**:
- Each cell's variant profile = bitset over ~30 informative MT sites (post-filter). ≤ 1 uint32 per cell.
- Pairwise Jaccard = `__popc(A & B) / __popc(A | B)` — one instruction pair.
- Full pairwise matrix for 20k cells: 20k² / 2 × (popcount + division) = 200M ops = **~10 ms** on V100S.
- Hierarchical clustering via device single-linkage on the Jaccard graph — O(n² log n) but fully parallel.

**Expected wall**: ~50 ms for full pipeline on 20k cells vs mgatk's ~10 min. **12000× headline.**

**Gate**: ARI ≥ 0.95 vs mgatk on the informative-variant subset.

### 4b. Device SCITE-style tree inference (approx)

SCITE (Jahn et al. 2016) is the gold-standard probabilistic mutation tree. CPU-only, MCMC-based. Published runtimes: hours on 1000 cells.

**Approximate GPU variant**: instead of full MCMC, use **ML tree reconstruction via neighbor-joining on Jaccard distance** followed by a **single local-rearrangement pass** (SPR — subtree prune and regraft — in a small neighborhood around each internal node). This is the standard "fast phylogeny" shortcut used in RAxML-NG.

- Neighbor joining on 20k cells via GPU-parallel distance updates: ~5 sec.
- SPR local rearrangements: parallel across internal nodes, ~2 sec.
- Total: ~7 sec vs SCITE's multi-hour MCMC.
- Tree quality (R-F distance to SCITE): expected within 2–5 edges on 1000-cell reference.

**Gate**: R-F distance ≤ SCITE + 5 edges, wall ≤ SCITE × 0.001.

### 4c. Heteroplasmy-weighted clone confidence (singlify bonus)

Because singlify's `mt_alleles.1pz` gives **per-site allele counts** (not just binary presence), we can weight clone assignments by heteroplasmy fraction confidence. A cell with 80% variant allele at MT:1234 is more confident about its clone membership than a cell with 20% at the same site.

Published tools threshold heteroplasmy first (losing information); we keep the continuous fraction as a weight in the Jaccard.

**Weighted Jaccard**: `w_J(A, B) = sum_i min(h_A[i], h_B[i]) / sum_i max(h_A[i], h_B[i])` where `h` is heteroplasmy. Published as "fuzzy Jaccard" in some clone-tracing papers.

**Expected gate**: improved ARI on cells at clonal boundaries (typically 5–10% of cells in a real sample), verified on a mixed-donor reference.

## 5. Autonomy pass (Rule 31)

No-args `mt_lineage::infer(path)` returns clone labels + tree with:

| Config | Auto |
|---|---|
| `min_heteroplasmy` | Auto: 0.05 (mgatk default). |
| `min_cells_per_variant` | Auto: 10. |
| `variant_filter` | Auto: composite of QC metrics (strand bias, position variance, coverage uniformity). |
| `distance` | Auto: `weighted_jaccard` if heteroplasmy fractions present, else binary `jaccard`. |
| `clustering` | Auto: `single_linkage` for ≤10k cells, `average_linkage` for larger. |
| `tree_method` | Auto: `nj_spr` (neighbor-joining + SPR rearrangement). |
| `n_clones` | Auto: dendrogram cut at the largest gap in merge distances. |

## 6. OOC streaming contract

MT lineage is small — the variant-site matrix is ~30 sites × 20k cells = 600k entries, entirely resident. Streaming only needed at >10M cells, which is unlikely for MT applications. Document, no implementation needed.

## 7. Determinism contract

- Jaccard distance computation is deterministic.
- Neighbor joining is deterministic up to tie-break (stable cell-id ordering).
- SPR rearrangement is deterministic with a fixed acceptance threshold.
- MCMC fallback is seeded.

## 8. Phase E dispatch spec

Dispatch `gpu-kernel-dev` with:
- Read 14-mt-lineage.md + this doc + singlify `pileup_writer.h` for MT allele format.
- Implement variant calling + weighted Jaccard + NJ+SPR tree + heteroplasmy confidence weighting.
- Correctness test against mgatk on a reference sample with known mixed donors.
- Dispatch `gpu-bench` in parallel for the 7-config bench table.

## 9. Open questions

- **What MT site filter does mgatk use?** Their filter is based on strand bias, position variance, and minimum coverage. Port the exact filter for correctness comparison.
- **NJ+SPR tree quality vs full MCMC**: empirical, needs validation. If R-F distance is too high, escalate to a simulated-annealing MCMC variant bounded at 1000 steps (still seconds vs hours).
- **Reference dataset with known clonal hierarchy**: hard to obtain. Use a simulated dataset with injected MT mutations + mgatk + SCITE as the "gold standard".

## 10. Links

- Original: `state/designs/14-mt-lineage.md`
- mgatk: Lareau et al. 2021 "Massively parallel single-cell mitochondrial DNA genotyping"
- MQuad: https://github.com/single-cell-genetics/MQuad
- SCITE: Jahn et al. 2016 "Tree inference for single-cell data"
- singlify MT output: `singlify/include/singlet-pileup/pileup_writer.h`
