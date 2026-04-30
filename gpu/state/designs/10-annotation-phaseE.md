---
feature: 10
module: anno/marker_score.h, anno/reference_map.h, anno/logistic_head.h
cycle: 71 or later
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy
extends: 10-annotation.md
status: draft
---

# Feature 10 — Cell-type annotation Phase E (SingleR + CellTypist + marker scoring)

Feature 10 owns reference-based cell-type annotation. Two complementary approaches: centroid-distance (SingleR-style) and logistic-head (CellTypist-style). Plus marker-gene scoring for unsupervised annotation.

## 1. Success metrics

On GSM4037629 annotated against the Human Cell Atlas reference:

- **SingleR-style wall p50**: ≤ `SingleR::SingleR` (R) × 0.01 (100× headline).
- **CellTypist-style wall p50**: ≤ `celltypist.annotate_data` × 0.1 (10× headline — CellTypist is already Python+sklearn so the wall gap is smaller).
- **Label accuracy** vs gold-standard annotation: F1 ≥ SingleR × 0.98 AND ≥ CellTypist × 0.98.
- **Peak dev mem**: ≤ 200 MB for 20k cells × 20k genes × 50 reference cell types.

Frontier gate: dominance on both backends, accuracy match, Rule 31 autotune delta ≤ 10%.

## 2. SOTA baselines

| Baseline | Path | Notes |
|---|---|---|
| **SingleR** | R Bioconductor | Centroid-correlation reference |
| **CellTypist** | Python / scikit-learn | Logistic-regression head, widely used |
| **scmap** | R | Alternative reference mapping |
| **scArches / SCVI scANVI** | Python | Deep-learning-based label transfer (future covered in scanvi-phaseE not yet drafted) |
| **Seurat label transfer (`TransferData`)** | R | R gold-standard transfer |

## 3. Bench configurations

Scales:
- **small**: GSM4037629 (20k cells) × Human Cell Atlas subset (50 cell types, 30k genes)
- **medium**: 100k concat × full HCA (200 cell types)
- **large**: 1M cells × full HCA

Configurations:

1. `ours_singler_style_centroid` — reference port
2. **`ours_closed_form_calibration`** — Rule 30 §4a
3. `ours_celltypist_logistic`
4. **`ours_batched_logistic_head`** — Rule 30 §4b
5. `ours_marker_auc` — unsupervised marker scoring
6. `ours_auto` — Rule 31
7. `singler_r`
8. `celltypist_python`
9. `seurat_transferdata` — R subprocess, small scale only

Metrics: wall p50, peak dev mem, label F1 vs gold, confusion-matrix diagonal, per-cell confidence distribution.

## 4. Novel pursuit (Rule 30)

### 4a. Closed-form centroid calibration

**SingleR approach**: compute per-gene Spearman correlation between each query cell and each reference cell-type centroid. Rank, threshold, return the highest-correlation label. Per-cell cost: O(n_genes × n_ref_types × log(n_genes)).

**The published bottleneck**: Spearman rank computation dominates — each cell needs per-gene ranking, which is O(n_genes × log n_genes) sort per cell, then a weighted sum against each centroid rank profile.

**Novel variant**: replace Spearman with **Pearson on rank-transformed reference centroids**. Pre-rank the reference centroids ONCE at setup time (cost: O(n_ref × n_genes × log n_genes), done once). Per-cell: use the cell's raw counts but in a pre-computed rank-basis via a batched GEMM: `Query_raw @ Ref_rank^T / norm`. This is a single batched GEMM across all cells × all ref types — fully fused.

Bonus: closed-form calibration of confidence scores via the per-type Pearson distribution. SingleR's "fine" vs "coarse" calibration becomes a single multi-level inference in one kernel pass.

**Expected wall**: ~1 sec for 20k cells × 50 ref types on H100 vs SingleR's ~10 min. **600×.**

**Gate**: F1 ≥ SingleR × 0.98 on held-out annotation.

### 4b. Batched logistic head (CellTypist alternative)

CellTypist trains a logistic regression per cell type on the reference, then infers on query cells. Training is serial across cell types in sklearn.

**Novel variant**: batched logistic regression on device via closed-form Newton's method with the Hessian inverse precomputed per cell type. All cell types trained simultaneously via batched cuBLAS.

Inference is a single `Query @ W + b` GEMM → softmax → argmax.

**Expected wall (training + inference)**: ~2 sec for 20k query × 50 ref types vs CellTypist's ~30 sec. **15× headline.**

**Gate**: F1 ≥ CellTypist × 0.98.

### 4c. Marker scoring via rank-AUC (unsupervised)

For samples without a reference, use marker-gene AUC scoring:
- For each candidate cell type, load its marker gene set from `data/genesets/cellmarker.1pz` (loaded via Cycle 56).
- Per cell: compute the AUC of marker genes in the cell's gene ranking (same kernel as AUCell from 11-gsea).
- Assign the cell type with the highest AUC above a confidence threshold.

Zero training. Fully reference-free.

**Expected wall**: ~500 ms for 20k cells × 200 cell types (using CellMarker 2.0). Useful as a first-pass annotation before the more expensive reference-based methods.

**Gate**: label agreement with reference-based methods ≥ 0.8 on the cell types where CellMarker has reliable signatures.

## 5. Autonomy pass (Rule 31)

No-args `anno::classify(counts, reference)` returns labels with:

| Config | Auto |
|---|---|
| `method` | Auto: `centroid` for small ref (≤100 cell types), `logistic` for large ref, `marker_auc` if no reference provided. |
| `reference` | Auto: detect built-in references at `data/references/hca_core.1pz` if none provided. |
| `min_confidence` | Auto: 0.7 (cells below get `unknown`). |
| `hierarchical_labels` | Auto: true if reference has label hierarchy in metadata. |
| `reject_unknown` | Auto: true. |
| `batch_aware` | Auto: true if multiple samples with different batches detected. |

## 6. OOC streaming contract

Annotation is per-cell — embarrassingly streamable. Per-chunk: compute labels for cells in the chunk, emit results. No global accumulator needed. Works natively with feature 16's streaming driver.

## 7. Determinism contract

- Centroid method is deterministic (closed form).
- Logistic regression with fixed seed is deterministic.
- Marker AUC is deterministic.

## 8. Phase E dispatch spec

Single worker (gpu-kernel-dev → gpu-bench chain):
- Implement closed-form centroid calibration + batched logistic head + marker AUC scoring.
- Port HCA core reference as a `.1pz` accessory (one-off tooling).
- Correctness test against SingleR R subprocess on GSM4037629.
- Bench 9-config × 3-scale.

## 9. Open questions

- **HCA reference bundle**: need to package the Human Cell Atlas core dataset as a `.1pz` with cell-type labels in the metadata TLV. One-off packaging script in `tools/`.
- **Label hierarchy handling**: SingleR supports coarse → fine labels via a tree. Pre-encode in the reference `.1pz` TLV sidecar.
- **`unknown` rejection calibration**: SingleR uses z-scored Pearson for its "pruneScores" step. Port the exact criterion for correctness match.

## 10. Links

- Original: `state/designs/10-annotation.md`
- SingleR: Aran et al. 2019
- CellTypist: Domínguez Conde et al. 2022
- Human Cell Atlas: https://www.humancellatlas.org/
- CellMarker 2.0: Hu et al. 2023
- Mandate v2 §A (EDA parity): Annotation row
