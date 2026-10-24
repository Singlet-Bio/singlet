# E1: How much of a cell is predictable from the rest of the cell? (the state-information measurement)

> **Status 2026-10-22: complete.** Results in `notes/e1.md`: the state term is 0.04-0.12 bits per cell nonzero on
> typical files (k_eff 0-62), well below the anticipated 0.5-1.5; neighbours carry it, the cell's own half adds ~0.

Context: `notes/theory_fable.md` sections 1-3. The cells payload is sampling entropy H(x | π) plus state
information I(x; π | S). We cannot observe π, but we can measure how much *one half of a cell's UMIs* predicts the
other half beyond what depth and the population model predict. That mutual information is a lower bound on the
state information and, with reasonable models, close to the reachable gain of sequential/neighbour conditioning.
It decides whether the cells codec target is ~7 bits/nnz (done) or ~3-4 bits/nnz (big win).

## Design

For each corpus file (exon and intron, called cells only; empties separately as a control), for each cell with
N ≥ 50 UMIs: randomly split the cell's UMIs into halves A and B (per UMI, i.e. per unit of count: a feature with
count 3 contributes three UMIs assigned independently). Hold B out. Score the cross-entropy of B (as a multiset of
UMIs: Σ over UMIs of −log2 q(feature) with the multiset correction, and separately as a sparse presence+count
vector) under a ladder of predictors q, all computed *without B*:

| level | predictor of feature probabilities for B | what it measures |
|---|---|---|
| L0 | population per-feature frequency (this file, all cells, excluding the cell) | order-0 |
| L1 | L0 × depth-bucket adjustment (per-feature presence vs depth; 16 buckets) | today's best static model |
| L2 | L1 shrunk toward the mean profile of the cell's k nearest neighbours in A-space (k = 10, 30, 100; neighbours found with the *other cells' full* profiles, distance on log1p of top 2000 genes) | neighbour conditioning (Idea 2 ceiling) |
| L3 | Dirichlet posterior: (α · L2 profile + counts_A) / (α + N_A), α swept {10, 30, 100, 300} | sequential within-cell conditioning (the cell's own A half as context) |
| L4 | L3 where the prior is the cell's own *full* profile smoothed (α·pop + counts_A + counts_B) | in-sample oracle: biased low; brackets the sampling floor |

Report per file: bits per held-out UMI and per held-out nonzero for each level; the drop L1 → L2, L2 → L3
(these are the gains available to the two kinds of conditioning), and L3 → L4 (unreachable, ≈ remaining state
information + sampling). Also report by depth bucket (shallow cells should show the largest L1→L3 gains) and the
effective k implied by fitting gain ≈ (k/2) log2 N per cell.

Second measurement, cheaper and complementary: **entropy of a cell's expression distribution** H(π_c), estimated
by the Chao-Shen / coverage-adjusted estimator on each cell's counts and by the L3 posterior; report the
distribution across cells and the implied H(Multinomial(N, π)) via the formula in theory_fable.md section 2.

## Deliverables

- `tools/e1_split.py` (numpy only; `python -P`; corpus files are untrusted input), run on all 14 files; output
  `results/e1_split.tsv` (file, subset, depth bucket, level, params, n_cells, held-out UMIs, held-out nnz,
  bits/UMI, bits/nnz) and `results/e1_hpi.tsv` (per-file distribution of H(π_c) estimates).
- `notes/e1.md`: the ladder table per file, the implied effective k, the implied sampling floor bracket, and a
  one-paragraph verdict: is the state term closer to 0.3 or 3 bits/nnz, and for which depth regimes?

## Anticipated result

PBMC-depth cells (N ~ 700): L1→L3 gain 0.5-1.5 bits/nnz, concentrated in cells below 500 UMIs; nuclei at
N ~ 1400: 0.3-0.8; T cells (dense, homogeneous): < 0.3. Effective k ≈ 50-100. If the gains are instead > 2
bits/nnz, sequential/neighbour conditioning becomes the primary codec lever and E2 must include it.

## Constraints

Single-threaded; cap cells per file at ~5,000 (random subset) to keep runtime under ~20 min per file; run in the
background. Only `tools/e1_*`, `results/e1_*`, `notes/e1.md` are yours to write.
