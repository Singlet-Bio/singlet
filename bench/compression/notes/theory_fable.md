# A reasoning system for lossless compression of single-cell count data

Fable 5.1, 2026-10-21. Working note, revised as agent results land. Companion to `brainstorm_fable.md`
(ideas) and `../plans/*.md` (investigations). Numbers marked (m) are measured in this study; (e) are estimates
with the assumption stated; (b) are bounds.

The goal is to be able to say, for a `.singlet` count matrix, *this many bytes are redundancy, this many are
information about the biology, and this many are irreducible sampling noise*, and to design a decoder that reaches
the sum of the second and third at memory-bandwidth speed. "Ultradeep" or "recursive" compression is then a precise
statement: approach H(X | everything the decoder legitimately knows) with a decoder that does O(log m) work per
nonzero.

## 1. The object and the decomposition

A droplet count matrix is a set of columns. Column c is a barcode b_c, a depth N_c (total UMIs) and a sparse
count vector x_c over features (here: exon or intron intervals, grouped into genes). Write S for the decoder's
side information: the reference (feature table, gene structure, barcode whitelist), any shipped model, and
everything already decoded from this file (and, in a study bundle, from earlier samples).

The expected length of any lossless code is at least

    L  >=  H(X | S)  =  Σ_c [ H(b_c | S) + H(N_c | S) + H(x_c | N_c, S) ]            (chain rule)

and the three terms behave very differently:

- **Barcodes.** A sorted subset of n from a whitelist of U: log2 C(U, n) ≈ n (log2(U/n) + 1.44) bits.
  For 96k of 737k that is 4.4 bits/barcode, 53 KB (b). Negligible once coded as whitelist index deltas (today
  they are 16-char strings inside a zstd-3 blob: 0.47 MB (m)).
- **Depths.** ~1-2 bytes per column after the per-file distribution is learned; negligible.
- **The count vector.** Everything else. The rest of this note is about H(x_c | N_c, S).

Under the standard generative model a cell's counts are a multinomial draw from its (unknown) expression
proportion vector π_c:  x_c ~ Multinomial(N_c, π_c). Then, exactly,

    H(x_c | N_c, S)  =  H(x_c | N_c, π_c)  +  I(x_c ; π_c | N_c, S)                   (1)

The first term is **sampling entropy**: given the cell's true state, which UMIs happened to be captured. No decoder
can beat it because π_c is never part of S (it is the quantity the file is communicating). The second term is
**state information**: how much the observed counts tell the decoder about π_c beyond what S already implies. This
is the only term a better model, a bigger atlas prior, neighbour conditioning or "recursive" conditioning can reduce,
and it is reduced exactly by making S more informative about π_c.

Everything in this study is a measurement of, or an attempt to approach, the two terms of (1), plus the pure
redundancy that sits on top of them in today's files.

## 2. The sampling floor, numerically

H(Multinomial(N, π)) = N·H(π) − log2 N! + E[Σ_g log2 x_g!]. The −log2 N! term is the multiset saving: we code a
set with counts, never the order of UMIs; every sparse coder gets it automatically.

For brain_sn_v3_deep exon, called cells (m): 998 nonzeros per cell, mean count 1.43, so N ≈ 1428. log2 1428! ≈
12,900 bits; the Σ log2 x_g! term ≈ 110 bits. Hence (e)

| H(π_c) assumed | H(Mult) per cell | bits per nonzero |
|---|---|---|
| 11 bits (perplexity ~2k features) | ~2,900 | 2.9 |
| 12 bits | ~4,340 | 4.3 |
| 13 bits | ~5,760 | 5.8 |

Today's order-0 per-feature cost on the same cells is 7.3 bits/nnz (m, Track D), the adaptive per-feature × depth
coder 8.07 on all columns (m, Track C c_r2) and ≈ 7 on cells. So **between 20% and 60% of what we spend on called
cells is sampling entropy and cannot be removed by any model**, and the rest is state information plus model
mismatch. Which end of that range we are at depends on H(π_c), the entropy of a single cell's expression
distribution at interval resolution, which no simple estimator gives (the plug-in from the cell's own counts is
biased low by thousands of unseen features). Plan E1 measures it indirectly.

Interval resolution adds a fixed cost: H(interval | gene) ≈ 1.15 bits per UMI in-sample (m, Idea 3, tumor_3p), i.e.
H(π_interval) ≈ H(π_gene) + 1.15. Out of sample the static per-gene interval tables cost twice that (m): where a UMI
lands within a gene depends on the protocol (3' vs 5' bias) and must be learned per file. Adaptive per-gene
interval tables (m: 419k bits vs 313k in-sample vs 639k static-OOS on tumor exon) are the right tool.

## 3. The state-information term and why cell-type models have been disappointing

How much can S learn about π_c from x_c? If π_c lies on (or near) a k-dimensional manifold (cell types and
continuous programs; k ≈ 30-50 is the usual "number of informative PCs", k ≈ 300 if one counts every gene
program), then the minimax regret of learning k smooth parameters from N observations is the classic
(k/2) log2 N + O(k) (Rissanen; Clarke-Barron). That is an upper bound on I(x_c; π_c | N_c, S) when S contains the
manifold but not the coordinates:

| k | N = 700 (PBMC, 410 nnz) | N = 1428 (nuclei, 998 nnz) | N = 10,000 (3,000 nnz) |
|---|---|---|---|
| 50  | 237 bits = 0.58 bits/nnz | 262 bits = 0.26 bits/nnz | 332 bits = 0.11 bits/nnz |
| 300 | 1,420 bits = 3.5 bits/nnz | 1,570 bits = 1.6 bits/nnz | 1,990 bits = 0.66 bits/nnz |

Three consequences:

1. **A discrete cell-type label is nearly worthless as side information.** I(x_c; type) ≤ H(type) ≤ log2(#types)
   ≈ 4 bits per cell ≈ 0.01 bits/nnz. Track D's k-topic mixtures (mult1/4/16/64) gaining ≤ 0.2 bits/nnz over
   per-feature × depth (m) is exactly what this predicts. Any scheme that transmits "which cluster" buys nothing.
2. **Transmitting a per-cell latent vector as side information cannot help on net.** If h is a function of x_c,
   coding h costs ≥ I(x_c; h) and conditioning on it saves exactly I(x_c; h) (data processing inequality). The
   marginal p(x) = Σ_h p(h) p(x | h) is always at least as good as any (h, x | h) split. Latent side information is
   a *computational* device for approximating the marginal, never an information-theoretic gain. (Idea 4 should be
   judged as such: its value is in making the marginal tractable, not in the side info.)
3. **The value of model sophistication falls like log N / N per nonzero.** Deep cells are near-optimally coded by
   a per-gene model; shallow cells are where cell-state conditioning can matter. Whether it matters at all depends
   on the effective k, which is an empirical question (Plan E1).

The honest range for the state term is therefore 0.3-3.5 bits/nnz for PBMC-depth cells. That is also the honest
range for the *reachable* gain over a per-feature × depth adaptive coder: **somewhere between "essentially done"
and "another 40%".** Plan E1 is designed to collapse this range with one measurement.

## 4. In-file adaptation versus shipped models (settled by data)

A shipped out-of-sample model saves the regret of learning the per-feature rates from the file itself. For a
per-feature × 16-depth-bucket Bernoulli model with hierarchical shrinkage, that regret was measured at 0.036
bits/nnz on brain_sn_v3_deep exon (c_r2 with model 8.075 vs c_r2_nomodel 8.111, m) and the in-sample "upper bound"
model gains only 0.016 more. Conclusion: **the file is its own model**; a shipped prior is worth < 0.5% and is only
useful for very small files (tumor_3p, 0.25M nnz: 10.52 vs 10.93, m). This removes the entire "8 MiB L3-resident
model" complexity from the design and the cross-validation caveats from the claims. The reference vocabulary
(names, gene structure, whitelist) remains the one piece of shipped knowledge, and it is worth 3-5 MB per file.

Corollary for "recursion": the only recursive structure with information-theoretic content is *conditioning on
what has already been decoded* (cells on earlier cells, intervals on genes, introns on exons, empties on cells).
Second-pass compression of a near-entropy bitstream returns ratio ≥ 0.97 for every general-purpose codec and a
context-mixing coder (m, Track D), and the residual 1-3% is the first pass's own modelling slack, not hidden
structure.

## 5. Empty droplets: a floor we can reach, and half the file

(Corrected 2026-10-21 23:30 against Track D and Idea 2.) Empties are multinomial draws from the ambient profile
p_amb with k UMIs. The exact ambient-multinomial code (profile learned from already-decoded columns, Idea 2 stage 1)
lands within 1-6% of the Monte-Carlo entropy H(Mult(k_j, p̂)) on every file (m), and that entropy reproduces the
in-sample code length to 0.0-0.2 bits/nnz (m, Track D): **shallow droplets (k < 32) are, to measurement precision,
exchangeable multinomial draws** (cross-fitted Pearson dispersion 1.00-1.02). Today's zstd-3 chunks spend
1.5-3.5 bits/nnz *more* than that floor on empties (m), so the ambient code is a real 20-30% win on the empties
share, which is 6-61% of today's payload.

Two measured departures from the multinomial, in opposite directions: (i) k ≥ 64 "empties" on deep samples have
dispersion 2-9 (brain_sn_v3_deep exon: 4.8 at k 128-255, 9.4 at k ≥ 256): uncalled cells and debris with
cell-type structure, to be coded with the cell model (a per-droplet mixture weight; Idea 2 stage 2); (ii) real
droplets show fewer distinct features per UMI and more repeats than multinomial draws (z = −16 to −355): UMI capture
is bursty at the molecule level, so a count model with within-droplet clumping (Dirichlet-multinomial-like) beats
the pure multinomial by a few percent on empties and, by the same physics, on cells.

Because empties are 48% of the nonzeros in brain_sn_v3_deep exon and 39% in pbmc_v2_shallow (m) and will sit at
their entropy floor once ambient-coded, **no further codec work will make them small**. The lever is layout: put
called cells in the primary member and empties in a second member that is fetched only on demand. This is not
compression and does not change the lossless guarantee of the bundle; it changes what a typical user transfers.
(E4 measured how much: because today's columns are already roughly depth-sorted, the empties split is worth 1-15%
of the all-cells transfer for most samples and 25-40% where cells and empties interleave; the larger latency win is
removing the redundant blocks and moving the chunk index to the front, 79-97% of the saving. See frontier_fable.md
§6.)

## 6. Where the bytes go: a budget for a typical file

brain_sn_v3_deep/exon_counts.1pz, 13.27 MB today (m):

| component | today | floor / target | how |
|---|---|---|---|
| feature names | 4.65 MB | 0 (reference vocabulary id) | Track A, done in principle |
| row permutation | 1.06 MB | ~0.07 MB (row counts) or 0 (model order) | Track A, measured 13x |
| barcodes | ~0.44 MB | ~0.05 MB (whitelist deltas) | Track A |
| empties payload | ~3.9 MB (3.07M nnz) | ~3.7 MB; tierable | floor; layout |
| cells payload | ~2.9 MB (3.28M nnz, 7.3 b/nnz) | 1.2-2.6 MB (2.9-6.5 b/nnz) | E1 decides; adaptive gene × depth × state coder |
| counts > 1, depths, ptrs | ~0.2 MB | ~0.15 MB | cheap |
| **total transferred for cells-first use** | **13.27 MB** | **≈ 1.4-2.8 MB** | 5-9x |
| **total, everything** | 13.27 MB | ≈ 5.1-6.5 MB | 2-2.6x |

The 5-9x is almost entirely redundancy removal plus layout; the codec frontier contributes a further 1.1-2.2x on
the cells payload. This is the shape of the result to expect, and the claim to defend.

## 7. Thought experiments

**T1. The atlas as decoder prior.** Suppose the decoder holds singlet.bio's 240M-cell atlas and can find, for
each cell, its nearest atlas neighbours. The gain is I(x_c; π_c | S) with S now containing a very good manifold
estimate, i.e. the state term drops toward the manifold-coordinate cost (k/2) log2 N; it cannot go below the
sampling floor. For a 700-UMI PBMC cell that is at most a few hundred bits. The atlas cannot make a file
"ultradeep"; it can only remove the last 0.3-3 bits/nnz of state information, and in-file neighbours already
remove most of it. Prediction: atlas-prior gain over in-file adaptive neighbour conditioning < 0.3 bits/nnz.

**T2. Two samples from the same donor.** Cross-sample conditioning (sample 2 given decoded sample 1) is in-file
adaptation with a warm start. The gain is the learning regret of sample 2's model, ≈ 0.04 bits/nnz for a large
sample, more for a small one (≈ 0.4 on tumor_3p). Prediction: warm-starting the adaptive model from earlier samples
of a study is worth 0-5%; cheap to do, not a headline.

**T3. The deep cell.** At 50k UMIs over 8k genes, presence is nearly certain for expressed genes and the cost moves
to counts: H(Poisson(λ)) ≈ 0.5 log2(2πe λ) per gene. A per-gene × depth model is within (k/2) log2 N ≈ 0.1
bits/nnz of optimal. Prediction: for deep plate-based or Smart-seq data the codec question is value coding, not
presence; conditioning buys nothing measurable.

**T4. Recursing on the bitstream.** Let the first-pass coder spend −log2 q(x) bits. A second pass with any byte
model r over the output bits composes to a model q' on x; it gains iff KL(p‖q) > KL(p‖q'). Nothing about the
second pass being "recursive" matters; it is one more model. Since q is already adaptive on the data's own
statistics, r has only coder slack to find (m: 1-3%). The correct "recursion" is to improve q with structure we know
exists (gene → interval, exon ↔ intron, cell ↔ neighbours), which is the chain-rule hierarchy, and each layer's
gain is the corresponding conditional mutual information, measurable in advance without building the coder.

**T5. A derived matrix costs zero bits.** splice_psi is a function of sj_counts; column sums of indptr; the
permutation of row counts; names of the reference. Anything deterministic from (reference, counts) is Kolmogorov-
free. The bundle should store the generating inputs and recompute the rest (Plan E4 audits which members qualify
and whether recomputation is bit-exact, which for floats means fixing the formula and the evaluation order).

**T6. A lossy policy is not a codec.** Dropping empties, or storing them at lower resolution, changes H(X) by
changing X. It may be the right product decision (Plan E4 tiering keeps them, just elsewhere), but it must never be
presented as compression of the matrix.

## 8. What "the frontier" looks like, concretely

A decoder that (i) ships nothing but the reference, (ii) codes cells first, in blocks of ~1024, under a
per-gene × depth model that adapts between blocks (so block-static tables permit interleaved rANS at 200+ Mnnz/s),
(iii) optionally conditions each cell on a small set of already-decoded neighbour cells for the shallow-cell
regime where state information is worth > 0.5 bits/nnz, (iv) codes intervals given genes with adaptive per-gene
tables, (v) codes introns given exons at gene level, (vi) codes empties last against the ambient profile derived
from the decoded cells, and (vii) stores empties as a separately-fetchable member.

Predicted outcome versus today (brain_sn_v3_deep exon, 13.27 MB): cells-first transfer 1.4-2.8 MB (5-9x),
everything 5-6.5 MB (2-2.6x); decode at ≥ 100 Mnnz/s single-threaded for the block-adaptive path, ≥ 20 Mnnz/s if
per-cell neighbour conditioning is enabled. The cells payload lands within 10% of the E1-measured conditional
entropy, and the second-pass ratio on the output is ≥ 0.99.

What would change this picture: E1 finding that half of a cell predicts the other half much better than depth +
population do (large effective k, large state term), in which case neighbour/sequential conditioning becomes the
main codec lever and the cells payload target moves toward 3-4 bits/nnz.

## 9. Three arguments that need no data

**9.1 The floor, and what it assumes.** Claim: for any lossless code whose decoder knows S, the expected length
over files drawn from the generative model is ≥ Σ_c H(x_c | N_c, π_c) + (terms for barcodes, depths). Proof
sketch: L ≥ H(X | S) by Shannon; H(X | S) ≥ H(X | S, Π) where Π = (π_c)_c, because conditioning never increases
entropy; given Π the columns are independent multinomials, so H(X | S, Π) = Σ_c H(x_c | N_c, π_c). What this
assumes: (i) that there is a well-defined π_c, i.e. UMIs within a cell are exchangeable draws from one
distribution. UMI deduplication makes PCR duplicates irrelevant; residual violations (UMI collisions, barcode
errors, index hopping) are small perturbations of π_c, not structure a coder could exploit. (ii) That S does not
contain π_c, which is true by construction: π_c is the biology the file communicates. The floor is a property of
the measurement process, not of any coder; it is what makes "asymptotically impossible to compress further" a
statement about physics rather than about our ingenuity. The one way around it is to change X (drop or coarsen
columns), which is a policy, not a codec (T6).

**9.2 Feature order is irrelevant to a model-driven coder, so permutation blocks are an artefact of LZ.** Under a
product model Π_f Bernoulli(p_f) for presence, the ideal code length of a presence set is Σ_f [present ?
−log2 p_f : −log2(1−p_f)], a sum over features that does not depend on the order in which they are coded. The
survival-gap coder (Idea 1) attains exactly this sum: coding "next present feature is j, given previous was i" at
cost Σ_{i<t<j} −log2(1−p_t) − log2 p_j telescopes to the same total. Hence the row-frequency permutation that today
costs 0.7-1.06 MB per file (m) exists only because zstd is a byte-context model that needs small gaps to see
structure. A model-driven coder needs no permutation at all; it needs the model, which the file provides
adaptively. (Track B's finding (m) that interpolative coding beats gap+zstd without any entropy coder is the same
fact seen from the set-coding side: interpolative coding is near the uniform-set bound log2 C(m, k); the model-driven
coder is near the non-uniform bound, which is lower by exactly the KL between uniform and p.)

**9.3 Block adaptation loses little.** Let the model have K effective parameters (features with non-negligible
counts × depth buckets) and let the file have B blocks. Per-symbol Bayesian adaptation pays the parametric regret
≈ (K/2) log2 n once. Block adaptation with tables frozen within a block and refit between blocks pays, in addition,
the information the block's own symbols would have contributed to their own prediction: for a block with n_b
symbols of a feature whose rate is r, the loss is ≈ the KL between the rate estimated before the block and the
rate estimated after it, ≈ (1/2) · Var(estimate) · Fisher ≈ 1/(2 n_prev) nats per symbol when n_prev symbols were
seen before. Summed over blocks this is ≈ (K/2) Σ_b n_b / n_{<b} ≈ (K/2) ln B. For K ≈ 10^5 active parameters,
B = 83 blocks: ≈ 3 × 10^5 bits ≈ 0.05 bits/nnz on a 6M-nnz file. Doubling the block size halves B and the cost
barely moves (ln). So E2's expectation of 0.1-0.2 bits/nnz regret is conservative; the first block (flat prior) is
the only expensive one, and only for small files.

## 9b. The interval split is sampling noise too (learned from Idea 3)

Rows are intervals of genes; which interval of a gene a UMI lands in is set by where the fragment was captured on
the transcript. Idea 3 measured (m) H(interval | gene) at 0.5-1.3 bits per UMI for exon matrices and 1.2-2.4 for
intron matrices, essentially unchanged by conditioning on the (cell, gene) count (< 0.04 bits), and **not
transferable between studies** (shipped tables cost 1.3-2.7 bits/UMI; adaptive Dirichlet-multinomial tables learned
from the file itself match the in-sample static optimum within 0.05-0.1). Per interval nonzero the split is 10-30%
of the exon payload and 17-41% of the intron payload. This is a second sampling-entropy term, physical (fragment
position), irreducible, and it explains why intron matrices are so much more expensive per nonzero than exon
matrices (intron UMIs spread over ~20 intervals per gene with weak positional bias). Decomposition (1) therefore
has three parts for this format: H(gene multiset | π_c) + H(interval | gene) + I(x; π_c | S). Only the last is
the codec's to win.

Likewise, exon ↔ intron conditioning at gene level is worth only 0.06-0.17 bits per intron gene-nonzero once the
intron model conditions on depth and the gene's rate class (m; 0.42 for 5' T cells, ~1 bit only on the tiny tumour
file): in nuclei the intron matrix is the dominant signal and the exon matrix is too sparse to predict it. Plan E3's
anticipated 15-30% joint saving is falsified; the gene-level hierarchy keeps its other advantages (model size,
out-of-sample transfer of gene rates, one code path for both matrices) but is not a bits lever.

## 9c. Evidence log: predictions versus measurements

| prediction (section) | measurement | verdict |
|---|---|---|
| discrete cell type ≈ worthless; latent gain ≤ (k/2) log2 N (§3) | k-topic mixtures +0.1-0.3 bits/nnz (Track D); NMF k=16 gross 0.3-0.4, net +0.13 after 16 B/cell (Idea 4) | supported; k_eff is small or cells too shallow |
| shipped model worth < 0.5% (§4) | c_r2 vs c_r2_nomodel 0.036 bits/nnz (Track C); Idea 2's margins-only rank-1 prior lands on the CV optimum on every file | supported |
| second pass ≥ 0.97 (§4, T4) | 0.976-1.004 general codecs, 0.987-0.996 CM; 0.992-0.999 once names/perm are model-side (Track D) | supported |
| per-chunk orchestrator oracle gap tiny (T4) | 0.003-0.18% among nine coders (Track D) | supported |
| feature order irrelevant to a model coder; permutation is an LZ artefact (§9.2) | row counts re-derive the permutation 13x cheaper; a model coder needs none (Idea 1, 2, Track C) | supported |
| empties at floor (§5, original) | today's coder is 1.5-3.5 bits/nnz above the ambient floor; the ambient code reaches it | corrected (§5) |
| exon ↔ intron joint saving 15-30% (E3) | 0.06-0.17 bits per intron gene-nonzero | falsified (§9b) |
| state information 0.3-3.5 bits/nnz, likely 0.3-1 for PBMC depth (§3) | E1 split-half: 0.04-0.12 bits per cell nonzero on typical files, k_eff 0-62 (median ~20); neighbours carry it, the cell's own half adds ~0; one outlier (+0.35, degenerate library) is burstiness | below the predicted range; the in-file frontier is reached |
| fast model-driven decode ≥ 100 Mnnz/s (§8) | Track B final: 73 Mnnz/s measured native (≈172 at 3 GHz), 34 canonical; Idea 1 exact survival coder 10.7-13.9 at 3 GHz (instruction-bound); E2 scalar ≈ 6 | met only by static-table rANS in native order; the sort is the bottleneck |
| block adaptation costs ≤ 0.1-0.2 bits/nnz (§9.3) | E2 ascending-order block-refit coder within 0.10-0.12 of the static rank-order coder on two large files | supported (partial run) |
| shipped model worth < 0.5% (§4) | Track C final: about 1.5% | roughly supported; small either way |

**Final status (2026-10-22).** The study ended when the account's weekly limit stopped all agents. The curated
synthesis with final numbers, the per-file frontier table and the status of every experiment is
`frontier_fable.md`; this file keeps the derivations.

## 10. Open questions I cannot settle by thought alone

1. H(π_c) and the effective k (E1).
2. Whether high-k "empties" are better coded as cells (Idea 2, running).
3. The exact exon ↔ intron conditional gain for nuclei (Idea 3, running).
4. The speed/size trade of per-symbol versus per-block adaptation under rANS (E2).
5. Whether the pipeline's float members (PSI, heteroplasmy) can be recomputed bit-exactly (E4).
