# Idea 2: the file is its own model (neighbour-conditioned cells, ambient-coded empties)

Status: in progress, 2026-10-21. Hypothesis under test: **H2** -- adaptive neighbour mixing beats the static
per-feature x depth model by >= 0.5 bits/nnz on called cells, and empties land within 2% of the multinomial floor,
with no shipped model.

Files: `kernels/idea2_core.h` (integer maths + range coder), `kernels/idea2.h` (codec), `kernels/idea2_order.h`
(encoder-side embedding/clustering, floats allowed because the result is transmitted), `harness/idea2_bench.cpp`
(bench driver, `bin/idea2_bench.exe`), `tools/idea2_*.py`, `results/idea2_*.tsv`.

## 1. What is implemented (all decode-side arithmetic is integer)

Stream = header (m, n, nnz, 24 u32 params) + names (zstd-3, as the reference codecs do; not the object of study)
+ one range-coded payload (64-bit carry-less range coder, 16-bit binary probabilities, 32-bit multi-symbol CDFs).

Payload, in decode order:

1. **Column headers, original column order**: per column the UMI total k_j (adaptive Elias-gamma), the
   called-cell flag (adaptive bit, context = quarter-octave bucket of k_j: the flag is almost a deterministic
   function of depth, 200-2000 bits per file), and for called cells the "excess" k_j - nnz_j (context log2 k_j).
   So the split is transmitted as a bitmap costing ~0.02 bits/column.
2. **Per-feature nnz over called cells z_f** (adaptive Elias-gamma, context = previous feature's z; ~1.4 bits per
   *supported* feature). This is the whole "prior": both sides rank features by z and fit, by integer alternating
   projection (`poisson_fit`), the rank-1 Poisson presence model p_{fj} = 1 - exp(-exp(L_f + L_j)) whose fitted
   per-feature and per-cell margins reproduce z_f and nnz_j exactly. Table `ptab` = 1 - exp(-exp(L)) in Q.32 over
   L in [-24, 8) at 1/1024 resolution, built with an integer exp.
3. **Called cells**, coded in a transmitted-implicit order (sort by (cluster label, k_j)): for the top T=4096
   ranked features one binary decision each with probability from the prior or, with `mask != 0`, from an integer
   logistic mixer over {prior, (feature x depth-bucket) counter, (feature x cluster) counter, fast EMA, slow EMA};
   the loop stops once nnz_j hits are found. The tail (ranks >= T) is coded as a *set* with the order-statistic
   coder: the next present rank is the minimum of r iid draws from the static per-rank mass table, decoded by
   binary search over the prefix-mass table (O(log m) per nonzero, no O(m) scan). Counts given presence: the
   excess k_j - nnz_j is known, so each entry's extra UMIs are coded as "u >= t?" decisions against a zero-truncated
   Poisson prior for lambda_{fj} = exp(L_f + L_j), adapted by (lambda-bucket, t) counters and per-feature counters;
   the last entry's count is implied.
4. **Empty droplets**, last: ambient profile = sum of decoded called-cell counts per feature (+1/8 pseudo-UMI),
   mixed with the running sum of decoded empty-droplet UMIs (prefix-mass table rebuilt when the running sum has
   grown by 25%); each droplet of k UMIs is coded as an exact *multiset* of k draws (order statistics with
   lo' = x, i.e. the Multinomial(k, p) likelihood plus nothing else).

Side information in the stream: the split flags, z_f, the labels (log2 K bits per cell when K > 1).
No shipped model (`model_bytes = 0`).

## 2. Stage 1: static per-file prior (`idea2_static`, mask = 0, K = 1), all 14 files

`results/idea2_stage1.tsv` / `.log` (reps = 2, all `matrix_ok = 1`). Bits are payload bits per nonzero of the column
class, *including* the side information attributed to that class (column k_j and flags, z_f table and nnz excess
for cells; k_j for empties). Comparison columns: today's zstd-3 chunks restricted to the same columns (Track D
`track_d_split.tsv`), Track D's static per-feature x depth model with cross-validated parameters (`M2_featdepth_cv`),
the exact multinomial entropy floor at the in-sample MLE ambient profile (`tools/idea2_floor.py`, plus the order-0
entropy of k_j that any code must pay) and Track D's leave-one-droplet-out sequential estimate (`amb_loo` + H0(k)).
Decode speed is single-thread, measured while 7 other agents shared the 4 cores (2-4x noise).

sample	member	codec	tag	cells_b/nnz	(top,tail,val,side_z,hdr)	cells_zstd3	cells_trackD_M2cv	empt_b/nnz(data)	empt_b/nnz(+k)	empt_zstd3	MC_floor	MC_mle	LOO

Reading: on called cells the file-level Poisson rank-1 prior (side info = z_f + per-cell nnz, 0.2-1.4 b/nnz) lands
*exactly* on Track D's static per-feature x depth cross-entropy on every file (within 0.1 b/nnz, better on the two
small tumor files where the CV model overfits), 0.4-1.6 b/nnz below today's zstd-3 chunks except tumor_3p intron
(60k nnz: the z_f table alone is 2.6 b/nnz). On empties the exact multinomial code is 1-6% above the in-sample floor
and already at or below Track D's LOO estimate on 10 of 14 files, with the untuned gamma = 1e5 ambient prior.
The exact floor is for a *known* profile; the gap is the cost of learning the ambient profile from the file.
## 3. Open items
- stage 1 across all 14 files + comparison with Track D floors (M2_featdepth_cv, amb_* floors)
- stage 2/3: K in {8, 32, 128} cluster labels, EMA neighbour counters, orderings (stored / depth / cluster+depth /
  greedy NN chain with explicit permutation)
- empties: two-component ambient (ambient + nearest-cluster profile; depth-dependent weight) vs the floor
- stage 4: speed
