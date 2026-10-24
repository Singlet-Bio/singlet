# Idea 4 ("pp" variant): low-rank manifold prior, interval-level Poisson-process coder

Status: in progress (2026-10-21/22). Hypothesis H4: a shipped gene-loading model with a per-cell factor vector beats the
static per-gene x depth model by >= 1 bit/nnz on called cells, out of sample, net of side information.

IMPORTANT (duplicate agent): a second Idea-4 session (a fork of the same brief, apparently started from my context at
~22:35) has been writing into the same ownership patterns at the same time. It owns `notes/idea4.md`, `kernels/idea4.h`
(its own `I4M3` codec: gene-level coder with calibrated presence/count tables, codebook theta), `kernels/idea4_core.h`,
`harness/idea4_bench.cpp`, `harness/idea4_tables.cpp`, `tools/idea4_export.py`, `tools/idea4_ce.py`. To avoid
clobbering each other (it overwrote my first `kernels/idea4.h`, `tools/idea4_export.py` and `harness/idea4_bench.cpp`)
my implementation lives in separate files, namespace `cs::ipp`: `kernels/idea4_pp_core.h`, `kernels/idea4_pp.h`,
`kernels/idea4_pp_codec.h`, `harness/idea4_pp_bench.cpp` (-> `bin/idea4_pp_bench.exe`), `tools/idea4_pp_*.py`,
`tools/idea4_pp_folds.txt`, `results/idea4_pp_*`. Shared and untouched since the fork: `tools/idea4_lib.py`,
`tools/idea4_nmf.py`, `tools/idea4_em.cpp` (EM trainer), `corpus_idea4/`. The two implementations are independent
(different probability models, different coders), so agreement between them is a cross-check.

## 1. What is implemented

Counts of a column are an inhomogeneous Poisson process over the interval axis (310,797 human / 386,028 mouse exon
intervals) with rate `lambda_j = Nhat * pi_g(j) * q_j`: `pi` over genes (38,606 / 78,298), `q` = split of a gene's mass over
its intervals. Presence of the next non-zero uses the survival function of that process:
`P(next present = j | last present i) = exp(-(M(j-1)-M(i))) - exp(-(M(j)-M(i)))` with `M` the cumulative hazard
(`M(j) = P_g + mu_g * qcum_j`, gene prefix sums built once per cell, so the interval axis costs nothing); symbols are
cut from a 2^28 scale with a per-symbol floor so every position stays codable, and the decoder finds `j` by a
two-level binary search on `M` followed by an exact check against the same integer cdf (decode cost O(log m) per
non-zero). Counts given presence: 16-symbol table indexed by the half-octave bucket of the interval's own lambda
(+ Elias-gamma escape). Everything on the decode path is integer: Q62 `exp(-x)` by an exact 128-bit Taylor table
(max error one Q30 ulp, monotone, 200 M random pairs), rANS64 (reverse-encoded), integer normalisation with 128-bit
products. `Nhat` is a transmitted scale class (1/8 octave for called cells, 1/4 octave for empties): the Poisson
process needs no exact depth (the sum constraint of the multinomial costs ~0.5 log2(2 pi e N) bits per cell, which is
the same price as coding N exactly).

Model variants (`K` topics): `pi_g(cell) = pi0_g * sum_k theta_k u_gk / Z`, `u` uint8 with quadratic companding per
topic, `theta` = 15 geometric levels (sqrt 2 spacing) relative to the max component, 16 symbols/component under
adaptive frequency models initialised from a trained histogram. `K = 0` is the static per-gene x depth baseline.
Encoder: EM for theta (20 iterations, float), level quantisation + coordinate refinement on the true likelihood.
Empty droplets (and, in the first version, every column whose cost is lower): ambient profile built by the decoder from
the already decoded called cells (+ decoded empties, rebuilt on a doubling schedule) with pseudo-counts toward the shipped
profile. Optional `adapt` mode: the *base* gene profile `pi0` and interval split `q` of called cells are themselves
re-estimated by the decoder from the called cells decoded so far (doubling rebuild schedule, pseudo-counts toward the
shipped model), so the shipped part is only the topic ratios.

## 2. Findings so far

(numbers are filled in below as runs complete)

## 3. Notes / pitfalls

- Bug found with the verify tool: `(cum << 31) / tot` for the per-gene interval cdf overflowed u64 for genes with
  >= 8 intervals (encoder and decoder disagreed through different non-monotone `M`). Fixed with 128-bit arithmetic.
- The OS is shared with ~8 busy agents, so all wall-clock numbers here are noisy; decode timings are min-of-reps.
