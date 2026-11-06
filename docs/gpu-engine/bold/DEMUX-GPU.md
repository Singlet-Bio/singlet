# Donor demux on the GPU: port sketch

**Status:** design note. Numbers marked "estimate" are not measured on a device. The CPU baseline and the tolerance proposal come from `bench/demux-truth/`.

## What the CPU engine does

`run_demux` in `include/singlet/pileup/donor_demux.h` assigns called cells to donors from the SNP pileup (`--snps`, `--n-donors`). It runs in five stages.

1. **Covered SNPs.** `extract_covered_snps` keeps SNPs with any depth and rebuilds each cell as a sparse list of (SNP, alt depth, total depth). This is serial and allocates one vector triple per cell.
2. **K selection.** With `--n-donors -1` an EM fit with a BIC penalty runs on a 500 cell subsample for K = 1 to 8. This is the slowest part in the host timing study (about 0.68 core-seconds per million read pairs).
3. **Variational Bayes (VB) fit.** Beta posteriors on allele frequency per SNP and donor, and a categorical posterior per cell. Up to 10 random restarts run, one thread each. The best restart then continues with all threads.
4. **Assignment.** A cell is assigned when its top posterior reaches 0.9. Otherwise it is "unassigned".
5. **Ambient back-assignment.** Uncalled barcodes get one E-step with the fitted posteriors.

The engine has no doublet model. `prob_doublet` is always 0.

## What maps to the device

The pileup is already resident on the device. The demux needs only (cell, SNP) counts, so it can stay there.

| Stage | Device mapping | Notes |
|---|---|---|
| Covered SNPs | Mark SNPs with depth by a scatter flag, prefix-sum to a reduced index, then compact the pileup triples in place | No per-cell vectors. Output is CSR over cells with int32 SNP index and uint8 AD and DP |
| M-step | Segmented reduction over the SNP axis: s1[j,k] += q[c,k]·ad, s2[j,k] += q[c,k]·(dp−ad) | One thread block per SNP chunk. Use a CSC copy of the same triples so each SNP row reduces without atomics |
| Digamma cache | One thread per (j,k), three digamma calls | Tiny. ns·K entries per iteration |
| E-step | One warp per cell. Gather d1, d2, d12 for the cell's SNPs, accumulate K log-likelihoods, softmax | The dominant kernel. Memory bound on the gather |
| ELBO | Fused into the E-step and a reduction over (j,k) | Needed only for the convergence test and restart ranking |
| Restarts | Batch all restarts as an extra grid axis | Replaces the host thread pool |
| K selection | Same kernels with K as a batch axis, EM instead of VB | Keep BIC on the host. It needs a few scalars per K |
| Back-assignment | One E-step kernel over uncalled cells | Reuses the E-step |
| PCA seeding and k-means++ | Stay on the host | The seed only needs the per-cell allele fraction of about 10^4 informative SNPs. Download that small matrix once |

Per iteration the work is about 2·nnz·K multiply-adds, where nnz is the number of (cell, SNP) entries. For one sample that is a few million entries and K up to 8. A device iteration should take well under a millisecond. The fit needs about 100 iterations per restart.

## Numerical points

- Reductions in a different order change float sums. Results will not match byte for byte. This is the reason for the tolerance below.
- Keep digamma and lgamma in float32 with the same asymptotic series as the host. Do not switch to a fast approximation without re-running the benchmark.
- Ties and near-ties decide a small fraction of assignments at the 0.9 threshold. Assignment agreement is the metric that will move first.
- The host seeds restarts with `seed + 1000·i + 100000·K`. A port that uses a different random generator will find other local optima on hard mixtures. Compare against the seed variation in the benchmark, not only thread variation.

## Speed expectation (estimate)

- **VB and E-step:** the CPU spends tens of core-seconds per sample here. A device should finish in well under one second. Expect 50 to 200 times on this part.
- **K selection:** also under a second once batched.
- **Covered-SNP extraction:** moves from a serial host pass to a device compaction. It becomes bandwidth bound, about tens of milliseconds.
- **Transfers:** with the pileup resident only the K posteriors per cell (ncells·K floats), the assignments and the per-SNP Beta parameters leave the device. These are megabytes.
- **End to end:** the demux wall time per sample should fall from roughly 30 to 70 seconds on the host to a few seconds. The host core-seconds should fall to near zero, apart from the seeding step and the writers.

Risks to the estimate are the many small kernel launches in the early-stopping loop, and uneven cell depth in the E-step. Check the convergence test on the device with one scalar read per iteration, or every few iterations.

## Acceptance

Judge the port with the mixtures in `bench/demux-truth/` and the thresholds in the benchmark summary (`$W/jobs/demux-truth/results/overnight/summary.md`). The proposal is based on the observed variation between repeats, thread counts and seeds of the CPU engine. The port passes when every metric stays inside its band on all mixtures.

## Order of work

1. Covered-SNP compaction and the E-step on the device, with the CPU M-step. Check agreement.
2. Device M-step, digamma cache and ELBO. Check the ELBO trace against the host.
3. Batched restarts and K selection.
4. Back-assignment. Then add the doublet term, since it reuses the same kernels.
