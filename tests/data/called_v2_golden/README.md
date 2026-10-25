# `called_v2` golden vectors

Inputs and expected outputs of the `called_v2` cell call (cellcall 2.1: docs/called-v2.md, docs/called-v2-truth.md), shared by
the Python implementation (`python/singlet/v2/cellcall.py`, checked by `tests/python/test_v2_cellcall.py`) and the C++ port
(`include/singlet/pileup/called_v2.h`). Both must reproduce every case with default parameters.

Format id: `singlet-called-v2-golden/1`. All files are plain text with LF line endings (`.gitattributes`).

## Cases

| directory | what it exercises |
|---|---|
| `synth_knee_below_plateau` | Kang-like rank curve (plateau `lower` 7, a debris band at 8-14 UMIs): the unbounded 2.0 knee is 11, so all 85 tested barcodes were retained untested; the bounded knee is 1,088 and only the 30 low-RNA cells are called |
| `synth_tail_extrapolation` | 830 tested barcodes, 30 low-RNA cells below every simulated path: without the tail extrapolation BH passes none of them at FDR 0.001 (fewer than n_tested / 10 at the Monte Carlo floor) |
| `synth_mixed` | the unit-test simulation at small scale; the unbounded knee (176) is again inside the tested range, the bounded one 3,591 |
| `real_hgmm1k_v3` | 45 called cells after subsampling, so the 50 excluded top ranks reach into the tail: an edge case for the curve bounds (knee 386, lower 100) |
| `real_hgmm1k_v3`, `real_neuron1k_v3`, `real_pbmc1k_v3` | 10x benchmark samples (pipeline-benchmark `singlet-modfix-t8` runs on Clipper), subsampled from the stored exon + intron counts (below) |

`case.json` of each case has a one-line `description` and the `source`. `generate.py` rebuilds every case
(`python -P generate.py synthetic OUT`, `python -P generate.py real NAME RUN_DIR OUT`); the real ones need the Singlet output
folders on Clipper (`bench/called_v2_truth/verify.sbatch` runs both).

Real cases are subsampled to stay small: a uniform share of the nonzero barcodes (which keeps the shape of the log-log rank
curve; hgmm1k 5%, neuron1k and pbmc1k 10%, seed 20261025), the genes with the most counts over the kept uncalled barcodes
(hgmm1k 100, the others 150), plus one last row `OTHER` that holds the sum of all other genes, so every barcode keeps its
total. Called barcodes enter the call only through their totals (plateau `lower`
and knee), so their counts are stored as a single entry in the `OTHER` row. The subsampled matrix is the input; the expected
values are the call on that matrix, not on the full sample.

## Files per case

### `counts.mtx`: the input matrix

MatrixMarket `coordinate integer general`, genes x barcodes, 1-based, entries sorted by column then row, no explicit zeros.
This is exactly the matrix `call()` consumes: gene-level counts, exon + intron, after aggregation to genes (the C++
`GeneCounts`). Barcodes with total 0 are left out (they never enter the call). Row names, where known, are in
`case.json["genes"]` (real cases only); they play no part in the call.

### `called.txt`: the stored `called` flag

One line per barcode (column order of `counts.mtx`): `1` = in the stored `called` set, `0` = not.

### `case.json`

| key | content |
|---|---|
| `format` | `singlet-called-v2-golden/1` |
| `protocol` | protocol string passed to the call (`10xv3`: droplet family, validated) |
| `options` | parameter overrides; `{}` everywhere = defaults (plateau `lower`, test floor 50, FDR 0.001, knee retain, 10,000 iterations, seed 20261023) |
| `n_genes`, `n_barcodes`, `n_called` | shape of `counts.mtx`, number of `1` in `called.txt` |
| `expected.lower` | plateau `lower` (integer) |
| `expected.retain_total` | the knee used for `retain` (within DropletUtils' curve bounds), or `null` when there is none (barcodes at or above it get p = 0) |
| `expected.alpha` | Dirichlet-multinomial concentration (MLE over the ambient pool) |
| `expected.n_ambient`, `expected.ambient_umi` | ambient pool: uncalled barcodes with 0 < total <= `lower`, and their summed UMIs |
| `expected.n_tested` | uncalled barcodes with total > max(`lower`, 50) |
| `expected.n_tail` | tested barcodes whose p-value came from the tail extrapolation |
| `expected.n_called_v2`, `expected.n_added` | size of `called_v2`, and how many it adds to `called` |
| `expected.method`, `expected.version` | `singlet-emptydrops`, `2.1` |
| `knee_unbounded` | the 2.0 knee (spline over every total > `lower`), for reference only |
| `tolerance` | how the tests compare (below) |
| `description`, `source`, `genes` | provenance |

JSON has no NaN: a missing number is `null`.

### `expected.tsv`: the tested barcodes

Tab-separated with a header line, one row per tested barcode in column order:

| column | content |
|---|---|
| `col` | 0-based column in `counts.mtx` |
| `total` | the barcode's total count (column sum) |
| `pvalue` | the p-value: 0 at or above `retain_total`; `(n_le + 1) / (niters + 1)` from the Monte Carlo (n_le = simulated paths with log-probability <= the observed one at that total); `min(Phi(z), 1/(niters + 1))` when n_le = 0, with z = (observed - mean) / sd over the 10,000 simulated log-probabilities at that total (population sd); `1/(niters + 1)` above 50,000 UMIs (not simulated) |
| `fdr` | Benjamini-Hochberg over the tested barcodes |
| `called_v2` | `1` when fdr <= 0.001 |

Numbers are the shortest decimal that round-trips a double (`repr`). Barcodes not listed are not tested: their p-value and FDR
are NaN, and `called_v2` = `called`. So the full `called_v2` set = `called` OR (listed with `called_v2` = 1).

## Comparing

* `called_v2`, `lower`, the tested set, `n_*`: exact.
* Monte Carlo p-values (`pvalue >= 1/(niters+1)`): exact, i.e. the same n_le. The C++ port draws with numba's MT19937
  streams, so the simulated paths are the same.
* p-values of 0 (retained) and FDRs of 0: exact.
* Tail p-values and all FDRs, on the log scale: `|ln got - ln expected| <= 1e-5 * max(1, |ln expected|)`
  (`tolerance.pvalue_tail_log_rel`, `tolerance.fdr_log_rel`). A tail p of 1e-150 has z near -26 and changes by |z| times
  any change in z, so tiny differences in `alpha` or in the summation order of the mean and sd give relative differences of
  up to 1e-4 between machines. `alpha` comes from numpy's `polyfit` (LAPACK) and cannot be bit-identical in C++; its 1e-6
  tolerance propagates to ~2e-6 in ln p (2.2e-6 measured on `real_pbmc1k_v3`), hence 1e-5. Everything else above stays
  exact.
* `alpha`: relative 1e-6 (bounded scalar minimisation, `xatol` 1e-4 in log alpha; Linux/numpy 2.5 and Windows/numpy 2.2
  agree to 1e-7). `retain_total`: relative 1e-9 (it is 10^log10(total) of one of the observed totals).

The synthetic ambient droplets are Dirichlet-multinomial (concentration 300; 30 in the knee case, whose ambient droplets hold
only 1-3 UMIs). With purely multinomial (or weakly overdispersed) ambient the alpha MLE sits at or near infinity, the
likelihood is flat and the fitted alpha depends on rounding, which made the expected values differ between machines.

The committed files are the reference, not `generate.py`'s output on any machine: numpy's gamma/multinomial streams differ
slightly between numpy versions, so a rerun elsewhere can give a different (equally valid) synthetic input.

The expected values were generated on Clipper (Linux, numpy 2.5, scipy 1.18, numba 0.68) and checked on Windows (numpy 2.2,
scipy 1.15, numba 0.64).
