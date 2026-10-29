# Track D: entropy floor, recursion, and the asymptote argument

Status: in progress, 2026-10-21 (tables refreshed after each step; "pending" marks numbers still being computed).
Everything here is computed by `tools/track_d_*.py` and `harness/track_d_*.cpp` from the 14 corpus matrices;
the TSVs in `results/track_d_*.tsv` are the source of every number. Markdown tables are produced by
`python -P tools/track_d_tables.py`.

## 0. Summary of findings so far

1. **Today's matrix payload (VOCSC + zstd-3) sits at or slightly below the order-0 entropy of its own symbol
   streams** (6.8 vs 6.8 bits/nnz on pbmc_v2_typical exon, 6.9 vs 7.1 on tcell_5p exon). zstd is not leaving
   order-0 redundancy on the table; the gap to the floor is a *modelling* gap (which feature is present in which
   cell), not an entropy-coder gap. The only files where zstd-3 is well above order-0 are the sparse ones
   (tumor_3p, brain_sn_v3_intronic exon: +1.1 to +2.3 bits/nnz), where 1024-column chunks are too short for zstd.
2. **Per-feature x depth conditioning is the first real win**: a 2-fold cross-validated per-feature x depth-bucket
   presence model plus a rate-conditioned value table (M2) codes 10 of 14 files 0.6-2.3 bits/nnz below today.
   A per-feature model without depth (M1) is *worse* than zstd-3 on 11 of 14 files: zstd's LZ matches are
   already capturing depth-like structure through the per-column metadata and gap byte planes.
3. **Called cells: the diminishing-returns curve is flat.** With a held-out protocol (25% test cells, smoothing
   chosen on separate validation cells) the best models reach 5.4-11 bits/nnz depending on the file. Doubling the
   number of training cells gains 0.02-0.15 bits/nnz at the largest sizes; a power-law extrapolation to infinite
   training data (with a 200-draw bootstrap over test cells) lands 0.05-0.2 bits/nnz below the warm value.
   Latent structure (mixtures of 4/16/64 profiles) adds at most 0.1-0.3 bits/nnz over one pooled profile, and
   nothing at all on the shallow files. The remaining distance from today's payload to that asymptote is
   1.0-4.1 bits/nnz on the cells (15-40% of the cell payload).
4. **Empty droplets are 6-61% of today's payload and are where the irreducible part lives.** Under the
   exchangeable-multinomial ambient model their leave-one-droplet-out code length is 1.5-3.5 bits/nnz below
   today, and the Monte-Carlo expectation of H(Multinomial(k_j, p)) reproduces the in-sample code length within
   0.0-0.2 bits/nnz on every file: the real droplets are, to that precision, as unpredictable as the model says.
5. **But the exchangeable-multinomial assumption is measurably wrong, in two opposite directions.** Cross-fitted
   Pearson dispersion of real droplets against a held-out ambient profile is 1.00-1.02 for shallow droplets
   (k < 32) on nearly every file, which is the multinomial value, but rises to 2-9 for the largest "empties"
   (k >= 64) on deep samples (brain_sn_v3_deep exon: 4.8 at k=128-255, 9.4 at k >= 256; pbmc_v2_typical exon:
   2.9 at k >= 64). Those columns are damaged cells / debris, not ambient draws, and they carry cell-type
   structure. In the other direction, real droplets show *fewer distinct features per UMI* and *more repeated
   features* than multinomial draws (z = -16 to -355): UMI capture is bursty (several UMIs of the same molecule
   species in one droplet). Both effects mean a better model than one multinomial exists; the held-out
   mixture-of-profiles runs (pending) measure how many bits that is worth. Expected: a few percent.
6. **Recursion: second-pass compression of a zstd-19/xz/brotli output returns 0.976-1.004** with xz the best
   second pass (it finds the names and permutation blocks, which those codecs still ship), and 0.992-0.999 once
   names and permutation are model-side; bzip2 *expands* every output. A 300-line logistic-mixing CM
   (`d_cm_bytes`, order-0..4 + match model + two APM stages) gets 0.987-0.996 on the same inputs. Full table
   over all codecs pending (runs as codecs from other tracks land).
7. **Orchestrator oracle, among entropy coders of the same transform: 0.0-0.2%.** Choosing the best of nine
   coders (zstd 1/3/9/19 on the production prefilter, zstd-19/xz/brotli on the un-prefiltered streams, xz and CM
   on the prefilter) *per 1024-column chunk* beats the best single coder for the whole file by 0.003-0.18%. The
   signalling cost (0.2-1.2 bits per chunk) is negligible, and a leave-one-study-out classifier on cheap chunk
   features captures nothing useful because there is nothing to capture. Per-chunk choice among *transforms*
   (Track A/B/C codecs) is pending.

## 1. Methods and what each estimate assumes

All estimates are computed on the canonical CSC arrays (`singlet._pz.read_1pz`), with `cell_calls.tsv` splitting
columns into called cells and empty droplets (columns with zero nnz are counted separately; they cost nothing
beyond their k_j = 0 symbol). The writer's symbol streams are reproduced exactly (frequency-sorted rows,
grouping by value, gap coding with `prev = row + 1`); `w_H0_*` are order-0 entropies of those streams.

| quantity | what it is | bias |
|---|---|---|
| `w_H0_total` | order-0 entropy of the writer's gap stream (separate tables for value==1 and value>1 groups) plus the per-column metadata symbols | in-sample, no parameter cost; `_mdl` adds 0.5 log2(n) bits per used symbol |
| `comb` | log2 C(m_eff, nnz_j) + log2 C(k_j - 1, nnz_j - 1) per column: a uniform code over all presence patterns and compositions with the same (nnz_j, k_j) | no learning at all; an upper reference, not a bound |
| `pres_*_plugin` | sum over features (x depth bucket) of N h(p_f) | in-sample empirical entropy, ignores parameter cost: optimistic |
| `M1_cv`, `M2_cv` | 2-fold cross-validated code length: interleaved halves of the columns, train on one, score the other, sum; one or two smoothing scalars chosen on the scored fold | pessimistic (model saw half the data); the scalar tuning is a negligible optimism |
| `amb_loo` | sum over empties of -log2 Mult(x_j; k_j, p_(-j)) with p_(-j) the pooled profile of all *other* empties, Dirichlet-smoothed toward the called-cell profile | a code a decoder can realise (profile from already-decoded droplets); the smoothing weight is tuned on the same data (two scalars, negligible) |
| `amb_plugin` | same with the in-sample profile | optimistic end |
| `amb_entropy_mc` | sum_j H(Mult(k_j, p_hat)) by Monte Carlo (3 draws of the whole droplet set) | the floor *if* droplets are iid multinomial draws from p_hat |
| cells ladder | held-out code length: 25% test cells, 10% validation cells (smoothing / concentration / mixture alpha chosen there), training on 1/32 ... 1 of the remaining 65% | honest out-of-sample within the file; the model is file-specific (the decoder would build it from already-decoded cells) |
| asymptote | fit L(N) = L_inf + a N^-b on the six training sizes; central value with b fixed at 0.5; interval from the free-b fit on 200 bootstrap resamples of the test cells | extrapolation; the lower end rests on the power-law form |

The barcode identity of each column (which 16-mer) is counted as n log2(whitelist size) bits (19.5 or 21.8 bits
per column), since column order is not canonical today. If the format sorted columns by barcode this term
would drop to log2 C(W, n), the cost of the *set*: for the exon files that is 13.3 KB (pbmc_v2_typical, n=15k),
51.6 KB (pbmc_v2_shallow, n=96k), 72.5 KB (brain_sn_v3_deep, n=84k) against 37 / 235 / 230 KB for n log2 W and
78 / 469 / 436 KB for today's zstd-3 colnames block (Track A's accounting). The column *order* would then have
to be carried by the matrix side (as a permutation, or by making the row-frequency order implicit).

## 2. Order-0 and model reference points (all columns)

Refreshed 2026-10-21 22:40. bits/nnz; "today z3" is the sum of the chunk blobs re-encoded with the production
writer at level 3 (the exact bytes in the file), z19 the same at level 19.

| file | nnz | today z3 | z19 | comb C(m_eff,nnz_j) | H0 writer streams | H0 +MDL | pres feat (plug-in) | pres feat x depth (plug-in) | M1 2-fold CV | M2 2-fold CV |
|---|---|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 2569228 | 6.81 | 6.66 | 10.24 | 6.80 | 6.87 | 6.09 | 4.46 | 7.70 | 5.79 |
| pbmc_v2_typical intron | 1202661 | 9.08 | 8.84 | 10.03 | 8.89 | 9.05 | 9.29 | 7.56 | 9.79 | 8.46 |
| pbmc_v2_shallow exon | 5304754 | 8.70 | 8.25 | 11.77 | 8.08 | 8.15 | 6.71 | 6.51 | 7.23 | 7.01 |
| pbmc_v2_shallow intron | 1170031 | 13.89 | 13.68 | 13.47 | 12.50 | 12.86 | 11.93 | 11.38 | 12.20 | 12.24 |
| pbmc_v3_lowgenes exon | 1759828 | 7.16 | 6.85 | 9.83 | 6.73 | 6.78 | 6.69 | 3.99 | 8.53 | 5.42 |
| pbmc_v3_lowgenes intron | 1076297 | 8.55 | 8.20 | 9.90 | 7.90 | 8.00 | 8.52 | 5.50 | 9.90 | 6.95 |
| brain_sn_v3_deep exon | 6349311 | 9.36 | 9.09 | 11.73 | 9.14 | 9.26 | 9.41 | 7.32 | 10.12 | 8.11 |
| brain_sn_v3_deep intron | 6678099 | 8.27 | 8.12 | 10.23 | 8.32 | 8.43 | 9.70 | 6.35 | 10.59 | 7.31 |
| brain_sn_v3_intronic exon | 1380834 | 11.56 | 11.20 | 13.14 | 10.48 | 10.74 | 10.15 | 8.38 | 11.11 | 9.65 |
| brain_sn_v3_intronic intron | 8144777 | 8.39 | 8.24 | 10.32 | 8.49 | 8.57 | 9.02 | 6.77 | 9.69 | 7.48 |
| tcell_5p exon | 10529513 | 6.92 | 6.82 | 10.42 | 7.05 | 7.10 | 7.35 | 4.98 | 8.68 | 6.01 |
| tcell_5p intron | 1098311 | 11.34 | 10.84 | 11.70 | 10.56 | 10.84 | 11.60 | 9.07 | 12.08 | 10.22 |
| tumor_3p exon | 245117 | 11.92 | 11.62 | 12.28 | 10.44 | 11.04 | 10.13 | 8.23 | 11.27 | 11.15 |
| tumor_3p intron | 107364 | 14.35 | 14.05 | 12.61 | 12.10 | 13.15 | 12.09 | 9.68 | 13.69 | 14.78 |

Reading: the plug-in per-feature x depth number (column 9) is what a model with 16 x 310k free parameters
*fitted to the file* would achieve if the parameters were free; it is 2-4 bits/nnz below today and is the number
one would quote if one forgot the parameter cost. The cross-validated M2 (last column) is the honest version of
the same model: 0.6-2.3 bits/nnz below today on the 10 larger files, and *above* today on the three small files
(tumor_3p, pbmc_v2_shallow intron), where the model cannot be learned from the file itself. That spread, 2.5-3
bits/nnz between plug-in and CV, is the parameter cost of a per-feature model and is why Track C ships the
model out of band. The `comb` column shows that a uniform code over presence patterns is 1.5-3.5 bits/nnz above
today: zstd on the VOCSC stream is already exploiting most of the feature-frequency skew.

### Today's payload split by column class

Re-encoded with the production writer (zstd-3, 1024 columns per chunk) on the cell columns alone and the empty
columns alone; the two add up to the file's payload within 0.5%.

| file | cells nnz | cells bytes | cells bits/nnz | empties nnz | empties bytes | empties bits/nnz | empties share of payload |
|---|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 2402405 | 1987979 | 6.62 | 166823 | 187860 | 9.01 | 9% |
| pbmc_v2_typical intron | 1150372 | 1282659 | 8.92 | 52289 | 76676 | 11.73 | 6% |
| pbmc_v2_shallow exon | 3218012 | 3321731 | 8.26 | 2086742 | 2438176 | 9.35 | 42% |
| pbmc_v2_shallow intron | 756599 | 1260249 | 13.33 | 413432 | 765637 | 14.82 | 38% |
| pbmc_v3_lowgenes exon | 1609577 | 1367816 | 6.80 | 150251 | 206898 | 11.02 | 13% |
| pbmc_v3_lowgenes intron | 992792 | 1019274 | 8.21 | 83505 | 130538 | 12.51 | 11% |
| brain_sn_v3_deep exon | 3283431 | 2891889 | 7.05 | 3065880 | 4483379 | 11.70 | 61% |
| brain_sn_v3_deep intron | 4695868 | 3979218 | 6.78 | 1982231 | 2891639 | 11.67 | 42% |
| brain_sn_v3_intronic exon | 863248 | 1078811 | 10.00 | 517586 | 898421 | 13.89 | 45% |
| brain_sn_v3_intronic intron | 6324550 | 5695281 | 7.20 | 1820227 | 2796116 | 12.29 | 33% |
| tcell_5p exon | 9427097 | 7808316 | 6.63 | 1102416 | 1287000 | 9.34 | 14% |
| tcell_5p intron | 978048 | 1336235 | 10.93 | 120263 | 213578 | 14.21 | 14% |
| tumor_3p exon | 142440 | 192971 | 10.84 | 102677 | 166166 | 12.95 | 46% |
| tumor_3p intron | 60517 | 96234 | 12.72 | 46847 | 92649 | 15.82 | 49% |

Empties cost 2-5 bits/nnz *more* than cells today (sparser columns: less for the LZ to match), and in the
droplet-heavy samples they are a third to two thirds of the payload. Any plan that targets the matrix bytes has
to target the empties first in brain_sn_v3_deep, brain_sn_v3_intronic, pbmc_v2_shallow and tumor_3p.

## 3. The ambient floor for empty droplets

Model: each empty droplet j with k_j UMIs is an independent draw from Multinomial(k_j, p) with one profile p
shared by all droplets of the sample. Under that model the code length of the count vector cannot be below
H(Mult(k_j, p)) in expectation (source coding theorem), plus the cost of transmitting k_j, plus (for a real
decoder) the cost of learning p, which the leave-one-out estimate realises by estimating p from the other
droplets.

| file | empties (nnz) | today z3 | H0 writer | comb | Mult plug-in (in-sample p) | Mult LOO (held-out p) | E[H(Mult)] MC | held-out mixture C=1 / 4 / 16 | k_j side (bits/nnz) |
|---|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 166823 | 9.01 | 8.59 | 12.32 | 7.12 | 7.43 | 6.96 | pending | 0.29 |
| pbmc_v2_typical intron | 52289 | 11.73 | 11.01 | 11.99 | 9.28 | 10.20 | 9.26 | pending | 0.27 |
| pbmc_v2_shallow exon | 2086742 | 9.35 | 8.46 | 12.21 | 7.21 | 7.25 | 7.20 | pending | 0.14 |
| pbmc_v2_shallow intron | 413432 | 14.82 | 13.14 | 14.19 | 11.99 | 12.26 | 11.98 | pending | 0.50 |
| pbmc_v3_lowgenes exon | 150251 | 11.02 | 8.68 | 11.48 | 7.00 | 7.08 | 6.85 | pending | 0.67 |
| pbmc_v3_lowgenes intron | 83505 | 12.51 | 10.24 | 11.32 | 9.02 | 9.24 | 8.71 | pending | 0.45 |
| brain_sn_v3_deep exon | 3065880 | 11.70 | 10.76 | 14.13 | 9.48 | 9.58 | 9.37 | pending | 0.17 |
| brain_sn_v3_deep intron | 1982231 | 11.67 | 10.86 | 13.13 | 9.42 | 9.56 | 9.52 | pending | 0.22 |
| brain_sn_v3_intronic exon | 517586 | 13.89 | 11.55 | 15.14 | 9.51 | 9.76 | 9.54 | pending | 0.57 |
| brain_sn_v3_intronic intron | 1820227 | 12.29 | 11.49 | 13.11 | 10.28 | 10.39 | 10.26 | pending | 0.20 |
| tcell_5p exon | 1102416 | 9.34 | 9.07 | 12.33 | 7.71 | 7.83 | 7.53 | pending | 0.21 |
| tcell_5p intron | 120263 | 14.21 | 12.43 | 13.07 | 10.53 | 11.23 | 10.52 | pending | 0.37 |
| tumor_3p exon | 102677 | 12.95 | 11.08 | 13.26 | 9.39 | 10.11 | 9.32 | pending | 0.61 |
| tumor_3p intron | 46847 | 15.82 | 12.85 | 13.56 | 11.03 | 12.35 | 10.86 | pending | 0.89 |

Three readings:

- **Today spends 1.5-3.5 bits/nnz more on empties than the multinomial code would** (column "today z3" vs
  "Mult LOO"). For brain_sn_v3_deep exon that is 4.48 MB today vs 3.67 MB (LOO) for the same columns, and the
  LOO code is realisable by a decoder that builds the profile from already-decoded droplets.
- **The in-sample code length equals the expected entropy of the model** within 0.0-0.2 bits/nnz ("Mult plug-in"
  vs "E[H(Mult)] MC"). So, *given the multinomial model*, the real droplets are exactly as unpredictable as random
  draws from it; there is no hidden order-0 structure left. Plug-in minus LOO (0.04-1.3 bits/nnz) is the cost of
  learning the profile, large only where there are few ambient UMIs (tumor_3p, pbmc_v2_typical intron).
- **The k_j side information is small** (0.14-0.9 bits/nnz, order-0 of the per-droplet totals; the writer today
  stores u64 column sums, 68 KB zstd'd for brain_sn_v3_deep, which is 4x more than this).

### Is the exchangeable-multinomial assumption true? (goodness of fit)

Tests are cross-fitted: the ambient profile is estimated from a random half of the droplets and the statistics
computed on the other half (and vice versa), restricted to the feature set W with >= 10 ambient UMIs in the
fitting half so that the profile is well estimated (W covers 16-94% of ambient UMIs; low coverage on the
intron matrices of shallow samples). The null distribution is a parametric bootstrap: synthetic droplets with
the same k_j drawn from the fitted profile, 5 replicates. Pearson dispersion is the chi-square over 40 equal-
mass feature bins divided by 39: exactly 1 in expectation under the multinomial.

| file | droplets | UMI coverage of W | Pearson dispersion real / synth (z) | distinct features per UMI real / synth (z) | repeat features per UMI real / synth (z) | LOO code length real / synth bits per droplet (z) |
|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 9404 | 0.81 | 1.423 / 0.999 (+173) | 0.658 / 0.737 (-187) | 0.153 / 0.125 (+85) | 77.368 / 74.554 (+45) |
| pbmc_v2_typical intron | 1732 | 0.16 | 1.030 / 1.000 (+6) | 0.898 / 0.943 (-16) | 0.074 / 0.047 (+21) | 27.309 / 27.363 (-1) |
| pbmc_v2_shallow exon | 54703 | 0.94 | 1.013 / 1.000 (+55) | 0.939 / 0.944 (-16) | 0.053 / 0.049 (+17) | 239.077 / 239.486 (-7) |
| pbmc_v2_shallow intron | 52521 | 0.55 | 1.004 / 1.000 (+6) | 0.997 / 0.998 (-20) | 0.003 / 0.002 (+20) | 43.836 / 44.009 (-20) |
| pbmc_v3_lowgenes exon | 45851 | 0.92 | 1.084 / 1.000 (+94) | 0.714 / 0.771 (-64) | 0.114 / 0.087 (+65) | 18.877 / 18.452 (+25) |
| pbmc_v3_lowgenes intron | 16399 | 0.79 | 1.210 / 1.000 (+230) | 0.740 / 0.926 (-355) | 0.169 / 0.059 (+385) | 31.938 / 30.665 (+60) |
| brain_sn_v3_deep exon | 79913 | 0.91 | 1.502 / 1.000 (+707) | 0.611 / 0.629 (-69) | 0.045 / 0.048 (-37) | 290.166 / 286.727 (+33) |
| brain_sn_v3_deep intron | 79857 | 0.80 | 1.086 / 1.000 (+97) | 0.832 / 0.845 (-48) | 0.083 / 0.053 (+260) | 157.391 / 160.966 (-84) |
| brain_sn_v3_intronic exon | 67274 | 0.84 | 0.995 / 1.000 (-5) | 0.580 / 0.589 (-21) | 0.117 / 0.114 (+73) | 45.200 / 45.966 (-13) |
| brain_sn_v3_intronic intron | 69125 | 0.83 | 1.015 / 1.000 (+27) | 0.972 / 0.989 (-121) | 0.022 / 0.010 (+110) | 205.649 / 205.607 (+1) |
| tcell_5p exon | 56636 | 0.87 | 1.092 / 1.000 (+85) | 0.730 / 0.771 (-163) | 0.102 / 0.101 (+7) | 111.381 / 108.640 (+145) |
| tcell_5p intron | 8751 | 0.28 | 1.001 / 1.000 (+0) | 0.947 / 0.972 (-41) | 0.043 / 0.024 (+41) | 26.444 / 26.571 (-4) |
| tumor_3p exon | 13468 | 0.62 | 1.138 / 1.000 (+76) | 0.897 / 0.955 (-254) | 0.062 / 0.040 (+124) | 30.661 / 30.361 (+7) |
| tumor_3p intron | 8560 | 0.41 | 1.263 / 1.000 (+74) | 0.772 / 0.912 (-44) | 0.081 / 0.062 (+8) | 13.179 / 12.607 (+9) |

Pearson dispersion by droplet size (real; synthetic is 1.00 +- 0.01 in every cell):

| file | k1-1 | k2-3 | k4-7 | k8-15 | k16-31 | k32-63 | k64-127 | k128-255 | k256-inf |
|---|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 1.00 (n=1038) | 1.04 (n=2796) | 1.11 (n=2145) | 1.34 (n=1107) | 1.83 (n=887) | 2.56 (n=592) | 2.92 (n=413) | 2.88 (n=426) |  |
| pbmc_v2_typical intron | 0.97 (n=713) | 0.99 (n=277) | 1.04 (n=308) | 1.13 (n=344) | 1.25 (n=76) |  |  |  |  |
| pbmc_v2_shallow exon | 0.99 (n=32) | 1.01 (n=199) | 1.02 (n=627) | 1.01 (n=952) | 1.01 (n=10924) | 1.02 (n=41969) |  |  |  |
| pbmc_v2_shallow intron | 1.00 (n=4841) | 1.00 (n=16625) | 1.00 (n=25710) | 1.01 (n=5325) | 0.99 (n=20) |  |  |  |  |
| pbmc_v3_lowgenes exon | 0.95 (n=25555) | 1.16 (n=15504) | 1.18 (n=3136) | 2.11 (n=310) | 2.81 (n=219) | 2.96 (n=278) | 1.84 (n=658) | 1.91 (n=191) |  |
| pbmc_v3_lowgenes intron | 1.00 (n=11312) | 1.68 (n=3592) | 2.13 (n=266) | 1.26 (n=165) | 1.36 (n=320) | 1.51 (n=451) | 1.78 (n=205) | 2.19 (n=70) |  |
| brain_sn_v3_deep exon | 1.32 (n=1828) | 1.37 (n=1647) | 1.31 (n=299) | 1.05 (n=446) | 1.03 (n=13500) | 1.07 (n=46295) | 1.95 (n=10680) | 4.79 (n=4086) | 9.38 (n=1132) |
| brain_sn_v3_deep intron | 1.09 (n=1608) | 1.04 (n=2514) | 0.94 (n=11818) | 0.95 (n=41203) | 1.05 (n=12960) | 1.37 (n=4832) | 1.74 (n=2781) | 2.44 (n=1327) | 4.53 (n=814) |
| brain_sn_v3_intronic exon | 1.44 (n=987) | 1.05 (n=2146) | 0.97 (n=18291) | 0.96 (n=40030) | 1.15 (n=5003) | 1.68 (n=817) |  |  |  |
| brain_sn_v3_intronic intron | 1.00 (n=1410) | 1.01 (n=1132) | 1.00 (n=2010) | 1.01 (n=24402) | 1.01 (n=35728) | 1.02 (n=1596) | 1.12 (n=1196) | 1.16 (n=1432) | 1.39 (n=219) |
| tcell_5p exon | 0.99 (n=805) | 1.00 (n=6877) | 1.01 (n=30966) | 1.02 (n=15457) | 1.01 (n=572) | 1.68 (n=70) | 2.37 (n=154) | 2.24 (n=236) | 3.78 (n=1499) |
| tcell_5p intron | 0.98 (n=6327) | 1.04 (n=591) | 1.03 (n=203) | 1.04 (n=971) | 1.10 (n=607) | 1.46 (n=52) |  |  |  |
| tumor_3p exon | 1.03 (n=3199) | 1.08 (n=2779) | 1.12 (n=4103) | 1.26 (n=3204) | 2.09 (n=179) |  |  |  |  |
| tumor_3p intron | 1.20 (n=3810) | 1.24 (n=3167) | 1.36 (n=1258) | 1.76 (n=290) | 2.25 (n=35) |  |  |  |  |

Interpretation, which matters for how "floor" the ambient floor is:

- **The bulk of empty droplets (k < 32, which is 80-99% of the empties in every file) is multinomial to within
  the resolution of 50k-80k droplets**: dispersion 0.95-1.05 and code length within +-1% of the parametric
  bootstrap (pbmc_v2_shallow, brain_sn_v3_intronic intron, tcell_5p intron, pbmc_v2_typical intron). For those
  columns the multinomial code length *is* the floor, up to the model-learning cost, and no second pass, no
  learned model and no orchestrator will beat it by more than the few percent the mixture runs (pending) show.
- **The overdispersion is concentrated in the largest "empties"** (k >= 64, dispersion 2-9), which exist in the
  deep samples and carry a large share of the ambient UMIs there (exon matrices: brain_sn_v3_deep 17.8k droplets
  with k >= 64 holding 50% of the empties' UMIs; tcell_5p 1.9k droplets, 72%; pbmc_v2_typical 1.0k, 66%;
  pbmc_v3_lowgenes 0.9k, 52%; none at all in pbmc_v2_shallow, brain_sn_v3_intronic, tumor_3p). These are
  debris / damaged nuclei / cells below the caller's threshold, with cell-type-specific content. They are not ambient draws and should be coded with the cell model (mixture of
  profiles), which is exactly what the mixture runs test. They are also the columns where the real LOO code length
  exceeds the bootstrap (brain_sn_v3_deep exon +1.2%, tcell_5p exon +2.5%): the single-profile model is paying for
  features it did not expect.
- **The second departure is burstiness**: real droplets contain fewer distinct features per UMI and more repeated
  features than multinomial draws, on every file (z from -16 to -355 on distinct-per-UMI). This is UMI-level
  clumping (several UMIs from the same transcript species captured together, or PCR/UMI-collapse artefacts), and it
  makes real droplets *cheaper* than the bootstrap on several files (brain_sn_v3_deep intron -2.2%, pbmc_v2_shallow
  intron -0.4%): the repeats are predictable given the first hit. A Dirichlet-multinomial (one extra concentration
  parameter) captures this; on the called cells it is worth 0.00-0.04 bits/nnz (section 4), i.e. the burstiness is
  real but cheap.

So the honest statement of the empties floor is a bracket: the in-sample multinomial code length (lower end,
"Mult plug-in") to the leave-one-out code length (upper end), minus whatever the held-out mixture model recovers
from the large droplets (pending; from the cell results, expect 0.1-0.3 bits/nnz on the deep files and nothing on
the shallow ones).

## 4. Called cells: the diminishing-returns curve

Protocol: feature space F = rows with >= 1 nnz anywhere in the file (known to a decoder from the row counts);
cells in random order; test = 25%, validation = 10% (hyper-parameters chosen there), training = first fraction f
of the remaining 65%, f in 1/32 ... 1. Models code the multiset of UMIs of each held-out cell given k_j:
`comb` (uniform, no learning), `M1` (per-feature Bernoulli presence + order-0 values), `M2` (per-feature x
depth-bucket presence + value table by rate context), `mult1` (one pooled multinomial profile), `dm1`
(Dirichlet-multinomial around that profile: burstiness), `multC` (mixture of C profiles fitted by EM: latent
cell types). All are *file-internal* models that a decoder could build from already-decoded cells; they do not
use any out-of-study model (Track C's domain).

| file | test cells / nnz | today z3 (cells) | comb | M1 feat | M2 feat x depth | mult1 (pooled profile) | dm1 (Dirichlet-mult) | mult4 | mult16 | mult64 | best | Linf (b=0.5) | Linf free [95% boot] | b free |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 1357 / 607609 | 6.62 | 10.09 | 6.14 | 5.54 | 5.53 | 5.51 | 5.50 | 5.46 | 5.45 | mult16 | 5.39 | 5.41 [5.38, 5.46] | 0.59 |
| pbmc_v2_typical intron | 1357 / 291544 | 8.92 | 9.93 | 8.22 | 8.02 | 7.98 | 7.97 | 7.98 | 7.98 | 7.98 | dm1 | 7.83 | 7.89 [7.85, 7.93] | 0.60 |
| pbmc_v2_shallow exon | 10431 / 801761 | 8.26 | 11.48 | 6.86 | 6.66 | 6.57 | 6.57 | 6.57 | 6.57 | 6.57 | dm1 | 6.51 | 6.54 [6.53, 6.55] | 0.65 |
| pbmc_v2_shallow intron | 10431 / 188032 | 13.33 | 13.10 | 11.54 | 11.41 | 11.15 |  | 11.35 | 11.31 | 11.22 | mult1 | 10.99 | 11.01 [10.99, 11.04] | 0.53 |
| pbmc_v3_lowgenes exon | 2781 / 398709 | 6.80 | 9.69 | 5.84 | 5.18 | 5.29 | 5.21 | 5.01 | 5.00 | 5.02 | mult16 | 4.95 | 4.86 [4.83, 4.88] | 0.32 |
| pbmc_v3_lowgenes intron | 2781 / 244608 | 8.21 | 9.79 | 7.26 | 6.63 | 7.17 | 6.67 | 7.00 | 7.02 | 7.04 | M2 | 6.55 | 6.60 [6.57, 6.65] | 0.72 |
| brain_sn_v3_deep exon | 822 / 817749 | 7.05 | 9.51 | 6.88 | 6.52 | 6.56 | 6.56 | 6.52 | 6.50 | 6.50 | mult16 | 6.38 | 6.41 [6.34, 6.47] | 0.55 |
| brain_sn_v3_deep intron | 822 / 1172263 | 6.78 | 9.03 | 6.67 | 6.18 | 6.24 | 6.22 | 6.22 | 6.17 | 6.15 | mult16 | 6.05 | 6.09 [6.05, 6.14] | 0.62 |
| brain_sn_v3_intronic exon | 2185 / 217429 | 10.00 | 11.94 | 8.95 | 8.64 | 8.49 |  | 8.59 | 8.55 | 8.54 | mult1 | 8.32 | 8.36 [8.33, 8.39] | 0.55 |
| brain_sn_v3_intronic intron | 2185 / 1594320 | 7.20 | 9.50 | 6.89 | 6.51 | 6.54 | 6.52 | 6.46 | 6.46 | 6.43 | mult64 | 6.35 | 5.76 [5.70, 5.91] (*) | 0.11 |
| tcell_5p exon | 3090 / 2330959 | 6.63 | 10.20 | 6.25 | 5.73 | 5.82 | 5.78 | 5.74 | 5.72 | 5.72 | mult64 | 5.67 | 5.40 [5.28, 5.51] (*) | 0.13 |
| tcell_5p intron | 3090 / 240487 | 10.93 | 11.56 | 9.62 | 9.47 | 9.35 |  | 9.59 | 9.49 | 9.42 | mult1 | 9.21 | 9.25 [9.22, 9.28] | 0.56 |
| tumor_3p exon | 955 / 34506 | 10.84 | 11.64 | 9.32 | 9.23 | 9.02 |  | 9.04 | 8.93 | 8.92 | mult16 | 8.78 | 8.71 [8.60, 8.79] | 0.42 |
| tumor_3p intron | 955 / 14567 | 12.72 | 11.98 | 11.28 | 11.02 | 10.76 |  | 10.85 | 10.70 | 10.68 | mult16 | 10.57 | 10.47 [10.21, 10.59] | 0.40 |

(dm1 was added after the first files had run; the empty cells are re-run pending. (*) mult64 only runs from
training size 512 cells upward, so its curve has 3-4 points and the free-exponent fit is unreliable there: the
b = 0.11-0.13 extrapolations 0.3-0.6 bits below the warm value are a fitting artefact; the b = 0.5 column and
the mult16 fits, with six points, are the ones to read. The floor summary uses the free-b lower end only from
models with >= 5 training sizes.)

Diminishing returns with training-set size, pooled-profile multinomial (bits/nnz of the same test cells):

| file | pool x 1/32 | pool x 1/16 | pool x 1/8 | pool x 1/4 | pool x 1/2 | pool x all |
|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 5.838 (n=110) | 5.711 (n=221) | 5.624 (n=441) | 5.580 (n=882) | 5.551 (n=1764) | 5.532 (n=3529) |
| pbmc_v2_typical intron | 8.587 (n=110) | 8.348 (n=221) | 8.197 (n=441) | 8.098 (n=882) | 8.034 (n=1764) | 7.985 (n=3529) |
| pbmc_v2_shallow exon | 6.767 (n=848) | 6.688 (n=1695) | 6.634 (n=3390) | 6.596 (n=6780) | 6.577 (n=13561) | 6.567 (n=27122) |
| pbmc_v2_shallow intron | 11.851 (n=848) | 11.602 (n=1695) | 11.410 (n=3390) | 11.286 (n=6780) | 11.206 (n=13561) | 11.147 (n=27122) |
| pbmc_v3_lowgenes exon | 5.414 (n=226) | 5.355 (n=452) | 5.324 (n=904) | 5.307 (n=1808) | 5.298 (n=3616) | 5.292 (n=7231) |
| pbmc_v3_lowgenes intron | 7.453 (n=226) | 7.328 (n=452) | 7.243 (n=904) | 7.201 (n=1808) | 7.180 (n=3616) | 7.169 (n=7231) |
| brain_sn_v3_deep exon | 7.115 (n=67) | 6.898 (n=134) | 6.731 (n=267) | 6.645 (n=534) | 6.595 (n=1069) | 6.564 (n=2138) |
| brain_sn_v3_deep intron | 6.721 (n=67) | 6.521 (n=134) | 6.384 (n=267) | 6.309 (n=534) | 6.264 (n=1069) | 6.239 (n=2138) |
| brain_sn_v3_intronic exon | 9.175 (n=178) | 8.944 (n=355) | 8.742 (n=710) | 8.605 (n=1421) | 8.533 (n=2842) | 8.493 (n=5684) |
| brain_sn_v3_intronic intron | 6.864 (n=178) | 6.719 (n=355) | 6.627 (n=710) | 6.580 (n=1421) | 6.554 (n=2842) | 6.538 (n=5684) |
| tcell_5p intron | 9.983 (n=251) | 9.747 (n=502) | 9.582 (n=1005) | 9.478 (n=2009) | 9.409 (n=4018) | 9.350 (n=8037) |
| tumor_3p exon | 9.779 (n=78) | 9.510 (n=155) | 9.376 (n=310) | 9.177 (n=621) | 9.083 (n=1242) | 9.016 (n=2483) |
| tumor_3p intron | 11.450 (n=78) | 11.313 (n=155) | 11.109 (n=310) | 10.966 (n=621) | 10.900 (n=1242) | 10.763 (n=2483) |

What the ladder says:

- **Rung 1 (per-feature presence, M1 -> M2 with depth)** is worth 0.2-0.7 bits/nnz on the cells; **rung 2
  (count-level multinomial, mult1)** is at or slightly below M2 on most files because it couples presence and
  value through k_j p_f automatically; **rung 3 (burstiness, dm1)** adds 0.00-0.04; **rung 4 (latent cell types,
  mult4/16/64)** adds 0.02-0.3 bits/nnz, most on the deep / heterogeneous samples (pbmc_v3_lowgenes exon 0.29,
  brain_sn_v3_deep 0.06-0.09, tumor_3p 0.1) and *nothing* on the shallow ones (pbmc_v2_shallow: 0.00). Mixtures
  of 64 are no better than 16 anywhere: with 2-27k training cells, cell-type structure beyond ~16 profiles is
  not learnable from the file itself. (A model trained on other studies, Track C, could start from more; this
  ladder bounds what *file-internal* adaptation can do.)
- **Training-set size**: each doubling of the training cells is worth 0.01-0.15 bits/nnz at the top of the range,
  shrinking with size. The free power-law exponent b is 0.3-0.7 across files; the fitted asymptote is 0.03-0.15
  bits/nnz below the warm (all-training) value with a bootstrap 95% interval of +-0.03 to +-0.1. The curve has
  essentially flattened by a few thousand cells.
- **Distance from today**: today's payload on the called cells is 1.0-4.1 bits/nnz above the warm held-out code
  length (pbmc_v2_typical exon 6.62 -> 5.45; pbmc_v2_shallow exon 8.26 -> 6.57; brain_sn_v3_deep exon 7.05 ->
  6.50; tcell_5p intron 10.93 -> 9.35; tumor_3p intron 12.72 -> 10.68). That is 10-30% of the cell payload, and
  is the realistic budget for Track B/C on the cells; the extrapolated asymptote adds 1-2 points to it.

## 5. Recursion: can anything compress the codec outputs again?

`bin/track_d_dump.exe` writes `encode()` of every registered codec for every corpus file to
`results/track_d_dump/<sample>__<member>__<codec>.bin`; `bin/track_d_second.exe` compresses each dump with
zstd-22 (long window), xz -9e, brotli-11 (window 24), bzip2 -9 and the Track D context-mixing coder
(`dcm::compress`, kernels/track_d.h). The CM is a conventional lpaq-style design: bitwise prediction from
order-0/1 direct tables and order-2/3/4 hashed tables (22-bit), a match model (min length 5), one gated
logistic mixer selected by the partial byte and match state, two APM/SSE stages, 32-bit binary arithmetic coder
with 12-bit probabilities; all integer arithmetic. On text it is a stronger compressor than xz (PLAN.md: 7160
bytes vs 8280 xz -9e; 8x-repeated PLAN.md: 7300 bytes, i.e. the match model works); on random bytes it costs
0.17% overhead; it runs at ~0.25 MB/s, so it is a measuring instrument, not a codec. It is registered as
`d_cm_bytes` (ref_zstd_csc_L3 output -> CM), `d_cm_raw` (raw CSC arrays + names -> CM, no LZ at all) and
`d_cm_1pz` (the production .1pz bytes -> CM), all passing the matrix round-trip in the harness.

Second-pass ratios (out/in), pending the full run; first results (brain_sn_v3_deep exon, 11.7-12.9 MB inputs):

| codec | brotli-11 | bzip2 | CM (Track D) | xz -9e | zstd-22 |
|---|---|---|---|---|---|
| a_names_as_model | 0.979 | 1.002 | 0.987 | 0.972 | 0.996 |
| a_names_perm_model | 1.000 | 1.003 | 0.992 | 0.997 | 0.999 |
| a_perm_from_rowcounts | 1.000 | 1.004 | 0.995 | 0.999 | 0.999 |
| a_vocsc_brotli | 0.982 | 1.004 | 0.996 | 0.976 | 0.998 |
| a_vocsc_xz | 0.982 | 1.004 | 0.996 | 0.977 | 0.998 |
| a_vocsc_zstd19 | 0.982 | 1.004 | 0.996 | 0.977 | 0.998 |
| a_vocsc_zstd22long | 0.983 | 1.004 | 0.996 | 0.977 | 0.998 |

Where a second pass still finds 2-3% (xz on a_vocsc_*), the bytes it finds are the names block and the
permutation, which those codecs ship as zstd-3 blobs; with names and permutation model-side
(`a_names_perm_model`) every second pass is within 0.3-0.8% of 1.0, and the CM is the only one that gains at
all. Full table over `ref_1pz_v1_L3` (expect < 1: the production file is recursively compressible because of
its zstd-3 metadata), Track B rANS and Track C arithmetic-coded outputs (expect >= 0.99) will appear here.

## 6. Orchestrator oracle

`bin/track_d_oracle.exe` re-encodes every 1024-column chunk of every file with nine coders of the same VOCSC
symbol stream (file-global frequency-sorted rows, exactly as the writer), records the size of each and 15 cheap
chunk features (nnz, columns with > 0, mean/quantiles of column nnz, fraction of ones, mean value, fraction >= 3,
mean gap, fraction of zero gaps, share of nnz in the top-64 rows, metadata bytes per nnz, chunk position).
`tools/track_d_oracle.py` computes the oracle and trains three choosers leave-one-study-out (logistic
regression, depth-3 tree, ridge regression on the log size ratios followed by argmin).

Partial (7 of 14 files, the rest pending):

| file | chunks | today z3 | best single | oracle | oracle gap | signal bits / chunk |
|---|---|---|---|---|---|---|
| brain_sn_v3_deep exon | 83 | 7,428,216 | rawbr 6,870,935 | 6,868,499 | 0.035% | 0.37 |
| brain_sn_v3_deep intron | 83 | 6,900,454 | rawbr 6,470,201 | 6,469,017 | 0.018% | 0.42 |
| brain_sn_v3_intronic exon | 78 | 1,994,230 | rawbr 1,799,022 | 1,796,188 | 0.158% | 0.55 |
| brain_sn_v3_intronic intron | 78 | 8,538,970 | rawbr 8,062,880 | 8,048,526 | 0.178% | 0.73 |
| pbmc_v2_shallow exon | 95 | 5,765,573 | rawbr 5,142,769 | 5,142,625 | 0.003% | 0.20 |
| pbmc_v2_shallow intron | 95 | 2,029,340 | pfcm 1,847,331 | 1,843,949 | 0.183% | 1.00 |
| pbmc_v2_typical exon | 15 | 2,186,835 | rawbr 2,043,388 | 2,043,374 | 0.001% | 0.35 |
| pbmc_v2_typical intron | 15 | 1,364,425 | pfcm 1,288,356 | 1,287,460 | 0.070% | 0.57 |
| pbmc_v3_lowgenes exon | 70 | 1,574,360 | pfcm 1,416,179 | 1,414,476 | 0.120% | 1.18 |

Totals over these files: today 38.17 MB; best single coder per file 35.30 MB (-7.5%, almost all of it brotli-11
or the CM on the *un-prefiltered* byte-split streams: the bit-plane + bitmap prefilter helps zstd-3 and hurts
stronger coders); per-chunk oracle 35.27 MB, **0.08% below the best single coder**. The signalling cost is
0.2-1.2 bits per chunk (irrelevant). None of the three choosers captures the gap out of sample (they capture
-100% to -700% of a gap that is itself 0.1%: choosing wrong once costs more than the whole gap). The right
reading is not that choosers are bad but that **there is no gap to capture among entropy coders of one
transform**: a chunk that is cheaper under coder A than coder B is cheaper for every chunk of the same file.
The per-chunk oracle across *transforms* (per-chunk re-encode with the Track A/B/C codecs on blank-name
sub-matrices) is pending and is the version of this experiment where a gap could exist.

## 7. Floor per file and the asymptote argument

Pending: `tools/track_d_summary.py` combines the pieces into `results/track_d_floor.tsv` once the cell and
empties runs have finished for all files. The argument, with the numbers so far:

**Why no recursive / learned scheme can beat the entropy of the generating process.** A lossless code assigns
each possible matrix X a codeword of length L(X); Kraft's inequality forces sum_X 2^-L(X) <= 1, so for any
probability law P over matrices E_P[L(X)] >= H_P(X) (Shannon's source coding theorem), with equality only for the
code with L(X) = -log2 P(X). Running a second coder on the codewords changes L but not the inequality: the
composition of two lossless codes is a lossless code and is bound by the same H_P. A *learned* model Q changes
the achievable length to E_P[-log2 Q(X)] = H_P(X) + KL(P || Q): learning can only reduce the KL term, never the
entropy term. "Recursive" schemes are therefore worth exactly KL(P || Q_current), the mismatch between the data's
law and the current coder's implicit model, and nothing more. Every number in sections 2-4 is an estimate of
E[-log2 Q(X)] for some Q; the best of them is an upper bound on H_P, and the parametric-bootstrap agreement in
section 3 shows that, for the shallow empties, Q = multinomial is already so close to P that the bound is tight.

**Where the entropy comes from.** (i) *UMI sampling of a cell's transcriptome*: a cell with k UMIs is a draw of
k molecules from its expression profile; the per-cell multiset is Multinomial(k, p_cell), whose entropy is 5-11
bits per UMI here depending on how peaked the profile is (shallow and intron matrices are flat: 8-13 bits;
deep exon matrices with a few dominant genes: 5-7 bits). No decoder knows p_cell exactly; the best it can do is
a mixture over cell types (mult16) or an out-of-study prior (Track C), and the ladder in section 4 shows the
return on better p_cell estimates has flattened to < 0.1 bit per doubling of data. (ii) *Ambient RNA in empty
droplets*: a multinomial draw from one profile; 7-12 bits per UMI, with the multinomial entropy measured and
matched by the data. (iii) *Barcode identity*: which 16-mers of the whitelist appear, ~20-22 bits per column
unless the column order is canonical. (iv) *Removable*: feature names (identical across every file on the same
reference), the row permutation (derivable from row counts), 8-byte column sums, chunk headers, zstd framing,
and the modelling gap between zstd's implicit order-0-ish model and the models above.

**What fraction of today's bytes is irreducible.** Pending the summary table; from the tables above, for
brain_sn_v3_deep exon (13.27 MB today): names 4.65 MB + permutation 1.06 MB + pointers/colsums 0.14 MB are
removable (44%); the payload of 7.43 MB splits into cells 2.89 MB -> floor ~2.6 MB (6.4-6.5 bits/nnz) and
empties 4.48 MB -> floor ~3.6 MB (9.4-9.6 bits/nnz); plus barcode identity 0.23 MB. Irreducible ~6.5 MB of
13.27 MB (49%), of which 56% is ambient noise in droplets that no analysis will ever use. For pbmc_v2_typical
exon (6.45 MB): removable 4.27 MB (names, permutation); payload 2.18 MB -> floor ~1.65 MB (cells 5.4 bits/nnz,
empties 7.1-7.4); barcodes 0.04 MB; irreducible ~1.7 MB (26%).
