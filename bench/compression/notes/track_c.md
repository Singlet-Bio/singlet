# Track C: learned, out-of-sample conditional model + range coding

Status (2026-10-21): rungs 1, 1b and 2 implemented, bit-exact, cross-validated leave-one-study-out on all 14
corpus files (98 encode/decode round trips, `matrix_ok=1` everywhere, also through `bench.exe`). Rung 3 exists as
`c_r3` (section 4) and does not beat rung 2 out of sample (-0.14% best, +0.8% worst); rung 4 is measured and has no headroom. Owner: Track C agent.

Headline: **`c_r2` (file classes sent as side info through a trained CPT + online-adapted presence table) is 8.3-23.0%
smaller than today's chunk payload on every one of the 14 files (mean -14.1%, total -13.8% of bytes), at 5.5-6.0
bits/nnz on the normal PBMC/T-cell exon files versus 6.8-7.2 today**, with an integer-only decoder, a 1 MB model
(limit 8 MiB), and a coded size that equals the model cross-entropy to 0.0004 bits/nnz. It decodes at only
0.6-1.4 Mnnz/s (section 5). The out-of-sample model buys little once the codec adapts (section 3).

Files: `kernels/track_c.h` (registry), `kernels/track_c_codec.h` (model loader, encoder, decoder, engine),
`kernels/track_c_core.h` (binary range coder, Q40 fixed-point exp/log, adaptive bit), `kernels/track_c_tables.h`
(generated integer tables, `tools/track_c_gen_tables.py`), `harness/track_c_eval.cpp` (driver: payload-only
accounting, ideal-cost breakdown, per-bucket and per-column diagnostics), `tools/track_c_train.py` (model builder,
numpy), `tools/track_c_train_all.sh`, `tools/track_c_exp*.py` (design experiments, listed in section 3),
`tools/track_c_report.py` (renders the CV table), `corpus_train.tsv` / `corpus_train/` (9 extra training samples
from 7 other studies: GSE165080 x2, GSE285335, GSE244515, GSE268210 for human GRCh38-2024-A; GSE249382 x2,
GSE245201, GSE254360 for mouse GRCm39-2024-A; the 7-sample test corpus is untouched),
`results/track_c_model_*.bin` (models: {human,mouse} x {exon,intron} x {none, one per held-out test study, plus
`pbmconly` and `xhuman`}), `results/track_c_samples.tsv` (barcode-list fingerprint -> study id, so a codec picks
the leave-one-study-out model automatically), `results/track_c_cv.tsv` (the CV table below),
`results/track_c_xdomain.tsv`, `results/track_c_bench_*.tsv` (bench.exe self-test),
`results/track_c_baseline_chunks.tsv` (today's chunk payload per file, from `tools/pz_breakdown.py`).

Run: `bash tools/track_c_train_all.sh` (trains the 14 LOSO models, ~1 min each, one thread);
`bin/track_c_eval.exe --names-level 1 --codec c_r2 --out results/x.tsv corpus/*/exon_counts.1pz` (names use
zstd-19 by default, `--names-level 1` only speeds up experiments; payload numbers do not depend on it);
`bin/bench.exe --no-file-check --codec c_r2 ...` gives the harness-level matrix_ok/timing/total.

## 1. The coder

All numbers are **payload only**: the range-coded matrix stream (depth symbols + per-feature side info + presence +
counts), excluding names/kv/header (those use `ref_zstd_csc` stream helpers at zstd-19 and are reported as
`names_bytes`; they are Track A's problem). Today's reference is the `.1pz` chunk payload (6.8-14.4 bits/nnz).

Columns are coded in file order (adjacent columns have similar depth: correlation of log nnz between neighbours is
0.73-0.98, but no other exploitable structure was found). Per column the encoder sends

1. the **depth bucket** b: 33 symbols (empty column, then 32 log-spaced buckets of nnz per column, thresholds
   generated into `track_c_tables.h`), a 6-bit adaptive binary tree with the previous column's bucket group as
   context (0.01-0.27 bits/nnz; the context buys nothing measurable, the delta-to-previous variant is worse);
2. the **presence set**. Features are partitioned into *classes* (features of one class are exchangeable for the
   model) and laid out class by class. In a class run of length L with hazard h = -ln(1 - p(b, class)) the coder
   sends "any more present in this run?" with P = 1 - exp(-h R), then the offset of the next present feature by
   galloping + bisection, each step with P(left) = E(h, span_left) / E(h, span), E(h, x) = 1 - exp(-h x). This is
   the survival form of independent per-feature Bernoulli coding: O(log span) binary decisions per present
   feature and **zero cost for absent features individually** (their mass is folded exactly into the "any more?"
   flags), so the decoder is O(nnz log m) rather than O(m n) (m n = 3e10 for the shallow files);
3. the **count** given presence: unary over 7 levels (1..8+) with adaptive bits per (rate class group, depth
   group) initialised from trained tables, Exp-Golomb escape for >= 8.

Everything on the decode path is integer: Q40 fixed point, 1 - exp(-x) from a 1,421-entry table with linear
interpolation plus a Taylor branch for small x, -ln from a 1,025-entry table, 2^-x/2048 from a 2,049-entry
table, LZMA-style range coder with 16-bit probabilities (`bound = (range * p) >> 16`, no precision loss). `double`
appears only in the encoder's cost accounting and in the choice of each feature's class (the class is transmitted,
so the decoder never recomputes it). Tables are generated by Python and embedded as integers; the decoder uses
`unsigned __int128` for a few divisions. Coded size and ideal cross-entropy agree within 0.0004 bits/nnz on all
98 (file, codec) pairs, i.e. the coder itself has no measurable overhead.

**Classes.** `c_r1` uses classes from the *training profile* (108 classes: 96 x 0.25 octave of the training share
of nnz, 12 for intervals never seen, by gene level) -- rung 1 as written in PLAN.md. `c_r1b`/`c_r2` use **file
classes**: the encoder counts k_f = number of columns in which feature f is present, quantises it (exact for
k <= 8, then 0.25-octave bins; 121 symbols, 0 = unused) and sends the class of every feature (310k / 272k /
386k / 308k symbols) with a 7-bit binary tree whose node probabilities come from a trained CPT
`P(class | profile level of f shifted by log2 nnz, gene-position state)` (105 x 3 contexts x 127 nodes). The CPT is
copied into adaptive bits with confidence 8, so it also adapts to the file (this alone took `side` from 0.101 to
0.046 bits/nnz on pbmc_v3_lowgenes). This is the two-part (semi-static) code: k_f is the sufficient statistic of the
per-feature Bernoulli model, and the permutation of Track A is exactly the sort of these counts.

**Presence table.** P(present | bucket b, class c): `c_r1`/`c_r1b` use a trained static table (indexed by the
class's rate k/n); `c_r2` starts from it and updates exact integer counts (A present events, M = columns x run
length) per (b, c): p = (A + kappa) / (M + kappa / p0), kappa = 1, row b recomputed after each of the first 64
columns of that bucket and every 16th after (period 1/16/64 changes < 0.002 bits/nnz). The decoder mirrors every
update from the decoded data.

**Models** (`results/track_c_model_<org>_<member>_<holdout>.bin`, 0.92-1.26 MB each): `plevel[m]` (u8 profile
level), `cls1[m]` (u8 rung-1 class), `gfirst[m]` (first interval of its gene -- the row names encode the gene),
`p0_r1[33 x 108]`, `p0_r1b[33 x 121]` (u16 log-domain, isotonic-smoothed in the class direction and calibrated
leave-one-study-out *inside* the training pool), `cpt`, `v0_r1`, `v0_r1b` (initial count probabilities). All
integers. The loader checks the row-name hash, so a model can only be applied to the reference build it was
trained on; `ver=2` models (`mouse_*_xhuman`) carry only the feature-independent tables.

| codec | classes | presence table | adaptation | model | tests |
|---|---|---|---|---|---|
| `c_r1` | trained profile classes | trained, static | none | LOSO | rung 1 as in PLAN.md |
| `c_r1b` | file classes (side info) | trained, static (by rate k/n) | none | LOSO | two-part code, no adaptation |
| `c_r2` | file classes | trained prior + **online update** | yes | LOSO | rung 2 (the headline codec) |
| `c_r2p` | profile classes | trained + online | yes | LOSO | rung 2 without side info |
| `c_r2_nomodel` | file classes, adaptive CPT, prior p0 = k/n | online | yes | none | what the trained model buys |
| `c_r1_insample`, `c_r2_insample` | as above | | | trained incl. the test study | labelled upper bounds |
| `c_r3` | `c_r2` + 2,048 head features coded per feature under (cluster, tier) tables | | yes | LOSO + cluster file | rung 3, section 4 |

## 2. Cross-validation table (leave-one-study-out)

Held-out study = the test file's own study (both samples of GSE142444 and of GSE285936 are held out together); the
model is trained on the other test studies of the same organism plus `corpus_train/`. Payload bits/nnz;
`today` = `.1pz` chunk payload (zstd-3 VOCSC, pointer blocks included in both). Human pools: 4 test + 4 train studies;
mouse pool: 3 train studies only (GSE285936 is the only mouse test study). Source: `results/track_c_cv.tsv`
(98 rows, `tools/track_c_report.py`). Model size per file: 1.03 MB (human exon), 0.92 MB (human intron), 1.26 MB (mouse
exon), 1.03 MB (mouse intron); an exon+intron pair needs 2.0-2.3 MB.

| sample | member | held-out | nnz | today | c_r1 | c_r1b | **c_r2** | c_r2p | c_r2_nomodel | c_r1_insample | c_r2_insample | c_r2 vs today | c_r2 xent | c_r2 total b/nnz (names zstd-19) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| brain_sn_v3_deep | exon | GSE285936 | 6.35M | 9.35 | 9.50 | 8.55 | **8.07** | 8.97 | 8.11 | 8.75 | 8.05 | -13.7% | 8.071 | 13.12 |
| brain_sn_v3_intronic | exon | GSE285936 | 1.38M | 11.56 | 10.65 | 9.73 | **9.48** | 10.22 | 9.61 | 10.12 | 9.44 | -18.0% | 9.484 | 32.51 |
| pbmc_v2_shallow | exon | GSE142444 | 5.30M | 8.70 | 9.19 | 7.46 | **6.99** | 8.82 | 7.01 | 8.05 | 6.98 | -19.6% | 6.994 | 11.89 |
| pbmc_v2_typical | exon | GSE142444 | 2.57M | 6.80 | 7.86 | 6.27 | **5.79** | 7.75 | 5.83 | 7.01 | 5.78 | -14.8% | 5.794 | 14.67 |
| pbmc_v3_lowgenes | exon | GSE244263 | 1.76M | 7.17 | 11.10 | 6.50 | **5.52** | 10.64 | 5.53 | 8.73 | 5.51 | -23.0% | 5.518 | 19.81 |
| tcell_5p | exon | GSE152522 | 10.53M | 6.92 | 7.41 | 6.30 | **6.03** | 7.39 | 6.05 | 7.07 | 6.02 | -12.9% | 6.029 | 8.39 |
| tumor_3p | exon | GSE145370 | 0.25M | 11.92 | 13.31 | 11.99 | **10.37** | 12.74 | 10.55 | 11.63 | 10.14 | -13.0% | 10.370 | 104.76 |
| brain_sn_v3_deep | intron | GSE285936 | 6.68M | 8.26 | 8.09 | 7.36 | **7.27** | 7.74 | 7.31 | 7.60 | 7.25 | -12.0% | 7.265 | 11.19 |
| brain_sn_v3_intronic | intron | GSE285936 | 8.14M | 8.38 | 8.14 | 7.54 | **7.49** | 7.85 | 7.53 | 7.76 | 7.48 | -10.6% | 7.489 | 10.68 |
| pbmc_v2_shallow | intron | GSE142444 | 1.17M | 13.89 | 12.98 | 12.53 | **12.21** | 12.92 | 12.27 | 12.58 | 12.12 | -12.1% | 12.208 | 32.20 |
| pbmc_v2_typical | intron | GSE142444 | 1.20M | 9.10 | 9.14 | 9.21 | **8.35** | 9.08 | 8.46 | 8.78 | 8.31 | -8.3% | 8.346 | 25.15 |
| pbmc_v3_lowgenes | intron | GSE244263 | 1.08M | 8.56 | 11.92 | 7.54 | **7.02** | 11.76 | 7.04 | 10.05 | 6.99 | -18.0% | 7.017 | 27.99 |
| tcell_5p | intron | GSE152522 | 1.10M | 11.34 | 10.85 | 10.09 | **9.94** | 10.79 | 10.07 | 10.49 | 9.91 | -12.3% | 9.943 | 30.26 |
| tumor_3p | intron | GSE145370 | 0.11M | 14.35 | 15.13 | 13.84 | **13.09** | 14.95 | 13.48 | 13.53 | 12.76 | -8.8% | 13.086 | 204.51 |

Summary over the 14 files (mean change of payload vs today): `c_r1` +8.4% (beats today on 5/14), `c_r1b` -8.6%
(12/14), **`c_r2` -14.1% (14/14; best -23.0%, worst -8.3%)**, `c_r2p` +5.6% (7/14), `c_r2_nomodel` -13.3% (14/14),
`c_r2_insample` -14.6%. Payload sum over the 14 files: 43.2 MB vs 50.1 MB today. The "total" column includes the
zstd-19 names (2.6-4.0 MB per file); Track A replaces them by a vocabulary id.

Where the `c_r2` bits go (pbmc_v2_typical exon, 5.79): presence 4.57, counts 1.09, class side info 0.13,
depth symbols 0.01. tumor_3p exon (10.37): presence 8.88, counts 0.33, side 0.89, depth symbols 0.27 (a file of 25k
mostly-empty droplets with ~10 nnz each; side info and bucket symbols are amortised over few non-zeros).

### 2a. Cross-domain and within-domain (`results/track_c_xdomain.tsv`, exon member, payload bits/nnz)

"restricted" = model trained on a narrower pool; "mixed" = the LOSO model of the CV table above.

| test sample | model pool | c_r1 | c_r1b | c_r2 | note |
|---|---|---|---|---|---|
| pbmc_v2_typical | mixed (other 3 test studies + 4 train) | 7.86 | 6.27 | 5.79 | |
| pbmc_v2_typical | **PBMC training studies only** (within-domain) | 7.51 | 6.16 | 5.79 | |
| pbmc_v2_shallow | mixed | 9.19 | 7.46 | 6.99 | |
| pbmc_v2_shallow | PBMC only (within-domain) | 9.14 | 7.27 | 6.99 | |
| pbmc_v3_lowgenes | mixed | 11.10 | 6.50 | 5.52 | |
| pbmc_v3_lowgenes | PBMC only (within-domain) | 11.15 | 6.57 | 5.52 | |
| tcell_5p | mixed | 7.41 | 6.30 | 6.03 | |
| tcell_5p | PBMC only (cross-tissue: sorted CD4 T, 5') | 6.95 | 6.35 | 6.03 | |
| tumor_3p | mixed | 13.31 | 11.99 | 10.37 | |
| tumor_3p | PBMC only (cross-tissue: tumour) | 12.87 | 11.87 | 10.37 | |
| brain_sn_v3_intronic | mouse LOSO (3 mouse train studies) | 10.65 | 9.73 | 9.48 | |
| brain_sn_v3_intronic | **human-trained tables only** (cross-species) | n/a | 10.55 | 9.60 | profile/CPT unusable: different rows |
| brain_sn_v3_deep | mouse LOSO | 9.50 | 8.55 | 8.07 | |
| brain_sn_v3_deep | **human-trained tables only** (cross-species) | n/a | 8.24 | 8.10 | |

Reading: (1) restricting the pool to PBMC changes `c_r2` by < 0.005 bits/nnz on every human file, and even
cross-tissue (tumour, T cells) the PBMC-only model costs nothing, because online adaptation overrides the prior
(and `c_r1` is not hurt by a narrower, more PBMC-like pool: it is mostly noise, +-0.5). (2) Cross-species: the
feature-independent part of the model (presence table in rate-class space, count priors) learned on **human** data
transfers to mouse nuclei with a loss of 0.03 (deep) to 0.12 (intronic) bits/nnz for `c_r2`, i.e. almost nothing;
what does not transfer is the per-feature profile (different rows), which only affects the side info. (3) The
static `c_r1b` table, which relies on the prior alone, is the only rung sensitive to the training pool (+-0.2-0.8).

### 2b. Harness self-test (`bench.exe`, `results/track_c_bench_exon.tsv`, `track_c_bench_intron.tsv`)

`bin/bench.exe --no-file-check --reps 1 --codec c_r2,c_r1b,c_r1 corpus/*/exon_counts.1pz` and the same for
`intron_counts.1pz`, full codecs including the zstd-19 name streams: 42 rows (14 files x 3 codecs), **`matrix_ok = 1`
on all 42**; `model_bytes` = 1,260,568 (largest shipped model). The file-level roundtrip is not claimed (`--no-file-check`,
the production writer is not byte-reproducible across zstd builds, see PLAN section 1). The same 98 (file, codec)
round trips of the CV table were also verified by `bin/track_c_eval.exe` (`D == M` on all fields).

## 3. What each rung bought (out of sample)

- **Rung 1 as specified (profile classes + static trained table, `c_r1`): beats today on only 5/14 files, mean
  +8.4%.** The per-feature rate learned on other studies is a poor predictor: a feature's presence count drifts by
  2-4x between studies, and 1-10% of the non-zeros sit on intervals never seen in training (10% for tcell_5p, 5'
  chemistry). pbmc_v3_lowgenes (2.6k genes detected) is the extreme: 11.1 bits vs 7.2, because all its features
  are "rare" in the training profile. The in-sample bound `c_r1_insample` (model trained with the test study) is
  only 0.3-2.4 bits better, so this is not a training-set-size problem. A static cross-study feature rate is the
  wrong model.
- **Rung 1b (file-class side info), -14% vs `c_r1` (mean), beats today on 12/14.** k_f costs 0.05-0.3 bits/nnz on
  normal files (0.9-1.5 on the two tiny tumour files) through the trained CPT; in exchange the table is indexed
  by the true rate. Model-free, coding k_f adaptively costs 0.02-0.4 bits/nnz more (`c_r2_nomodel`), the clearest
  place the trained model earns its bytes.
- **Rung 2 (online adaptation of the (bucket, class) table): a further -5.9% (mean)**, biggest on the degenerate
  and deep files (pbmc_v3_lowgenes 6.50 -> 5.52, tumor 11.99 -> 10.37, pbmc_v2_typical 6.27 -> 5.79). Adaptation
  makes the codec nearly model-free: `c_r2_nomodel` is 0.01-0.40 (mean 0.09) bits/nnz worse than `c_r2`, and
  `c_r2_insample` is only 0.005-0.33 (mean 0.06) better. **The out-of-sample model is worth 0.1 bits/nnz (1.5%)
  on average, almost all of it the class CPT.** kappa 1-16 changes < 0.03 bits/nnz.
- **Rung 2 on profile classes (`c_r2p`) does not recover rung 1b** (7.75 vs 5.79 on pbmc_v2_typical): adapting a
  per-class rate cannot repair a wrong class assignment, because the class decides which features are exchangeable.

Design experiments that did not earn their keep (numpy, in-sample tables unless said otherwise; bits/nnz):

| idea (script) | result | kept? |
|---|---|---|
| depth bucket on total UMI vs nnz; 16/32/56 buckets (`exp2`) | nnz buckets 0.03-0.05 better; 32 vs 16: 0.02; 56: none | nnz, 32 |
| second column covariate z = UMI/nnz for presence (`exp12`) | -0.003..-0.013 presence, +0.003..+0.031 side info | no |
| z for the count model (`exp9`) | -0.01..-0.03, side 0.00-0.03 | no |
| count contexts: previous value in column, running mean (`exp6`) | -0.01..-0.03 | no |
| 2-D feature class (k in shallow vs deep columns) (`exp8`) | -0.003..-0.06, before side info | no |
| coarser file classes (0.5-octave, octave bins) (encoder flag) | side -0.00..-0.03, presence +0.01..+0.05 | no |
| gene-level model: gene presence + interval choice as multinomial (`exp3`) | 8.2 vs 6.3 bits for the interval model | no |
| gene presence + interval given earlier intervals of the gene (`exp5`, `exp15`) | 4.67 vs 4.56 (exon); introns: indep 7.769 vs 7.766 (PBMC), 6.927 vs 6.914 (nuclei) | no |
| richer side-info contexts: last used interval's class in the gene, etc. (`exp10`) | 0.659 -> 0.637 on tumor, 0.116 -> 0.114 typical | no |
| bucket context = previous column's bucket / zig-zag delta | 0.266 vs 0.264 / 0.337 | context kept (harmless), delta dropped |
| adaptive (vs frozen) class CPT | side 0.929 -> 0.892 (tumor), 0.101 -> 0.046 (lowgenes) | **kept** |

The gene-level and within-gene results kill the "model at gene level" suggestion of PLAN.md for this data: UMIs of
a gene land on different intervals nearly independently given each interval's rate; pooling to the gene loses
more than the dependence recovers.

## 4. Rung 3 (latent structure) and rung 4 (ambient empties)

**Rung 3.** Evidence first (numpy, `tools/track_c_exp7.py`, `exp13.py`, `exp14.py`; head = 2,000 most frequent
features = 55-70% of the non-zeros of called cells; bits per *head* non-zero, pbmc_v2_typical exon):

| model of P(head feature present given the cell) | bits/head-nnz | side info |
|---|---|---|
| per-feature x depth tier, rates from the file's other half (what rung 2 converges to), K=1 | 2.80 | 0 |
| + k-means on the file's own other half, K=8 / 16 / 32 (hard cluster id per cell) | 2.76 / 2.78 / 2.82 | 0.010 / 0.013 / 0.016 |
| (tcell_5p same, K=1 / 16) | 2.70 / 2.65 | 0 / 0.010 |
| rates trained on other studies, K=1 (no adaptation) | 4.21 | 0 |
| K=4 / 16 clusters trained on other studies, cell assigned by max likelihood | 2.50 / 2.40 | 0.010 / 0.019 |
| Poisson NMF K=4 (loadings from other studies, per-cell 4-bit factors) | 2.70 (vs 3.91 for K=1 under the same link) | 0.067 |

Clustering the file's own cells gives -1% (typical) to -2% (tcell) of head bits; the large OOS gains (4.21 -> 2.40)
are mostly a calibration effect (the K=1 out-of-sample table is mis-specified), not cell types, and the question is
what survives on top of rung 2. A parallel Track C session implemented exactly that in C++ (`c_r3`: 2,048 head
features coded per feature under an adaptive (cluster, depth-tier) table initialised from OOS k-means profiles, a
4-bit cluster id per column with >= 128 nnz as side info; `tools/track_c_train_r3.py`, models
`results/track_c_r3_*.bin`, 139 kB each). Full leave-one-study-out run on the 7 exon files (`results/track_c_cv_r3.tsv`, `track_c_cv_r3a/b.log`):

| exon file | `c_r2` | `c_r3` | change |
|---|---|---|---|
| pbmc_v2_typical | 5.794 | 5.786 | -0.14% |
| pbmc_v3_lowgenes | 5.518 | 5.516 | -0.04% |
| tcell_5p | 6.029 | 6.028 | 0.0% |
| pbmc_v2_shallow | 6.994 | 7.034 | +0.6% |
| tumor_3p | 10.370 | 10.449 | +0.8% |
| brain_sn_v3_deep | 8.071 | 8.089 | +0.2% |
| brain_sn_v3_intronic | 9.484 | 9.543 | +0.6% |

(An earlier single-file number in `results/track_c_r3_dev.tsv`, 5.780 on pbmc_v2_typical, came from a previous
version of the r3 model/code; the table above is the final code and models.) Conclusion: **rung 3 does not beat rung 2 out
of sample** (best -0.14%, worst +0.8%); cell-type structure is real but, once feature rates are known per file, the
head-feature gain is smaller than the cost of the 4-bit cluster id per cell and of the diluted adaptive statistics.
The feature-rate drift between studies that the OOS cluster experiment appears to fix (4.21 -> 2.40) is already
removed by the transmitted k_f classes. Rung 3 is dropped; `c_r3` stays in the registry as documentation.

**Rung 4.** Empty/low-depth droplets are 6-48% of the non-zeros and cost 7.3-9.95 bits/nnz (more than called cells).
Bound (`tools/track_c_exp11.py`, per-column cost dump of `c_r2`): code every non-called column as a Multinomial(k UMIs) draw
from the pooled profile of the *other half* of the non-called columns (alpha = 0.05 smoothing) plus the entropy
of the per-column UMI total k:

| file (exon) | non-called nnz share | `c_r2` bits/nnz on non-called | multinomial-ambient bound (+ total-UMI cost) |
|---|---|---|---|
| pbmc_v2_typical | 6.5% | 7.45 | 7.49 (+0.30) = 7.79 |
| pbmc_v2_shallow | 39% | 7.35 | 7.28 (+0.14) = 7.42 |
| tumor_3p | 42% | 9.97 | 10.16 (+0.61) = 10.77 |
| brain_sn_v3_deep | 48% | 9.50 | 9.61 (+0.17) = 9.78 |

`c_r2` is already at or below the ambient-multinomial bound on all four files (and the bound ignores the cost of
sending the ambient profile). The non-called columns are mostly low-depth cells, not pure ambient, so one profile
fits them worse than the (bucket, class) table does. **Rung 4 is dropped: no headroom.**

## 5. Decode speed

Single-threaded decode of the full stream (min of 3, release, machine shared with three other agents):
`c_r1` 1.0-1.15, `c_r1b` 1.2-1.4, `c_r2` 0.6-0.9 Mnnz/s (pbmc_v2_typical / pbmc_v2_shallow exon). During the loaded
CV the same code measured 0.05-0.5 (`dec_Mnnz_s` in `track_c_cv.tsv`, min of 1 on a saturated machine; treat
those as lower bounds). That is 10-20x below the production reader and 10-60x below Track B's `b_rans16` (17 Mnnz/s
canonical, 62 native, AVX-512). Where it goes: one range-coder decision per bisection step (~8-12 per non-zero)
with two Q40 `E(h, x)` evaluations and a 64-bit division per step, plus the "any more in this run?" flag for each
(column, non-empty class) (121 per column: ~25% of decisions for typical columns), plus the per-column table
refresh and the class CPT pass over all m features (7 adaptive decisions each, ~0.1-0.3 s per file). How to make it fast without changing the model:

1. Replace the per-step bisection by direct gap coding under the same survival law: the gap inside a class run is
   geometric with the run's hazard, so it can be coded as one multi-symbol rANS symbol from a per-(bucket, class)
   quantised CDF (tables counted in the model), i.e. one decode per non-zero instead of ~10 (expected 10-20x).
2. Precompute p16 for the power-of-two spans per (bucket, class) at refresh time (the bisection only asks for the
   stride and halves of it), so the loop has no E40 evaluation or division.
3. Interleave 8-32 rANS streams (state per bucket is frozen between refreshes) and SIMD them, as Track B does.
4. Skip empty classes in a column with one decision per group of classes (hierarchical survival), and refresh
   tables every 64 columns (< 0.002 bits/nnz).

The rate-distortion question to measure before committing: how much the quantised geometric CDF costs (the
current coder is at the exact survival law).

## 6. Gap to Track A / Track B

All three read the same transform-free matrix, so the comparison is bits/nnz of the matrix payload:

| file | today | Track A best (brotli, `track_a_sweep`) | Track B `b_rans16` payload (excl. row-count side info) | **Track C `c_r2` (all-in: side info included)** |
|---|---|---|---|---|
| pbmc_v2_typical exon | 6.80 | 6.45 | 5.73 | **5.79** |
| pbmc_v2_shallow exon | 8.70 | 7.86 | 7.06 | **6.99** |
| pbmc_v3_lowgenes exon | 7.17 | 6.50 | 5.61 | **5.52** |
| brain_sn_v3_deep exon | 9.35 | n/a | 8.06 | **8.07** |
| brain_sn_v3_intronic exon | 11.56 | 10.54 | 9.45 | **9.48** |
| tcell_5p exon | 6.92 | n/a | 6.01 | **6.03** |
| tumor_3p exon | 11.92 | 10.70 | 9.87 | **10.37** |

Track B's number excludes its per-row-count side info (its `side_bytes`: 0.15-0.25 bits/nnz on normal files,
1.6 on tumor_3p), my number includes the equivalent (class symbols 0.05-0.9). Equal accounting: `c_r2` is within
+-1% of `b_rans16` on the normal files and ahead on the small ones, while `b_rans16` decodes 15-60x faster. In
other words **a static, position-bucket x depth rANS (Track B) already reaches the same entropy as an adaptive
model with a trained prior (Track C)**, and the transform + entropy-coder gap to zstd (about 0.7-1.4 bits/nnz,
10-20%) is the entire prize for count matrices of this kind: the content-dependent structure that a learned
model could exploit beyond per-feature rate x depth (cell-type clusters, within-gene dependence, column covariates,
ambient profiles) is worth < 0.5% (section 3-4).

## 7. Lessons

1. **The file's own per-feature presence count is worth far more than any out-of-sample feature rate.** Sending
   k_f (0.05-0.3 bits/nnz) fixes the 2-4x cross-study drift in feature rates that makes a static trained table lose
   to zstd (`c_r1`: +8%); after that the trained model's job shrinks to coding that side info cheaply and
   initialising adaptive state (worth ~1.5% on average, and cross-species/cross-tissue transfer costs nothing).
2. **Per-feature Bernoulli coding is cheap and exact once exchangeable features are grouped into runs**: survival
   coding makes it O(nnz log m) with all absent mass coded exactly, the coder sits on the model's cross-entropy to
   0.0004 bits/nnz, and every remaining bit is a modelling question. The modelling questions have short answers on
   this data -- per-feature rate x depth is the floor to within < 1% (no gain from gene structure, column
   covariates, 2-D classes, cell-type clusters or ambient profiles) -- so the real lever for the format is
   speed (Track B's lane rANS) and moving names/permutation to the model side (Track A).
