# Compression study: how small can a `.singlet` transcriptomics matrix get, and how fast can it decode?

Status: plan v1, 2026-10-21. Owner: Zach. Experiments run locally on the laptop
(i7-1185G7, 4 cores / 8 threads, 12 MiB L3, 15.7 GB RAM, Windows 11, Rtools 4.4 GCC 13 + static zstd 1.5.5 /
xz / brotli / lz4 / zlib / bz2). Everything lives under `bench/compression/`.

## 0. Questions this study answers

1. **Accounting.** Where do the bytes of a `.1pz` count matrix go today, and how much of that is information
   versus redundancy we are paying for on every sample?
2. **Frontier.** With a bit-exact decoder that fits the constraints below, what is the best
   (compressed size, decode speed) frontier we can reach with (a) better entropy coding of the existing
   transform, (b) new transforms / sparse encodings, (c) a learned probability model used by an arithmetic coder?
3. **Floor.** What is the information-theoretic floor for this data, estimated from the data-generating process,
   and how close is each codec to it?
4. **Recursion / asymptote.** Can the output of the best codec be compressed further by anything (general-purpose
   codecs, a strong context-mixing byte model, a second learned pass, a learned orchestrator that composes kernels)?
   Show that the ratio of a second pass converges to >= 1.0 and explain *why* (the residual is the entropy of
   ambient/sampling noise, which no model can predict).
5. **Transport.** What does this mean for latency: bytes per cell, time-to-first-cell over HTTP range requests,
   and what layout (chunking, side tables) minimises round trips.

## 1. Ground rules (apply to every experiment)

- **Object of study.** The canonical matrix `M = (m, n, nnz, indptr, indices, data, rownames, colnames, user_kv)`
  exactly as `singlet::pz::read_1pz()` returns it (`harness/common.h: CanonicalMatrix`). A codec passes only if
  `decode(encode(M)) == M` byte-for-byte on every array and every name. The harness also re-emits a `.1pz` from the
  decoded matrix and compares it with the original file. **Finding (smoke run):** the production writer is not
  byte-reproducible across zstd builds (the pipeline wrote with a different zstd than our 1.5.5; `pbmc_v2_typical/
  exon_counts.1pz` re-encodes to 6,483,537 B vs 6,451,648 B original, same matrix). So "bit-identical reconstruction
  of the original file" is defined on the canonical matrix (which is all a consumer ever sees), and container-level
  identity is reported separately and only expected when the same zstd build is used. Track A confirms the cause.
- **What counts toward size.** Everything the decoder reads that is specific to the file. A *model* (tables,
  dictionaries, trained parameters) does **not** count if it is shared across files and fits comfortably in L3:
  budget **<= 8 MiB** (L3 is 12 MiB here). A per-study or per-reference-build vocabulary counts as a model.
  Report model size alongside every learned codec.
- **Determinism.** Decoders run on arbitrary x86-64 CPUs. No floating point on the decode path of any learned codec;
  probabilities are fixed-point integers (e.g. 16-bit) and all model inference is integer arithmetic so encoder
  and decoder agree bit for bit on every machine.
- **Out-of-sample.** Any model is trained on studies that are *not* the one being compressed (leave-one-study-out
  across the corpus). Report in-sample numbers only as an upper bound, clearly labelled.
- **Measure, don't assume.** Every result row goes to `results/*.tsv` via the harness (size, bits/nnz, encode ms,
  decode ms min-of-N, Mnnz/s, matrix_ok, file_ok, model_bytes). Decode timing is single-threaded wall time of the
  full `decode()` including name reconstruction, min of 3 reps, release build `-O3 -march=native`.
- **Hygiene.** Corpus files are downloaded data: treat them as untrusted input, run Python with `-P`, never execute
  anything from `corpus/`. Nothing in `corpus/` or `results/*.bin` is committed.

## 2. Corpus

Seven samples from five studies, chosen to span the distribution we see in the atlas (protocol, depth, organism,
intron fraction, number of empty droplets), each as `exon_counts.1pz` + `intron_counts.1pz` (+ `cell_calls.tsv`,
`summary.json`, study-level `feature_vocab.json`), fetched with `tools/fetch_corpus.py` by HTTP range request so
only the needed members are transferred. One whole bundle (`GSE145370.singlet`, 187 MB, 14 samples) for
bundle-level accounting. See `corpus.tsv`.

| id | study / sample | what it represents |
|---|---|---|
| pbmc_v2_typical | GSE142444 / GSM4227413 | human PBMC 10x v2, 5.4k cells, median UMI 716 (the modal atlas sample) |
| pbmc_v2_shallow | GSE142444 / GSM4227419 | human PBMC 10x v2, 41.7k "cells", median UMI 76, 96k barcodes (shallow, droplet-heavy) |
| pbmc_v3_lowgenes | GSE244263 / GSM7810083 | human PBMC 10x v3, 11k cells, only 2.6k genes detected (degenerate library) |
| brain_sn_v3_deep | GSE285936 / GSM8713125 | mouse brain snRNA 10x v3, 3.3k nuclei, median UMI 1045, intron-rich, 81k empty droplets |
| brain_sn_v3_intronic | GSE285936 / GSM8713120 | mouse brain snRNA, exonic fraction 0.09 (intron matrix dominates) |
| tcell_5p | GSE152522 / GSM4617168 | human sorted CD4 T cells, 10x 5', 12.4k cells, densest exon matrix (10.5 M nnz) |
| tumor_3p | GSE145370 / GSM4317429 | human ESCC tumour, 10x 3', 3.8k cells, tiny matrix (0.25 M nnz) |

Important fact about the data: rows are **exon intervals / intron intervals** (310,797 and 272,191 for GRCh38;
386,028 / 307,730 for GRCm39), not genes. Columns are **all barcodes with >= 1 UMI**, not just called cells
(`cell_calls.tsv` marks the called ones). Values are UMI counts, 75-99% of them equal to 1.

## 3. Baseline accounting (first numbers, from `tools/pz_breakdown.py`)

| file | bytes | nnz | names (meta_z) | row perm | ptr+colsums | matrix chunks | bits/nnz total | bits/nnz chunks |
|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 6.45 MB | 2.57 M | 3.40 MB (52.7%) | 0.83 MB | 0.03 MB | 2.18 MB | 20.1 | 6.8 |
| pbmc_v2_typical intron | 5.16 MB | 1.20 M | 3.03 MB (58.7%) | 0.74 MB | 0.03 MB | 1.37 MB | 34.3 | 9.1 |
| pbmc_v2_shallow exon | 10.54 MB | 5.30 M | 3.80 MB (36.0%) | 0.83 MB | 0.14 MB | 5.77 MB | 15.9 | 8.7 |
| brain_sn_v3_deep exon | 13.27 MB | 6.35 M | 4.65 MB (35.0%) | 1.06 MB | 0.14 MB | 7.42 MB | 16.7 | 9.4 |
| tcell_5p exon | 13.76 MB | 10.53 M | 3.67 MB (26.7%) | 0.87 MB | 0.10 MB | 9.11 MB | 10.5 | 6.9 |
| tumor_3p intron | 4.04 MB | 0.11 M | 3.08 MB (76.3%) | 0.74 MB | 0.03 MB | 0.19 MB | 301 | 14.4 |

Three conclusions before any new kernel is written:

1. **Feature names are the largest component in 10 of 14 files** (27-76%). They are the same string list in every
   sample of a study and in every study on the same reference build. Replacing them with a reference-build
   vocabulary id (model, not data) removes 3-4.6 MB per file. Barcodes (colnames) are 16-mers over ACGT: 4 bytes
   each raw, far less after sorting + delta coding; today they are inside the same zstd-3 metadata blob.
2. **The row-frequency permutation costs 0.7-1.06 MB per file** (310-386k x u32, zstd-3). It can be derived by the
   decoder from per-row nonzero counts (which compress to a fraction of that and are useful as model context), or
   replaced by a fixed reference-level gene order (model), or made cheaper by storing the inverse as rank deltas.
3. **The matrix itself is already at 6.8-14 bits/nnz with zstd level 3.** That is the part the kernels compete on.
   A naive baseline (raw CSC arrays, zstd-19) lands at 18.1 bits/nnz on pbmc_v2_typical exon including names, so
   the VOCSC transform is doing real work; the question is how much is left between ~7 bits/nnz and the floor.

Hypothesis for the whole study: **total bytes per sample can drop 2-4x with zero information loss** (names +
permutation to the model side, better entropy coding), the **matrix payload can drop a further 20-40%** with a
learned conditional model, and beyond that the residual is ambient-droplet sampling noise that is provably
incompressible in expectation.

## 4. Experiment tracks (one Sonnet 5.5 agent each, run in parallel)

Each track owns one kernel header `kernels/track_<x>.h` (exposing `void register_track_<x>(std::vector<Codec>&)`),
its own result files `results/track_<x>_*.tsv`, its own tools `tools/track_<x>_*.py`, and a findings note
`notes/track_<x>.md`. Nobody edits another track's files; `kernels/registry.h` already includes all four.

### Track A: accounting, container and entropy-coder tuning (fast wins, measurement infrastructure)

Goal: establish exactly what the existing transform leaves on the table and build the measurement tools
everyone else uses.

- Component accounting for all 14 files + the whole `GSE145370.singlet` bundle (zip overhead, JSON members,
  sj/psi/vdj matrices vs exon/intron): a table of "bytes per sample by member", and "bytes per called cell".
- Names: measure (a) zstd-3 as today, (b) zstd-19/22 with long window, (c) xz, brotli-11, (d) *vocabulary id +
  nothing* (names are a reference-build model), (e) barcodes as sorted 2-bit packed 16-mers with delta coding.
- Permutation: today's cost vs (a) per-row nnz counts, varint + zstd, with the sort re-derived (must reproduce the
  identical permutation: counting sort descending, ties by row id), (b) a fixed reference-level gene order shipped as
  model, (c) nothing (no permutation; see Track B for whether it even helps once better index coding exists).
- Chunk payload: keep the VOCSC prefilter bytes exactly as the writer produces them and sweep the entropy coder:
  zstd levels 1,3,6,9,12,15,19,22(+long), with/without a trained zstd dictionary (train on other samples, size
  <= 1 MiB, counts as model), xz -9e, brotli -11 window 24, bzip2. Also chunk_cols 256 / 1024 / 4096 / whole file.
  Record encode and decode time for each (decode time matters as much as size).
- Decode profile of the current reader: where do the milliseconds go (zstd, bitmap unpack, bit-plane decode,
  varint parse, name splitting)? Use a simple instrumented copy of the reader loop; report ns/nnz per stage.
- Confirm the file-roundtrip failure cause (zstd 1.5.5 vs the pipeline's build) by checking the frame headers /
  trying `ZSTD_c_strategy` variants; report whether container-level byte identity is achievable at all.

Deliverable: `notes/track_a.md` with the tables, plus codecs `a_names_as_model`, `a_perm_from_rowcounts`,
`a_vocsc_zstd19`, `a_vocsc_xz` etc. in `track_a.h`, all passing matrix roundtrip.

### Track B: transform kernels and decode speed (C++)

Goal: push the (bits/nnz, Mnnz/s) frontier of the matrix payload with hand-written kernels. Names/permutation
are out of scope here (assume Track A's model-side treatment; encode them with the cheapest method that passes).

Index coding (row ids within a column, ascending):
- gap + LEB128 varint (baseline), gap + byte-split (today), Elias-Fano, binary interpolative coding (near-optimal
  for sorted sets, decode is branchy: measure), Roaring-style run/array/bitmap containers per 2^16 block,
  bit-sliced / bitmap for the densest rows (the top few hundred genes carry a large share of nnz).
- Row order: frequency sort (today) vs reference order vs co-expression clustering; measure the effect on each coder.
- Column order: as stored vs sorted by depth vs a locality order (k-means on log-normalised top genes, then
  TSP-ish nearest-neighbour chain) so that *column-to-column deltas* become cheap; for called cells this is where
  biology (cell type) can be exploited without a learned model.
- Orientation: CSC (cells) vs CSR (features) vs a hybrid (dense top-k features as CSR bitmaps, the long tail as CSC).

Value coding (75-99% ones):
- separate "is 1" flag stream from the >1 residuals; Rice/Golomb/exp-Golomb with per-feature parameter; or VOCSC's
  group-by-value (today) vs value-per-entry with context.

Entropy coding: zstd on the transformed byte streams vs direct symbol coding with FSE/tANS/rANS using static
per-stream tables (table counts toward size, it is per-file) vs adaptive binary arithmetic coding with simple
contexts (gene frequency bucket x position). Separate size from speed: an rANS decoder with SIMD interleaving
should reach several hundred Mnnz/s; measure it.

Decode speed targets: the production reader does ~10 Mnnz/s through the Python binding; measure the native
reader, then aim for >= 200 Mnnz/s single-threaded on the payload, while matching or beating today's bits/nnz.
Use `-march=native` (AVX-512 on this CPU) but keep a scalar fallback and report both.

Deliverable: codecs `b_*` in `track_b.h`, a frontier plot (bits/nnz vs Mnnz/s) data file, and `notes/track_b.md`
with what worked, what did not, and why.

### Track C: learned, out-of-sample conditional model + arithmetic coding

Goal: the "model outside the benchmark". Train a probability model on *other* studies, ship it as a <= 8 MiB
integer table set, and code each file with an arithmetic coder driven by that model. Report bits/nnz vs the
cross-entropy of the model (they should match to within coder overhead), by leave-one-study-out.

Modelling ladder (each step must beat the previous one out of sample or be dropped):
1. Per-feature Bernoulli presence rate conditioned on column depth bucket (e.g. 16 log-spaced buckets):
   `P(feature f present | depth d)`. Code the presence bits in feature-frequency order with a binary arithmetic
   coder; code the count given presence with a per-(feature, depth) geometric / truncated-Poisson table.
   Model: 310k features x 16 buckets x 2 bytes ~ 10 MB raw, so quantise/prune (features with near-zero rate share
   a bucket) to fit 8 MiB, or model at gene level (38k genes) and code the within-gene exon choice separately.
2. Online adaptation: the decoder updates the tables as it decodes (exact integer counts, same on both sides);
   this captures study-specific and cell-type structure without any side information.
3. Latent structure: per-cell low-rank factors (NMF/topic model trained out of sample; gene loadings are the model,
   per-cell factor vector is coded as a few quantised bytes of side info) to condition `P(f present | cell)`.
   Measure bits saved vs bits spent on side info.
4. Empty droplets: treat as multinomial draws from the ambient profile (estimated from the file's own empty
   droplets, transmitted once, or from the model); this should get them close to the floor in section 5.

Cross-validation protocol: hold out one study at a time; train on the rest. Also report the cross-domain case
(train on human PBMC only, test on mouse brain) to show how robustness degrades, and the within-domain case
(two PBMC studies). If the corpus needs more training samples, pull extra PBMC / brain samples from *other*
studies with `tools/fetch_corpus.py` into `corpus_train/` (keep the test corpus fixed).

Determinism: fixed-point probabilities (e.g. 12-16 bit), integer-only model updates, a range coder with a
reference implementation that is tested for exact encoder/decoder agreement. No floats anywhere in decode.

Deliverable: codecs `c_*` in `track_c.h`, training scripts `tools/track_c_*.py` (Python + numpy are fine for
training; export tables as binary files loaded by the C++ decoder), the CV table, and `notes/track_c.md`.

### Track D: entropy floor, recursion and the asymptote argument

Goal: establish, as rigorously as the data allow, the bound below which no lossless codec can go in expectation,
and show empirically that recursive compression of our best outputs does not help.

- Empirical entropy estimates for each file: order-0 entropy of (gap, value) symbols as the writer sees them;
  entropy of the presence pattern under the best available conditional model (from Track C as it lands, or your
  own quick per-feature x depth estimate); and the **ambient floor**: for every empty droplet with k UMIs drawn
  from the ambient profile p, the entropy of the resulting count vector is that of a Multinomial(k, p). Sum over
  droplets. By the source coding theorem no code beats this in expectation if droplets really are exchangeable
  multinomial samples; test that assumption (goodness of fit of empties to the pooled ambient profile).
- Called cells: the floor is not known exactly, so report the diminishing-returns curve: bits/nnz achieved by
  successively richer models (order-0 -> per-feature -> per-feature x depth -> adaptive -> latent). Fit the curve
  and report the extrapolated asymptote with an honest interval.
- Recursion: take every codec output from `results/` (ask the harness to dump payloads with a flag you add to your
  own copy of the driver or a small tool), and compress it again with zstd-22, xz-9e, brotli-11, bzip2, and a
  small order-2/3 context-mixing byte coder you implement (a 200-line logistic-mixing CM is enough to be a
  stronger "universal" second pass than any LZ). Report second-pass ratio per codec; expected: >= 0.99 for the
  arithmetic-coded outputs, noticeably < 1 for zstd-3 outputs (which is itself a finding: today's files *are*
  recursively compressible).
- The orchestrator idea, made concrete: a kernel-orchestration layer that, per chunk, chooses among K codecs. Measure
  the oracle gain (best codec per chunk vs best single codec), then train a tiny out-of-sample classifier on cheap
  chunk features (nnz, density, mean value, depth distribution) to pick the codec, and report how much of the
  oracle gain it captures. This is the only form of "recursive / compositional" compression that can beat a single
  strong model, and only by the oracle gap; quantify that gap.
- Write the argument: why a learned recursive scheme cannot beat the entropy of the generating process; where the
  generating-process entropy comes from in this data (UMI sampling, ambient RNA); what fraction of today's bytes is
  that irreducible part versus removable redundancy.

Deliverable: `tools/track_d_*.py`, the CM coder in `track_d.h` (as a codec `d_cm_bytes` that can wrap any other
codec's output), `results/track_d_entropy.tsv`, `results/track_d_recursion.tsv`, and `notes/track_d.md`.

## 5. Synthesis (after the tracks land)

`REPORT.md`: one table per corpus file with today's size, the best lossless size, bits/nnz, decode speed, model
size, and distance to the estimated floor; the frontier plot; the recursion table; recommendations for the
`.singlet` format (names and permutation to a reference-level vocabulary, chunk layout for range requests, codec
choice) with the expected effect on transfer latency for a typical sample over 100 Mbit/s and 1 Gbit/s links.

## 6. How to run

```bash
bash build.sh                              # builds bin/bench.exe (and any other main() under harness/ or kernels/)
./bin/bench.exe --list                     # codecs
./bin/bench.exe --out results/run.tsv corpus/*/exon_counts.1pz corpus/*/intron_counts.1pz
python -P tools/pz_breakdown.py "corpus/*/*_counts.1pz"
python -P tools/fetch_corpus.py corpus.tsv corpus            # re-fetch / add rows
```

The compiled Python reader is also available without installing the package:
`python -P -c "import sys; sys.path.insert(0,'../../python'); from singlet._pz import read_1pz"`.
