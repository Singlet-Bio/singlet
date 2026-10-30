# Internal peer review, round 1: response

Three independent reviews of draft 1 (2026-10-29): data compression and information theory, single-cell practice, and
benchmarking methodology. All three recommended major revision. The reports are not in the repository; this file
records what changed. Commits are on `paper/s1-benchmark-results`.

## Measurement fixes (code changed, everything timed again)

| Concern (reviewer) | Change |
|---|---|
| Competitors' timed reads included the harness's conversion and checks (`canon_from_scipy`, `Canon.check`); `.s1`'s did not (all three) | Every format now has a timed `load()` (the library's own read into its native object) and an untimed `to_canon()`. Decode times `load()` only; the write stage's round trip still checks every format exactly. |
| `.s1` headline in stored gene order; the others in source order | `.s1` is compared in source gene order; stored order is a separate row (`dec-s1native-*`). |
| The AVX-512 result was AVX2 (the kernel needs VBMI2; silent fallback) (compression) | `decode_bench` and the Python codec rows record `isa_used`; the report keeps only rows whose requested kernel ran. The text says AVX-512 was not measured. |
| Eq. 1 counted disk I/O twice (cold local read as decode rate) (benchmarking) | `wire_section` uses warm, in-memory rates; reports the crossover bandwidth per matrix. "Never the bottleneck" removed. |
| rclone `--bwlimit` let several MiB through after idle; Toxiproxy limits per connection (benchmarking) | New `bwproxy.py`: one token bucket for all connections; Toxiproxy adds latency only. The first version (wait for 64 KiB of tokens per send) lost the event loop's timer overshoot on every step and delivered about 0.4 of the 1 Gbit/s tiers' rate; the bucket now goes into debt and holds max(64 KiB, 5 ms of the rate), delivers 0.10, 1.00 and about 9 Gbit/s at the three tiers, and the network tiers were rerun. The report records the delivered rate per tier (`delivered-*`). |
| TileDB-SOMA ML and scDataset filled 65,536- and 8,192-cell buffers before the first batch; shuffling not equalised (single-cell, benchmarking) | Both are configured to the shared workload: 64-cell blocks, one 512-cell batch per read, no larger buffer. |
| BPCells unpinned; text said it was handed to PyTorch (benchmarking) | Pinned with `taskset`; text says it is timed to an R matrix. |
| Raw times and repeat counts dropped; a long warm-up could be reported as the timed run (benchmarking) | Decode rows keep `n`, `times_s` and `isa_used`; `util.repeat` never reports a warm-up time. |
| Reduced protocol for large matrices applied by hand (benchmarking) | `decode.big_nnz` and `decode.big_formats` in `config.yaml`; the decode stage applies them itself. |
| Download estimates computed but never reported; capped tier not flagged (benchmarking) | Table 5 includes the estimates (marked with a superscript e) and the capped tiers (`\nm{cappedTiers}`). |
| Ratios of medians over different subsets (benchmarking) | Speed-ups are medians of per-matrix ratios. |
| No checksums recorded or verified (benchmarking) | Every downloaded file's SHA-256 and size is in `datasets.yaml` (`source.checksums`); `fetch` verifies each file. |
| One `env.json` overwritten by every task (benchmarking) | Each dataset's run keeps `results/env/<dataset>.json`. |
| `Read10X_h5` returns a bare matrix (single-cell) | Added `CreateSeuratObject(Read10X_h5(...))` as the like-for-like baseline for `singlet::load(as = "seurat")`. |

## Claims rewritten (no new measurement)

- **The floor and the sampling share.**
  - The floor is now "reference code lengths", not a bound: the models are charged nothing, fitted on half of the
    cells, and there are no latent models. The sampling share is a plug-in estimate, for UMI matrices only, relative to
    the whole-matrix payload.
  - The shrinkage weight is fitted on the sampled cells, and the methods now say so.
  - Fig. 3's title, legend and caption follow.
  - Bits-back coding and latent models are discussed as the route to stronger codes.
- **The payload ladder.** Its last rung combines reordering, contexts and rANS; the text says so.
- **Format description, corrected against the code** (Supplementary Section S1, a normative specification):
  - chunk size is a default, not a limit;
  - the prefix size is stated as it is;
  - objects are named by a 128-bit digest prefix;
  - decoding takes a few lookups per symbol, not one;
  - the integrity check covers entropy-coded symbols, not raw bits, and the canonical digest is available for full
    decodes;
  - the writer, not the container, rejects floats.
- **Related work** added (30 verified references):
  - integer and index compression: ANS index compression, partitioned Elias–Fano, interpolative coding;
  - generic codecs: Blosc and bitshuffle, zstd dictionaries;
  - other sparse data: genotype compressors, graph reordering;
  - storage and access: object-store latency, cloud-native chunking;
  - modelling and decoding: bits-back coding, GPU ANS.
- **Limitations, now explicit:**
  - shared nodes and the controller cache;
  - the scalar decoder on ARM;
  - no DataLoader workers;
  - missing dictionary and Blosc2 baselines;
  - annotations not in the format;
  - gene-wise access needs a full decode;
  - one implementation;
  - writes not timed.

## Not done (stated as limitations or future work)

- Stronger count models (scVI, GLM-PCA) and prequential, model-charged code lengths for the floor.
- Ladder rungs that separate reordering, contexts and entropy coding; zstd dictionaries and Blosc2 at chunk granularity.
- DataLoader workers; NVMe hosts; other CPU architectures; real cloud object stores.
- Write timings.
- Test vectors for the specification, which the supplement proposes.
