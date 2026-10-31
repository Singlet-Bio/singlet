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

# Internal peer review, round 2: response

Three independent Opus 5.5 reviews of draft 4 (2026-10-30), written as reviewers for Nature Methods (single-cell
infrastructure), Genome Biology (formats, compression and reproducibility) and PVLDB (storage, compression, cloud data
systems), plus a survey of how 24 format, compression and data-systems papers present results. All three recommended
major revision. Their reports are not in the repository.

## Done in this revision (no new measurements)

**Presentation**
- Fig. 1a shows one point per matrix (size relative to `.s1`) with the median marked, instead of median bars.
- Fig. 1d shows thread scaling from the existing sweep, instead of throughput by storage tier, which duplicated Fig. 2c
  and joined categorical tiers with lines.
- New Table 3 gives every format on the same matrices: median size, geometric-mean size relative to `.s1`, smallest-file
  count, median cold read rate and fastest-read count.
- New table of format properties beyond size and speed: reads in part over HTTP, integrity, data digest, members and
  cells table, integer guarantee. A paragraph states what is specified but not evaluated, including that there is no
  checksum per chunk.

**Numbers and wording**
- The "most compact alternative" ratio is now paired by matrix: on each matrix, the smallest other file. It was a ratio
  of medians.
- The prefix size is stated with the chunk index: 24 bytes per chunk, about half a megabyte at a billion non-zeros.
- The `singlet` version in Table 1: the cited release's package code is the code that ran, and the version strings the
  runs recorded are named.

**Fixes**
- The broken `\textsuperscript` in the Table 5 caption.
- The stray "Tables" heading.
- The 4-versus-8-cores wording.
- The Ref column running off the page in Table 2.

## Proposed (new measurements; not done yet)

1. **Decode protocol.** Import each library before timing, report the first cold repeat and the median separately, and
   rerun the decode stage. Today the first repeat includes imports, and repeats 2–3 are warmed by the RAID controller's
   cache.
2. **Latency.** Sweep round-trip times from 0 to 200 ms (5 or 6 values), sweep requests in flight from 1 to 64, and plot
   per-minibatch CDFs.
3. **A real object store.** Cloudflare R2 or S3, read from a cloud VM: measured transfers, requests per epoch, and dollars.
4. **Random-access costs.** Over-read bytes, index bytes and requests per minibatch, and a chunk-size sweep.
5. **Baselines.**
   - Sharded Zarr v3.
   - The same chunks and index with a generic codec (zstd or Blosc2), to separate the codec from the container.
   - scDataset with its default `fetch_factor`.
   - Loaders under DataLoader workers.
6. **Hardware.** An AVX-512 VBMI2 CPU and an ARM CPU (codec only, cycles per non-zero).
7. **Write.** Write time and peak memory on the timing host.
8. **Format.** Optional per-chunk checksums, a self-contained specification of the cells table and packs, test vectors,
   and a bit-flip experiment.

# Internal peer review, round 3: measurements for the round-2 proposals

All on Clipper. The campaign ran on the paper's branch, the decode stage on the b001–b004 nodes and the rest on the same
hardware; `singlet` 1.3.3 (PyPI) was released during the campaign and is the package for the new loader runs.

## Done

1. **Decode protocol.** Before timing, each process writes and reads a 60×40 matrix in the same format, untimed. The
   first cold repeat is now a median 1.00× the median of three (90th percentile 1.08×, 1,238 configurations). The
   controller cache cannot be flushed from a job (a file just written reads at about 5 GB/s after 16 GB of other I/O,
   against 1.4 GB/s from the disk). Methods says so, and "cold" is defined as evicted from memory.
2. **Latency and requests in flight.** Added round-trip times of 0–200 ms at 1 Gbit/s, with CDFs at 30 ms and 1–64
   requests in flight with 0–8 minibatches ahead (Fig. `fig_reads` a–c). The `.s1` reader with 1.3.2's settings fell
   from 57,000 cells/s to 16,000 at 30 ms; 64 requests restore it to 59,000 (70% of the link). This led to **1.3.3**: 32
   requests and four minibatches ahead by default, and a fix for `S1Cells` under forked DataLoader workers.
3. **A real object store.** Cloudflare R2 through a Worker that serves the bucket by range (Table `tab:r2`). Read from a
   cluster node over the public internet, **not from a cloud VM**: about 70 ms per request, 15 ms of it to the edge.
   On HLCA core: first minibatch 0.67 s (`.s1`), 1.5 s (AnnData Zarr), 2.1 s (Zarr v3), 101 s (download the h5ad).
   Sustained: 6,100 / 15,100 / 23,200 cells/s for `.s1` at 1.3.2 settings, 1.3.3 defaults and 64×8, against 540 and 340
   for the two Zarr layouts. Per epoch: 0.92 GiB in 8,835 requests, $0.086 at S3 prices, against $1.76 for AnnData Zarr.
   Limitations names the single path, time of day and list prices.
4. **Random-access costs.** Bytes, requests and over-read per minibatch from each format's own chunk index, and a
   chunk-size sweep on three matrices (Table `tab:access`, Fig. `fig_reads` d), 29 matrices × 200 minibatches. `.s1`:
   761 KiB in 8 range requests, over-read 1.4. h5ad gzip: 2,533 KiB in 22 requests. AnnData Zarr: 8,164 KiB in 22,
   over-read 7.9. On HLCA core, 4,096-non-zero chunks bring the over-read to 1.0 for an 8% larger file, and
   393,216-non-zero chunks raise it to 2.0 for a 0.5% smaller file, so the 49,152 default stays.
5. **Baselines.**
   - Sharded Zarr v3 control: CSR arrays in 49,152-element chunks (one `.s1` chunk), 64 per shard, Blosc zstd with
     bit shuffling. It is 2.3× larger than `.s1` (12.0 against 5.3 bits per non-zero) and is in Fig. 1, Table 3, the
     minibatch table and the R2 table. This is the generic-codec-at-the-same-granularity control; we did not put a
     generic codec inside the `.s1` container itself.
   - scDataset with its default `fetch_factor` (16) added to the minibatch table.
   - DataLoader workers (Table `tab:workers`): scDataset gains 3.7× with four workers; `S1Cells` gains nothing,
     because its decoding already uses all cores.
7. **Write.** Write time and peak memory for every format on PBMC 10k, Tabula Muris FACS and HLCA core (Table
   `tab:writecost`). The `.s1` writer is among the faster ones (119 s on HLCA core against 205 s for h5ad gzip and
   915 s for TileDB-SOMA). It needs more memory than most (31 GB beyond the matrix against 9 GB for h5ad gzip). The
   text says both.

## Not done

6. **Hardware** (AVX-512 VBMI2 and ARM). Clipper has neither; still a limitation.
8. **Format** (per-chunk checksums, a self-contained specification of the cells table and packs, test vectors, a
   bit-flip experiment). This needs format work, not measurement. It remains in Limitations and future work.
