# Track B: transform kernels and decode speed

> **COORDINATION NOTE (written 2026-10-21 22:42 by the kernel session; resolved - the other session stopped editing at 23:15, Part II was written after)**: two Track B sessions are writing to this
> directory (edits to `track_b_util.h`, `track_b_bench.cpp`, `track_b_prof.cpp`, `track_b_canon.cpp`, `notes/track_b.md`,
> `tools/track_b_coldelta.py` appeared that I did not make). To avoid clobbering each other: the kernel session keeps
> `kernels/track_b_*.h`, `harness/track_b_bench.cpp` and `harness/track_b_index.cpp` (it is mid-edit: 32-lane build, fused flag list,
> AVX-512 canonicalisation, LEB/zstd codecs `b_leb_*`, cfg selection by histogram `estimate_payload`) and the final full-corpus
> timing runs; the other session please limit itself to `tools/track_b_*.py` analyses and review. Append, do not rewrite, this
> file; re-read a file before editing it. If you are the other session, reply here with one line saying what you are doing.
> Running background jobs of mine: `bin/track_b_index.exe` (index survey -> results/track_b_index.tsv), `bin/track_b_cm.exe`.


Status: complete as of 2026-10-22 (sections 1-7 written by the resumed session on 2026-10-21; Part II below holds the final, authoritative runs and supersedes any conflicting number above). Everything below is measured; "entropy" rows are
ideal static-model sizes (in-sample, tables excluded unless stated), codec rows are real bit-exact roundtrips through
`harness/track_b_bench.cpp` (matrix roundtrip verified on every field, including names).

Files: `kernels/track_b.h` (registration) -> `track_b_codecs.h` (codec factories, side info) -> `track_b_decode.h`
(chunk decoder + canonicalisation) -> `track_b_lane.h` (format, encoder, scalar decoder) + `track_b_avx512.h`
(AVX-512 kernels) + `track_b_rans.h` (static rANS tables) + `track_b_util.h`. Survey coders: `track_b_index.h`;
canonicalisation variants: `track_b_canon.h`, `track_b_sort.h`. Drivers: `harness/track_b_bench.cpp` (payload-only
accounting and timing), `track_b_index.cpp` (index-coder survey), `track_b_prof.cpp` (phase profile),
`track_b_canon.cpp` (canonicalisation micro-bench), `track_b_explore.cpp` (entropy estimates). Python:
`tools/track_b_models.py` (values / row order / CSR entropy), `track_b_bern.py`, `track_b_cluster.py`,
`track_b_gene.py`, `track_b_stats.py`, `track_b_hybrid.py`, `track_b_refrank.py`, `track_b_coldelta2.py`; result tooling `track_b_summarize.py`, `track_b_frontier.py`, `track_b_vs_today.py`, `track_b_run_all.sh`. Also `kernels/track_b_leb.h` (gap+LEB128 codecs `b_leb_raw/z3/z19`), `harness/track_b_selftest.cpp` (edge cases), `harness/track_b_cm.cpp` (adaptive context-mixing floor). Final results: `results/track_b_codecs_final.tsv`, `track_b_variants2.tsv`, `track_b_frontier.tsv/.md`, `track_b_index.tsv`. Earlier results: `results/track_b_dev.tsv` (codec runs), `track_b_index.tsv`
(survey), `track_b_models_*.txt`.

## Measurement caveat

The laptop is shared with three other agents. A 1-cycle dependency-chain calibration shows this thread getting the
equivalent of ~1.0 GHz of a 3.0 GHz core during most runs (OS time slicing + SMT sharing), so wall-clock Mnnz/s is
2-3x pessimistic and noisy. Every table reports the calibrated clock next to the wall time where available; TSC
ticks (constant 3 GHz) from `-DTB_PROFILE` builds are the most stable per-phase numbers.

## 1. Payload accounting (what the codec competes on)

Payload = column counts + coder tables + chunk streams. Side info = per-row nnz counts (LEB128+zstd-3, the
frequency permutation is re-derived from them: counting sort descending, ties by row id, identical to the writer).
Names/kv = zstd-3 of NUL-joined strings (Track A's problem; included in the total only).

## 2. Frontier so far (codec rows, bit-exact)

`b_rans16` = 32-lane lock-step static rANS over (gap class x value>1 flag) symbols with (position bucket x column
depth) contexts, extra gap bits as uniform rANS symbols, residual (value-2) classes in a second 32-lane stream;
probability precision pb=10 (pb 12 -> 10 costs <0.01 bits/nnz and keeps the tables in L2). Decode: AVX-512
(F/BW/DQ/CD/VBMI2) with a bit-identical scalar fallback (`b_rans16_scalar`).

| file | codec | payload bits/nnz | total bits/nnz | enc ms | dec ms (full) | payload dec, canonical order | payload dec, native (rank) order | today (chunks only) |
|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical exon (2.57 M nnz) | b_rans16 (AVX-512) | **5.73** | 16.60 | 18860* | 472 | 150 ms, 17 Mnnz/s | 41 ms, **62 Mnnz/s** | 6.8 bits/nnz, 11 Mnnz/s via Python |
| pbmc_v2_typical exon | b_rans16_scalar | 5.73 | 16.60 | 16625* | 642 | 413 ms, 6.2 Mnnz/s | 305 ms, 8.4 Mnnz/s | |
| tumor_3p exon (0.245 M nnz) | b_rans16 (AVX-512) | **9.87** | 124.3 | 2172* | 327 | 16.5 ms, 14.9 Mnnz/s | 3.3 ms, **74 Mnnz/s** | 14.4 (intron) |
| tumor_3p exon | b_rans16_scalar | 9.87 | 124.3 | 875* | 348 | 46 ms, 5.3 Mnnz/s | 16.5 ms, 14.9 Mnnz/s | |

(*) encoder tries all five context configurations and uses an O(steps x lanes) scan for the residual order; it is
being replaced by histogram-based selection. Full decode time is dominated by the names (zstd-3 of 3.4 MB of
strings, ~250 ms) which is Track A's problem. The 16-lane predecessor (`b_rans8`/first `b_rans16` rows in
`results/track_b_dev.tsv`) reached 33 Mnnz/s native; going to 32 lanes (2 zmm vectors in flight) doubled it.

Full corpus (`results/track_b_codecs.tsv`, all 28 rows roundtrip, both the AVX-512 and the scalar decoder give
identical bytes). Payload bits/nnz of `b_rans16` vs today's chunk-only bits/nnz (PLAN section 3 / Track A):

| file | nnz | b_rans16 payload | today's chunks | change | total bits/nnz (names zstd-3) |
|---|---|---|---|---|---|
| brain_sn_v3_deep exon | 6.35 M | 8.06 | 9.4 | -14% | 14.12 |
| brain_sn_v3_deep intron | 6.68 M | 7.26 | | | 12.03 |
| brain_sn_v3_intronic exon | 1.38 M | 9.45 | | | 36.84 |
| brain_sn_v3_intronic intron | 8.14 M | 7.51 | | | 11.38 |
| pbmc_v2_shallow exon | 5.30 M | 7.06 | 8.7 | -19% | 12.93 |
| pbmc_v2_shallow intron | 1.17 M | 12.12 | | | 36.17 |
| pbmc_v2_typical exon | 2.57 M | 5.73 | 6.8 | -16% | 16.60 |
| pbmc_v2_typical intron | 1.20 M | 8.18 | 9.1 | -10% | 28.97 |
| pbmc_v3_lowgenes exon | 1.76 M | 5.61 | | | 22.51 |
| pbmc_v3_lowgenes intron | 1.08 M | 7.15 | | | 32.09 |
| tcell_5p exon | 10.53 M | 6.01 | 6.9 | -13% | 8.90 |
| tcell_5p intron | 1.10 M | 9.84 | | | 34.50 |
| tumor_3p exon | 0.25 M | 9.87 | | | 124.3 |
| tumor_3p intron | 0.11 M | 12.20 | 14.4 | -15% | 244.9 |

Decode timings of that run were taken under heavy machine load (thread-cycles/nnz: native 32-58, canonical
66-119 for the AVX-512 path; scalar 175-327 / 324-567) and are superseded by the table in section 9 once the
renormalisation fix lands.

"Native order" = row ids in frequency-rank order within each column (what a redesigned format would store, with
rownames permuted once at write time); "canonical" = today's row order, which costs an extra sort per column
(see section 6). Both numbers are under the shared-machine clock (~1 GHz effective); phase profiles below are in
TSC ticks.

Phase profile, pbmc_v2_typical exon, 16-lane build, pb=10, TSC ticks (3 GHz) per nnz: main pass 21, flag scan 15
(now folded into the main pass), residual ctx 3.7, residual decode 3.6, scatter 1.6; canonicalisation ~40-50.

## 3. Index coding survey (`results/track_b_index.tsv`, 7 exon files, 1024-column chunks, index bits only)

Mean idx bits/nnz over the 7 exon files; decode = median Mnnz/s (shared-machine wall clock, includes the zstd
decompression where applicable) and median calibrated cycles/nnz.

| space | coder | raw bits | +zstd-3 | +zstd-19 | raw Mnnz/s (cyc/nnz) | zstd-3 Mnnz/s | zstd-19 Mnnz/s |
|---|---|---|---|---|---|---|---|
| rank | gap + LEB128 | 10.01 | 7.64 | 7.56 | 126 (5.3) | 57 | 36 |
| rank | gap + byte-split (today's layout) | 18.08 | 7.74 | 7.48 | 250 (3.7) | 67 | 54 |
| rank | Elias-Fano | 12.06 | 10.46 | 10.02 | 80 (14) | 30 | 24 |
| rank | binary interpolative | **7.29** | 7.11 | **7.09** | 7.4 (52) | 18 | 18 |
| rank | Roaring containers | 17.58 | 11.85 | 10.95 | **439 (1.4)** | 68 | 44 |
| rank | head bitmap 2048 + LEB tail | 9.83 | 7.74 | 7.65 | 109 (7.7) | 68 | 62 |
| rank | head bitmap 8192 + LEB tail | 10.28 | 7.77 | 7.66 | 130 (7.2) | 69 | 46 |
| orig | gap + LEB128 | 15.06 | 13.32 | 13.03 | 69 | 25 | 33 |
| orig | gap + byte-split | 24.30 | 12.96 | 12.53 | 166 | 58 | 63 |
| orig | Elias-Fano | 13.19 | 13.04 | 12.77 | 83 | 63 | 54 |
| orig | binary interpolative | 12.50 | 12.50 | 12.47 | 17 | 18 | 20 |
| orig | Roaring | 20.72 | 14.89 | 13.07 | 251 | 33 | 25 |
| rank | **static rANS, (pos x depth) ctx (`b_rans16`, index+flag part)** | ~5.0 (pbmc) | - | - | 62 native | | |

Intron files (7, mean idx bits/nnz raw / +zstd-3 / +zstd-19, rank space): LEB 11.32 / 9.69 / 9.67; byte-split
18.88 / 9.70 / 9.49; Elias-Fano 12.50 / 11.89 / 11.67; interpolative 9.29 / 9.12 / 9.11; Roaring 18.64 / 14.37 /
13.62; head bitmap 2048 11.66 / 9.93 / 9.88. Original order: LEB 15.77 / 14.33 / 14.30, interpolative 12.98 flat.
Same ranking as the exon files, ~2 bits/nnz higher (sparser columns, longer gaps). All 504 survey rows roundtrip.

Per file, rank space (LEB+zstd-3 | byte-split+zstd-3 | interpolative raw | interpolative+zstd-19 | Roaring raw
Mnnz/s): brain_sn_v3_deep exon 8.82 | 8.80 | 8.01 | 7.98 | 460; intron 7.71 | 7.67 | 7.01 | 6.95 | 451;
brain_sn_v3_intronic exon 10.23 | 10.23 | 10.11 | 9.62 | 318; intron 8.10 | 8.02 | 7.35 | 7.34 | 401;
pbmc_v2_shallow exon 7.77 | 7.97 | 7.31 | 7.24 | 439; intron 13.11 | 13.10 | 12.70 | 12.55 | 194;
pbmc_v2_typical exon 5.59 | 5.59 | 4.94 | 4.90 | 810; intron 9.01 | 8.85 | 8.19 | 8.19 | 734;
pbmc_v3_lowgenes exon 4.98 | 5.11 | 4.94 | 4.62 | 438; intron 6.59 | 6.70 | 6.43 | 6.31 | 337;
tcell_5p exon 6.06 | 6.04 | 5.39 | 5.36 | 732; intron 10.85 | 10.74 | 9.86 | 9.85 | 366;
tumor_3p exon 10.06 | 10.44 | 10.31 | 9.93 | 169; intron 11.93 | 12.29 | 12.69 | 12.07 | 93.

Entropy of the gap classes under a static (position bucket x depth) model, pbmc_v2_typical exon: 4.57 bits/nnz
(no context: 5.63; position only: 4.79; adding previous symbol or elements-so-far: 4.55, i.e. nothing left in
local context). Same file, original row order: 10.5.

Takeaways: (a) frequency-rank space is worth 5-6 bits/nnz for every coder; (b) zstd behind a structured coder is
worth ~2.4 bits/nnz on LEB and 10 on byte-split: today's layout only works because zstd re-learns the gap
distribution; (c) interpolative coding is the best universal coder (7.1) but 10x slower than LEB; (d) a static
rANS with a position x depth context beats all of them by 1.5-2 bits/nnz because the gap distribution depends
strongly on where in the rank space you are (that is what zstd cannot see); (e) Roaring/bitmaps are the fastest
raw decoders (439 Mnnz/s) but 2.4x larger: bitmaps only pay for the top few hundred rows and zstd cannot recover
the loss; (f) Elias-Fano is poor here because the within-column distribution is far from uniform.

## 4. Row order (`results/track_b_models_order.txt`, static pos x depth gap entropy, bits/nnz)

| order | pbmc_v2_typical exon | brain_sn_v3_deep exon | pbmc_v2_shallow exon |
|---|---|---|---|
| frequency sort (today) | 4.567 | 7.494 | 6.521 |
| frequency class (4 per octave), genomic order inside class | 4.573 | 7.497 | 6.525 |
| frequency class (1 per octave), genomic inside | 4.635 | 7.540 | 6.576 |
| gene frequency, exons by frequency | 6.820 | 9.477 | 8.422 |
| gene frequency, exons genomic | 6.940 | 9.571 | 8.439 |
| original (genomic) order | 10.537 | 11.444 | 13.432 |
| random | 10.941 | 11.942 | 13.682 |

Only the coarse frequency class matters (+0.006 bits/nnz for 4 classes per octave). Consequence: the permutation
can be replaced by a reference-level order plus a tiny per-file class table, and within a class the rows stay in
genomic order, which makes canonicalisation a k-way merge of pre-sorted runs instead of a sort (section 6).
Grouping by gene (gene-level presence + per-gene exon mask with contexts) costs 6.16 bits/nnz vs 4.57: exons of
a gene are not predictable enough from each other to beat per-exon frequency coding.

Co-expression / column order: held-out Bernoulli(rank, depth-bucket) model on the top-4096 rows, pbmc_v2_typical
exon: 3.470 bits per head nnz; the same with a k-means cell-cluster id (K=4..32 on log-normalised top-1000 genes,
PCA-32) as side information: 3.446 best (K=4, +0.006 side info), worse for K>=16. The gap coder with
(pos x depth) context (3.469 held-out on the same elements) already reaches the independent-Bernoulli model;
cluster structure buys <1%. Column-to-column delta coding is therefore not pursued as a kernel (a learned
conditional model, Track C, is the only way to get more than a few % here).

## 5. Orientation and values

CSC (rows by frequency) vs CSR (cells by depth), static gap entropy + count stream, bits/nnz:
pbmc_v2_typical 4.567+0.043 vs 4.546+0.185; brain_sn_v3_deep 7.494+0.086 vs 7.465+0.189; pbmc_v2_shallow
6.521+0.119 vs 6.545+0.099. Identical to within 0.1 bits: orientation is a layout choice (random column access
wants CSC), not a compression lever. Hybrid (dense rows as CSR bitmaps) is covered by the head-bitmap survey rows:
no gain.

Values, pbmc_v2_typical exon (75% ones), bits/nnz: flag "value>1" with no context 0.82, with (own rank bucket)
0.51, with (own rank bucket x depth) 0.47, joint with the gap symbol 0.46 (what `b_rans16` does). Residual
(value-2) classes: no context 0.69, (own rank bucket/2 x depth class) 0.60, (own x depth) 0.58. Single-symbol
value class with context 1.05; Rice/Golomb with per-rank-bucket k 1.62; VOCSC-style group-by-value 5.74 index
+ 0.20 headers = 5.94 vs flag+gap 5.03 + residual 0.60 = 5.63. Group-by-value loses ~0.3 bits/nnz because each
value group restarts the gap chain. A per-column sum(value-1) hint costs 0.03 and buys 0.01-0.02: not worth it.

## 6. Canonicalisation (rank order -> original ascending row ids)

`harness/track_b_canon.cpp`, pbmc_v2_typical exon (86% of nnz in columns of 257-1024 elements), wall ns/nnz on
the shared machine (`results/track_b_prof_run2.txt`): AVX-512 bitonic sort16 + 16x16 bitonic merge runs on
packed (orig id << 13 | position) keys (`track_b_sort.h`, `CanonVec`) **13 ns/nnz**; two-level bitmap over
original ids (`Canon`) 40-48; u16 value table variant 45-50; std::sort of packed 64-bit keys 120-156;
identity-permutation probe of the bitmap (no random `perm[]` gather) 24-28. The merge sort is 3.4x faster than
the bitmap and is now the canonical path of `b_rans16` (scalar fallback stays `Canon`). The redesign answer is
to not need it at all: store rows in frequency-class order (section 4) and permute rownames once at write time.

## 6b. Where the main-pass cycles go (`results/track_b_prof_run2.txt`, thread cycles, pbmc_v2_typical exon)

Chunk decode = 42.6 thread-cycles/nnz (70 Mnnz/s at the 3.0 GHz nominal clock) and the gap-table size does not
matter: 1920 KB (480 contexts x 2^10, cfg 0), 480 KB (cfg 1, pb 9), 160 KB (cfg 3), 80 KB (cfg 4) all give
23-29 TSC ticks/nnz in the main pass. Size cost of the smaller tables: cfg 0 5.731 bits/nnz, cfg 0/pb 9 5.744,
cfg 1/pb 9 5.811, cfg 3 5.983, cfg 4 6.007. So the gather is not the bottleneck; the per-element phase costs are
main pass ~24, value fill 2.7, residual ctx 2.4, residual decode 2.4, value scatter 5.7 TSC ticks/nnz.

## 7. What worked / what did not / why

Worked: frequency-rank space (5-6 bits/nnz on every coder); position-in-rank-space x depth contexts with a
static coder (-1.5 to -2 bits/nnz vs zstd on the same gaps); folding the value>1 flag into the gap symbol
(0.46 bits/nnz, cheaper than any separate flag stream); lock-step multi-lane rANS with one shared word stream
(expand-loads make renormalisation branch-free; 32 lanes = 2 vectors in flight hides gather latency);
pb=10 tables (L2 resident) instead of pb=12.

Did not work: Elias-Fano and Roaring (distribution is far from uniform within a column); gene-level grouping;
cluster side information; CSR; per-column value hints; richer local contexts (previous symbol, elements so far):
the gap sequence is memoryless given position and depth.

(Resolved in section 6b and Part II section 10: table size is not the limiter.) Open: the main pass is still ~21 TSC ticks/nnz; the gather into a 1.9 MB table (cfg 0, 480 contexts x 1024
slots) is the suspected bottleneck (fits L3 only), to be tested against 40-80 KB tables (fewer contexts, pb=9).


---

# Part II: final measurements (kernel session, 2026-10-22)

Everything in Part II was produced by the final build (`bin/track_b_bench6.exe` = `harness/track_b_bench.cpp`, `bin/track_b_selftest.exe`) after the numbers in
sections 1-7 above were written; where a number differs from sections 2-6 this part supersedes it. Reproduce with
`tools/track_b_run_all.sh` (codecs on all 14 files + knob grid), `bin/track_b_bench6.exe --codec b_rans16,b_rans16_scalar ...`,
`bin/track_b_selftest.exe`, `bin/track_b_index.exe`, `python -P tools/track_b_frontier.py ...`.

## 8. Timing method and what the Mnnz/s columns mean

The laptop is power/thermally throttled and shared: a 1-cycle dependency-chain calibration (`calib_ghz` in the drivers) measured
0.8-1.4 GHz of effective core clock during the runs (3.0 GHz base, 4.8 GHz turbo), with all 8 hardware threads busy. The drivers
therefore run at HIGH process priority, take the minimum of 5 repetitions, and report three things per decode:
(a) wall-clock Mnnz/s (what this machine did), (b) thread TSC ticks per nnz (`QueryThreadCycleTime`, excludes time scheduled out),
(c) **est@3GHz** = 3000 / (wall ns/nnz x calibrated core GHz): what the same instruction stream would deliver on an unthrottled
3 GHz core, assuming the cycle count is frequency independent. (c) is an estimate (memory latency does not scale with the clock, so it
is mildly optimistic, and the calibration is itself noisy); (a) is the measured fact. Treat run-to-run noise as +-25%.
"native" = payload decode with rows left in frequency-rank order inside each column (no canonicalisation); "canonical" = rows mapped
back to original ids and sorted ascending per column, i.e. what `decode() == M` requires. Both are payload only: they include the rANS table build, the column-count stream and (canonical) the sorter set-up, and exclude names, the row-count
side info (permutation re-derivation) and, because the output vectors are reused across repetitions, their allocation (the harness `decode_ms` includes names, which
cost ~250 ms per file and dominate every codec's full-decode time).

## 9. Final codec table (14 files, `results/track_b_codecs_final.tsv`, every row verified `matrix_ok = 1`)

| codec | what | mean payload bits/nnz | byte-weighted vs today | native Mnnz/s wall (est@3GHz) | canonical Mnnz/s wall (est@3GHz) |
|---|---|---|---|---|---|
| `b_rans16` | 32-lane static rANS, AVX-512, auto chunk size, best of 5 context sets | **8.23** | **-15.1%** | **72.6 (172)** | **34.0 (80)** |
| `b_rans16_c1024` | same, fixed 1024-column chunks | 8.29 | -15.0% | 72.2 (172) | 37.3 (88) |
| `b_rans16_scalar` | bit-identical scalar fallback | 8.23 | -15.1% | 16.1 (36) | 9.2 (22) |
| `b_leb_z19` | rank-space gap LEB128 + (value-1) LEB128, zstd-19 per 1024-column chunk | 9.67 | +0.9% | 51.5 (124) | 31.4 (73) |
| `b_leb_z3` | same with zstd-3 | 9.99 | +5.1% | 48.4 (114) | 30.6 (73) |
| `b_leb_raw` | same bytes, no second stage | 19.06 | +112% | 99.3 (229) | 43.2 (106) |
| today's `.1pz` (chunks + pointer block) | VOCSC + zstd-3 | 10.00 (mean of the 14) | 0 | not re-measured here (the production reader decodes ~11 Mnnz/s through Python, the C++ reference codec in the harness 11 Mnnz/s including names) | |

"Today" payload = `chunks_z + ptr_z + 4 x chunks` from the file header (what `tools/track_b_vs_today.py` computes). The mean over files is unweighted;
the byte-weighted column is the size change over the whole corpus. Per file (payload bits/nnz; `tables+counts` and `row-count side` are
included in / excluded from the payload respectively):

| file | nnz | b_rans16 | of which tables+counts | row-count side info (not in payload) | LEB+z19 | LEB+z3 | LEB raw | today |
|---|---|---|---|---|---|---|---|---|
| brain_sn_v3_deep exon | 6.35 M | 8.06 | 0.10 | 0.20 | 9.60 | 9.87 | 18.75 | 9.44 |
| brain_sn_v3_deep intron | 6.68 M | 7.26 | 0.09 | 0.19 | 8.65 | 9.01 | 18.01 | 8.34 |
| brain_sn_v3_intronic exon | 1.38 M | 9.39 | 0.34 | 0.59 | 11.40 | 11.63 | 20.20 | 11.87 |
| brain_sn_v3_intronic intron | 8.14 M | 7.51 | 0.08 | 0.15 | 8.82 | 9.11 | 18.18 | 8.45 |
| pbmc_v2_shallow exon | 5.30 M | 7.05 | 0.13 | 0.13 | 8.54 | 8.89 | 18.03 | 8.81 |
| pbmc_v2_shallow intron | 1.17 M | 12.01 | 0.47 | 0.61 | 13.76 | 13.79 | 22.92 | 14.34 |
| pbmc_v2_typical exon | 2.57 M | 5.74 | 0.08 | 0.25 | 6.90 | 7.40 | 16.94 | 6.85 |
| pbmc_v2_typical intron | 1.20 M | 8.18 | 0.14 | 0.59 | 9.46 | 9.65 | 18.61 | 9.18 |
| pbmc_v3_lowgenes exon | 1.76 M | 5.56 | 0.19 | 0.07 | 6.71 | 7.37 | 16.82 | 7.35 |
| pbmc_v3_lowgenes intron | 1.08 M | 7.07 | 0.27 | 0.21 | 8.27 | 8.80 | 17.43 | 8.81 |
| tcell_5p exon | 10.53 M | 6.02 | 0.05 | 0.10 | 7.17 | 7.67 | 17.06 | 6.96 |
| tcell_5p intron | 1.10 M | 9.75 | 0.28 | 0.65 | 11.41 | 11.54 | 20.66 | 11.60 |
| tumor_3p exon | 0.25 M | 9.75 | 0.60 | 1.62 | 11.24 | 11.42 | 20.29 | 12.50 |
| tumor_3p intron | 0.11 M | 11.92 | 1.11 | 3.04 | 13.51 | 13.71 | 22.92 | 15.47 |

`b_rans16` beats today's chunk payload on every file (-11% to -24%); a plain gap/varint stream behind zstd-19 only matches it
(+0.9% overall; it is 3-13% below today on the sparse lowgenes / tumor / shallow-intron files and 1-10% above on the dense ones). The row-count side stream
(from which the frequency permutation is re-derived) is not free on small matrices (1.6-3 bits/nnz on the two tumor_3p files); see
section 11 for why it still pays.

## 10. Knob study (`results/track_b_variants2.tsv`, 3 files, same run as the 1024-chunk baseline)

payload bits/nnz / native Mnnz/s wall / canonical Mnnz/s wall (est@3GHz native / canonical):

| codec | pbmc_v2_typical exon | tumor_3p exon | brain_sn_v3_deep intron |
|---|---|---|---|
| `b_rans16_c1024` (baseline: best-of-5 contexts, pb 10) | 5.73 / 57 / 40 (134 / 104) | 9.87 / 79 / 31 (178 / 71) | 7.26 / 67 / 36 (155 / 81) |
| chunk 256 columns | 5.76 / 81 / 46 (190 / 106) | 10.38 / 58 / 28 (147 / 63) | 7.32 / 87 / 39 (220 / 96) |
| chunk 4096 columns | 5.72 / 48 / 30 (109 / 66) | 9.75 / 78 / 28 (193 / 73) | 7.25 / 58 / 32 (158 / 72) |
| pb 11 | 5.73 / 67 / 34 (154 / 80) | 9.88 / 65 / 29 (168 / 65) | 7.26 / 68 / 34 (190 / 80) |
| pb 12 | 5.73 / 47 / 34 (107 / 84) | 10.06 / 21 / 14 (53 / 36) | 7.26 / 61 / 32 (148 / 72) |
| ctx 0: position(40) x depth(12), pb 10 | 5.73 / 71 / 43 (157 / 91) | 9.97 / 42 / 20 (98 / 54) | 7.26 / 68 / 33 (166 / 78) |
| ctx 1: position(40) x depth(6) | 5.79 / 63 / 43 (154 / 108) | 9.92 / 75 / 24 (172 / 69) | 7.33 / 78 / 37 (179 / 92) |
| ctx 2: position(20) x depth(6) | 5.81 / 72 / 42 (164 / 95) | 9.87 / 85 / 29 (195 / 66) | 7.35 / 78 / 37 (176 / 93) |
| ctx 3: position(40) only | 5.98 / 83 / 41 (189 / 96) | 10.10 / 91 / 35 (219 / 74) | 8.00 / 79 / 37 (226 / 92) |
| ctx 4: position(20) only, pb 10 | 6.01 / 78 / 35 (186 / 79) | 10.09 / 82 / 32 (211 / 75) | 8.02 / 76 / 37 (195 / 93) |
| ctx 4, pb 9 | 6.08 / 77 / 40 (196 / 97) | 10.12 / 95 / 27 (218 / 62) | 8.10 / 82 / 37 (190 / 101) |

Reading: probability precision between 9 and 12 bits is size-neutral (+-0.01 bits/nnz) but pb 12 (16 KB per context) is the one setting
that is clearly slower; context richness is the only size lever (position x depth at 5.73 -> position only at 5.98-6.01, +4-5%), and
buying that back as speed gives only ~+15-25% native decode: the decoder does not look cache-bound (tables from 80 KB to 1.9 MB decode at about the same speed); by my uop count it executes ~110-130
vector uops per 16 elements, which is where the time goes. Chunk size trades output-array locality (256 columns = 90 KB-class working
set, fastest on dense columns) against per-chunk overhead (lane states: +5% on the sparse tumor file), so the shipped default picks
the number of columns per chunk from the mean column density (~96 K nnz per chunk), which keeps sizes at or below the 1024-column
numbers on every file (`b_rans16` row of section 9).

## 11. More experiments

**Row order must be file specific.** `tools/track_b_refrank.py`: code each human exon file with the rank order of the pooled row counts
of the *other* four studies (leave-one-study-out, same reference build, so the rows align) instead of its own row counts:
static position x depth gap cost, bits/nnz, own order (+ shipping the row counts) vs leave-one-out reference order:
pbmc_v2_shallow 6.52 (+0.13) vs 7.94; pbmc_v2_typical 4.57 (+0.25) vs 6.15; pbmc_v3_lowgenes 3.99 (+0.07) vs 8.77;
tcell_5p 5.03 (+0.10) vs 8.18; tumor_3p 8.67 (+1.62) vs 12.86 (`results/track_b_refrank_exon.txt`). A shared reference order costs +1.3 to +4.7
bits/nnz because the *support* (which 3-36% of the 310k rows a sample ever touches) is what the file-specific order encodes; the
row-count side stream is a bargain even at 1.6 bits/nnz. If the permutation is to become a model, it has to be a prior over supports
(Track C territory), not a frequency list.

**Hybrid orientation (dense rows as raw CSR bitmaps over called cells + CSC remainder)**, `tools/track_b_hybrid.py` /
`results/track_b_models_hybrid.txt`, bits/nnz incl. counts: pbmc_v2_typical baseline 4.61; bitmap rows (theta = share of called cells that
contain the row) 0.15/0.25/0.40/0.60 absorb 44/37/29/22% of nnz and give 5.00/4.86/4.83/4.83 raw (4.75/4.73/4.72/4.73 if the bitmaps were
entropy coded). tcell_5p baseline 5.07 -> 5.45/5.27/5.23/5.22 raw. brain_sn_v3_deep baseline 7.58 -> 7.75/7.63/7.59/7.58. A hybrid is
3-8% larger than pure CSC for the same reason bitmaps lose to gap coding anywhere: the presence probability of a row depends on the
cell's depth, a raw bitmap cannot use that.

**Adaptive context mixing** (`harness/track_b_cm.cpp`, `results/track_b_cm.log`): binary decomposition of the gap class with up to five mixed
models (position x depth, position x previous class, position x elements-so-far, exact previous rank for the first 4096 ranks,
finer position): 4.584 bits/nnz on pbmc_v2_typical exon vs 4.567 for the static table (+ 0.03 tables) and the same on tumor_3p
(8.73) and pbmc_v2_shallow intron (11.49); extra contexts move it by < 0.01. So the "size regardless of speed" floor for this transform is the static
model; any further gain must come from information outside (position, depth): Track C's latent models.

**Column order / column-to-column deltas** (`tools/track_b_coldelta2.py`, `results/track_b_coldelta.txt`, pbmc_v2_typical exon, 5340 called cells k>=200, top-4096 rows = 75% of nnz, static in-sample entropies, bits per head nnz): independent (rank bucket x depth) model 3.318; adding the same row's presence bit in the previous column gives 3.317 in stored order, 3.316 sorted by depth, 3.318 random, and 3.306 (-0.4%) with cells chained by nearest neighbour in a 32-d PCA of the log counts (per-rank oracle: 3.292 -> 3.269, -0.7%). Reordering cells would also cost a permutation (about log2(n) bits per cell). Locality order + column deltas is worth less than 1%; dropped. (The first version of this script, `track_b_coldelta.py`, used `np.add.at` and did not finish in 27 CPU-minutes; `coldelta2` is the same experiment with bincount.)

## 12. Robustness (`harness/track_b_selftest.cpp`)

280 codec runs over 20 synthetic and edge-case matrices (empty, zero columns/rows, single entry, fewer columns than lanes, ties in row
counts, all columns k=1, 1025 columns across chunk boundaries, columns of 8191/8192/8193/16000/20000 nnz around the sorter's 8192
limit, values up to 2e5, m = 524287 at the format limit): 0 failures. Documented rejections (exceptions, never silent corruption):
the lane codec supports m <= 2^19 rows and value-2 < 2^19 (gaps and values are coded as class + <= 16 uniform bits); the LEB codecs have
no such limit.

## 13. Frontier, target status and recommendations for the format redesign

Frontier (payload only, bits/nnz vs single-thread Mnnz/s, native / canonical):

| point | codec | bits/nnz (byte-weighted vs today) | native wall (est@3GHz) | canonical wall (est@3GHz) |
|---|---|---|---|---|
| best size | `b_rans16`, size-optimal context set (5.56-12.0, mean 8.23) | -15.1% | 73 (172) | 34 (80) |
| faster, smaller context set | `b_rans16` ctx 3/4 (position only), 3 files | -4% to -19% | 76-95 (186-226) | 27-41 (62-101) |
| best raw speed, any size | `b_leb_raw` (full codec, +112%); index-only survey: Roaring containers 419 Mnnz/s, byte-split gaps 211, both ~18 bits/nnz | +112% | 99 (229) full codec | 43 (106) |
| balanced | `b_rans16` default | -15% | 73 (172) | 34 (80) |
| scalar fallback | `b_rans16_scalar` | -15% | 16 (36) | 9 (22) |
| yardstick | gap+LEB+zstd-19 | +0.9% | 52 (124) | 31 (73) |

Target ">= 200 Mnnz/s single-thread on the payload while matching today's bits/nnz": **met on the estimate, not on this machine's wall clock, and only in native order.**
Measured here: 73 Mnnz/s median native (up to 101 on pbmc_v2_shallow exon) at ~1.3 GHz effective; the same code estimates to
172 Mnnz/s (up to ~250) at 3 GHz; canonical order halves that (34 measured / 80 estimated). Size beats today by 15% (11-24% per file).

Insights for the format redesign:

1. **Row order is a decode-speed decision as much as a size decision.** Frequency-rank order saves 6 bits/nnz for every coder
   (10.5 -> 4.6 static gap entropy), but the production file hands back original row order, so every column has to be sorted again
   after decoding. That sort (a 16-wide bitonic merge sort here: 13 ns/nnz vs 38 for a two-level bitmap and 100-129 for std::sort on the same
   machine) costs as much as the entropy decoding itself: canonical 34 Mnnz/s vs native 73. A format that stores the matrix in the file's own
   frequency-class order (coarse classes lose only +0.006 bits/nnz at 4 per octave; rows in the same class stay in genomic order) and
   permutes the rownames once per file removes the sort for every consumer that does not care about gene order, and the row counts it is
   derived from are already in the file. Do not try to replace the per-file order by a reference-build order: that costs 1.3-4.7 bits/nnz.
2. **Position-in-rank-space x column depth is the whole model, and a static table coded by rANS reaches its floor.** The gap
   stream is memoryless given those two contexts (held-out Bernoulli per (rank, depth) 3.470 vs gap model 3.469 bits per head nnz; adaptive
   context mixing with five more contexts gains < 0.01); gene grouping, cluster side information, column neighbours, CSR/hybrid bitmaps and
   better numeric coders cannot beat it (index bits/nnz on pbmc_v2_typical exon, counts excluded: static position x depth rANS 4.57, binary
   interpolative 4.94, gap+LEB+zstd-3 5.59, Elias-Fano 11.5, Roaring 16.2; over all 14 files interpolative 8.2, LEB+zstd-3 8.6), and zstd behind
   any of them cannot see the position. The remaining 4.6 bits/nnz of index information is sampling noise; the next
   gain is a learned latent model (Track C), not a better kernel.
3. **Fold the value flag into the index symbol and keep the rest of the value stream small.** `value > 1` coded jointly with the gap
   class costs 0.46 bits/nnz on pbmc_v2_typical (separate flag stream with the best context: 0.47; VOCSC group-by-value: +0.3 bits/nnz
   worse because every value group restarts the gap chain); the residual (value-2) with (rank bucket x depth) contexts costs another 0.6.
   Values are ~1.06 bits/nnz = 18% of the payload and per-column hints (sum of counts) buy 0.03 bits for 0.03 bits of side information.
   Lay the stream out as 32 independent lanes with one shared word stream, chunk blobs that are independently decodable (range requests),
   and keep a scalar decoder: the vector decoder is 3.7-4.5x faster than the scalar one on the same bytes.
