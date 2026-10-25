# Singlet store: measured performance against the legacy format

What changed for storage, network transfer, decode time and memory when `.singlet` bundles (ZIP64 of TP1Z `.1pz` members)
are replaced by store per-sample objects (`docs/format-store.md`). Every number below comes from a run recorded in
`bench/compression/results/` or the production pilot (`docs/store-migration.md`, status log 2026-10-22). Machine: laptop,
Intel i7-1185G7 (4 cores / 8 threads, AVX-512), Windows, Python 3.11, R 4.4.1, a home connection measured at about 32 MB/s on one
stream to `data.singlet.bio`. Laptop timings are min of 3 fresh processes at 20 to 45% background load unless stated; read them as
upper bounds on what a quiet machine does.

## 1. Bytes stored

| what | legacy | store | ratio | source |
|---|---|---|---|---|
| GSE145370, 14 samples (10x 3', human tumour) | 187,066,350 B | 31,846,608 B (+ 90 KB manifest; packs shared) | **5.87x** | production pilot, campaign code |
| 58 samples, 45 studies, all 6 reference builds, 10+ protocols | 1,777.8 MB | 443.0 MB | **4.01x** | `notes/m_w2.md` (6,351 verify checks, 0 failures) |
| of which: deep 10x and plate-based samples | 978.6 MB | 383.0 MB | 2.56x | |
| of which: rewritten (gen B) bundles with STARsolo outputs | 612.5 MB | 28.2 MB | 21.7x | derived members regenerate exactly |
| 14 corpus count matrices alone (exon + intron) | 114.09 MB | 44.11 MB | 2.59x | `notes/m_w1.md` (names excluded: packs) |
| live atlas (8,202 studies) | about 2.7 TB | about 0.8 to 1.4 TB | 1.9x (verified levers) to 3.5x (all levers) | `PROPOSAL.md` section 3, projection |

Where the bytes went: feature names now live in shared packs (5.42 MB for all six builds, shipped in the Python wheel), row
permutations, column pointers and column sums are derived, barcodes are whitelist indices, and 14 recipes regenerate derivable
members (PSI, cell calls, STARsolo filtered matrices, raw barcode lists, features, ambient profiles, gene-expression and per-cell QC
tables) byte for byte. The count payload itself is 15% smaller (class-order rANS instead of VOCSC + zstd-3).

## 2. Bytes and requests to get data (GSE145370, measured through HTTP Range)

| task | legacy | store | bytes ratio |
|---|---|---|---|
| one sample (GSM4317426, 2,928 cells) | the whole study bundle: 187,066,350 B, 1 request (the client always downloads the study) | **2 requests, 2,101,298 B** | 89x fewer |
| one sample, manifest already cached (304) | same | 2 requests, 2,010,651 B | 93x fewer |
| whole study, 14 samples | 187,066,350 B | **15 requests, 11,696,362 B** (exon, intron, cells members only) | 16x fewer |
| 1,000 cells from one sample | the whole bundle | **4 requests, 1,917,176 B** (only the chunk runs covering those cells) | 98x fewer |

Network time on this connection (model from the measured time to first byte and throughput; `results/m_w4_netmodel.txt`):

| task | legacy | store |
|---|---|---|
| one sample | 6.39 s | **0.61 s** |
| whole study | 6.17 s | **0.91 s** |
| 1,000-cell slice | 6.2 s (whole bundle) | **0.82 s** |

Server side: legacy bundles were never edge-cached (`cf-cache-status: DYNAMIC`, and half the bytes sit in objects over the 512 MB CDN
limit). store objects are immutable, content-addressed, at most 132 MB, and served by the `singlet-data` Worker with CORS and the Cache
API, so repeat reads come from the edge. Stale offsets, which broke partial downloads for half the atlas on 2026-10-22, cannot occur
with content-addressed keys.

## 3. Time to an in-memory matrix

Python (`singlet.load`, AnnData with X, spliced and unspliced layers; legacy rows exclude its 187 MB download):

| task | legacy | store local file | store over HTTP | speedup |
|---|---|---|---|---|
| one sample | 1.84 s | 0.38 s | **0.36 s** | 5x (17x counting the download) |
| whole study (40,614 cells x 38,606 genes) | 20.4 s | 0.86 s | **0.92 s** | 22x (29x counting the download) |
| 1,000-cell slice | n/a | | **0.32 to 0.38 s** | |
| production pilot, real internet, cold edge cache | 20.12 s local decode + about 6 s download | | **4.29 s** | about 6x |

The production pilot's store AnnData is identical to the legacy reader's: X, both layers, cell and gene names in the same order, every obs
column.

R (`singlet::load`, local files, `results/m_w4_r_time.txt`):

| task | legacy | store | speedup |
|---|---|---|---|
| one sample | 2.31 s | **0.23 s** | 10x |
| whole study | 27.28 s | **2.42 s** | 11x |

Matrix decode kernel alone (payload, single thread unless stated, quiet window; `results/m_w1/timing_final.tsv`):

| decoder | Mnnz/s |
|---|---|
| legacy production reader (full decode incl. per-column sort and name parsing) | 4 to 21 |
| store, AVX-512, native (class) row order | **258 to 299** (334 to 459 on a boosted core) |
| store, AVX-512, native order, 4 threads | **640 to 1,084** |
| store, AVX2 | 161 to 252 |
| store, scalar (any CPU) | 26 to 48 |
| store, canonical row order (legacy ordering), 1 / 4 threads | 56 to 71 / 181 to 213 |

All three ISAs produce identical bytes and identical decoded matrices; the build needs no `-march` flag (runtime dispatch).

## 4. Memory

| task | legacy | store | change |
|---|---|---|---|
| Python, one sample: peak RSS growth | 184 MB | **97 MB** | -47% |
| Python, one sample: peak traced Python allocations | 152 MB | **79 MB** | -48% |
| Python, whole study: peak RSS growth | 277 MB | **224 MB** | -19% (the result itself is about 150 MB) |
| R, one sample: peak heap growth over the 402 MB baseline | 315 MB | **144 MB** | -54% |
| R, whole study: peak heap growth | 504 MB | **454 MB** | -10% |

The legacy path decoded each exon and intron matrix twice per sample, wrote temporary files, copied arrays element by element in the
binding and built Python strings for every feature name; store decodes once into caller-owned arrays and resolves names lazily from packs.

## 5. Caveats

* Laptop measurements under background load; a quiet core decodes 10 to 50% faster.
* The atlas-wide size is a projection from samples (section 1, last row); the per-study numbers are measured.
* The network model uses this connection's measured constants; links with higher latency gain more from store's fewer requests.
* The cluster campaign that migrates the remaining studies records per-study sizes and timings in `store/ledger/<GSE>.jsonl`.
