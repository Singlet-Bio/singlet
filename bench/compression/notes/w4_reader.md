# w4_reader: the end-user path from `singlet.load("GSE...")` to an in-memory matrix

Status: COMPLETE for the coordinator's cut scope (hard stop 2026-10-22 07:40 EDT). Owner: the `w4_reader` agent. Production
code was only read. Delivered: (1) stage-time table for one sample, (2) network model, (3) a small decoder prototype with
timing, (4) MCP chatter analysis, prioritized list, and a short R timing (section 1.2).

Files: `tools/w4_reader_stages.py` (stage timers, checked equal to production), `tools/w4_reader_profile.py` (cProfile and
whole-bundle timings), `tools/w4_reader_netmodel.py` (network model), `harness/w4_reader_native.cpp` (C++ stage timer on the
production `pz_reader.h`), `kernels/w4_reader_dec.h` + `harness/w4_reader_dll.cpp` + `tools/w4_reader_proto.py` (prototype), `results/w4_reader_stages.tsv` (14 samples), `results/w4_reader_stages_check.tsv` (the one
sample below), `results/w4_reader_netmodel.{txt,tsv}`, `corpus_w4_reader/` (two extracted members of GSM4317426).

Measurement conditions for every number below: laptop i7-1185G7, CPU load 96-100% from other agents (checked before and
after most timed runs, reported clock 2995 MHz which does not show throttling). Timing lock: held for the whole-bundle stage run,
the C++ stage timer (1.1) and the prototype timing (3); **not** held for the one-sample table of section 1 (the equality-check
run, load about 90%), the exploratory `to_anndata` wall times, or the R run (1.2). Treat
**shares as reliable and absolute milliseconds as inflated by an unknown factor (2-3x vs an idle machine)**.

## 0. Findings from reading the code

Python `singlet.load("GSE...")` is `_loader._load_one` -> `download()` -> `SingletBundle.open(path).to_anndata()`:

1. `download()` fetches the **whole bundle** with one `urllib` GET: 64 KiB reads, no Range, no resume, no parallelism, no
   conditional request, `timeout=120`. A one-sample `load("GSM...")` pays one API call (`/gsm?q=`, a second on fallback)
   and then the whole parent bundle. For a gene-level matrix only exon + intron + cell_calls + vocab are needed: 54% of a
   median atlas bundle's bytes (p10 23%, p90 82%; 180-bundle sample in `results/w4_atlas_*`), 9.6 MB of 187 MB for one
   sample of GSE145370. Atlas bundle sizes: p50 181 MB, p90 709 MB, p99 1.56 GB; p50 4 samples per bundle; gene-level
   members of one sample p50 15 MB.
2. After the download, `verify_bundle(hashes=True)` reads every member through `zipfile` (CRC-32) and SHA-256 (1.35 s for
   187 MB here: SHA-256 432 MB/s, `zlib.crc32` 419 MB/s). Then the decoder re-verifies: **zip CRC-32, SHA-256 (manifest),
   whole-file TP1Z CRC-32 (6.8 ms per 5 MB member), then per-chunk CRC-32**: four passes over the same bytes.
3. `to_anndata(layers=True)` (the default through `load()`) decodes **each exon and intron `.1pz` twice per sample**:
   `_load_gsm_gene_counts` builds `X`, then `_gene_level_part` -> `read()` decodes the same members again for the
   `spliced` / `unspliced` layers, building a pandas `Index` of 300k+ feature names each time. Every decode goes through
   a temp file (write, native read, unlink). `X` is recomputed instead of being `spliced + unspliced`.
4. The feature-to-gene map is rebuilt **in pure Python for every matrix of every sample, twice**:
   `[r.split("_")[0] for r in rownames]` then `gene_id_to_row.get(...)` over 310k-386k names. On the 14-sample bundle that is
   16.3 M `str.split` and 16.3 M `dict.get` calls, although all samples of a bundle share one feature list (E4: identical
   across the 14 samples).
5. The native decode (`pz_reader.h`) is single-threaded although chunks are independent once the ptr block is known. Per
   chunk it does a bit-by-bit bitmap unpack, an 8-iteration-per-element bit-plane decode, and a `std::sort` of
   `pair<uint32,uint32>` per column. The pybind binding then copies indptr, indices and data element by element into three
   new numpy arrays (two uint32 -> int32, data kept uint32 even when `vt_code` says 8 or 16 bits) and converts every row
   name and barcode into a Python `str`, whether or not the caller needs them. The shipped `_pz.pyd` comes from
   `setup.py` (`/utf-8`, default optimisation, no `-march`), not from the AVX-512 build used by the study harness.
6. R (`load.R`, `bundle.R`, `pz_io.cpp`): `download.file()` (single stream) plus a header probe (`curlGetHeaders`), `unzip()`
   of the members to a temp directory, `pz_read_1pz_raw` (indptr/indices to R integer, data to **double**, names to
   `CharacterVector`), `sparseMatrix(i = indices + 1L)` (another copy), `proj %*% mat` (SpGEMM) per matrix,
   `exon_g + intron_g`, three `[, keep]` column subsets, `cbind` per assay; three assays (`counts`, `spliced`, `unspliced`)
   each a `dgCMatrix` with double values (12 bytes per nonzero).
7. Nothing revalidates a cached bundle: a cache hit checks size and zip structure only (`verify_bundle(hashes=False)`,
   13.7 ms), never an ETag or `Last-Modified`. A re-packed study stays stale until `force=True`.

## 1. Stage times for loading ONE sample (exon + intron), production Python path, download excluded

Sample GSM4317426 of `corpus/bundles/GSE145370.singlet` (2,928 called cells, 63,995 columns in each matrix, 310,797 exon
and 272,191 intron rows, 0.60 M + 0.66 M nonzeros). `tools/w4_reader_stages.py` re-implements `_load_gsm_gene_counts` +
two `_gene_level_part` calls line for line with a timer around each step; its X and both layers are **identical to the
production output** (`equal_to_production=True`). One run; total 6.65 s. Rows marked (split) divide the pyd call using the
separate C++ and Python timings of section 1.1.

| stage | ms | share |
|---|---:|---:|
| zip extract of 4 STORED members (2 matrices x 2 passes; CRC-32 inside `zipfile`) | 70 | 1.1% |
| cell_calls.tsv (zip read, `pd.read_csv`, called list) | 18 | 0.3% |
| temp-file write + unlink (4x) | 35 | 0.5% |
| **native decode inside `read_1pz` (4 calls)** | 1,134 | 17.0% |
| &nbsp;&nbsp;of which names block: zstd + TLV parse into `vector<string>` (split) | 702 | 10.6% |
| &nbsp;&nbsp;of which column chunks: zstd, unpack, sort, emit (split) | 373 | 5.6% |
| &nbsp;&nbsp;of which whole-file CRC-32 + perm/ptr blocks (split) | 59 | 0.9% |
| **pybind conversion inside `read_1pz` (split, by difference)** | 839 | 12.6% |
| &nbsp;&nbsp;names to Python `str` lists (estimate, upper bound 640) | 400-640 | 6-10% |
| &nbsp;&nbsp;element-wise int32/uint32 array copies (remainder) | 200-440 | 3-7% |
| scipy `csc_matrix` construction | 2 | 0.0% |
| barcode dict and column index | 30 | 0.5% |
| gene aggregation for `X`: column subset `mat[:, idx]` | 65 | 1.0% |
| gene aggregation for `X`: feature name -> gene, Python split/dict loops | 786 | 11.8% |
| gene aggregation for `X`: indicator matmul + `astype(int32)` | 255 | 3.8% |
| `X = exon_gene + intron_gene`, `.T.tocsc()` | 20 | 0.3% |
| layers: gene_order list + dict rebuilt (3x per sample, 111 ms total with the one in `X`) | 111 | 1.7% |
| layers: `AnnData(mat.T.tocsr())` + `pd.Index(names)` + `is_unique` | 1,530 | 23.0% |
| layers: feature name -> gene, Python split/dict loops | 1,104 | 16.6% |
| layers: projector, `X[idx] @ projector`, `astype`, `tocsr` | 456 | 6.9% |
| untimed glue between the timers (barcode `str` lists, `set()`, list comprehensions) | 198 | 3.0% |
| **total** | **6,654** | 100% |
| vstack of 14 samples to CSR (bundle level, not in the total) | 62 | |

Groupings that matter:

- **Python-level name handling is 51% of the sample**: the two split/dict loops (1,890 ms, 28.4%) plus the pandas index
  of names (1,530 ms, 23.0%). None of it is needed if the feature-to-gene map is computed once per bundle.
- **The second decode pass is about 1.0 s (16%)**: half of the pyd call plus its temp file and zip extraction (the
  `AnnData` it builds is counted in the 23% pandas row above).
- **Native decode proper (chunk decode) is 5.6%.** The names block costs more than the matrix (702 vs 373 ms), so the
  compression study's "names are 27-76% of a file" is also a CPU cost: 62% of the native time for these two matrices.
- Everything the pyd does together (native + conversion) is 29.6%.

Whole bundle (14 samples, 187 MB, `layers=True`, same code): sum of the stage totals 60.05 s (`results/w4_reader_stages.tsv`);
the same stage table has name loops 22.5 s (37.5%), layer `AnnData`/pandas 12.5 s (20.9%), pyd call 16.7 s (27.8%).
Production `to_anndata` wall: 60.3 s with layers, 23.5 s with `layers=False` (exploratory runs, load about 90%). So
**the layer columns cost 37 s of 60 s**, and `layers=False` is already 2.6x faster with no code change. Download is not
included; the whole-file integrity pass `verify_bundle(hashes=True)` adds 1.35 s.

### 1.1 Native versus binding, the same file (`harness/w4_reader_native.cpp`, `bin/w4_reader_native.exe`)

GCC 13 `-O3 -march=native`, production `pz_reader.h` unchanged, min of 7, whole file already in memory:

| member | nnz | native total | file CRC-32 | perm+ptr | names zstd+parse | chunks | chunk rate | `_pz.read_1pz` (pyd) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exon_counts.1pz | 597,300 | 272 ms | 6.8 | 7.8 | **177** | 80 | 7.4 Mnnz/s | 473 ms |
| intron_counts.1pz | 656,832 | 295 ms | 6.7 | 8.3 | **174** | 106 | 6.2 Mnnz/s | 481 ms |

The pyd costs 190-200 ms more than the native decode on each member. It is built by a different compiler, so this includes
a compiler difference of unknown size, but the pybind work (three element-wise loops over nnz, 310k name strings to `str`)
is real. A Python loop decoding 310k names to `str` takes 139 ms (upper bound for the pybind conversion).

### 1.2 R, the same sample (`tools/w4_reader_r_time.R`, one run, NOT under the timing lock, machine load 100%)

The installed R package named `singlet` on this laptop is a different package (no `pz_read_1pz_raw`; `library(singlet)`
alone took 75 s here), so the script loads the repo's prebuilt `r/src/singlet.dll` and sources `r/R/read.R` and `r/R/bundle.R`.
Same two members of GSM4317426, min of 3:

| step | exon | intron |
|---|---:|---:|
| `pz_read_1pz_raw` (Rcpp decode, copies, double values, names to `CHARSXP`) | 1,980 ms | 2,930 ms |
| `read_1pz` (raw + `sparseMatrix(i = indices + 1L)` + dimnames) | 2,270 ms | 2,840 ms |
| `.bundle_aggregate_to_vocab` (name split/`match` + `sparseMatrix` + `proj %*% mat`), 38,606 genes | 2,390 ms | not run |

So R spends about 5.1 s on the two reads and about 4.8 s on the two aggregations (second one extrapolated from the first)
before the called-cell subsets, `exon_g + intron_g`, three `[, keep]` copies and the `cbind` of three assays: about
10 s for this sample against 6.65 s for Python on the same machine state. The 310,797 x 63,995 `dgCMatrix` is 46.7 MB for
597,300 nonzeros (7.2 MB of `x`+`i`): by subtraction the other 39 MB is the row and column names as R strings. The native decode is the same
code as Python's, so the same findings apply (names dominate, double values, copies); what differs is that R has no
Python-loop problem (the name split is vectorised `sub`/`match`) but pays `CHARSXP` creation for every name and a sparse
matrix product per matrix.

## 2. Network model

Measured constants (coordinator, this laptop to data.singlet.bio): TTFB 0.18-0.27 s per range request (model uses 0.22),
single stream 32 MB/s, **8 parallel streams only 18 MB/s aggregate**, RTT 50 ms. Atlas tail requests (180 bundles, this
study): p50 TTFB 0.197 s, p90 0.33 s; 174 of 180 central directories fit the 64 KiB tail (p50 8.6 KB, p90 31 KB, max 197
KB); 21 of 180 ETags are 3-part multipart.

What today's client issues for one sample: Python `load("GSM...")` = 1-2 API GETs + 1 whole-bundle GET (one stream),
total 2-3 requests, but all the bytes of the bundle. R adds one `curlGetHeaders` probe (a HEAD-style request, more if it redirects) before the same single-stream download. A client that uses the
service's member index issues: 1 revalidation (ETag/size), then one ranged GET per coalesced member range. In a ZIP the
exon and intron members of a sample are adjacent, so a sample needs 4 ranges: exon+intron, cell_calls, vocab, manifest
(vocab and manifest are per-study and cacheable).

`tools/w4_reader_netmodel.py`: time = handshake (2 RTT) + TTFB + slow-start loss + bytes / rate (+ hashing 7.2 ms/MB today,
about half of that for SHA-256 only). A **model, not a measurement**; link efficiency 95%.

Scenario A, one sample of GSE145370 (9.6 MB needed of 187 MB):

| link | today: whole bundle, one stream, verify | index, member ranges, 1 connection | index + 8 connections (observed 18 MB/s aggregate) | index + 8 connections (projection, streams scale) |
|---|---:|---:|---:|---:|
| 100 Mbit/s | 17.6 s | 2.2 s | 1.4 s | 1.4 s |
| 1 Gbit/s | 7.7 s | 1.7 s | 1.2 s | 0.8 s |

Scenario B, a 500 MB study, all samples at gene level (54% = 270 MB needed, 41 coalesced ranges):

| link | today: whole bundle, one stream, verify | index, member ranges, 1 connection | index + 8 connections (observed) | index + 8 connections (projection) |
|---|---:|---:|---:|---:|
| 100 Mbit/s | 46.2 s | 33.2 s | 24.7 s | 24.7 s |
| 1 Gbit/s | 19.7 s | 18.9 s | 17.0 s | 4.3 s |

Reading the tables:

- A single sample: **member ranges cut the time 8-12x at 100 Mbit/s and 4.5-10x at 1 Gbit/s**, almost entirely by
  fetching 5% of the bytes. Parallelism adds little because the request count is 4, not hundreds.
- A whole study at 100 Mbit/s is link-bound (12 MB/s): the only lever is fewer bytes (54% needed, and the proposed
  container is about 4.5x smaller than that). Parallel connections do not help.
- At 1 Gbit/s the observed 18 MB/s aggregate for 8 streams is **below** the 32 MB/s single stream, so on this laptop the
  best measured option is one or two connections with coalesced ranges; the 4.3 s projection needs a client where
  parallel streams scale. Measure the connection count on the user's machine at first use instead of fixing it
  (start with 2, add streams while aggregate throughput rises).
- The per-request TTFB of 0.2 s is the cost that matters for small transfers. Everything that can be batched in one
  range (adjacent members, a chunk index placed first so the first fetch carries the data it describes) saves 0.2 s each.

### 2.1 Stale member index (coordinator's note)

The service's cached member index is stale for rewritten bundles (about half the atlas; GSE111113's index says
324,676,242 bytes, the live object is 170,777,036). A client that trusts the index blindly reads wrong offsets. Required
protocol for a client:

1. Send every ranged GET with `If-Match: <etag from the index>`; a mismatch returns 412 and the client falls back to
   reading the live central directory (EOCD in the last 64 KiB: 174/180 bundles fit; 1 more request otherwise).
2. Check `Content-Range: bytes a-b/TOTAL` in the first 206 against the index's total size at no extra request.
3. Check the local file header signature (`PK\x03\x04`) and name at the start of each member range, then the manifest
   SHA-256 of the member (the manifest is itself a member; its checksum must come from a fetch validated by 1 and 2).
4. The service should return the ETag with the index and refresh the index when the object's ETag changes (the 180-bundle
   scan shows ETag and Last-Modified per object, so the check is free for the service).

## 3. Prototype: threaded zero-copy decoder (built, verified bit-identical, decode stage timed)

Files: `kernels/w4_reader_dec.h` (decoder), `harness/w4_reader_dll.cpp` (C ABI), `tools/w4_reader_build.sh` (builds
`bin/w4_reader_dll.dll`, GCC 13, static zstd), `tools/w4_reader_proto.py` (ctypes driver: checks, then timing),
`results/w4_reader_proto.tsv`. It copies the chunk loop of `decode_vocsc_section` and keeps the production kernels (bitmap
unpack, bit planes, byte unsplit) **unchanged**, so every speedup below comes from the structure, not from new kernels.

What it does differently from `read_1pz` + the pybind wrapper: never parses the names block; whole-file CRC-32 optional
(per-chunk CRC-32 kept); chunks decode in parallel (`std::thread`, atomic chunk counter, thread-private scratch, own zstd
context per thread) straight into caller-owned numpy buffers (int32 indices; int32, uint16, uint8, float32 or float64
values; no uint32 -> int32 copy loops); a chunk range `[c0, c1)` can be decoded alone; optional packed-key (row<<32|value)
sort instead of `std::sort` of pairs. API (ctypes): `w4r_open(buf, n, verify_file, err, errlen) -> handle`, `w4r_shape`,
`w4r_indptr(h, int32*)`, `w4r_decode(h, c0, c1, int32* indices, void* values, value_kind, threads, packed_sort, err, errlen)`,
`w4r_close`. Bug found on the way: a `thread_local ZstdDCtx` inside a static-linked MinGW DLL crashes (access violation);
the context lives in the per-thread scratch instead. Production hazard to note for any threaded patch: `CRC32::get_table8()`
fills its static table lazily without a lock (data race); the prototype builds the table before spawning threads.

**Correctness**: output arrays (indptr, indices, values) are **identical to `singlet._pz.read_1pz`** on all 5 files below,
for 1, 4 and 8 threads and both sort variants (`identical_to_production=True`). Not yet run on all 14 corpus matrices or
through a full `to_anndata`.

**Timing**: `tools/w4_reader_proto.py --reps 5`, min of 5, timing lock held, **CPU load 100% before and after, other agents
running** (so thread scaling is unreliable: 2 threads beat 1 only on some files, e.g. GSM4317426 exon t1 97 ms, t2 383 ms,
t4 220 ms, t8 109 ms; single-thread rows are the trustworthy ones). Production time is the pyd `read_1pz(path)` including
the file read, names and conversions; the prototype is given the bytes already in memory (as from a zip read or mmap) and
returns int32 values, so the comparison is "decode to arrays a scipy matrix can wrap". Milliseconds:

| input | nnz | production pyd | proto 1 thread | proto 4 threads | proto 8 threads | speedup 1 thread | speedup best |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM4317426 exon (63,995 cols, mostly empty droplets) | 0.60 M | 537 | 97 | 220 | 109 | 5.5x | 5.5x |
| GSM4317426 intron | 0.66 M | 483 | 111 | 99 | 129 | 4.3x | 4.9x |
| tumor_3p exon (sparse, names-dominated) | 0.25 M | 440 | 45 | 38 | 27 | 9.7x | 16x |
| pbmc_v2_typical exon (typical density, 169 nnz/col) | 2.57 M | 794 | 371 | 127 | 125 | 2.1x | 6.4x |
| tcell_5p exon (dense, 152 nnz/col, 69k cols) | 10.53 M | 2,472 | 1,695 | 522 | 515 | 1.5x | 4.8x |

Reading: on name-dominated files (sparse, or many empty-droplet columns) skipping names and the conversion loops is the
whole gain (4-10x with one thread). On decode-bound files (dense cells) the unchanged kernels run at 6.2 Mnnz/s per thread
(10.5 M nnz in 1.7 s) and only threads help: 3.2x with 4 threads on tcell_5p, nothing beyond 4 on this 4-core laptop. The
packed-key sort is worth 3-11% on decode-bound files (tcell_5p 1,902 -> 1,695 ms at 1 thread; noise elsewhere). Adding the
whole-file CRC-32 costs 5-10 ms per 5 MB member. The next lever is the kernels, which this prototype does not touch: bit-plane
decode (8 branchy iterations per element) and bitmap unpack (bit by bit) have SIMD, `pdep` and 8x8-bit-transpose forms; the
rANS payload of Track B (73 Mnnz/s per thread in native order) would make this stage negligible.

**Not measured**: end-to-end `load()` with the prototype wired into the Python glue. As a bound: replacing the four
`read_1pz` calls of the one-sample table (1,973 ms) with two prototype calls (97 + 111 ms at one thread) would cut the
6.65 s sample load by about 1.8 s (27%) alone; with the glue changes of section 5 (items 1 and 2) the projection is a
sample load of about 0.4-0.6 s (a **projection**).

Gene-level aggregation inside the decoder (not built): pass `feature_to_gene[m]` (computed once per bundle) and emit
gene-level columns directly; if the feature order follows the gene order the merge is linear and replaces the
`indicator @ matrix` products (255 + 456 ms in section 1) and the float64 detour. To be verified on the vocabulary.

## 4. MCP chatter (from `python/singlet/mcp/server.py`; hosted-tool outputs were not observed)

Caveat: `python/singlet/mcp/server.py` is the **local** server (12 tools: `singlet_stats`, `singlet_nl_search`,
`singlet_search`, `singlet_qc`, `singlet_load`, `singlet_browse`, ...). The live hosted tools named in the brief
(`search_datasets`, `get_study`, `list_bundle_files`, `get_partial_download`, `get_download_url`, `get_modalities`,
`get_sample_qc`, `export_manifest`) are not in the repo, I did not call them, and I did not read their schemas, so the
byte counts below are estimates from the local server's field lists, labelled as such.

Observed in the code:

- Every response is `json.dumps(result, indent=2)`: indentation adds roughly 25-40% to bytes and tokens over compact JSON.
- `singlet_search` and `singlet_browse` return **whole API rows** (about 18 fields: ids, organism, protocol, modality,
  status, tissue, cell_type, title, source, mapping_rate, n_cells, median_genes, median_umis, mt_pct, qc_flag,
  singlet_version, pipeline_date) for up to 100 rows. At roughly 450-600 B compact per row a 20-row search is 9-12 KB
  (about 3-4k tokens) before indentation; the model usually needs ids, title and two or three numbers per row.
- `singlet_qc` and `singlet_load` each call `_lookup_gsm` (1-2 HTTP requests) and then repeat nine of the same fields
  (`gsm_id`, `gse_id`, `organism`, `protocol`, `status`, `title`, `tissue`, `cell_type`, `web_url`); asking for QC and then
  load info for one sample costs 2-4 requests for one database row. `singlet_load` returns a 250 B static `python_code`
  string every time. There is no cache, no ETag, and no "unchanged" answer.
- `singlet_nl_search` returns the interpreted filters plus all accessions plus a static `load_hint`.

Redundancy to expect across the hosted sequence `search_datasets -> get_study -> get_sample_qc -> get_modalities ->
get_partial_download -> export_manifest` (inference from the tool names, to be checked against real outputs): the per-sample
list (GSM ids, titles, cell counts) repeated by `get_study`, `get_sample_qc`, `list_bundle_files` and `export_manifest`;
member names and sizes in `list_bundle_files` repeated in `get_partial_download`; gene/feature names if any response carries
them (they are a reference constant, 310k-386k strings, and must never travel in a tool response).

Minimal-chatter protocol (design):

1. **Pack ids and ETags.** Every study response carries `{bundle_id, etag, size, index_version}`. Any later call may send
   `if_etag`; if equal the server answers `{"unchanged": true}` (about 30 bytes) instead of the payload. The model keeps
   the id, not the data.
2. **Handles instead of repeats.** A reference pack (gene vocab, barcode whitelist, per-build feature list) is named by
   `pack_id` (content hash) once; responses say `vocab: "grch38-2024-a/3f9c..."` and never inline names.
3. **Field selection with compact defaults.** `fields=` on search/browse, defaults of 5 columns (id, title, organism,
   n_cells, quality tier), compact JSON, `limit` default 10; a `next_cursor` instead of repeated page metadata.
4. **One call per decision.** `get_partial_download` returns everything a client needs to fetch and decode a member without
   further requests:
   `{url, etag, member, offset, length, header_len, method:"stored", member_crc32, sha256, bundle_size, index_version,
   codec:{id:"tp1z"|"rans16", version, value_type, chunk_cols, n_chunks, chunk_table:{offset,length}, row_order:"perm"|"native"},
   vocab:"<pack_id>", barcodes:"<pack_id|member>"}`. `offset` is the first byte of member data (after the ZIP local header),
   so the client issues one ranged GET with `If-Match: etag` and verifies `sha256`; the codec block says which decoder to
   run and where the chunk table is, so a cell-subset read is another ranged GET of the chunks it needs.
5. **Freshness.** Because the index can be stale (section 2.1), the response states `index_version` and `indexed_at`, and
   the server treats a 412 on `If-Match` as a signal to rebuild that bundle's index.

## 5. Prioritized change list with expected effect

Effects are on one-sample load unless stated; "measured" means in this note, "projected" means arithmetic on measured pieces.

| # | change | where | expected effect | basis |
|---|---|---|---|---|
| 1 | Compute the feature-to-gene map once per bundle (numpy/pandas string ops or a dict built once; names are identical across samples) and reuse it for both matrices and both layer calls | `bundle.py` | removes 1.9 s of 6.65 s per sample (28%); 22.5 s of 60 s on the bundle | measured (section 1) |
| 2 | Decode each member once; build `spliced`/`unspliced` from that decode and `X = spliced + unspliced`; do not build `AnnData` or `pd.Index` of 300k names inside `read()` for the layer path | `bundle.py` | removes about 2.5 s (the second pass 1.0 s plus the pandas index 1.5 s): 38% | measured |
| 3 | Items 1 and 2 together | `bundle.py` | 6.65 s -> about 2.1 s per sample (3x), `to_anndata` 60 s -> about 15-20 s with no native change | projected from the stage table; `layers=False` already shows 60 -> 23.5 s |
| 4 | Skip the names block unless asked (`names=False`), return the feature map from a vocabulary id | `_pz.cpp`, reader | native decode 272 -> 95 ms per exon matrix (62-65% of it is names), plus the 140 ms `str` creation; prototype 4-10x on name-dominated members | measured (sections 1.1, 3) |
| 5 | Zero-copy output: write indices/indptr as int32 and values at native width straight into numpy buffers, no three-array element copy | `_pz.cpp` | removes 200-440 ms per sample | measured by difference; prototype writes int32/uint16 directly |
| 6 | Parallel chunk decode (63-95 independent chunks), faster bitmap/bit-plane/sort kernels | `pz_reader.h` | decode-bound files 3.2x with 4 threads (tcell_5p 1,695 -> 522 ms); small samples are name-dominated, so threads matter less | measured on a loaded machine (section 3) |
| 7 | Range fetch of only the needed members through the member index, with `If-Match`, coalesced ranges, ETag revalidation of the local cache | `_loader.py`, R `load.R`, service | 17.6 -> 1.4-2.2 s (100 Mbit/s), 7.7 -> 0.8-1.7 s (1 Gbit/s) for one sample of a 187 MB bundle; 5% of the bytes | model (section 2) |
| 8 | One integrity pass: SHA-256 per member as bytes arrive in the receive thread, per-chunk CRC kept, drop whole-file CRC-32 and zip CRC on the trusted path | `_integrity.py`, `pz_reader.h` | 1.35 s per 187 MB (7.2 ms/MB) -> hidden in the download; 6.8 ms per 5 MB member saved | measured per pass |
| 9 | Container v2: vocabulary id instead of names, chunk index first | format (other agents) | removes the 62-65% names cost from the CPU path as well as 27-76% of the bytes | compression study + section 1.1 |
| 10 | Sort-free layout (frequency-class order or E2 ascending order) with the 32-lane rANS decoder | format + decoder | the only way the chunk part gets below the sort cost (34 vs 73 Mnnz/s) | Track B |
| 11 | R: decode into `dgCMatrix` slots directly (double values written once), skip names (build dimnames from the vocabulary and whitelist once), compute `spliced`, derive `counts = spliced + unspliced` in one pass, replace `proj %*% mat` with the mapping done in the decoder, read members from the zip into memory instead of `unzip()` to a temp dir | `r/src/pz_io.cpp`, `bundle.R` | about 10 s -> under 1 s per sample is plausible (reads 5.1 s + aggregation 4.8 s are both removable); 39 of 46.7 MB of a matrix are names | measured parts (section 1.2), rest projected |
| 12 | MCP: compact JSON, field selection, ETags with `unchanged`, `get_partial_download` decode recipe | server | cuts search responses by roughly 2x and removes 1-3 requests per sample | estimate (section 4) |

Order to do them: 1+2 (a Python-only patch, about a day, 3x), then 7+8 (client network, 8-12x for single samples), then
4+5+6 (native, 5-10x on the decode part), then the format work (9, 10) that the other agents own.

## 6. Not done

- End-to-end `load()` with the prototype wired into the Python glue, and gene-level aggregation inside the decoder: not built.
  Resume: wire `tools/w4_reader_proto.py::proto` into a copy of `_load_gsm_gene_counts`, compute `feature_to_gene` once per
  bundle, compare `to_anndata` output to production, time the whole bundle under the lock.
- Prototype checks cover 5 files, not all 14 corpus matrices; thread scaling must be re-timed on an idle machine; the
  kernels (bit-plane decode, bitmap unpack, per-column sort) are production code and are the next optimisation target.
- R: only the read and aggregation steps of one sample were timed (section 1.2), without the timing lock and not through
  `load()`/`read_singlet()` (the installed `singlet` package is a different package and takes 75 s to attach here). Whole-study
  R timing and the `unzip()`-to-temp-dir cost are not measured.
- No patches were written to `results/w4_reader_patches/` (changes 1+2 would be a diff of `bundle.py`).
- Network constants are the coordinator's, not remeasured; no request was sent to data.singlet.bio by this agent.
- All timings are single runs or min-of-5 on a machine at 96-100% load.

## 7. Housekeeping

- Timing lock: at 07:21 my `mkdir results/.timing_lock` failed because another agent held it, and the `rmdir` that followed in
  the same command removed that agent's lock by mistake. Another agent had re-created it within seconds, so the window was
  short, but whoever timed during 07:21:24-07:21:49 should check their run. Later runs of mine either held the lock
  (sections 1.1, 3: before 07:20) or are marked as unprotected (section 1.2, the 07:27 R run).
- Files left on disk: `corpus_w4_reader/` (two extracted `.1pz` members and the bundle's `feature_vocab.json`, 17 MB),
  `bin/w4_reader_native.exe`, `bin/w4_reader_dll.dll`, `results/w4_reader_*`. Nothing was committed or installed; no
  production file was modified; no request was sent to any production service.
