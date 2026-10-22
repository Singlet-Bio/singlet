# m_w4: bindings + clients (W4): `singlet._pz2`, `singlet.v2.client`, R, local MCP

Owner: W4. 2026-10-22. Status: DONE for the brief, open items in section 8. Everything below was run on this laptop (i7-1185G7 4C/8T,
AVX-512, Windows 11, Python 3.11.9 / numpy 2.2 / scipy 1.15 / pybind11 3.1, R 4.4.1 + Rtools 4.4 GCC 13.2). Nothing was committed by W4.

## 0. Headline numbers (GSE145370, 14 samples, 187 MB legacy bundle; v2 = 31.8 MB migrated with the native codec)

Equality first: the v2 AnnData of the whole study equals the legacy one exactly (X int32, `spliced` and `unspliced` layers, `var`, obs index
in the same order, all nine obs columns, `uns["study_meta"]`): `tools/m_w4_equal.py`, 40,614 cells x 38,606 genes, all OK.

| | legacy (`SingletBundle`, local file) | v2 local files (mmap) | v2 over HTTP Range (loopback) |
|---|---|---|---|
| one sample (GSM4317426, 2,928 cells), wall | 1.84 / 1.95 / 2.06 s (per-sample stages only; the real `load("GSM..")` runs the whole study below) | 0.38 / 0.41 / 0.50 s | 0.36 / 0.36 / 0.42 s |
| one sample, peak RSS growth over the imports / tracemalloc peak | 184 MB / 152 MB | 98 MB / 79 MB | 97 MB / 79 MB |
| one sample, requests and bytes | whole bundle: 1-2 + 1 GET, 187,066,350 B | 0 | **2 requests, 2,101,298 B** (manifest 90,647 + one 2,010,651 B range) |
| 14-sample study (what `load("GSM..")` or `load("GSE..")` cost today), wall | **20.4 / 22.7 / 23.3 s** | **0.86 / 0.89 / 1.10 s** | 0.92 / 1.03 / 1.23 s |
| 14-sample study, peak RSS growth / tracemalloc peak | 277 MB / 198 MB | 219 MB / 190 MB | 224 MB / 189 MB |
| 14-sample study, requests and bytes | 1 GET, 187,066,350 B | 0 | **15 requests, 11,696,362 B** (16.0x fewer bytes) |
| 1,000 largest cells of one sample | needs the whole bundle | | two-phase plan 4 requests, 1,917,176 B, 0.32-0.38 s; default plan 2 requests, 2,101,298 B, 0.32-0.38 s |
| cells slice `n_umi_exon >= 300` (181 cells) | | | 4 requests, 1,917,176 B, 0.29-0.33 s |

Format min / median / max over 3 fresh processes per row (`tools/m_w4_measure.py`, timing lock taken per child, child imports excluded). CPU load
21-43% before each child (reported clock 2995 MHz; `results/m_w4_measure_*final.log` has the per-run values). An earlier pass on the same
data (older codec build, other agents saturating the CPU at 100%) gave legacy sample 3.6-5.8 s, legacy study 47-87 s, v2 sample
0.54-0.92 s, v2 study 1.1-1.7 s (`results/m_w4_measure.log`, `m_w4_measure_slices.log`). The HTTP rows use a loopback server in another process
(`tools/m_w4_serve.py`), so they contain no network time; the model below adds it. v2 numbers include `open_study` (0.04-0.12 s) and
the pack lookups (wheel packs, no download), and AnnData construction.

Network-time MODEL (`tools/m_w4_netmodel.py`; requests and bytes exact, times use the coordinator's constants TTFB 0.22 s, one stream 32 MB/s,
2 RTT handshake, and 18 MB/s aggregate for parallel streams on this link):

| scenario | requests | bytes | model s |
|---|---:|---:|---:|
| legacy one sample (API lookup + whole bundle) | 2 | 187,066,350 | 6.39 |
| legacy study | 1 | 187,066,350 | 6.17 |
| v2 one sample | 2 | 2,101,298 | 0.61 |
| v2 one sample, manifest revalidated with 304 | 2 | 2,010,651 | 0.60 |
| v2 study, 1 stream | 15 | 11,696,362 | 0.91 |
| v2 study, 4 connections (TTFB overlapped, 18 MB/s aggregate) | 15 | 11,696,362 | 1.85 (4 connections do not help on this link) |
| v2 1,000-cell slice, two-phase plan | 4 | 1,917,176 | 0.82 |

The legacy rows exclude the 4 integrity passes (1.35 s per 187 MB measured earlier) and the 20 s of decode above; the v2 rows exclude the 0.4-1.1 s of decode.
At these sample sizes (1.5-4.3 MB objects) fetching a sample whole in one request beats the two-phase plan (extra round trips); the two-phase plan
exists for large samples (default switch at 48 MB of cells+exon+intron).

R (same data, local files, `tools/m_w4_r_time.R`, min of 3, R 4.4.1): one sample legacy `.read_singlet_bundle(gsms=)` **2.31 s** vs v2 **0.23 s**;
14-sample study legacy **27.3 s** (1 rep) vs v2 **2.42 s** (3 reps). R heap peak (includes the 402 MB of attached packages): 717 -> 546 MB for one sample,
906 -> 856 MB for the study. Result objects are identical in size (31 MB / 123 MB): dgCMatrix with double values.

## 1. Dependencies and what I assumed from the others

* W1 (`include/singlet/pz2/pz2.h`, final headers 10:47): `Mode` has Native and Canonical only; the format's `gene` mode is implemented in the binding
  (`pz2_client.h`). `NamesKind::None` handled. Rebuilt and retested against the final headers: all pass; data migrated with the 10:08 build still decodes.
* W2 (`python/singlet/v2/`): I use `container.S2Reader` (V2Bundle only), `cells.CellsReader`, `packs.PackStore/VocabPack` (`row_gene()`, `gene_table()`),
  `manifest.study_file_bytes`, `sample.V2Sample` + `context.V2Context` + `rcp_adapter` (regenerating legacy members in `V2Bundle.read`). `codec.NativeCodec`
  calls my `encode`, `decode_columns`, `Member`: signatures agree (their migration with `--codec native` ran on my build, 113 s for the 14-sample study).
* Test data: `bench/compression/corpus/bundles/GSE145370.singlet`; migrated copy `bench/compression/results/m_w4_v2/native14b` (32 MB; the earlier copies were deleted).

## 2. `singlet._pz2` (binding): `src/bindings/python/_pz2.cpp`, helpers `src/bindings/common/singlet/pz2_client.h`

Build: `bash src/bindings/python/build_pz2_mingw.sh` (Rtools GCC 13, `-O2`, no `-march`, about 2 min on the loaded laptop; imports only `python311.dll` and the UCRT;
zstd, libstdc++, winpthread static) or `pip install .` (setup.py; the orchestrator added `-pthread` for manylinux; kept).

* `read_header(buf)`; `Member(source=None, fetch=None, prefix_hint=65536)` opened from bytes-like (whole member or at least its prefix) and/or a `fetch(off, len)`
  callable that supplies any range the decoder asks for (NeedRange handled inside; the callable runs with the GIL); properties `header, m, n, nnz, prefix_len, value_dtype, row_perm`;
  methods `chunk_index()`, `plan(c0, c1)`, `add_range(off, buf)`, `has(off, len)`, `column_counts`, `row_names()`, `col_names()`, `kv()`,
  `decode(c0, c1, *, mode='native'|'canonical'|'gene', threads, dtype, index_dtype, isa, cols, gene_map, n_genes)`.
* `decode(source, ...)`, `decode_columns(source, c0, c1, ...)` one-shot; `encode(indptr, indices, data, m, n, *, codec, chunk_cols, ..., row_names, col_names, kv, digest, compute_digest, threads, zstd_level) -> (bytes, stats)`;
  `digest(...)`; `aggregate_genes`, `select_columns` (kernels on arrays); `isa_report()`, `selected_isa()`, `best_isa()`, `isa_supported()`; `Pz2Error(.code, .off, .len)`, `NeedRange`.
* Output arrays are allocated once by numpy (uninitialised), filled by the decoder, and trimmed in place (`ndarray.resize`, no copy): `indptr` int32 (int64 if the bound is >= 2^31), `indices` int32,
  `data` in the stored width (uint8/16/32/float32) or the requested dtype. The GIL is released for decode, selection and aggregation.
* Mode `gene`: the canonical-row -> gene map is composed once with the member's row permutation, the decode runs in native order (no sort), rows are aggregated in place per column
  (bitmap scan for wide columns, sort of the touched genes for narrow ones), threaded over blocks of equal nnz; `cols` keeps a sorted subset of columns first; sums int32 (float64 for R); zero entries dropped.
* `PZ2_ISA=scalar|avx2|avx512` forces the path (read once per process); `isa=` overrides per call. The decoded bytes are identical on all three (tested through the env var in child processes).

## 3. `singlet.v2` client (`python/singlet/v2/{transport,client,bundle}.py`)

* `open_study(gse)`: one GET of `v2/m/<GSE>.json`, conditional on the cached ETag (`<cache>/v2/m/`); 304 costs no body; in-process memo 60 s; offline falls back to the cached copy with a warning;
  `StudyNotFound` (a `FileNotFoundError`) with a message naming `legacy=True` when the manifest is missing.
* `Study.to_anndata(samples, genes, obs_filter, layers, cell_order, concurrency, threads, verify, whole_max)`; `load(...)` is the same through an accession (GSM resolved via the API as before) or a local
  `.s2` / manifest path (`load_s2`: memory-mapped, zero copy of the file). `cells(...)` / `Study.cells(where, samples, columns, genes, layers, max_cells)`: `where` is a pandas query, a callable or a dict.
  `download_study` mirrors manifest + objects (whole-object sha256) + packs under `<cache>/v2/`, after which `load` reads them locally. `Study.request_plan()` predicts requests and bytes.
* Request plan: manifest + one coalesced range per sample (cells, exon, intron are adjacent in the object; only 64-byte alignment gaps are fetched in between). Samples above 48 MB (`whole_max`) use two phases:
  cells + matrix prefixes concurrently, then only the chunk runs that cover the wanted cells, concurrently. Large single ranges (> 32 MiB) are split into parallel parts. Samples are read by a pool (`concurrency`, default 4);
  decoder threads = cores // concurrency.
* Integrity: every ranged response must be 206 with `Content-Range` start/end equal to the request and total equal to the manifest's object size; a server that ignores Range, a wrong total or a short body is an error;
  the pz2 header digest must equal the manifest member digest; `verify=True` (or `SINGLET_V2_VERIFY=1`) recomputes the canonical digest of every decoded matrix. Transport: one persistent connection per thread, retries
  with backoff on 429/5xx/network errors (`Retry-After` honoured, one free retry for a stale keep-alive), redirects limited to https, `User-Agent: singlet-client/2` (the data host returns 403 to urllib's default).
* Names: row -> gene maps come from the vocabulary packs (`VocabPack.row_gene()`), cached per pack; wheel packs first, then `<cache>/v2/ref/`, then `v2/ref/<id>.pack` verified against `index.json` sha256.
  Matrices with inline names fall back to splitting at the first `_` (the legacy rule). Genes requested with `genes=` become a keep-map applied inside the aggregation (a one-gene load does almost no work).
* `V2Bundle(SingletBundle)` (`singlet.open_bundle("GSE..")`, `SingletBundle.open(manifest.json | .s2 | "GSE..")`): `manifest`, `study_meta`, `feature_vocab`, `gsm_ids`, `list_files`, `has`, `modalities`, `read` (matrices
  decoded canonical as uint32; tables/JSON regenerated by W2/W3's context: stored blob or verified recipe), `raw_counts`, `qc`, `to_anndata`, `to_h5ad`, `to_parquet`, and the convenience accessors. Verified against the legacy
  bundle for exon/intron/junctions/vdj matrices, summary/pileup_stats/provenance JSON, `cell_calls.tsv`, `raw_counts`.
* Loader routing (`_loader.py`): `load`, `download`, `open_bundle` default to v2; `legacy=True` selects the old code (`_download_legacy`); `load_sample`/`load_dir` (cluster) unchanged. New `load(..., layers, **v2_options)`.
  `_io.read_matrix` also reads a standalone `PZ2M` member (`read_pz2`) and a `.s2`. `fetch.py` docstrings point at v2.

**Decision for the orchestrator:** with `legacy=None` (default) a study without a v2 manifest raises `StudyNotFound` naming `legacy=True`; it never silently reads a legacy bundle (your brief). `SINGLET_LEGACY_FALLBACK=1` switches
on automatic fallback (with a FutureWarning) for the transition while studies are being migrated; flip the default only if you want it on for users.

**Known difference from the legacy reader (obs order).** The legacy reader lists called cells as `cell_calls.tsv` does (STARsolo: bytewise sorted barcodes; EmptyDrops: ascending column; top-N fallback: an unstable sort order).
The cells table keeps a `called` bitmap only, so `cell_order="legacy"` (default) sorts by barcode unless the table carries `cc_pos` (then that order) or `cc_listed`/`cc_pvalue` (EmptyDrops: column order). On GSE145370
(STARsolo) this reproduces the legacy order exactly (checked, all 14 samples). `cell_order="column"` gives matrix-column order (largest libraries first).

## 4. R (`r/src/pz2_io.cpp`, `r/R/v2.R`, routing in `load.R`/`bundle.R`)

* Rcpp binding (all verified by testthat): `pz2_header_`, `pz2_plan_`, `pz2_chunk_first_cols_`, `pz2_decode_` (decodes straight into the `i`/`p`/`x` of a dgCMatrix: 0-based row ids, double values, optional
  column selection and gene aggregation, threads), `pz2_names_`, `pz2_cells_` (CEL2 tables incl. whitelist-coded barcodes), `pz2_pack_info_`, `pz2_pack_genes_` (gene table + exons per gene: the feature -> gene map is a `rep.int`),
  `pz2_whitelist_barcodes_`, `pz2_sha256_`, `pz2_zstd_`, `pz2_isa_report_`. Headers staged by `scripts/sync_r_headers.py` (now also `pz2/` and `pz2_client.h`, with `--check`).
* R functions: `singlet_study(accession)` (manifest, ETag revalidation, `tools::R_user_dir` cache), `load()` / `singlet_load()` (v2 by default; `legacy = TRUE`; a `.json` manifest path is a v2 study; `.singlet` stays legacy),
  `download()` (v2 mirror, returns the manifest path), `read_singlet()` (also a manifest), `singlet_cells(x, where, samples, columns, layers, max_cells)` (`where` is an unevaluated expression over the cell-table columns).
  Transport: curl (Suggests, required for ranged HTTP; a local directory needs nothing), `Content-Range` checks, retries, `curl::multi` pool for concurrent ranges; packs verified by sha256.
* `R CMD INSTALL` builds clean with Rtools 4.4; `R CMD check --no-manual` on a scratch copy: code, Rd, compiled-code and cross-reference checks all OK for the new code. Two pre-existing problems show up on R 4.4.1: WARNING
  `tools::sha256sum` is not exported (it exists from R 4.5; used by `integrity.R`) and the resulting 9 failures in `test-integrity.R` / `test-aliases-gsm.R`. Not touched.
* Not done in R: `singlet_read()` / `singlet_modalities()` on a v2 study (the modality readers still take legacy bundles); the whole-member `pz2_names_` fallback for matrices with inline row names in the two-phase plan.

## 5. Local MCP server (`python/singlet/mcp/server.py`)

Compact JSON (`separators=(",", ":")`, no indent); `singlet_load` returns, for a migrated study, `{format, rev, manifest, object, bytes, sha256, n_columns, n_called, members: {name: [off, len, kind]}, packs, gene_level: {ranges, bytes, requests}}`
and the one-line `load`; `if_rev` answers `{unchanged: true, rev}` (about 60 bytes); a study without v2 manifest keeps the legacy `bundle_url`. New tool `singlet_slice_cells(gse_id, where, samples, max_cells, save_to)`: `where` is a conjunction
(`n_umi_exon >= 5000 and called`) parsed by a small grammar, never evaluated; returns per-sample counts, requests and bytes used, median UMIs, optional `.h5ad`. The hosted tools of `m_w5.md` (CS4) were not written when I
finished, so the names/arguments of `slice_cells` here are mine and need aligning with W5's.

## 6. Tests (all pass)

* Python: `test_v2_client_native.py` 15 (header errors, `PZ2_ISA` forced in child processes, aggregation and selection against scipy), `test_v2_client_codec.py` 27 (round trips for both codecs / native, canonical, gene
  modes / thread counts / column ranges / value widths / dtype conversion / float32 / empty matrices; fetch-callable reads exactly the planned range once; NeedRange; names kinds; kv; corrupt and truncated input; GIL released;
  identical decodes under `PZ2_ISA=scalar|avx2|avx512` end to end), `test_v2_client.py` 24 (synthetic 3-sample study with the real codec and packs: independent gene-level computation from the row names, request/byte plans,
  two-phase plan, cells slices, ETag/offline, retries, wrong total, ignored Range, bad digest, `verify`, local `.s2` and manifest, `download` then offline, V2Bundle interface, loader routing, legacy switch),
  `test_v2_client_mcp.py` 8, `test_v2_client_corpus.py` 4 (GSE145370: migrate 2 samples with the native codec, equality with the legacy reader local and over HTTP, genes/cells slices, V2Bundle modalities; skipped without the corpus).
  W2's `test_v2_container.py` + `test_v2_migrate.py` (87) pass on my build. The existing loader tests that exercised the legacy download path (`test_loader_unit.py`, `test_download_integrity.py`,
  `test_loader_paths.py`: 22 tests) now call `_download_legacy` / `legacy=True` (mechanical edits, outside my paths); with that, `test_loader*`, `test_download_integrity`, `test_find`, `test_bundle_export`, `test_mcp_server`,
  `test_api_surface`, `test_bundle_modalities`, `test_1pz` all pass (`test_codec_properties.py` needs `hypothesis`, not installed).
* R: `r/tests/testthat/test-v2.R`, 9 tests / 61 expectations (dir transport; HTTP when `SINGLET_V2_TEST_URL` is set), all pass; the study comes from `python -P tests/python/_v2_synth.py <dir>` (`SINGLET_V2_TEST_ROOT`, or built on the fly when
  `python` + `singlet` are available). `R CMD check` test run on the scratch copy: [FAIL 9 | WARN 3 | SKIP 15 | PASS 267]; all 9 failures are in `test-integrity.R` / `test-aliases-gsm.R` (the pre-existing R 4.4.1 problem above), none in the new tests.

## 7. Files

New: `src/bindings/python/_pz2.cpp`, `src/bindings/python/build_pz2_mingw.sh`, `src/bindings/common/singlet/pz2_client.h`, `python/singlet/v2/{transport,client,bundle}.py`, `r/src/pz2_io.cpp`, `r/R/v2.R`, `r/man/{singlet_study,singlet_cells,pz2_*}.Rd`,
`r/tests/testthat/test-v2.R`, `tests/python/{test_v2_client*.py,_v2_synth.py,_v2_range_server.py}`, `bench/compression/tools/m_w4_*.{py,R}`, this note, `results/m_w4_*`.
Edited: `setup.py`, `MANIFEST.in`, `pyproject.toml` (package-data for `v2/packs`), `python/singlet/{_loader,bundle,_io,fetch}.py`, `python/singlet/mcp/server.py`, `scripts/sync_r_headers.py`, `r/R/{load,bundle,RcppExports}.R`, `r/src/RcppExports.cpp`,
`r/NAMESPACE`, `r/man/{download,load,read_singlet}.Rd`, `r/inst/include/singlet/{pz2/,pz2_client.h}` (generated), three legacy test files (section 6).

## 8. Not done / how to resume

* `singlet/__init__.py` does not export `open_study` / `cells`; they are `singlet.v2.client.open_study` / `.cells` (not my path). `docs-python/api.rst`, README snippets still describe the `.singlet` flow.
* The hosted `slice_cells` contract (W5 CS4) was not available; align names/arguments of `singlet_slice_cells`.
* MSVC and NEON builds of `_pz2` untried (no MSVC here); the Linux wheel path (`pip install .`) was not run on this laptop; only the MinGW build was.
* Peak memory of `to_anndata(layers=True)` holds the per-sample parts, the stacked X and two stacked layers (about 3x the result); consuming parts while stacking would cut it by about a third.
* Cells table barcodes are decoded for the whole table before selecting the called ones (30 ms for 64k cells); decoding only the selected indices would be faster for huge samples.
* The two-phase plan was verified for correctness and request counts on 4 MB samples only; its byte saving matters above ~48 MB, which the corpus does not contain (the largest production sample is 132 MB).
* R: modality readers for v2, `Seurat` path untested on v2 (the existing `.sce_to_seurat` is reused), pool-based concurrent ranges tested against one local server only.
* Resume: `PYTHONPATH=python python -P -m pytest tests/python/test_v2_client*.py`; rebuild the binding with `bash src/bindings/python/build_pz2_mingw.sh`; R: `python scripts/sync_r_headers.py`, then install `r/` with R 4.4.1 + `RTOOLS44_HOME=C:/RBuildTools/4.4`
  (build a scratch copy: `R CMD INSTALL` writes `.o`/`.dll` into `r/src`), `SINGLET_V2_TEST_ROOT=<dir from tests/python/_v2_synth.py>`; measurements: `python -P bench/compression/tools/m_w4_measure.py --v2 <native14b dir> --reps 3 --out ...`.
