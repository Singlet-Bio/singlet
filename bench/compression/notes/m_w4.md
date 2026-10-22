# m_w4: bindings + clients (W4): `singlet._pz2`, `singlet.v2.client`, R, local MCP

Owner: W4. Started 2026-10-22. Status: IN PROGRESS (this file is kept current; newest facts first in each section).

## 0. Status board (2026-10-22 10:30)

| step | what | state |
|---|---|---|
| 1 | `src/bindings/python/_pz2.cpp` (pybind11) + `src/bindings/common/singlet/pz2_client.h` + `src/bindings/python/build_pz2_mingw.sh` | DONE, 24 codec tests + 15 helper tests pass (real codec) |
| 2 | `python/singlet/v2/transport.py`, `client.py`, `bundle.py` (V2Bundle) | DONE, 24 end-to-end tests pass on a synthetic study; equality with the legacy reader checked on GSE145370 (below) |
| 3 | loader routing (`_loader.py`, `bundle.py`) with `legacy=` | DONE; existing loader tests not yet re-run (they test the legacy download path) |
| 4 | `setup.py` extension, `scripts/sync_r_headers.py` | not started |
| 5 | R: `r/src/pz2_io.cpp`, `r/R/v2*.R` | not started (R 4.4.1 + Rtools 4.4 + Rcpp/Matrix/curl/httr2/testthat/SingleCellExperiment are installed: can build and test) |
| 6 | MCP: v2 download info, compact JSON, `slice_cells` | not started |
| 7 | measurements (time, peak memory, bytes, requests; legacy vs v2) | waiting for the native-codec migration of GSE145370 (`bench/compression/results/m_w4_v2/native14`) |

## 1. Dependencies and what I assume from other workstreams

* W1 (`include/singlet/pz2/pz2.h`): codec landed 09:54. `Mode` has Native and Canonical only; `gene` mode is implemented in the binding
  (aggregation by a stored-row -> gene map, `pz2_client.h`). New `NamesKind::None` is handled.
* W2 (`python/singlet/v2/`): I use `container.S2Writer` (tests only), `cells.CellsReader`/`CellsTable`, `packs.PackStore`/`VocabPack`
  (`row_gene()`, `gene_table()`, `names()`), `manifest` (study-file entries), `context.V2Context` + `rcp_adapter` (regenerating legacy
  members for `V2Bundle.read`), `sample.V2Sample`. `codec.NativeCodec` calls my `encode`/`decode_columns`/`Member`: signatures agree.
* Test data: `bench/compression/corpus/bundles/GSE145370.singlet` (14 samples, 187 MB); migrated copies under
  `bench/compression/results/m_w4_v2/` (`py2`: 2 samples, `py14`: all 14 with the pure-Python codec 2, `native14`: all 14 with the native codec).

## 2. `singlet._pz2` (binding)

Built with `bash src/bindings/python/build_pz2_mingw.sh` (Rtools GCC 13, `-O2`, no `-march`, ~2 min on the loaded laptop; links only
`python311.dll` and the UCRT; zstd, libstdc++, winpthread are static). `setup.py` builds it with the other extension (step 4).

API (full doc comment at the top of `_pz2.cpp`):

* `read_header(buf) -> dict` (m, n, nnz, codec, value_type, row_order, chunk_cols, n_chunks, names kinds + ids, digest hex, `prefix_len`, sections).
* `Member(source=None, fetch=None, prefix_hint=65536)`: opened from bytes-like (a whole member or at least its prefix) and/or a
  `fetch(off, len)` callable that supplies any range the decoder asks for (the NeedRange protocol is handled inside, the callable runs
  with the GIL). Attributes `header, m, n, nnz, prefix_len, value_dtype, row_perm`; methods `chunk_index()`, `plan(c0, c1)`,
  `add_range(off, buf)`, `has(off, len)`, `column_counts`, `row_names()`, `col_names()`, `kv()`, `decode(c0, c1, *, mode, threads, dtype, index_dtype, isa, cols, gene_map, n_genes)`.
* `decode(source, ...)`, `decode_columns(source, c0, c1, ...)`: one-shot forms (source may be the fetch callable).
* `encode(indptr, indices, data, m, n, *, codec, chunk_cols, classes_per_octave, lanes, prob_bits, value_prob_bits, ctx_kind, value_type,
  row_names, col_names, kv, digest, compute_digest, threads, zstd_level) -> (bytes, stats)`; names are `list[str]` (inline) or
  `{kind: inline|pack|pack_subset|cells|none, pack_id, pack_index, names}`. `digest(...) -> 16 bytes`.
* `aggregate_genes`, `select_columns`: the two post-decode kernels on arrays (used by the tests as scipy cross-checks).
* `isa_report()`, `selected_isa()`, `best_isa()`, `isa_supported(name)`; `Pz2Error` (`.code`, `.off`, `.len`), `NeedRange`.

Behaviour: output arrays are allocated once by numpy (uninitialised) and filled by the decoder; `indptr` int32 (int64 if the bound
is >= 2^31), `indices` int32, `data` in the stored width (uint8/uint16/uint32/float32) or the requested `dtype`; the GIL is released
for decode, selection and aggregation; arrays are trimmed in place (`ndarray.resize`, no copy) to the exact nnz. Mode `gene`
composes `gene_map` (canonical row -> gene, -1 drops) with the member's row permutation once, decodes in native order (no sort), and
aggregates in place per column with a bitmap (wide columns) or a sort of the touched genes (narrow ones), threaded over column blocks
of equal nnz; sums are int32 (or float64 for R). Column selection (`cols`, ascending, relative to `c0`) compacts in place.
Zero-valued entries are dropped by aggregation, as a sparse product would.

## 3. `singlet.v2.client` (read path)

(to be completed: API summary, request plan, behaviour on failure; see the module docstring until then)

* `open_study(gse)`: one GET of `v2/m/<GSE>.json`, conditional on the cached ETag (`<cache>/v2/m/`), 304 costs no body; in-process memo 60 s;
  offline falls back to the cached manifest with a warning. `StudyNotFound` (a `FileNotFoundError`) when the manifest does not exist.
* `Study.to_anndata(samples, genes, obs_filter, layers, cell_order, concurrency, threads, verify, whole_max)`; `load(...)` is the same through the accession.
* `cells(...)` / `Study.cells(where, samples, columns, genes, layers, max_cells)`.
* `load_s2(path)` / `load(path.s2)` / `load(manifest.json)`: local, mmap, zero copy of the file.
* `download_study`: mirrors the manifest, objects (whole-object sha256 verified) and packs under `<cache>/v2/`; `load` then reads them locally.
* Request plan (verified by tests): manifest + 1 coalesced range per sample (cells + exon + intron are adjacent in the object, 64-byte alignment gaps only);
  samples larger than 48 MB (`whole_max`) use a two-phase plan: cells + matrix prefixes (concurrently), then only the chunk runs that cover the wanted cells.
* Integrity: HTTP 206 + `Content-Range` start/end/total against the manifest size on every ranged read (wrong total, ignored Range, short body are errors),
  pz2 header digest == manifest member digest, `verify=True` (or `SINGLET_V2_VERIFY=1`) recomputes the canonical digest of every decoded matrix.
* Transport: persistent connection per thread, retries (429/5xx/network, exponential backoff, `Retry-After`), `User-Agent: singlet-client/2`, redirects limited to https.

Decision flagged for the orchestrator: `legacy=None` (default) never reads a legacy bundle when a study has no v2 manifest: it raises
`StudyNotFound` with a message naming `legacy=True`. `SINGLET_LEGACY_FALLBACK=1` switches on an automatic fallback (with a FutureWarning)
for the transition while studies are being migrated; the default is the strict behaviour the brief asks for.

Known difference from the legacy reader: obs order. The legacy reader orders called cells as `cell_calls.tsv` does (STARsolo: bytewise sorted
barcodes; EmptyDrops: ascending column; top-N fallback: unstable sort order). The v2 `cells` table keeps a `called` bitmap only, so
`cell_order="legacy"` (default) sorts by barcode unless the cells table carries `cc_pos` (then that order), or `cc_listed`/`cc_pvalue`
(EmptyDrops: matrix column order). On GSE145370 (STARsolo) this reproduces the legacy order exactly (checked).

## 4. R

(not started)

## 5. MCP

(not started)

## 6. Tests

* `tests/python/test_v2_client_native.py` (15): header errors, ISA selection (`PZ2_ISA` forced in child processes), `aggregate_genes` and `select_columns` against scipy.
* `tests/python/test_v2_client_codec.py` (24): round trips (codecs, shapes, native/canonical, thread counts, column ranges), value widths, dtype conversion, float32,
  empty matrices, gene mode with selection against scipy, ISA identity, fetch-callable behaviour (exactly the planned range, once), NeedRange, names kinds, kv, corrupt/truncated inputs, GIL release.
* `tests/python/test_v2_client.py` (24, ~80 s): synthetic 3-sample study (`_v2_synth.py`: real codec, real packs, W2's container writer), served by `_v2_range_server.py`.

## 7. Measurements

(to be filled in when `native14` is migrated)

## 8. Not done / how to resume

(to be completed)
