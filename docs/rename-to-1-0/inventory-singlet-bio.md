# singlet-bio: inventory of "v2" branding (for the Singlet 1.0.0 rename)

Repo: singlet-bio, `origin/main` at 5c279cd. Read-only survey, nothing edited.

Method and caveats
* All counts come from `git grep -c` (text files, `-I`) and so count **matching lines**, not tokens. Pathspec excludes noted per section. `-i v2` is case-insensitive.
* Whole repo: 3,127 tracked files; 635 text files (684 with binaries) contain `v2` (case-insens.) on 47,583 lines. Of those lines: `bench/compression` 38,401 (290 files), `bench/libtype_screen/libtype_suspects.tsv` 6,191 (one TSV, all 10x chemistry), `papers` 472, `notebooks` 54. Everything that is the format/rename scope is therefore only about 2,400 lines.
* `r/inst/include/singlet/**` is a **verbatim mirror** of `include/singlet/pz2/**` and `src/bindings/common/singlet/pz2_client.h`, produced by `scripts/sync_r_headers.py`. Rename once in the source and re-run the sync (or CI's staleness check will fail).
* This repo has **no** website, Worker, D1 migration or wrangler files. D1 tables, the `data.singlet.bio` Worker and R2 key layout are only *referenced* here (campaign scripts, tests, docs). The authoritative definitions live in the website repo.
* Python/R/C++ code never embeds the version in a CLI named `singlet v2 ...`. The only CLIs are `python -m singlet.v2.migrate|verify|packs_build` (section 2). `[project.scripts]` has only `singlet-mcp` and `singlet`.

Things whose rename **changes stored or served bytes** (the high-risk list; details in sections 4, 7, 10):
1. `"format": "singlet-v2"` in every study manifest, and `"singlet-v2-packs"` in `v2/ref/index.json` (and the packaged copy `python/singlet/v2/packs/index.json`).
2. R2 key prefix `v2/` (all of `v2/s/`, `v2/m/`, `v2/ref/`, `v2/ledger/`, `v2/cidx/`, `v2/a/`, `v2/cf/`) and the URL `data.singlet.bio/v2/...`: objects are already published under it.
3. Magic bytes `SINGLET2`, `PZ2M`, `SPK2`, `CEL2`, `SCX1`, and hash domain strings `PZ2CANON1`, `CEL2CANON1`: changing them changes content digests, so every published object's name (`<d16>.s2`) and the manifest `rev` would change. Treat as wire format, not branding, unless a full re-migration is planned.
4. The cells-table columns `called_v2`, `called_v2_fdr` and cells meta key `called_v2` stored in each `.s2`.
5. Manifest/D1 field values `v2_*` (D1 tables `v2_study`, `v2_sample`, `v2_migration`, migrations 032/033) and the `/api/ingest/v2-study` endpoint (defined in the website repo).

---

## 1. Python package `python/singlet/v2/`

**Contents (git-tracked):** 33 `.py` modules, 16 `packs/*.pack` + `packs/index.json`, 1 `recipes_data/features_order_GRCm39-2024-A.u32.zst`. Directory: `python/singlet/v2/`.

Modules (v2 text hits, case-insens., inside the package: 295 lines in 26 files): `__init__` 3, `atlas` 25, `builtin_recipes` 1, `bundle` 21, `cellcall` 22, `cells` 1, `client` 59, `codec` 5, `container` 1, `context` 4, `junctions` 9, `legacy` 3, `manifest` 3, `matrix` 1, `migrate` 68, `packs` 9, `packs_build` 3, `rcp_adapter` 4, `recipes` 1, `recipes_ctx` 3, `recipes_pzw` 4, `sample` 13, `transport` 3, `util` 3, `verify` 25. (No hits: `recipes_cells`, `recipes_extra`, `recipes_mtx`, `recipes_psi`, `recipes_qc`, `recipes_star`, `recipes_tables`, `recipes_tsv`.)
Intra-package imports are relative (`from . import ...`, 100 lines) so they survive a directory rename; absolute `singlet.v2` self-references inside the package: 29 lines (`atlas` 6, `verify` 5, `rcp_adapter` 4, `client` 3, `migrate` 3, `context` 2, `builtin_recipes`/`codec`/`legacy`/`packs`/`packs_build`/`recipes`/`util` 1 each).

**Import / reference sites outside the package** (`singlet.v2`, `from .v2`, `.v2`; 235 lines in total across the repo excluding bench/compression):

| group | files:count |
|---|---|
| python core | `python/singlet/_loader.py` 14, `_io.py` 4, `bundle.py` 3, `mcp/server.py` 3, `__init__.py` 2, `celllabels/io.py` 4, `features/io.py` 1 |
| scripts | `scripts/v2_campaign.py` 11, `v2_migrate.sbatch` 4, `v2_junction_table.py` 3, `v2_publish.py` 2, `v2_cidx_build.py` 1, `v2_retire.py` 1, `annotate_sample.py` 1, `features_run.py` 1 |
| bench | `bench/junctions/bench_table.py` 5, `called_v2/validate.py` 3, `called_v2/aggregate.py` 2, `called_v2_truth/verify_cellcall.py` 2, `called_v2_truth/revalidate.py` 1, `called_v2/run.sbatch` 1, `cidx/chunking.py` 2, `cidx/formats.py` 2, `cidx/measure.py` 1, `pz2_chunks/measure.py` 2, `junctions/census_stats.py` 1 |
| tests | `tests/python/test_v2_junctions.py` 12, `test_v2_container.py` 11, `test_v2_migrate.py` 11, `test_v2_recipes.py` 11, `_v2_synth.py` 8, `test_v2_client_corpus.py` 7, `test_v2_client.py` 4, `_cidx_fixture.py` 4, `test_v2_recipes_corpus.py` 4, `test_v2_campaign_common.py` 4, `test_v2_atlas.py` 3, `test_v2_cellcall.py` 3, `test_v2_modalities.py` 3, `test_v2_client_mcp.py` 2, `test_v2_campaign_catalog.py` 1, `test_cell_features.py` 1; `tests/data/called_v2_golden/generate.py` 2; `r/tests/testthat/test-v2-modalities.R` 2 |
| docs | `docs/format-v2-cell-index.md` 13, `format-v2.md` 3, `v2-migration.md` 1, `cell_annotation.md` 1, `cell_features.md` 1 |

By submodule (outside the package): `singlet.v2` bare (e.g. `from singlet.v2 import client as C`) 83, `.client` 19, `.migrate` 16, `.atlas` 14, `.verify` 9, `.bundle` 9, `.packs` 5, `.recipes_ctx` 4, `.matrix` 4, `.cellcall` 4, `.cells` 3, `.recipes_psi` 2, `.legacy` 2, one each `.sample .recipes .packs_build .manifest .junctions .context`. The only `import singlet.v2.x as y` form is `tests/python/test_v2_container.py:259`; no `from singlet import v2`.
Public re-export: `python/singlet/__init__.py:244-245` (`from singlet.v2.client import cells`, `open_study`).
Path strings `python/singlet/v2` (hard-coded filesystem path, 15 files): `pyproject.toml` 2, `v2_migrate.sbatch` 4, `v2_campaign.py` 1, `python/singlet/v2/packs_build.py` 2, `recipes_pzw.py` 1, `r/R/v2.R` 3, `bench/called_v2_truth/common.py` 2, `tests/python/test_v2_container.py` 2, `tests/data/called_v2_golden/README.md` 1, and docs (`format-v2.md` 3, `v2-migration.md` 2, `called-v2.md` 3, `called-v2-truth.md` 1, `format-v2-cell-index.md` 1, `CELL-ARCHITECTURE.md` 1).

**Packaging entries:**
* `pyproject.toml:108` `package-data`: `"v2/packs/*.pack", "v2/packs/index.json", "v2/recipes_data/*.zst"`.
* `pyproject.toml:153` `"python/singlet/v2/packs_build.py"` (per-file tool config list); `pyproject.toml:60` comment mentioning `called_v2` and `python/singlet/v2/cellcall.py`; `[tool.setuptools.packages.find]` at line 103 discovers the package automatically (no explicit `singlet.v2`).
* `setup.py:17-21` extension `singlet._pz2` from `src/bindings/python/_pz2.cpp` (the "v2 matrix codec"). `MANIFEST.in:12-13` `recursive-include include/singlet/pz2 *.h`.
* `CMakeLists.txt:35-42,167` target `singlet_pz2`, alias `Singlet::pz2`, install list.
* Pack file names are content digests (not branded), so the directory rename does not rename them.

---

## 2. Public Python API names containing "v2"

| identifier | where (file:line) | notes |
|---|---|---|
| `V2Bundle` (class) | `python/singlet/v2/bundle.py:39` | 21 uses repo-wide: `_loader.py` 6, `bundle.py` 3 (+ own file 5), `celllabels/io.py` 3, `tests/python/test_v2_*` rest. Returned by `open_bundle()` for v2 studies (public type) |
| `V2Sample` (class) | `python/singlet/v2/sample.py:24` | 17 uses |
| `V2Context` (class) | `python/singlet/v2/context.py:295` | 8 uses |
| `V2Error` (exception) | `python/singlet/v2/client.py:137` and a second, separate class `sample.py:20` | 32 uses |
| `read_v2` | `python/singlet/celllabels/io.py:219`, re-exported `celllabels/__init__.py:6,18,35` (`__all__`) | public; 8 uses incl. `bench/features/pilot.py:175`, `scripts/annotate_sample.py:92`, `features/io.py:88` |
| `is_v2`, `v2_gsms`, `iter_samples(..., v2=)` | `python/singlet/features/io.py:26, 71, 79-88` | 3+3+... ; `scripts/features_run.py:120-129` call them |
| `v2_options` kwarg, `_V2_OPTIONS`, `_load_v2`, `_no_v2_message` | `python/singlet/_loader.py:427,482,523,570,574,579,629,682,737` (`load(..., **v2_options)` documented in docstring 482) | `load()` public kwarg bucket: 8 lines |
| `_v2_sample_info` | `python/singlet/mcp/server.py:599` | private; but see wire output below |
| `Options.cell_call_v2`, `--no-cell-call-v2` | `python/singlet/v2/migrate.py:74,98,1276-1287` | see section 10 |
| `_add_called_v2` | `python/singlet/v2/migrate.py:463` | see section 10 |
| `FORMAT = "singlet-v2"` | `python/singlet/v2/__init__.py:7`, `manifest.py:25` | wire string, section 7 |
| `adata.uns["singlet_v2"]` | `python/singlet/v2/client.py:1671,1712,1847,2032` | **public output key** on every AnnData from a v2 read; R twin `S4Vectors::metadata(sce)$singlet_v2` at `r/R/v2.R:1022`, test `r/tests/testthat/test-v2-modalities.R:175` |
| `--v2 DIR` CLI flag | `python/singlet/v2/verify.py:4` (docstring), arg in `verify.py` main (~714-733); passed by `scripts/v2_campaign.py` | |
| `singlet.read_pz2`, `_detect_format()` returns `"pz2"` / `"s2"` | `python/singlet/_io.py:226, 26-41, 276-283`; `read_pz2` used 3 lines in `_io.py` only | `"pz2"`, `"s2"` are return values of a private helper |
| `singlet._pz2` extension module, `Pz2Error`, `NeedRange` | `src/bindings/python/_pz2.cpp:50-51,877,881`; imported as `singlet._pz2` in 30 files (counts: `test_v2_client_codec.py` 28, `test_v2_client_native.py` 18, `codec.py` 8, `_v2_synth.py` 8, `test_v2_junctions.py` 8, `atlas.py` 6, `_io.py` 5, `client.py` 4 ...) | `src/bindings/python/build_pz2_mingw.sh` also named after it |
| Container kind labels `KIND_PZ2`, `KIND_NAMES[1]="pz2"` | `python/singlet/v2/container.py:35,40`; `migrate.py:52-58` `KIND_LABEL` | `"pz2"` string lands in the ledger (`v2_kind`) and counters |

CLI entry points (all `python -m ...`): `singlet.v2.migrate` (`migrate.py:4,1235,1311`), `singlet.v2.verify` (`verify.py:4,714,733`), `singlet.v2.packs_build` (`packs_build.py:4,210`). Invoked from: `scripts/v2_campaign.py:9,13,2235-2240` (`--migrate-cmd` default `<python> -m singlet.v2.migrate`, `--verify-cmd`), `scripts/v2_migrate.sbatch:156,178,179`, `scripts/v2_publish.py:12`, `docs/format-v2.md:442`. There is no `__main__.py` inside `python/singlet/v2/`; `python/singlet/__main__.py` has no v2 content.

`python/singlet/_io.py` (v2 lines 7,8,9,22,23,29,227,230,253,263,276,281; 12 lines): magics `_PZ2_MAGIC = b"PZ2M"` (22), `_S2_MAGIC = b"SINGLET2"` (23), env `$SINGLET_V2_PACKS` doc (230), imports `singlet.v2.packs.default_store` (253, 263), `singlet.v2.client.load_s2` (281).
`python/singlet/_loader.py` (68 v2 lines, the heaviest outside the package): docstring URLs `https://data.singlet.bio/v2/m/<GSE>.json`, `/v2/s/<GSM>/<d>.s2` (7-8), `_DATA_BASE_DEFAULT = "https://data.singlet.bio"` (55), `.s2` suffix dispatch (388, 672), cache layout `<output_dir>/v2/` (205-207), `singlet_v2` etc.
`python/singlet/mcp/server.py` (v2 lines 11,155,167,212,315,599-625,645,664-673,737,746,768): **tool output wire fields**: `"format": "v2"` (623), `out["v2"]` (673), `out["v2_error"]` (669), `"manifest": ... f"v2/m/{gse_id}.json"` (625), generated code string `singlet.v2.cells("...")` (768). Lines 155, 167, 315 are `10xv2` protocol filters (false positives).
Other: `python/singlet/bundle.py` docstrings (5,6,8,1384,1385) + import (1390); `python/singlet/fetch.py:7,8,65` docstrings/messages (`https://data.singlet.bio/v2/`); `python/singlet/celllabels/pipeline.py:23` `cellguide-v2` is a **different** thing (annotation guide version), not the format.

---

## 3. R package (`r/`)

Total `v2` lines in `r/` (excluding `r/inst/include`): `r/R/v2.R` 228, `r/R/load.R` 21, `r/R/bundle.R` 3, `r/man/*.Rd` 20 (download 5, load 7, read_singlet 2, singlet_cells 1, singlet_study 3, singlet_study_modalities 2), `r/src/pz2_io.cpp` 11, `r/tests/testthat/test-v2.R` 57, `r/tests/testthat/test-v2-modalities.R` 34. No `v2` in `NAMESPACE`, `DESCRIPTION`, `NEWS.md`, `README.md`, `_pkgdown.yml`, `vignettes/`, `Makevars`.

* **Exported functions defined in `r/R/v2.R`** (from `r/NAMESPACE`): `singlet_study` (v2.R:275), `singlet_study_modalities` (:499), `singlet_cells` (:957), S3 `print.singlet_study` (:335, registered `S3method(print,singlet_study)`). **None has "v2" in its name**; the file name itself (`R/v2.R`, referenced in `r/man/singlet_cells.Rd:2`, `singlet_study.Rd:2`, `singlet_study_modalities.Rd:2` "Please edit documentation in R/v2.R", and `r/R/load.R:7` comment) is the only branded part. Other exports in NAMESPACE (`download`, `load`, `read_singlet`, `singlet_*`) have no v2 in their name but several Rd titles do: `singlet_study.Rd:5` "Open a study's v2 manifest", `singlet_study_modalities.Rd:5` "Modalities of a v2 study".
* **Internal `.v2_*` names** (51 definitions in `r/R/v2.R`, all non-exported): `.v2_assemble .v2_axis .v2_cell_order .v2_cells .v2_check_modalities .v2_check_response .v2_clear_memory .v2_coalesce .v2_count .v2_dgc .v2_download .v2_empty_gene_matrix .v2_gene_members .v2_gene_modalities .v2_get .v2_get_dir .v2_get_many .v2_handle .v2_hdr_range .v2_json .v2_junction_events .v2_legacy_fallback .v2_load_gse .v2_load_manifest .v2_make_sce .v2_mem .v2_member_gene_map .v2_member_has_data .v2_members_for .v2_modality_alias .v2_modality_members .v2_no_study_message .v2_normalize_modalities .v2_pack .v2_pack_dir .v2_psi .v2_read_sample .v2_read_study .v2_reader .v2_require_curl .v2_root .v2_runs .v2_sample_empty .v2_sample_modalities .v2_stats .v2_stats_reset .v2_stop .v2_study_file .v2_study_object .v2_transport .v2_vocab`. Callers outside v2.R: `r/R/load.R:417,422,579,581` (`.v2_download`, `.v2_legacy_fallback`, `.v2_load_gse`, `.v2_load_manifest`), `r/R/bundle.R:414` (`.v2_load_manifest`). Most used: `.v2_stop` 30, `.v2_get` ~, `.v2_mem` 27.
* **Condition classes** (user-catchable, so effectively public): `singlet_v2_error`, `singlet_v2_not_found`, `singlet_v2_integrity`, `singlet_v2_modality_unavailable` (`r/R/v2.R:16,113-168,294-304,455-482,882,1055,1075`; `r/R/load.R:418`).
* **Metadata key** `metadata(sce)$singlet_v2` (`r/R/v2.R:1022`; test `test-v2-modalities.R:175`). Manifest checks `"singlet-v2"` at `r/R/v2.R:323-324,915`.
* **Tests:** `r/tests/testthat/test-v2.R` (57), `test-v2-modalities.R` (34). Env vars used there: `SINGLET_V2_TEST_ROOT` (4), `SINGLET_V2_TEST_URL` (3) in test-v2.R; `SINGLET_V2_EXTRAS_ROOT` (3) in test-v2-modalities.R. (`SINGLET_LEGACY_FALLBACK` in v2.R:24,1078 is not branded.)
* **Man pages mentioning v2:** the six listed above, plus 12 generated `r/man/pz2_*_.Rd` files (names carry `pz2`): `pz2_cells_ pz2_chunk_first_cols_ pz2_decode_ pz2_header_ pz2_isa_report_ pz2_names_ pz2_pack_genes_ pz2_pack_info_ pz2_plan_ pz2_sha256_ pz2_whitelist_barcodes_ pz2_zstd_` (38 `pz2` lines).
* **`r/src` and RcppExports pz2 functions:** `r/src/pz2_io.cpp` (83 `pz2` lines across it and `r/src/RcppExports.cpp` 60 / `pz2_io.cpp` 23); `r/R/RcppExports.R` 25. The 12 Rcpp-exported functions: `pz2_header_ pz2_chunk_first_cols_ pz2_plan_ pz2_decode_ pz2_names_ pz2_zstd_ pz2_isa_report_ pz2_sha256_ pz2_pack_info_ pz2_pack_genes_ pz2_whitelist_barcodes_ pz2_cells_`, each also as C symbol `_singlet_pz2_*_` in `RcppExports.cpp` (registered through `useDynLib(singlet, .registration = TRUE)`). `RcppExports.*` and the Rd files are generated (`Rcpp::compileAttributes()` + roxygen) from the `//'` comments in `pz2_io.cpp`; regenerate after renaming. Callers of the pz2_* functions: `r/R/v2.R` 16 lines, `r/tests/testthat/test-v2.R` 8 lines.
* R-side C++ error strings `"v2: truncated data"` etc. at `r/src/pz2_io.cpp:121,131-136,141,154,172,236` (user-visible messages).
* Headers mirrored at `r/inst/include/singlet/pz2/**` (14 files, 119 `pz2` lines) and `r/inst/include/singlet/pz2_client.h`.
* Cross-links: `r/R/bundle.R:389-390,414` doc/comment, `r/man/read_singlet.Rd:10-11`, `r/man/download.Rd:24-45`, `r/man/load.Rd:27-84`.

---

## 4. C++ (`include/singlet/**`, `src/**`, `tests/cpp/**`)

**Namespace / headers / files** (no class or function in the C++ tree has "v2" in its name; the branding is the `pz2` codec name):
* Namespace `singlet::pz2` (and `singlet::pz2::detail`) in all 14 headers (`include/singlet/pz2/pz2.h`, `detail/{api,codec2,common,container,isa,rans1,rans1_avx2,rans1_avx512,rans1_dec,rans1_enc,sha256,simd,values}.h`) and `src/bindings/common/singlet/pz2_client.h`. R mirrors: `r/inst/include/singlet/pz2/**`, `r/inst/include/singlet/pz2_client.h`.
* `pz2` references: `include/singlet/pz2` 116 lines (14 files), `r/inst/include` 119 (15), `tests/cpp` 70 (17), `src/bindings` 30 (3: `python/_pz2.cpp`, `common/singlet/pz2_client.h`, `python/build_pz2_mingw.sh`), `r/src` 83 (2), `bench/pz2` 37 (9), `scripts/sync_r_headers.py` 17, `docs/format-v2.md` 12, `CMakeLists.txt` 6, `setup.py` 4, `MANIFEST.in` 1. Whole repo outside bench/compression and papers: 829 lines. Tracked files with `pz2` in the path: 62 (outside bench/compression).
* `include/singlet/pz2/**` has 14 files that contain the literal "v2" only in comments pointing at `docs/format-v2.md`: `pz2.h` 6 (lines 2,4,59,125,195,367), `detail/api.h:2`, `codec2.h:2`, `container.h:2`, `rans1.h:2`; `rans1_avx2.h`/`rans1_avx512.h` hits are register names (`v20`, `v16`, ...), false positives.
* Macros and env: `PZ2_AINLINE` 18, `PZ2_ISA` 10 (env var, `isa.h:110,133`, also `sha256.h`, `pz2.h`, `_pz2.cpp`, tests), `PZ2_COMPILED_AVX512/AVX2/SHANI`, `PZ2_BEGIN_*`/`PZ2_END_*`, `PZ2_X86`, `PZ2_SCALAR_ONLY` (`isa.h`), `PZ2_SOURCE_DIR` (`tests/cpp/CMakeLists.txt:55`), `PZ2_CORPUS` (`tests/cpp/test_pz2_roundtrip.cpp:30`), `PZ2_TEST_MAX_MB` (`tests/cpp/CMakeLists.txt:63`, `test_pz2_roundtrip.cpp:186`).
* Other symbols: `Pz2Error`, `NeedRange` exception types (`_pz2.cpp:50-51,877,881`), `g_pz2_error`, `g_pz2_needrange`, `pz2c` (5), `Errc::BadMagic` ("not a pz2 member"), error prefix `"pz2: "` (`pz2.h:116`).
* CMake: `singlet_pz2` INTERFACE library (`CMakeLists.txt:36-42,167`), alias `Singlet::pz2` (37). `tests/cpp/CMakeLists.txt:49-68` macro `add_pz2_test` and tests `test_pz2_api`, `test_pz2_edge`, `test_pz2_roundtrip`, `test_pz2_multi_tu` (+ `_b.cpp`). Test files: `tests/cpp/test_pz2_api.cpp`, `test_pz2_edge.cpp`, `test_pz2_multi_tu.cpp`, `test_pz2_multi_tu_b.cpp`, `test_pz2_roundtrip.cpp`.
* `include/singlet/pz2/detail/api.h:401` `h.update("PZ2CANON1", 9)`: canonical digest domain string (changes digests if renamed; mirrored in `python/singlet/v2/matrix.py:124`, `verify.py:166`, `bench/pz2/ref_decode.py:381`, `docs/format-v2.md:145`, `tests/python/test_v2_container.py:223`).
* `src/bindings/python/_pz2.cpp` 875-951: module doc strings "bindings for the v2 matrix codec".

**Other "v2" C++ hits that are NOT the format** (172 lines in `include/singlet/**`, `src/**`, `tests/cpp/**` outside pz2): `include/singlet/fq/indrop_detect.h` (`InDropVersion::V2`, inDrop v2 chemistry), `fq/protocol.h` (`10x-3p-v2` etc.), `fq/species_registry.h`, `pileup/species_detect.h` (`ARS-UI_Ramb_v2.0`, `ASM276v2`, `CSAV2.0`), `pileup/species_kmer_db_gen.h`, `pileup/nonhost/min_sketch.h` (min-sketch file format v2), `pileup/doublet_detect.h:31` (algorithm v2), `pileup/pileup_engine.h:4` ("Pipeline V2"), `gpu/**/*.h` (`cublas_v2.h` x30, local variables `v2`), `star/htslib/**` (BCFv2.1), `tests/cpp/test_doublet_detect_v2.cpp` (+ `tests/cpp/CMakeLists.txt:135` `add_pileup_test(test_doublet_detect_v2)`: doublet-detector v2, NOT the format), `tests/cpp/test_ws2_protocol_hardening.cpp` 20, `test_indrop_detect.cpp` 10, `tests/cpp/gpu/*` and `tests/cpp/gpu/refs/sparse_eig_medium_ref.npz` (binary).

**Magic constants: every definition and reader check** (non-bench, non-paper; text files only):

| magic | definitions (file:line) | reader / validator checks | other mentions |
|---|---|---|---|
| `"SINGLET2"` (8 B, `.s2` header) | `python/singlet/v2/container.py:29` (`MAGIC`); `python/singlet/_io.py:23` (`_S2_MAGIC`); spec `docs/format-v2.md:40`, header docstring `container.py:6` | `container.py:192` (writer), `container.py:262` (`if magic != MAGIC`), `_io.py:40` (format detect), `python/singlet/v2/verify.py:130` (literal `b"SINGLET2"`), `scripts/v2_campaign.py:747-748` (literal + message "not a SINGLET2 object") | tests: `tests/python/test_v2_container.py:139`, `test_v2_campaign_common.py:5,63` (fabricates headers). **No C++ or R reader** of the `.s2` header |
| `"PZ2M"` (4 B, matrix member) | C++ `include/singlet/pz2/detail/container.h:9` (`kMagic[4] = {'P','Z','2','M'}`; mirror `r/inst/include/.../container.h:9`); `python/singlet/v2/codec.py:46` (`MAGIC`); `python/singlet/_io.py:22` (`_PZ2_MAGIC`); comment `include/singlet/pz2/pz2.h:53` (+ R mirror); spec `docs/format-v2.md:72`; `codec.py:17` | C++ `container.h:21` (write), `container.h:43` (`memcmp ... throw Errc::BadMagic`); `codec.py:196` (write), `codec.py:228` (check); `_io.py:38`; `tests/cpp/test_pz2_edge.cpp:605-606` (BadMagic test) | `bench/pz2/ref_decode.py:56`; `tests/python/test_v2_container.py:410` |
| `"SPK2"` (4 B, pack wrapper) | `python/singlet/v2/packs.py:41` (`MAGIC`); docs `docs/format-v2.md:249,377`; docstring `packs.py:4,6`, `junctions.py:9` | `packs.py:56` (write), `packs.py:60-61` (check "not an SPK2 pack"), `packs.py:798` (index scan); **R**: `r/src/pz2_io.cpp:209` (`memcmp(p,"SPK2",4)`; message "not an SPK2 pack"), 198 comment, 3 header comment, roxygen 488 | tests `tests/python/test_v2_campaign_common.py:106,109`; 16 packaged `*.pack` files start with SPK2 (binary) |
| `"CEL2"` (4 B, cells table) | `python/singlet/v2/cells.py:41` (`MAGIC`), docstring `cells.py:6`; spec `docs/format-v2.md:227` | `cells.py:224` (write), `cells.py:337` (`first[:4] != MAGIC`); **R**: `r/src/pz2_io.cpp:605` (`memcmp(base,"CEL2",4)`; message "not a cells table"), roxygen 592, `RcppExports.R:86`, `r/man/pz2_cells_.Rd:5,22` | none C++ in include/ |
| `"CEL2CANON1"` | `python/singlet/v2/cells.py:188` (`h.update(b"CEL2CANON1")`), docs `cells.py:25`, `docs/format-v2.md:240` | (digest input; verified by `verify.py` via `cells.py`) | |
| `"PZ2CANON1"` | `include/singlet/pz2/detail/api.h:401` (+R mirror), `python/singlet/v2/matrix.py:124`, `verify.py:166`, `bench/pz2/ref_decode.py:381`, spec `docs/format-v2.md:145` | digest recomputation in `verify.py:166`, `tests/python/test_v2_container.py:223` | |
| `"SCX1"` (4 B, cell-index shard) | `python/singlet/v2/atlas.py:86` (`SCX_MAGIC`), section banner 85; spec `docs/format-v2-cell-index.md:46,56,62` | `atlas.py:194` (write), `atlas.py:207` ("not an SCX1 shard"), `atlas.py:220-221` | mentions `bench/cidx/formats.py:11`, `bench/junctions/bench_table.py:100`, `docs/format-v2.md:285,458,468,498,528`; `tests/python/test_v2_atlas.py:2,109`. Related: `GATHER_MAGIC = b"SGA1"` `atlas.py:3221`, check `atlas.py:3444` (not v2-named, stored/served only on the wire) |
| `"SVP0"` (vocab pack body coder) | `python/singlet/v2/packs.py:255` (writer), `:278` (check), `:459`, `:715` (header `coder`); `r/src/pz2_io.cpp:523,533,544,559` | | stored in pack headers (packaged packs); `bench/compression/notes/w4_vocab.md` |

Counts for the magics repo-wide (excluding `bench/compression`, `papers`): `SINGLET2` 9 lines/7 files, `PZ2M` 10 lines/9 files, `SPK2` ~34 lines (+16 binary packs), `CEL2` 12 lines, `CEL2CANON1` 3, `SCX1` 33, `PZ2CANON1` ~10. In `bench/compression` (historical): 6 lines of the five magics in 3 files. Not part of the format (legacy): `python/include/singlepress/singlepress.h:63-64` ("sparsepress v2", `MAGIC_LEGACY = 0x53505A32` "SPZ2"), `tests/python/test_io_format_detection.py:25`.

---

## 5. Scripts (`scripts/`, plus sbatch elsewhere)

| script | v2 lines | what it is |
|---|---|---|
| `scripts/v2_campaign.py` | 130+ (largest; 12 references to the other v2 scripts) | migration campaign driver (`run`, `plan`, `summarize`, `catalog-sql`, `catalog-post`, `refs-index`, `retire-sql` sub-commands, lines 2212-2310); `singlet.v2.*` 11, `singlet-v2` format checks 763,1060-1061,2150-2151, UA `singlet-v2-campaign/1.0` 1321, D1 tables, `/api/ingest/v2-study` |
| `scripts/v2_publish.py` | 5 refs | |
| `scripts/v2_retire.py` | 9 refs | `singlet-v2` check at 91-92 |
| `scripts/v2_cidx_build.py` | 3 refs | cell-index builder; writes `v2/cidx/` |
| `scripts/v2_junction_table.py` | 6 refs | `singlet-v2-packs` check 620-621 |
| `scripts/v2_junction_table.sbatch` | 7 refs (job name `jxtab`) | |
| `scripts/v2_migrate.sbatch` | 8 refs | branch `feat/singlet-v2` (lines 3,26,47,48) |
| `scripts/singlet_orchestrator_v2.sh` | 2 self refs | "canonical v2 pipeline" = pipeline v2, probably **not** the format; no inbound references anywhere in the repo |

Scripts that *reference* the v2 scripts or package: `bench/junctions/census_stats.py` 1, `docs/format-v2-cell-index.md` 4, `docs/format-v2.md` 1, `docs/v2-migration.md` 1, `python/singlet/v2/atlas.py` 1, and tests (`tests/python/test_v2_campaign*.py` import `scripts/v2_*.py` by path: `test_v2_campaign.py` 7, `_catalog` 6, `_common` 5, `_publish` 8, `_retire` 8, `test_v2_junctions.py` 4). Other scripts with real format references: `scripts/annotate_campaign.py` (`v2_manifest_url` column 68, `_has_v2` 142-182, `/v2/m/{gse}.json` 145, `v2_study` 15), `scripts/annotate_sample.py:69,92`, `scripts/features_run.py:120-129,202-203`, `scripts/features_run.sbatch:7,29` (default branch `claude/v2-cell-features`), `scripts/sync_r_headers.py` (pz2 17).
Scripts total `v2` lines: 283 in 19 files; the 8 files above contain 249.

**sbatch job names (`#SBATCH --job-name`):** `scripts/v2_migrate.sbatch:35` `v2mig`; `scripts/v2_junction_table.sbatch:18` `jxtab` (not branded); `bench/called_v2/run.sbatch:11` `cv2`; `bench/called_v2_truth/run.sbatch:7` `cv2t`, `kang_genotype.sbatch:4` `cv2t-geno`, `verify.sbatch:7` `cv2fix` (all called_v2). Log paths are `/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out` (`%x` = job name, so logs inherit the name); work dirs: `$WORK/state/v2/plan-09fd92b149/plan.tsv` (`bench/scbasecount/coverage.sbatch:13`, `bench/scbasecount/coverage.py:4`; campaign work area `state/v2/`), `$WORK/state/called_v2/` (`docs/called-v2.md:217`). Unrelated: `state/gpu/archive/cycle76_lfc_fp64_v2.sh` (`--job-name=sg_cycle76_lfc_fp64_v2`, GPU cycle iteration 2), `cycle74_parity_v2.sh`: false positives.

---

## 6. Docs

**Files with v2 in the name (tracked) and total v2 line counts inside them:**

| file | v2 lines | inbound refs (file:count, excl bench/compression & papers) |
|---|---|---|
| `docs/format-v2.md` | 42 | 51 lines across 41 files: `CLAUDE.md` 1, `docs/CELL-ARCHITECTURE.md` 1, `docs/quickstart.md` 1 (line 37), `docs/called-v2.md` 1, `docs/format-v2-cell-index.md` 1, `docs/v2-migration.md` 1, `docs/v2-performance.md` 1, 5 headers in `include/singlet/pz2/**` (`pz2.h` 5, `api.h`, `codec2.h`, `container.h`, `rans1.h` 1 each) + the 5 R mirrors, `python/singlet/_loader.py` 1, `bundle.py` 1, 14 `python/singlet/v2/*.py` (`migrate.py` 4 others 1), `r/R/v2.R` 1, `r/src/pz2_io.cpp` 1, `src/bindings/python/_pz2.cpp` 1, `scripts/v2_campaign.py` 1, `scripts/v2_junction_table.py` 1, `scripts/v2_junction_table.sbatch` 1, `bench/pz2/ref_decode.py` 1, `tests/python/test_v2_junctions.py` 1 |
| `docs/format-v2-cell-index.md` | 52 | 11 lines: `docs/format-v2.md` 4, `docs/CELL-ARCHITECTURE.md` 1, `python/singlet/v2/atlas.py` 1, `scripts/v2_cidx_build.py` 1, `bench/cidx/{chunking,formats,synth}.py` 1 each, `docs/format-v2-cell-index-costs.md` 1 |
| `docs/format-v2-cell-index-costs.md` | 3 | 5: `docs/format-v2-cell-index.md` 2, `python/singlet/v2/atlas.py` 2, `tests/python/test_v2_atlas.py` 1 |
| `docs/v2-migration.md` | 23 | 11: `docs/format-v2.md` 2, `docs/called-v2.md` 1, `docs/called-v2-truth.md` 1, `docs/v2-performance.md` 1, `python/singlet/_loader.py:163` 1, `python/singlet/v2/__init__.py` 1, `r/R/v2.R` 1, `scripts/v2_retire.py` 1, `bench/called_v2_truth/{common.py,run.sbatch}` 1 each |
| `docs/v2-performance.md` | 19 | 0 inbound by file name |
| `docs/called-v2.md` | 39 | 14 (see section 10) |
| `docs/called-v2-truth.md` | 13 | 12 (section 10) |
| `docs/called-v2/{tables.md,samples.csv,outliers.jsonl}` | 10 + 10 + 5 | linked from `docs/called-v2.md` |

* `docs/SUMMARY.md` and `docs/book.toml`: **no v2 entry at all** (the book TOC lists only Install, Quickstart, pipeline, Canonical output format, API overview, Notebooks), so none of the v2 docs are in the mdBook. `docs/format-v2*.md` are not in `docs/api/` or `docs-python/` either (`docs-python`: 0 hits).
* `README.md` (root): no v2/pz2 mention. `CONTRIBUTING.md`, `SECURITY.md`: none. `CHANGELOG.md`: 2 lines, both unrelated (`:34` removed `singlet.pz_v2` / `1PZ02` legacy reader; `:105` "IO schema v2" for singlify directory layout). So the changelog has **no** entry about the format yet.
* `CLAUDE.md:5-6,10` (binding rules): `.s2` objects, `docs/format-v2.md`, `data.singlet.bio/v2`, "v2 `cells` table". `docs/CELL-ARCHITECTURE.md` (9 lines: 17,24,51,55,56,67,68,81,111; mirror copy of the website repo's canonical copy, so the website repo must change first per CLAUDE.md).
* Other docs that cite the format (not named v2): `docs/cell_annotation.md` 6 (13,23,190,306,373,383), `docs/cell_features.md` 8 (14,87,121,125,145,146,179 + `v2/cf/` key), `docs/empty-droplets.md` 11 (3,9,40,220,305 are format sense; 64-69,159 are 10x v2; 361 `called_v2`), `docs/library-type-screen.md` 2 (3,45), `docs/quickstart.md:37`, `docs/scbasecount-comparison.md` (13 "frozen v2 study list `plan-09fd92b149`" is the campaign; the rest 10x or called_v2).
* Doc cross-references to a **non-existent** file: `scripts/build_features_fbin.py:5`, `scripts/build_snp_sites_fbin.py:5` cite `docs/V2_IMPLEMENTATION_PLAN.md` ("Pipeline V2", not in the repo, not the format).
* Dangling reference: `python/singlet/v2/cellcall.py:21` and `tests/python/test_v2_cellcall.py:154` mention `include/singlet/pileup/called_v2.h` ("C++ port"); that header does **not** exist in the tree.

---

## 7. Wire / storage strings (code and tests; excludes `bench/compression`, `papers` unless stated)

**Format strings**
* `"singlet-v2"`: 62 lines. Defining constants: `python/singlet/v2/__init__.py:7` (`FORMAT`), `python/singlet/v2/manifest.py:25` (`FORMAT`). Hard-coded literals: `bundle.py:102`, `client.py:559-560,1706,1995`, `verify.py:264`, `legacy.py:8,30` (UA `singlet-v2/1.0`), `packs.py:46` (UA), `r/R/v2.R:323-324,915`, `scripts/v2_campaign.py:763,1060-1061,1321`, `scripts/v2_retire.py:91-92`, `scripts/v2_migrate.sbatch` (branch name), docs, tests (`test_v2_campaign_catalog.py` 2, `_common.py` 2, `test_v2_container.py` 2, `test_v2_migrate.py` 1). File counts: `python/singlet/v2` 8 files, `r/R/v2.R` 3, `scripts/v2_campaign.py` 6, `scripts/v2_junction_table.py` 2, `scripts/v2_migrate.sbatch` 4, `scripts/v2_retire.py` 2, `docs/format-v2.md` 1, `docs/v2-migration.md` 2, `docs/called-v2.md` 1, tests 4 files / 7 lines; `bench/compression` notes: 13 lines in 6 files (historical).
* `"singlet-v2-packs"` (pack index format): `python/singlet/v2/packs.py:960`, **data file** `python/singlet/v2/packs/index.json:2`, `scripts/v2_campaign.py:2150-2151`, `scripts/v2_junction_table.py:620-621`, `tests/python/test_v2_container.py:621`, `test_v2_campaign_common.py:112`.
* `adata.uns["singlet_v2"]`, `metadata()$singlet_v2`, MCP `"format": "v2"` (see 2, 3).
* User-Agent strings: `singlet-v2/1.0 (+https://singlet.bio)` `legacy.py:30`, `packs.py:46`; `singlet-v2-campaign/1.0` `scripts/v2_campaign.py:1321`; asserted in `tests/python/test_v2_container.py:823` (`startswith("singlet-v2")`) and `test_v2_campaign_catalog.py:731`. The server answers 403 to the default Python UA, so a UA rename needs no Worker change unless the Worker allow-lists a prefix (check website repo).

**R2 key prefixes** (all begin with `v2/`; the bucket path is the URL path under `data.singlet.bio`):

| prefix | repo-wide lines/files | defining code (file:line) |
|---|---|---|
| `v2/s/<GSM>/<d16>.s2` | 37 / 20 | `python/singlet/v2/migrate.py:436` (`"v2/s/%s/%s.s2"`); readers via manifest `sm.key` (`client.py:765` `startswith("v2/")`) |
| `v2/m/<GSE>.json` | 65 / 27 | `migrate.py:1199`; `client.py:606` (`key = f"v2/m/{gse}.json"`, also 1711, 1949-1952 parent-name checks `"m"`/`"v2"`); `bundle.py:46`; `atlas.py:1619`; `features/io.py:27`; `mcp/server.py:625`; `scripts/annotate_campaign.py:145`; `r/R/v2.R:286,918,1106-1110`; `scripts/v2_retire.py` 3 |
| `v2/ref/<id>.pack`, `v2/ref/index.json` | 51 / 15 | `python/singlet/v2/packs.py:45` (`REMOTE = "https://data.singlet.bio/v2/ref/"`); `client.py:329,342`; `junctions.py:718,730,750`; `r/R/v2.R:371,377` |
| `v2/ledger/<GSE>.jsonl` | 31 / 10 | `migrate.py:1052` |
| `v2/cidx/...` (`v2/cidx/obj/<d16>.scx`, `.../query`, `.../gather`) | 46 / 8 | `python/singlet/v2/atlas.py:71` (`CIDX = "v2/cidx"`), `:299` (obj key), `:3156,3192` (`/v2/cidx/query`), `:3706` (`/v2/cidx/gather`) |
| `v2/a/<GSM>/<d16>.parquet` (label sidecars) | 8 / 4 | `atlas.py:311-312` |
| `v2/cf/<GSM>/<set>.<d16>.parquet` (feature sidecars) | 7 / 6 | `atlas.py:736-737`; mentioned `scripts/features_run.py:203` |
| `v2/cells/` | 2 / 2 | `docs/format-v2.md`, `scripts/v2_campaign.py` (no code constant) |
| other `v2/` prefixes | none found (`v2/archive`, `v2/retired`, etc. do not appear) | |
| local cache/mirror dirs named `v2` | | `client.py:217-219` (`<cache>/v2`), `client.py:2051` (download mirror root), `migrate.py:997`, `verify.py:252` (`os.path.join(v2_dir, "v2")`), `packs.py:754` (`~/.cache/singlet/v2`), `recipes_pzw.py:114` (`tempfile.gettempdir()/singlet_v2_recipes`), `scripts/v2_campaign.py:738` |

Other URL/base strings: `data.singlet.bio/v2` literal 14 lines in 8 files (`CLAUDE.md` 1, `docs/CELL-ARCHITECTURE.md` 3, `docs/format-v2-cell-index.md` 1, `docs/v2-migration.md` 4, `python/singlet/_loader.py` 2, `python/singlet/fetch.py:8`, `python/singlet/v2/atlas.py:9`, `python/singlet/v2/packs.py:45`). Single base definitions (host only, without `/v2`): `python/singlet/v2/transport.py:30` `DEFAULT_BASE`, `python/singlet/_loader.py:55` `_DATA_BASE_DEFAULT`, `r/R/load.R:28`. There is also a retired `data.singlet.bio/v1` at `python/singlet/fetch.py:57`.
`x-served-by`: `X-Served-By: data-v2` appears only in prose (`docs/v2-migration.md:76`); nothing in code or tests reads the header. The Worker route name `data-v2` is website-repo.

**`.s2` extension:** 73 lines in 31 files (`docs`: `CLAUDE.md` 1, `CELL-ARCHITECTURE.md` 3, `format-v2.md` 4, `format-v2-cell-index.md` 3, `v2-migration.md` 3; python: `_io.py` 1, `_loader.py` 6, `bundle.py` 3, `fetch.py` 1, `v2/__init__ ` 1, `v2/bundle.py` 3, `v2/client.py` 6, `v2/container.py` 3, `v2/context.py` 1, `v2/migrate.py` 2, `v2/transport.py` 2, `v2/verify.py` 4; R: `r/R/load.R` 1, `r/R/v2.R` 1, `r/man/download.Rd` 1, `test-v2.R` 1; scripts: `v2_campaign.py` 2, `v2_publish.py` 1; tests: `test_v2_container.py` 7, `test_v2_campaign.py` 3, `_common.py` 3, `_catalog.py` 2, `_v2_synth.py` 1, `test_v2_client_mcp.py` 1, `test_v2_migrate.py` 1). Suffix test sites in code: `_loader.py:388,672`, `bundle.py` (~1390), `client.py:2002` (`origin: "s2"`), `_io.py:29,40`. The `.s2` name is derived from `Singlet2`; if renamed, every key and manifest `sm.key` changes.

**D1 tables / catalog (referenced only; defined in website repo):** `v2_study` 36 lines/7 files (`scripts/v2_campaign.py` 11, `tests/python/test_v2_campaign_catalog.py` 19, `docs/v2-migration.md` 2, `docs/format-v2.md` 1, `scripts/annotate_campaign.py` 1, `scripts/v2_publish.py` 1, `test_v2_campaign_publish.py` 1); `v2_sample` 43/6 (`v2_campaign.py` 18, `test_v2_campaign_catalog.py` 20, docs 3, `v2_publish.py` 1, `_publish` test 1); `v2_migration` 60/10 (`v2_campaign.py` 19, `test_v2_campaign_catalog.py` 20, `test_v2_campaign_retire.py` 8, `_publish` 3, `v2_retire.py` 2, `v2_migrate.sbatch` 2, `test_v2_campaign.py` 2, `v2_publish.py` 1, docs 3). Migration file names: `032_v2_catalog` and `033_v2_sample_rev_key` (`scripts/v2_campaign.py`, `tests/python/test_v2_campaign_catalog.py` 6 lines), index `idx_v2_sample_gsm` (3). Catalog columns/ledger keys: `v2_manifest_url` (`scripts/annotate_campaign.py:68`), ledger fields `v2_object`, `v2_name`, `v2_kind`, `v2_bytes` (`migrate.py:453,876-897,1018-1021,1195`; 35 lines in 9 files). Endpoint `POST /api/ingest/v2-study` (8 lines: `v2_campaign.py` 4, `v2_migrate.sbatch`, `v2_publish.py`, tests 2).

**Environment variables containing V2** (all in code; define at one place each unless noted):
`SINGLET_V2_PACKS` (`python/singlet/v2/packs.py` 2, `_io.py:230`), `SINGLET_V2_CODEC` (`python/singlet/v2/codec.py` 3), `SINGLET_V2_VERIFY` (`client.py` 2), `SINGLET_V2_RECIPES_CACHE` (`recipes_pzw.py` 1, `scripts/v2_campaign.py` 1), `SINGLET_V2_CACHE` (`packs.py` 1), `SINGLET_V2_FULL` (tests: `test_v2_migrate.py` 3, `test_v2_recipes_corpus.py` 2), `SINGLET_TEST_V2_CORPUS_ROOT` (`test_v2_client_corpus.py` 2), `SINGLET_V2_TEST_ROOT` (`r/.../test-v2.R` 4), `SINGLET_V2_TEST_URL` (`test-v2.R` 3), `SINGLET_V2_EXTRAS_ROOT` (`test-v2-modalities.R` 3). Mentioned only in bench/compression notes: `SINGLET_V2_THREADS`. Non-V2-named but pz2: `PZ2_ISA`, `PZ2_SCALAR_ONLY`, `PZ2_CORPUS`, `PZ2_TEST_MAX_MB` (section 4). `SINGLET_LEGACY_FALLBACK` is not branded.

---

## 8. Tests and fixtures

**`tests/python`** (26 files with v2, 529 lines): `test_v2_campaign_catalog.py` 90, `test_v2_campaign.py` 55, `test_v2_migrate.py` 47, `test_v2_campaign_retire.py` 40, `test_v2_campaign_publish.py` 34, `test_v2_junctions.py` 30, `test_v2_cellcall.py` 28, `test_v2_recipes_corpus.py` 21, `test_v2_client.py` 21, `test_v2_campaign_common.py` 21, `test_v2_container.py` 20, `_v2_synth.py` 20, `test_v2_client_mcp.py` 19, `test_v2_client_corpus.py` 18, `test_v2_recipes.py` 14, `test_v2_modalities.py` 12, `_cidx_fixture.py` 12, `test_v2_atlas.py` 9, `_v2_range_server.py` 1, plus `test_v2_client_codec.py` and `test_v2_client_native.py` (pz2/PZ2_ISA heavy: 33 and 21 `pz2` lines; ~0 literal v2 beyond the file name). Non-format v2 hits in other tests: `test_detect_protocol.py` 6, `test_detect_fastq.py` 4, `test_integration.py` 2, `test_catalog.py` 2, `test_catalog_internal.py` 1, `test_singlepress_comprehensive.py` 1 (10x chemistry), `test_cell_features.py` 1 (imports `singlet.v2`).
Files whose **name** contains v2 (27 tracked): the 19 `test_v2_*` / helper files above + `tests/cpp/test_doublet_detect_v2.cpp` (NOT format) + `tests/data/called_v2_golden/**` (14 files) + `r/tests/testthat/test-v2*.R` (2).
pz2 in tests/python: 95 lines across 15 files.
No fixture data under `tests/data` in a v2 *format*; `tests/data/called_v2_golden/` (README.md, generate.py, `.gitattributes`, 6 case dirs: `real_hgmm1k_v3`, `real_neuron1k_v3`, `real_pbmc1k_v3`, `synth_knee_below_plateau`, `synth_mixed`, `synth_tail_extrapolation`, each with `called.txt case.json counts.mtx expected.tsv`) is the called_v2 golden set (section 10; "v3" in dir names = 10x chemistry). Tests use `monkeypatch`-built synthetic samples (`_v2_synth.py`) rather than checked-in `.s2` fixtures; there is no tracked `.s2`, `.pz2` or `.pack` test fixture other than the 16 packaged packs.
CI selects tests by directory: `.github/workflows/ci.yml:112,118` run `pytest tests/python/ -x -q --ignore=tests/python/test_loader.py`; `pyproject.toml:116-117` `testpaths = ["tests/python"]`. Renaming test files needs no CI edit.
Scripts reading tests by path: `bench/called_v2_truth/verify.sbatch` and `docs/called-v2.md:213` point at `tests/data/called_v2_golden/`.

---

## 9. CI (`.github/workflows/*`, `.github/agents/*`)

No workflow job/step name, path filter, matrix key or artifact path mentions the format's v2. All 14 hits are false positives:
* `.github/workflows/R-CMD-check.yml:47,49,81,90`, `pages.yml:40,42,46`: `r-lib/actions/*@sha # v2` (7 lines).
* `.github/workflows/wheels.yml:57`: `pypa/cibuildwheel@... # v2.21.3`.
* `.github/agents/bio-exec.agent.md:74`, `ops-scout.agent.md:21,60`, `singlet.agent.md:67,163`: `10xv2` chemistry; `singlet.agent.md:195`: `pypa/cibuildwheel@v2`.
Also check: `.github/workflows/pz-hardening.yml` is about the legacy `pz` codec (no `pz2`). `wheels.yml`/`ci.yml` run `tests/python` and build the `singlet._pz2` extension via `setup.py` (so the extension module rename must be mirrored in `setup.py` only). The R CI (`R-CMD-check.yml`) builds `r/` including `r/inst/include` mirrored headers.

---

## 10. `called_v2` (second version of the cell call; a different "v2")

Total: **212 lines in 40 files** (outside `bench/compression`, which has 0). Broader regex including `cell-call-v2` etc.: 282 lines in 42 files when counted by the classifier.

| area | files:count |
|---|---|
| code | `python/singlet/v2/cellcall.py` 16, `python/singlet/v2/migrate.py` 21, `pyproject.toml:60` 1 |
| docs | `docs/called-v2.md` 31, `docs/called-v2-truth.md` 10, `docs/called-v2/samples.csv` 1, `docs/called-v2/tables.md` 1, `docs/empty-droplets.md:361` 1, `docs/format-v2.md:236,238` 2, `docs/scbasecount-comparison.md:329,355` 2 |
| bench | `bench/called_v2/{aggregate.py 2, call_v2.py 12, inspect_outliers.py 2, run.sbatch 5, truth.py 1, validate.py 3}`; `bench/called_v2_truth/{common.py 7, datasets.py 4, kang_genotype.py 3, kang_genotype.sbatch 5, revalidate.py 4, run.sbatch 6, verify.sbatch 9, verify_cellcall.py 2}` |
| tests | `tests/python/test_v2_cellcall.py` 7, `tests/python/test_v2_migrate.py` 8, `tests/data/called_v2_golden/**`: `README.md` 10, `generate.py` 9, `case.json` x6 (3+3+3+4+4+4), `expected.tsv` x6 (1 each) |

Files/dirs with the name: `bench/called_v2/` (6 files), `bench/called_v2_truth/` (8), `docs/called-v2.md`, `docs/called-v2-truth.md`, `docs/called-v2/` (3), `tests/data/called_v2_golden/` (14 files). Note `bench/called_v2/call_v2.py` (prototype). Job names `cv2`, `cv2t`, `cv2t-geno`, `cv2fix`. A local variable `cv2` (= called_v2 array) is used in `cellcall.py:450`, `migrate.py:476`, tests (not OpenCV). Branch name `feat/singlet-v2` is not called_v2.

**Stored-in-data strings (rename here changes `.s2` bytes and therefore digests/rev):**
* Cells-table columns **`called_v2`** (bool bitmap, dtype code 7) and **`called_v2_fdr`** (f32): `python/singlet/v2/cellcall.py:42` (`COLUMNS = ("called_v2","called_v2_fdr")`), written at `migrate.py:476-477` (`base.add("called_v2", cells_mod.DT_BOOL, cv2)`, `base.add("called_v2_fdr", ...)`); documented `docs/called-v2.md:166-167`, `docs/format-v2.md:236-238`.
* Cells member meta key **`called_v2`** (params dict) `migrate.py:478` (`base.meta["called_v2"] = params`), copied into the manifest sample entry `migrate.py:577` and `migrate.py:848-849` (`d["called_v2"] = meta["called_v2"]`) -> so it is also in the served `v2/m/<GSE>.json`.
* Params dict keys: `n_called_v2` (`cellcall.py:385,405,452`, `migrate.py:483`; `docs/called-v2.md:172`) plus `method` values `"singlet-emptydrops"` / `"stored-call"` (asserted `tests/python/test_v2_migrate.py:374`). Method strings are not branded v2.
* Golden vectors: `tests/data/called_v2_golden/*/expected.tsv` header column `called_v2`, `case.json` keys `n_called_v2` and `"called_v2": "exact"`.
* Not stored (rename is code/doc only): the Python option `cell_call_v2` / CLI `--no-cell-call-v2` (`migrate.py:74,98,1276-1287`), function `_add_called_v2` (463), log text, file/dir names, sbatch names, docs.
* Downstream notes: `docs/called-v2.md:187-189` says making `called_v2` the default for reads (gather, cell index, D1 `n_called`) is a separate decision; D1 `n_called` and the cell index (`v2/cidx`) do not currently carry a `called_v2` bit. A rename after objects are published would require re-migration (new digests) or reading both names.

---

## 11. False positives (not the format; do not rename)

Counts are case-insensitive `v2` **lines**; "files" in parentheses.

| category | count | examples |
|---|---|---|
| 10x chemistry / protocol names (`10xv2`, `10x-3p-v2`, `10x-5p-v2`, `3' v2`, `SC3Pv2`, `v1/v2/v3`, `10x-vdj-v2`, `10x-atac-v2`) | about 6,350 lines in about 45 files (6,165 in `bench/libtype_screen/libtype_suspects.tsv` alone; next: `tests/cpp/test_ws2_protocol_hardening.cpp` 15, `docs/called-v2/samples.csv` 9, `docs/called-v2/tables.md` 9, `docs/empty-droplets/samples.csv` 9, `src/pipeline/singlet.cpp` 9 (134-135,1023-1024), `include/singlet/fq/protocol.h` 8+, `tests/cpp/test_protocol_confidence_override.cpp` 7, `python/singlet/preprocessing/_detect.py` 6 (18-135), `tests/python/test_detect_protocol.py` 6, `docs/format-1fq.md` 5 (138-209), `docs/empty-droplets.md` 5, `docs/scbasecount-comparison.md` 5, `state/pilot-success-criteria.md` 4, `.github/agents/*` 5, `python/singlet/mcp/server.py:155,167,315`, `notebooks/protocol_detection.ipynb`, `notebooks/quickstart.ipynb`) | Note: `docs/called-v2/*` and `docs/empty-droplets/*` tables contain 10x v2 *and* v3 labels |
| Other chemistry/format versions inside Singlet: inDrop v2 (`include/singlet/fq/indrop_detect.h` 6, `tests/cpp/test_indrop_detect.cpp` 10), min-sketch file format v2 (`include/singlet/pileup/nonhost/min_sketch.h` 4), doublet-detector v2 (`include/singlet/pileup/doublet_detect.h:31`, `tests/cpp/test_doublet_detect_v2.cpp`), `reference_map.h` header v2, "Pipeline V2" (`pileup_engine.h:4`, `scripts/singlet_orchestrator_v2.sh`, `python/singlet/pipeline/_binary.py:45`, `docs/V2_IMPLEMENTATION_PLAN.md` citations in `scripts/build_*_fbin.py`), legacy `.singlet` "1PZ02 v2" format (`docs/format-singlet.md:178`, `CHANGELOG.md:34,105`, `singlet.pz_v2`), "sparsepress v2" (`python/include/singlepress/singlepress.h:63`), `cellguide-v2` (`celllabels/pipeline.py:23`, `celltype.py:14`), "guide v2", ML refinement "deferred to v2" (`gpu/variants/monopogen.py:16,118`, `_bind_variants.hpp:197`, `gpu/preprocess/deconv_size_factors.h:506`, `gpu/variants/monopogen.h:89`) | about 80 lines |
| Third-party libraries / APIs | about 175 lines | `#include <cublas_v2.h>` (about 30 headers in `include/singlet/gpu/**`), S3 `list_objects_v2` (`scripts/restrict_published_bundles.py:121`), GitHub Actions `@v2` (section 9, 8 lines), htslib BCFv2.1 (`include/singlet/star/htslib/htslib/vcf.h:681,689`, `src/star/htslib/vcf.c`, `src/star/vcf.c`), SIMDe (`include/singlet/star/opal/simde_avx2.h`, `src/star/opal/simde_avx2.h`), CellChat v2 (`include/singlet/gpu/comm/cellchat.h:82,86,216`), `cellrank2`, `cv2` (OpenCV-style names), `shapeit2_integrated_snvindels_v2a` 1000G VCF (`scripts/annotate_ancestry_validate.py:75`, `annotate_build_af.py:10`) |
| Genome / species assembly names | 20+ lines | `ARS-UI_Ramb_v2.0`, `ASM276v2`, `ASM584v2`, `ASM694v2`, `CSAV2.0`, `AnoCar2.0v2` (`species_detect.h`, `species_registry.h`, `species_kmer_db_gen.h` 43815-50068, `python/singlet/_species.py:23`) |
| Gene symbols / gene-set names (in text and binary data) | binary | `NDUFV2`, `VAV2`, `NAV2`, `CAV2`, `ENV2`, `TRICOV2*`, `HALLMARK_MYC_TARGETS_V2`/`myc_targets_v2`, MSigDB `v2026.2`; binary hits: `python/singlet/data/sample_index.parquet`, `python/singlet/features/data/abundance_{human,mouse}.parquet`, `python/singlet/celllabels/data/*.gz`, `python/singlet/features/data/hallmark.json:1`, 5 of the 16 packaged `.pack` files (random bytes/gene names), `tests/cpp/gpu/refs/sparse_eig_medium_ref.npz`, `docs/**/*.png`, `notebooks/*.png` |
| GSM/GEO accession-like tokens | 0 found | No `GSE`/`GSM` accession contains `v2`; the only accession-adjacent hits are prose such as "GSE145370 (v2, ESCC)" in `docs/cell_features.md:87,121`, which refers to the format migration (pilot study) and is a *format* reference, not a false positive |
| `bench/compression/**` (historical result tables, notes, corpora, tools) | 38,401 lines in 290 files (719 tracked files in the dir); 102 `pz2` lines in 16 files | file names `results/m_w4_measure_v2_final.tsv`, `results/track_b_dev2.tsv`; row labels `pbmc_v2_typical` (19,583), `pbmc_v2_shallow` (16,804), `m_v2_a`, `bn_v2`, `h_v2_a/b/c` etc.; the `notes/m_w*.md` files do mention real format strings (`singlet-v2` 13 lines/6 files, `.s2` 19/9, `v2/{s,m,ref,ledger,cidx}/` 19/5, `singlet.v2` 21/8, `format-v2` 6/3, magics 6/3). Summary only: do not rename (historical) |
| Papers / manuscripts | 472 lines in 31 files (`papers/manuscripts/singlepress-format/**`, `singletdb/**`) | `benchmark_gpu_v2.py`, `benchmark_threading_v2.py`/`.py.old`, `benchmark_r_formats_v2.R`, `gpu_benchmark_v2.csv`, `threading_benchmark_v2.csv`, `regev2017human`, `structural_v2`; no `singlet-v2`, `.s2`, `v2/s|m|ref`, `singlet.v2`, `format-v2` hits in papers; fig PDF/PNGs match as binary |
| Notebooks | 54 lines in 19 files (text 19 ipynb/png) | `notebooks/protocol_detection.ipynb` 7 (`10xv2`), `quickstart.ipynb` 8 (`10xv2` table), `corpus_analytics.ipynb` 6, `umap_qc.png` 40 and `umap_clusters.png` 25 (binary), `reprocess_gsm.ipynb:43`; none refer to the format |
| GPU cycle logs / archive | 22 lines in 8 files under `state/` | `state/gpu/archive/cycle76_lfc_fp64_v2.sh` (8), `cycle74_parity_v2.sh`, `cycle75b_hvg_fix2.sh:18`, `state/gpu/cycle-log.md` ("kNN v2", "Mandate-v2", "design doc v2"), `state/gpu/style-rules.md:712` (`make_view_object_v2`), `integration-notes.md:95` (streampress v2), `state/reference-manifest-v1.yaml:170` (`reference-manifest-v1` style versioning; 10x v1/v2/v3 whitelist line), `state/pilot-success-criteria.md` (10xv2) |
| Local variable names / math (`v2`, `inv2`, `prev2`, `sigma2`, `lv2`, `v2g`, `kv2`, `r2_sv2`, AVX registers `v20`) | about 80 lines | `include/singlet/fq/packing.h:50-64` (5), `include/singlet/gpu/enrich/decoupler_ulm.h:210-218`, `gpu/integrate/combat.h` (6), `pileup/donor_demux.h:79-81,1005-1073`, `rans1_avx2.h:75,127,217` and `rans1_avx512.h:63,108,194` (register `v20`, `v16`, `v65535`), `tests/cpp/test_pz2_edge.cpp:478-479` (`kv2`), `src/star/Genome_transformGenome.cpp` (6) |
| "bench" miscellany | about 30 lines | `bench/cycle57b_bench_job.sh:163` ("streampress v2"), `bench/features/pilot.py` (`GSE:v2:GSM` arg token for the migrated format = format-related), `bench/empty_droplets/run.sbatch:28` (`venvs/v2tools`: format-related), `bench/scbasecount/coverage.py:4` (the frozen "v2 study list" = migration campaign) |

---

## Summary table of renaming surface (for planning)

| surface | files | lines | stored/served bytes? |
|---|---|---|---|
| `python/singlet/v2/` package dir + modules | 50 (33 py + 17 data) | 295 text lines in 26 py | no (code); `packs/index.json` has `singlet-v2-packs` |
| imports/uses of `singlet.v2` elsewhere | about 50 files | 235 | no |
| public Python names (`V2Bundle`, `V2Sample`, `V2Context`, `V2Error`, `read_v2`, `is_v2`, `v2_gsms`, `v2_options`, `uns["singlet_v2"]`, MCP `format:"v2"`) | about 12 files | about 150 | MCP responses yes (wire) |
| R: `R/v2.R`, `.v2_*`, `singlet_v2_*` classes, tests, man | 10 | about 440 | no |
| pz2 codec naming (C++ ns/dir/headers/tests/R/bindings/CMake) | 62 files with pz2 in path; 829 lines | | no (naming); magic `PZ2M`/`PZ2CANON1` yes |
| magics `SINGLET2 PZ2M SPK2 CEL2 SCX1 CEL2CANON1 PZ2CANON1 SVP0` | about 25 | about 150 | **yes** |
| scripts/sbatch | 8 v2_* scripts + 5 referencing | 283 | no |
| docs | 7 docs + 3 in `docs/called-v2/` + 12 files citing them | about 460 | no |
| wire strings (`singlet-v2`, `v2/` keys, `.s2`, `v2_*` D1/ledger, env `SINGLET_V2_*`) | about 60 | about 600 | **yes** |
| tests | 27 files + 14 golden files | about 580 | no |
| CI | 0 | 0 | no |
| `called_v2` | 40 files | 212 | **yes** (columns + meta) |
