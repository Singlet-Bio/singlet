# m_w2: container + transcoder + verifier (`python/singlet/v2/`)

Owner: W2. Started 2026-10-22. Status: **DONE for the brief** (migrate + verify work end to end on all corpora; section 4 has the numbers, section 5 what is open).

Everything is pure Python + numpy + `zstandard`. Run with `PYTHONPATH=python` (the package is not pip-installed here) and `python -P`.

## 0. Interface to the recipes (W3) as implemented

`singlet.v2.rcp_adapter.load_recipes()` wraps `singlet.v2.recipes` (W3) plus a tiny built-in set (`builtin_recipes.py`, only
`vocab-json-v1` as a stand-in). The migration/verifier use exactly this surface of the recipes module (all present in W3's `recipes.py`):

```python
RECIPES                                                      # dict recipe_id -> object (informational)
plan(member_name, legacy_bytes, ctx) -> (recipe_id, params, needs) | None
regenerate(recipe_id, params, ctx) -> bytes                  # exact legacy bytes; may raise (NotByteExact) for derived .1pz
regenerate_matrix(recipe_id, params, ctx) -> Matrix          # derived .1pz members: canonical matrix criterion
cells_columns(member_name, legacy_bytes, ctx) -> {name: (dtype, ndarray)}
```

`member_name` = path relative to the sample directory (`exon_counts.1pz`, `star_Solo.out/Gene/raw/matrix.mtx`, `donor/donor_assignments.tsv`) for
sample members, the bundle-level name (`manifest.json`, `study_meta.json`, `feature_vocab.json`) for study files (then `ctx.sample_id == ""`).
`exon_counts.1pz` and `intron_counts.1pz` are never offered to `plan` (always pz2). Every other member, `.1pz` included, is.

Context = W3's protocol (`recipes_ctx.Context`: `matrix`, `member_bytes`, `cells`, `vocab_names`, `genes`, `whitelist`, `features_order`, `vector`,
`meta`, `has`, `sample_id`, `study_id`, `reference_build`, `protocol_name`), implemented by `singlet/v2/context.py`:

* `LegacyContext` (migration, `plan` + stage 1) and `V2Context` (stage 2, verify, readers: built from the finished `.s2` alone), `StudyContext` (study files).
* `ctx.matrix(name)` returns W3's `recipes_ctx.Matrix` (indptr, indices, data, shape, rownames, colnames, user_kv); `ctx.cells()` a dict of numpy arrays
  (`barcode` and string columns as object arrays, `called` bool); `ctx.whitelist(name)` the sorted barcodes of a whitelist pack as a numpy `S<k>` array
  (`3M-february-2018`, `737K-august-2016` (alias `737K-cratac-v1`), `737K-april-2014`, `gex_737K-arc-v1`); `ctx.genes()` / `ctx.vocab_names(kind)` from the sample's vocabulary pack.
* Extensions (same names W3's `DirContext` uses): inside `plan()` only, `ctx.provide_matrix(aux_name, matrix)`, `ctx.provide_blob(aux_name, bytes)`,
  `ctx.provide_vector(name, ndarray)`; read back with `ctx.matrix(aux_name)`, `ctx.blob(aux_name)`, `ctx.vector(name)` in both phases. Convention for the aux
  matrix of a text matrix: `"<member>#csc"`. An aux member is stored (pz2 / kind 3, flags = 0, not a legacy member) **iff** the `needs` of an accepted plan names
  it (`"matrix:<aux>"`, `"blob:<aux>"`, `"vector:<name>"`); otherwise it is dropped. Aux matrices may be nameless (`rownames`/`colnames` None -> names kind `None`).
* `ctx.features_order()` = `recipes_ctx.load_features_order(ctx.reference_build)` (`recipes_data/features_order_<build>.u32.zst`); a pack kind for it is not built (open item).
* Sample facts a recipe may ask for (`reference_build`, `protocol_name`) are recorded in the `meta["ctx"]` of the cells member and of every derived member, so a
  reader that has only the `.s2` gets the same context as the migration.

**Verify-or-store protocol (migration, per sample)**

1. decode every `.1pz` with `singlet._pz.read_1pz` (a non-canonical or undecodable matrix is stored verbatim, kind 3);
2. cells table: barcodes = columns of `exon_counts.1pz` (else `intron_counts.1pz`), `called` from `cell_calls.tsv` (rule of `SingletBundle`; no file -> all called),
   `n_umi_*` / `n_feat_*` from the matrices; whitelist pack chosen as the smallest one containing every barcode;
3. candidate optional columns from `cells_columns` of every member; `plan` of every member (stage 1: regenerate against the legacy context, compare bytes, or for
   derived `.1pz` the decoded canonical matrix (`regenerate_matrix`) and, separately, the container bytes -> `bytes_exact`);
4. assemble the `.s2`; only the cells columns named in the `needs` (`"cells:<col>"`) of accepted plans are kept, and only the aux members they name;
5. **stage 2**: open the finished bytes as a `V2Sample`, rebuild a `V2Context` from them alone and (a) decode every pz2 member and compare with the legacy matrix
   (shape, indptr, indices, data, names, key/values, digest), (b) regenerate every derived member through the recipes and compare, (c) check stored members;
   a member that fails is stored (kind 3) and the object rebuilt (repeat until stable; `bytes_exact` is dropped, not the derivation, when only the byte claim fails).
   The ledger says why (`note`).

Derived `.1pz` members (`splice_psi.1pz`, stub files): kind 4 with `meta.bytes_exact` true iff a writer reproduces the container bytes on this machine; the table's
`digest` is the canonical matrix digest of the legacy matrix, `out_len`/`crc32` are the legacy bytes' when `bytes_exact`, else 0.

## 1. Modules

| module | what |
|---|---|
| `util.py` | digest16, CRC, canonical JSON, thread-safe zstd (`zstandard` objects are not thread safe), vectorised LEB128 |
| `container.py` | `S2Writer`, `S2Reader` (bytes / path / `fetch(offset, length)`; header + table in one request, one request per range), kinds, flags |
| `matrix.py` | `Csc`, canonical digest (4.6, vectorised, blockwise) |
| `codec.py` | `MatrixCodec` interface, `PyCodec2` (pure-Python codec 2, W1's byte layout), `NativeCodec` (adapter for `singlet._pz2`: codec 1 + 2), `get_codec`, `decode_any`, `parse_header` |
| `cells.py` | cells table (5.3): writer, `CellsReader` (lazy columns), barcode column modes, `base_table`, `parse_cell_calls` |
| `packs.py`, `packs_build.py` | SPK2 wrapper, `VocabPack` (vectorised SVP0 decode, lazy `name_at`/`names_at`, `row_gene`, `vocab_json`), `Whitelist`, `PackStore`, build script |
| `packs/` | the 16 pack files + `index.json` (section 2) |
| `manifest.py` | study manifest, `rev`, study-file entries (derived / inline JSON / zstd+base64) |
| `legacy.py` | `LegacyBundle`: ZIP/ZIP64 central directory from a local path, HTTPS (range requests, `User-Agent: singlet-v2/1.0`), bytes, or a directory; CRC-checked reads |
| `sample.py` | `V2Sample`: open an object, cells, matrices with names resolved, column ranges via ranged reads, stored blobs |
| `context.py`, `rcp_adapter.py`, `builtin_recipes.py` | recipe contexts and adapter (section 0) |
| `migrate.py` | `python -m singlet.v2.migrate --bundle PATH_OR_URL_OR_DIR --out DIR [--gse ID] [--samples G,G] [--recipes MOD\|none] [--chunk-cols N] [--created-at T]` |
| `verify.py` | `python -m singlet.v2.verify --bundle ... --v2 DIR [--gse ID] [--samples ...] [--recipes ...] [--quick]`; exit 1 on any mismatch |

Output tree of `migrate` (key space of the bucket): `v2/s/<GSM>/<d16hex>.s2`, `v2/m/<GSE>.json`, `v2/ledger/<GSE>.jsonl`, `v2/ref/<packid>.pack` + `v2/ref/index.json` (only the
packs the objects use). Objects are deterministic (same input + `--created-at` -> same bytes, same `rev`).

`MatrixCodec` (pluggable): `encode(csc, EncodeOptions(chunk_cols, row_names, col_names, level, threads, digest)) -> bytes`, `decode(bytes, mode, cols=None) -> Csc`
(names living in a pack / the cells table / nowhere come back as `Csc.row_ref` / `col_ref`, resolved by `V2Sample`). Names specs: `"inline"`, `"none"`, `("pack", id8)`,
`("subset", id8, indices)`, `"cells"` (columns only) = W1's `NamesKind` Inline/None/Pack/PackSubset/Cells.

## 2. Packs (`python/singlet/v2/packs/`)

SPK2 wrapper per spec section 6, one pack per (build, member): exon vocabulary pack (kind 1, header `coder` `SVP0` = the w4_vocab structured body, **byte-identical** to the
measured `results/w4_vocab_packs/*.svp`), intron pack (kind 1, `SVPD`, 16-byte body: the digest16 of its exon pack; intron names are the gaps between exons), barnyard = kind 3
composites (components + species tags, 223 / 225 B), whitelists (kind 2: sorted 2-bit codes, LEB128 gaps, zstd-19). `feature_vocab.json` is `exon_pack.vocab_json()`.
Pack id = digest16 of the file; matrices store 8 bytes. `index.json` lists id, kind, build/protocol, member, bytes, sha256, `ship`.

| pack | id (digest16 hex) | bytes | ships in wheel |
|---|---|---|---|
| GRCh38-2024-A exon / intron | a34ebca652f38b28f0ab5c0ec2a13249 / fd3fcc82842baa3a1d4c8bc6a8f99b90 | 1,144,318 / 147 | yes |
| GRCm39-2024-A exon / intron | cef1acaf465ebfc4ff7404f308966e92 / 15854e80854ce93702dc4c716e6a35c5 | 1,529,459 / 147 | yes |
| GRCz11 exon / intron | b6abb72f333412054fd8d2afa76f35b4 / 09243cc59ea1b0223dcbf071e1df34e3 | 967,845 / 140 | no (fetch `v2/ref/<id>.pack`) |
| Mmul_10 exon / intron | ce5a05fc3282d0e3c39e0c7109619eb1 / 99a51e8db8bc54baede2092c8b828737 | 879,073 / 141 | no |
| mRatBN7.2 exon / intron | 94e3bff7460f96e92311ff009efae582 / 0bcd2ceb31fc6cad695901e449f667cc | 875,198 / 143 | no |
| barnyard exon / intron (composite) | b708a9dbd566355a876377babed1a4a1 / 32bbdc195ae2486a1257e38d3c4c161e | 223 / 225 | no |
| whitelist 3M-february-2018 / 737K-august-2016 (+ alias 737K-cratac-v1) | b3961cb6986f392743ea332b879d1d5a / c46618191a313da3983d934abe51e751 | 6,619 / 3,436 | yes |
| whitelist 737K-april-2014 / gex_737K-arc-v1 | 88579b0570adecbc3f0e8602d65d243f / 5b24205cb61a91db46ebbc980e8ee6bc | 4,453 / 8,684 | yes |

Total 5,420,251 B for all 16 files; shipped subset (human, mouse, 4 whitelists) 2,697,811 B. **Packaging (not my paths):** `pyproject.toml` `package-data` needs
`"v2/packs/*.pack", "v2/packs/index.json"` and an exclusion of the 8 non-shipping pack files listed above (or move them out of the wheel directory); `MANIFEST.in` likewise.
Verified in `packs_build.py`: expansion == source names for the representative sample of every build, `feature_vocab.json` regenerated byte for byte, lazy lookups == full expansion,
and **36 of 36** (study, matrix) name lists of the fetched vocab corpus equal a pack. Rebuild: `PYTHONPATH=python python -P -m singlet.v2.packs_build` (about 3 minutes).
Reading cost (pure numpy): decode+expand a human exon pack ~1 s; `name_at` after the arrays exist is microseconds (Python).

## 3. Format facts decided or found while implementing (please fold into docs/format-v2.md)

* **pz2 header is 256 bytes** (W1's `container.h`; the 160 in the draft cannot hold its own fields: they sum to 180). `PyCodec2` writes/reads W1's byte layout exactly
  (header with CRC-32, section order chunk_index, tables, rows, chunks, cols, kv, extra, names parts, codec-2 chunk streams `STREAM(counts) STREAM(gaps) STREAM(values)`,
  gaps = `row - previous - 1`), so members are interchangeable with the C++ codec; only zstd frames can differ between zstd builds. Default `chunk_cols` = W1's policy
  (~96k nnz per chunk, power of two in [128, 4096]); `value_type` = the legacy `vt_code` (1/2/3), recorded in the member meta as `legacy_vt_code`.
* Member table meta (JSON) of a pz2 member: `{m, n, nnz, legacy_vt_code, chunk_cols, names: {rows: {pack: <id32hex>, member: exon|intron|genes} | "inline" | "none",
  cols: "cells" | "inline" | "none" | {pack}}, legacy_size, legacy_crc32}`. Which vocabulary member a pack id refers to is NOT in the 8-byte header id: it is in this meta
  (exon and intron are separate packs, but an aux gene-level matrix uses `member: genes` of the exon pack).
* Cells table barcode column (dtype 9) is self describing: mode 1 whitelist indices (`pack id prefix 8 B, bits u8, LSB-first bit-packed`), mode 2 2-bit codes, mode 0 zstd strings.
  Digest = SHA-256 of `"CEL2CANON1"` then per column in name order `name\0 dtype logical-values` (bool as 0/1 bytes, strings NUL-terminated); implemented literally from 5.3.
* Cells columns computed at write time are u32 as specified; zstd (or delta+zstd) per column chooses the smaller body. Cost on GSE145370: 4.6 B/cell (39,217 cells: 179 KB),
  about half of it the barcode (22-bit whitelist index = 2.75 B). This is the price of search without decoding matrices (D10).
* `kind 4` derived members store no bytes; `meta = {recipe, params, needs, pipeline, ctx, bytes_exact?, legacy_vt_code?}`.
* Study files (manifest.json, study_meta.json, feature_vocab.json) live in the study manifest `study_files` (`manifest.py`): derived by recipe, inline JSON (when `json.dumps` with a recorded
  variant reproduces the bytes) or zstd+base64. `manifest.json` of a real bundle carries the SHA-256 of every member including the `.1pz` files, so it cannot be regenerated from v2; it is stored (inline JSON, 20 KB).
* `rev` = first 16 hex of SHA-256 over `SGMREV1\n`, the sorted sample keys, and `name\0sha256(file)` of the sorted study files (no timestamps).

## 4. Results

All numbers: `python -P -m singlet.v2.migrate` (default codec = compiled `singlet._pz2`, automatic codec 1/2 choice, W3's recipes) then the independent
`python -P -m singlet.v2.verify` on the output, one process, sequential, below-normal priority, on the shared laptop (4C/8T, loaded by other agents; **these are single wall-clock
runs, not lock-protected benchmarks**: read the seconds as upper bounds). Driver: scratch `run_corpora.py`; the per-sample table is below.

**Verification: 58 samples in 45 studies, 6,351 independent checks passed, 0 failed, 0 exceptions** (GSE145370: 1,533 checks over 14 samples; 40 loose corpus_w4_corpus samples: 4,050;
4 gen-B STARsolo samples: 768), plus 87 pytest tests (`test_v2_container.py` 68, `test_v2_migrate.py` 19).

| set | samples | legacy (zip bytes) | v2 sample objects | ratio | transcode (sum, s) |
|---|---|---|---|---|---|
| GSE145370 full bundle (gen A, 10x v2, GRCh38) | 14 | 186,744,105 | 31,828,624 | **5.87x** | 354 |
| corpus_w4_corpus (all builds, all protocols, 10 gen-A style members) | 40 | 978,556,341 | 382,969,540 | 2.56x | 1,026 |
| corpus_w4_mtx gen-B STARsolo samples | 4 | 612,543,025 | 28,177,843 | **21.7x** | 215 |
| all | 58 | 1,777,843,471 | 442,976,007 | 4.01x | 1,595 |

The whole GSE145370 study is 187,066,350 B as a legacy bundle; as v2 it is 31.83 MB of sample objects + a 90.6 KB manifest + the shared packs (1.15 MB human exon pack, fetched once for all studies of the build).
Same study with the pure-Python codec 2 instead of codec 1: 37.18 MB (5.02x). Low ratios in the loose corpus (1.7 to 2.5x) are deep or large samples whose legacy bytes are almost all count payload
(h_v3_c: exon 11.75 -> 6.21 MB, intron 9.92 -> 5.32, sj 4.93 -> 3.12); the packs remove the names and derivations remove psi / cell_calls, the rest is the codec. Members by kind over all 58 samples:
cells 57, pz2 matrices 187 (185 codec 1, 2 codec 2 = empty stubs), derived 170, stored 207; recipes used: cellcalls-v1 57, psi-v1-trunc 27, psi-v1-rint 22, stub-v1 15, tsvcols-v1 12, mtx-filter-v1 8, features-v1 8,
psi-v1-stub 5, ambient-v1 4, geneexpr-v1 4, whitelist-sorted-v1 4, mtx-text-v1 4. Stored: summary.json, provenance.json, pileup_stats.json (58/58/54), the small STARsolo logs and `star_SJ.out.tab` (4), and one
`cell_calls.tsv` (x_stub: 33k rows but a 0x0 matrix, nothing to derive it from). No recipe was ever rejected by the stage-2 check (`stage2_rounds` = 1 in every ledger).
v2 bytes by part over the 58 samples (443.0 MB): pz2 matrices 407.4 MB (incl. 8.9 MB of auxiliary raw-mtx matrices), cells tables 23.5 MB (5.3%), stored members 2.9 MB, derived 0.

Where a gen-B sample goes (GSE171161.GSM5220137, mouse 10x v2, 24 members): 72.2 MB -> 1.24 MB (58x): exon 5.73 -> 0.85 MB, intron 4.38 -> 0.23 MB, cells 0.04 MB, 12 members derived (ambient_profile 22.4 MB, raw/filtered
matrix.mtx 8.7 + 6.0 MB, raw barcodes 12.5 MB, features, gene_expression, cell_qc, ...), the raw matrix kept as an auxiliary pz2 matrix (1.0 MB, standalone; conditional coding on the exon matrix is v2.1).

Transcode time per sample: 10x gen-A sample with ~40k columns and 130k junctions 17 to 48 s (profile of one: 9.5 s native encode, 11 s psi planning in W3's recipes, 9 s stage 2 (psi regenerated again by the
writers), 7 s one-time writer calibration per process, 3 s one-time pack matching); the 323 MB STARsolo sample 110 s; typical gen-B sample 18 to 60 s. Read path: whole-matrix decode of a
39k x 310k exon member ~0.4 s (codec 2, Python); native codec 1 is several times faster.

Interop with W1's C++ codec (`include/singlet/pz2`, built with the Rtools GCC): 7 matrices (u8/u16/u32/f32, inline names, key/values, chunk_cols 16..4096, empty and 0x0): members written by `PyCodec2`
decode in C++ to identical indptr/indices/data/names, and the C++-recomputed canonical digest equals the Python digest; members written by the C++ `encode_matrix(codec=Generic)` decode in Python identically.
(Bytes differ by 0.2 to 3% only because the zstd builds differ; the two empty cases are byte-identical.) `test_native_codec_roundtrip_and_cross_decoding` keeps the Python-side half of this in pytest.

### Per-sample table (legacy zip bytes vs v2 object bytes)

| item | sample | legacy zip B | v2 B | ratio | members cells/pz2/derived/stored | transcode s |
|---|---|---|---|---|---|---|
| GSE145370 | GSM4317423 | 15,417,492 | 3,184,595 | 4.84 | 1/4/3/3 | 48.0 |
| GSE145370 | GSM4317424 | 12,329,963 | 1,747,412 | 7.06 | 1/4/3/3 | 23.3 |
| GSE145370 | GSM4317425 | 12,979,655 | 2,053,970 | 6.32 | 1/4/3/3 | 27.6 |
| GSE145370 | GSM4317426 | 17,421,814 | 4,277,523 | 4.07 | 1/4/3/3 | 39.1 |
| GSE145370 | GSM4317427 | 14,494,917 | 2,745,299 | 5.28 | 1/4/3/3 | 25.7 |
| GSE145370 | GSM4317428 | 11,290,627 | 1,466,644 | 7.70 | 1/4/3/3 | 16.5 |
| GSE145370 | GSM4317429 | 12,916,401 | 2,085,778 | 6.19 | 1/4/3/3 | 20.9 |
| GSE145370 | GSM4317430 | 11,930,057 | 1,611,605 | 7.40 | 1/4/3/3 | 19.2 |
| GSE145370 | GSM4317431 | 11,416,915 | 1,377,939 | 8.29 | 1/4/3/3 | 17.0 |
| GSE145370 | GSM4317432 | 14,903,339 | 3,041,237 | 4.90 | 1/4/3/3 | 25.3 |
| GSE145370 | GSM4317433 | 14,564,351 | 2,812,881 | 5.18 | 1/4/3/3 | 27.5 |
| GSE145370 | GSM4317434 | 12,549,319 | 1,866,068 | 6.73 | 1/4/3/3 | 21.1 |
| GSE145370 | GSM4317435 | 12,587,523 | 1,928,595 | 6.53 | 1/4/3/3 | 21.7 |
| GSE145370 | GSM4317436 | 11,941,732 | 1,629,078 | 7.33 | 1/4/3/3 | 21.3 |
| L_a | GSM9277685 | 14,332,142 | 1,393,404 | 10.29 | 1/3/2/3 | 15.7 |
| L_b | GSM8225104 | 37,939,133 | 22,780,187 | 1.67 | 1/3/2/3 | 38.3 |
| bn_dropseq | GSM8842816 | 24,011,814 | 4,312,556 | 5.57 | 1/3/2/3 | 20.9 |
| bn_v2 | GSM4729683 | 35,969,152 | 11,628,002 | 3.09 | 1/3/2/3 | 46.6 |
| bn_v3_a | GSM4601703 | 37,175,593 | 13,705,313 | 2.71 | 1/3/2/3 | 32.5 |
| bn_v3_b | GSM7024956 | 37,815,515 | 12,985,441 | 2.91 | 1/3/2/3 | 44.9 |
| h_arc | GSM9092649 | 18,627,804 | 6,703,263 | 2.78 | 1/3/2/3 | 16.2 |
| h_bd | GSM7029373 | 25,261,779 | 9,848,801 | 2.56 | 1/3/2/3 | 44.5 |
| h_celseq2 | GSM4875776 | 12,745,785 | 3,237,661 | 3.94 | 1/3/2/3 | 19.4 |
| h_dropseq | GSM5911106 | 32,148,092 | 10,278,065 | 3.13 | 1/3/2/3 | 62.0 |
| h_indrop | GSM8249687 | 15,918,531 | 4,305,177 | 3.70 | 1/3/2/3 | 12.9 |
| h_mars | GSM9678297 | 17,508,816 | 8,753,498 | 2.00 | 1/3/2/3 | 9.3 |
| h_microwell | GSM4008717 | 45,099,148 | 21,202,336 | 2.13 | 1/3/2/3 | 68.6 |
| h_quartz | GSM5082427 | 50,470,658 | 18,989,472 | 2.66 | 1/3/2/3 | 86.1 |
| h_split | GSM5618241 | 14,395,374 | 2,880,352 | 5.00 | 1/3/2/3 | 22.2 |
| h_v2_a | GSM5101015 | 17,461,575 | 5,454,367 | 3.20 | 1/3/2/3 | 26.1 |
| h_v2_b | GSM4037324 | 24,081,748 | 9,008,732 | 2.67 | 1/3/2/3 | 31.0 |
| h_v2_c | GSM7892469 | 19,391,921 | 8,883,614 | 2.18 | 1/3/2/3 | 14.8 |
| h_v3_a | GSM5467179 | 23,734,341 | 9,442,333 | 2.51 | 1/3/2/3 | 46.6 |
| h_v3_b | GSM8101887 | 22,882,629 | 11,583,517 | 1.98 | 1/3/2/3 | 19.9 |
| h_v3_c | GSM7818508 | 27,083,154 | 15,076,826 | 1.80 | 1/3/2/3 | 29.7 |
| h_visium | GSM7412940 | 11,972,019 | 1,911,726 | 6.26 | 1/3/2/3 | 9.1 |
| m_arc | GSM9991277 | 22,055,048 | 7,144,837 | 3.09 | 1/3/2/3 | 22.5 |
| m_celseq2 | GSM6283570 | 11,278,533 | 1,259,637 | 8.95 | 1/3/2/3 | 3.4 |
| m_dropseq | GSM7945758 | 38,539,719 | 18,589,371 | 2.07 | 1/3/2/3 | 29.5 |
| m_mars | GSM6922987 | 10,297,390 | 568,315 | 18.12 | 1/3/2/3 | 2.7 |
| m_v2_a | GSM7143129 | 21,589,676 | 6,618,488 | 3.26 | 1/3/2/3 | 20.3 |
| m_v2_b | GSM4735384 | 27,962,264 | 11,530,102 | 2.43 | 1/3/2/3 | 25.1 |
| m_v3_a | GSM7745727 | 24,247,341 | 8,922,294 | 2.72 | 1/3/2/3 | 23.6 |
| m_v3_b | GSM6956535 | 34,646,726 | 16,466,038 | 2.10 | 1/3/2/3 | 21.8 |
| m_v3_c | GSM6588436 | 32,123,666 | 14,715,258 | 2.18 | 1/3/2/3 | 19.2 |
| m_visium | GSM6742901 | 16,344,387 | 4,448,383 | 3.67 | 1/3/2/3 | 12.6 |
| mq_v2 | GSM7842435 | 25,610,180 | 11,774,800 | 2.17 | 1/3/2/3 | 29.7 |
| mq_v3 | GSM8597920 | 33,118,037 | 16,733,519 | 1.98 | 1/3/2/3 | 27.2 |
| r_v2 | GSM6685520 | 21,041,984 | 9,905,937 | 2.12 | 1/3/2/3 | 28.2 |
| r_v3 | GSM5026247 | 13,143,358 | 3,756,770 | 3.50 | 1/3/2/3 | 5.2 |
| x_stub | GSM6610876 | 1,658,508 | 165,146 | 10.04 | 0/2/2/4 | 0.9 |
| z_v2 | GSM3032173 | 26,107,398 | 12,009,105 | 2.17 | 1/3/2/3 | 12.6 |
| z_v3_a | GSM8215882 | 22,407,358 | 9,187,346 | 2.44 | 1/3/2/3 | 11.6 |
| z_v3_b | GSM8691391 | 30,358,045 | 14,809,551 | 2.05 | 1/3/2/3 | 12.5 |
| GSE171161.GSM5220137 | GSM5220137 | 72,220,998 | 1,243,934 | 58.06 | 1/3/12/11 | 19.6 |
| GSE263963.GSM8208009 | GSM8208009 | 338,201,811 | 19,226,678 | 17.59 | 1/3/12/11 | 110.0 |
| GSE309702.GSM9277757 | GSM9277757 | 119,426,755 | 1,823,691 | 65.49 | 1/3/12/11 | 28.3 |
| GSE311511.GSM9326201 | GSM9326201 | 82,693,461 | 5,883,540 | 14.06 | 1/3/12/11 | 57.2 |

## 5. Not done / how to resume

* Decisions folded in after the orchestrator's notes: loose-directory names `star_Solo.out__`, `donor__`, `mt__`, `nonhost__` map to `/`; `ctx.whitelist()` returns a numpy `S<k>` array (STARsolo sample 141 s -> 18 s);
  D15 (the pass criterion for every `.1pz` member is equality of the decoded canonical matrix; container bytes informational): `verify.py` reports it in `Report.container` (equal / differs / unavailable / not_applicable) and never
  fails on it; `ctx.features_order()` loads `recipes_data/features_order_<build>.u32.zst` through `recipes_ctx.load_features_order(ctx.reference_build)`, and the build comes from the object itself (`meta["ctx"]`), so the
  `.s2` reader path reaches it with no study manifest.
* `NativeCodec` is the default; `--codec py|native|rans|generic` or `SINGLET_V2_CODEC`. If the native encode refuses a matrix or the stage-2 check fails, the migration re-encodes that member with `PyCodec2`; only if that fails too is the member stored.
  `V2Sample.matrix(name, cols=...)` uses ranged reads for codec-2 members only; for codec-1 members it fetches the whole member and decodes the range (W4's client uses the binding's `Member` + `plan()` for true ranged reads).
* The raw STARsolo matrix is stored as a standalone pz2 aux matrix (D7); conditional coding on the exon matrix is v2.1.
* The per-build features permutation is a W3 resource file, not a pack. A pack kind would make `features-v1` independent of the `recipes_data` directory.
* No multi-process `--workers` in `migrate` (the campaign parallelises over studies); memory is about 3x the largest legacy member (the 323 MB STARsolo sample peaks near 2 GB; not measured precisely).
* Junction names of `sj_counts` rows are stored inline per sample (130k names, 0.4 to 0.7 MB zstd): a per-build junction pack is the next lever (E3). `splice_psi` rows are free (derived).
* W3 observations: `plan_splice_psi` recomputes the junction events 5 times per sample (9 s) and the writer calibration costs 7 s per process; both could be cached.
* Not tested: HTTPS beyond the live check (directory + member of the live GSE145370 bundle read identically to the local copy, 1 request for the directory, 1 per member); members > 4 GB (ZIP64 fields are parsed and unit tested with `force_zip64`).
* pyproject / MANIFEST (not my paths): `package-data` must list `v2/packs/*.pack` and `v2/packs/index.json` and exclude the 8 non-shipping pack files of section 2 (or the wheel grows by 2.7 MB).
* Resume: `pytest tests/python/test_v2_container.py tests/python/test_v2_migrate.py` (`SINGLET_V2_FULL=1` for all 14 + 40 + gen-B samples, ~1 h); migrate a study:
  `PYTHONPATH=python python -P -m singlet.v2.migrate --bundle <path-or-url> --out <dir>` then `python -P -m singlet.v2.verify --bundle <same> --v2 <dir>`.
