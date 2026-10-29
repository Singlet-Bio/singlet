# w3_total: one measured codec for the combined container, payload and empties design

Status: complete for the items below (2026-10-22); unfinished items are listed in section 9. Owner: the `w3_total` agent.
Files: `kernels/w3_total.h` (codec), `harness/w3_total_bench.cpp` (driver, `bin/w3_total_bench.exe`), `harness/w3_total_calls.cpp`
(called-set stream), `harness/w3_total_second.cpp` (second pass), `harness/w3_total_cnt.cpp` (row-count coder experiment),
`tools/w3_total_extract.py`, `tools/w3_total_bundle.py`, `tools/w3_total_tables.py`, `corpus_w3_total/`, `results/w3_total_*`.
This file is assembled by `tools/w3_total_notes_template.md` + the generated tables; numbers are copied from `results/`.

## 0. Findings in one page

1. **One codec, bit-exact, measured.** `w3_total_v1` (Track A container removal + `b_rans16` payload) takes the 14 corpus
   matrices from **114,090,620 B to 46,438,410 B (0.407, 2.46x; 7.80 bits/nnz)**. Track A alone was 53,534,153 B (0.469).
   `matrix_ok = 1` (decode(encode(M)) == M on indptr, indices, data, row names, column names, key-values) on all 14 files for
   all 8 codecs (3 yardsticks + 5 w3_total variants), plus 28 more matrices in the bundle run. The report's projection ("114 MB -> about 46 MB") **held**.
2. **The GSE145370 bundle did not reach the projected 37-41 MB, because only the exon and intron matrices were recoded.**
   Measured by building the new archive (`corpus_w3_total/GSE145370.w3.zip`): **187,066,350 B -> 48,164,455 B (0.2575, 3.88x)**.
   exon+intron go 125.77 MB -> 11.75 MB (10.7x); splice_psi (24.6 MB) and cell_calls.tsv are replaced by derivation and a 3 KB
   called-set stream; the 36.07 MB that is left (74.9%) is sj_counts + vdj + mt, which were **not recoded** (E5 was not run).
   With Track A's container treatment applied to those two (a projection from Track A's accounting, not measured) the bundle
   would be about 28.2 MB.
3. **Decode is 2-6x faster than today's reader, including the name vocabulary.** On the five exon files measured at a calibrated
   core clock of 2.6-2.9 GHz: today's reader 8-21 Mnnz/s, `w3_total_v1` full canonical decode 17.5-72 Mnnz/s (2.2-6.3x faster; 30 Mnnz/s on
   pbmc_v2_typical). The payload alone decodes at 92-159 Mnnz/s in canonical order and **195-308 Mnnz/s in native rank order**
   (1.8-2.6x faster; the sort back to row-id order is the whole difference).
4. **Frequency-class row order** (4 classes per octave, genomic order inside a class) is 0.20% smaller than exact rank order
   (46,344,217 vs 46,438,410 B): payload +0.09%, row-count stream -13%. Decode speed is unchanged because the
   canonicalisation still runs; the class order only pays if a consumer takes the native order.
5. **Empties through Idea 2's ambient multiset coder (`w3_total_v2`) compose cleanly but are not worth it.** The ambient coder saves
   2.9% of the empties bytes against `b_rans16` on the same droplets, but a separate empties member itself costs +0.54%
   (second row counts, tables, bitmap), so the whole file is **0.35% smaller than v1 (46,274,869 B; 0.45% with class order),
   and v1 wins on 3 of 14 files**; decode time for the 14 files rises from 1.76 s to 13.25 s (2.7-8.4 Mnnz/s on the dense ones). Recommendation: do not ship the ambient coder.
6. **A second pass finds nothing**: zstd-19 and xz -9e return 0.9991-1.0002 on every `w3_total_v1` stream.
7. **Where the v1 bytes are** (section 6): chunk streams (the rANS symbols) 91.3%, barcodes 5.0%, row counts 2.1%, column-nnz
   stream 1.2%, coder tables 0.3%, row-name id and headers 0.0%. The vocabulary (3.57 MB human, 4.47 MB mouse) and whitelists
   (6.5 KB, 3.3 KB) are model files, not counted; charging all 8.05 MB to the 14 files gives 54.49 MB (0.478).

## 1. What was built

`kernels/w3_total.h` owns only glue and the container; every piece comes from a read-only header.

| piece | where it comes from | used for |
|---|---|---|
| row names -> 8-byte vocabulary id; barcodes -> whitelist index (or the cheaper of 2-bit / sorted / inline); user kv | `ta::enc_names_model` / `ta::dec_names_model` (Track A) | the `names` section |
| rank space, frequency permutation, cfg search, rANS payload | `tb::perm_from_counts`, `to_rank_space`, `estimate_payload`, `encode_payload`, `decode_open`, `rans_payload_decode` (Track B, the exact functions `b_rans16` calls) | the matrix payload |
| adaptive Elias-gamma range code, order-statistic multiset coder, `Engine::code_empties` | `i2::NumCoder`, `i2::RcEnc/RcDec`, `i2::Engine` (Idea 2, `idea2_static` parameters) | row counts (one candidate), empties member, split bitmap |

Stream (`W3T1`): `magic, flags(order), m, vt` | names section | member = `n, nnz, chunk_cols, order, row-count section, payload_len, payload`.
The row-count section holds, per row, the nnz (or its frequency class), coded by the smallest of seven Track A style coders
(LEB128 plain/delta x zstd-19/xz/brotli) and Idea 2's adaptive Elias-gamma coder; the adaptive coder won on all 28 matrices
(`results/w3_total_cnt_experiment.tsv`; richer contexts gain at most 0.5% of that stream, so they were not adopted). The
permutation is re-derived from it exactly as in `b_rans16`. There is no ptr or colsums block: the column nnz stream inside the
payload gives the column pointers.
`W3T2` (v2): same header and names section (all columns), then `cells member` (called columns, b_rans16, its own row counts),
`empties member`, and a split bitmap coded by merge context (the decoder already holds both column lists, so each flag is
coded against the depth difference between the next unplaced cell and the next unplaced empty, in quarter octaves).
Empties member, coder 0: `n_e, nnz_e, len`, then one range-coded stream = adaptive Elias-gamma `k` (UMI total) per droplet followed by every
droplet as an exact multiset of `k` draws under the ambient profile (cells' counts plus the running sum of decoded empties);
coder 1 (`w3_total_v2r`): a second b_rans16 member.
The 64 zero bytes of read padding that `tb::encode_payload` appends are not part of the stream; the decoder copies the stream into
a padded working buffer (the copy is inside the timed decode, 0.5-2 ms). Called cells for v2 are read at encode time from
`cell_calls.tsv`; the split is transmitted in the bitmap.
Validation of the composition: the payload (column-nnz + tables + chunk streams) of `w3_total_v1` on brain_sn_v3_deep exon is
6,394,278 B = 8.06 bits/nnz, identical to Track B's `b_rans16` payload figure (8.06); the empties of the same file cost 9.82 bits/nnz
including `k`, identical to Idea 2's stage-1 figure (9.819).

Codecs registered: `w3_total_v1` (exact rank order), `w3_total_v1c` (frequency class), `w3_total_v2` (ambient empties),
`w3_total_v2r` (empties via b_rans16), `w3_total_v2c` (ambient, class order).

## 2. The 14 corpus matrices (item 1 and item 3)

{{T1}}

`today` is the sum of the 14 original files (114,090,620 B); the re-emitted `ref_1pz_v1_L3` is 114,382,406 B (zstd 1.5.5). Model
files are not in these numbers (section 6). Rows are sorted as in the study; every `matrix_ok` is 1.

## 3. Row order (item 2)

`b_rans16` has no class-order option (`tb::RowPerm` is the exact descending-nnz order), and Track B's notes only describe it
(section 4: +0.006 bits/nnz at 4 classes per octave). It needed no change to Track B code: the permutation routine only uses its
key as a sort key, so `w3_total_v1c` passes the class id `tb::g_cls(nnz)` as the key (descending, ties by row id, hence genomic
order inside a class) and transmits the class per row instead of the nnz.

| | exact rank order (`v1`) | frequency class (`v1c`) |
|---|---|---|
| 14-file total | 46,438,410 B | 46,344,217 B (-94,193 B, -0.20%) |
| row-count stream | 995,547 B | 863,790 B (-13.2%; -0.4 to -24 KB per file) |
| payload (column-nnz + tables + chunk streams) | 43,123,003 B | +37,564 B (+0.09%; -13 to +15,411 B per file) |
| payload decode, canonical (sorted ids) | 92-159 Mnnz/s | same permutation machinery, same speed |
| payload decode, native (rank ids, no sort) | 195-308 Mnnz/s | same |

Native order means `indices` are ranks and `rownames` are permuted to match; `w3::native_matches` maps it back and checks every
entry against the original (`native_ok = 1` on all 14 files, both orders). Native decoding is 1.8-2.6x faster than canonical
for the payload. The class order is smaller but does not by itself remove the sort: a consumer needs the native layout, or the
canonicaliser would have to merge the few dozen pre-sorted class runs per column instead of sorting; that merge was not
written here (it would replace the bitonic sort of `CanonVec`, which is 44-62% of the canonical payload time on the clean rows).

## 4. Empties as a second member (item 3)

Idea 2's `Engine::code_empties` composes without changes by giving it virtual column ids (cells first, then empties) and a
virtual matrix; decode builds the cells' profile from the decoded cells member exactly as Idea 2 did from its own cells.
Results (from the tables above and in section 6; `empt` = the droplets that `cell_calls.tsv` does not call):

| | w3_total_v1 | w3_total_v2 (ambient) | w3_total_v2r (two b_rans16 members) |
|---|---|---|---|
| 14-file total | 46,438,410 | 46,274,869 (-0.35%) | 46,687,379 (+0.54%) |
| empties bytes (14 files) | inside the single member | 13,907,815 (k + multiset) | 14,320,255 (counts + payload) |
| files smaller than v1 | | 11 of 14 (loses on brain_sn_v3_deep exon +5.5 KB, tcell_5p exon +1.2 KB, tumor_3p intron +0.8 KB) | 0 of 14 |
| full decode, sum of 14 | 1.76 s | 13.25 s (2.7-8.4 Mnnz/s on the dense files) | 2.81 s |

What the ambient coder adds: -2.9% of the empties bytes against `b_rans16` on the same droplets (-0.8% on brain_sn_v3_deep exon to
-19.6% on pbmc_v2_typical intron; 9.8 vs 9.9 bits/nnz on the empties of the largest file), which is only 0.9% of the whole file,
and the separate empties member costs 0.5% more than keeping them in the one matrix (a second set of row counts, tables and
column counts, and the bitmap). Net: -0.35% for 7.5x the decode time. The 1-6% gap above the floor that Idea 2 reported is
real, but the floor was never the competitor: b_rans16's depth-conditioned position tables already put low-depth columns in
low-depth contexts. Splitting empties out remains useful for *transfer* (E4: the cells arrive first), not for size.
The split bitmap costs 137-2,482 B per file (0.025-0.40 bits per column; 2,429 B for the 96k columns of pbmc_v2_shallow exon, about twice Idea 2's own
depth-context flag stream, which however needed a separate UMI total for every cell), and `k` per droplet 2.4-47 KB.

## 5. GSE145370 bundle (item 4)

Extracted with Python `zipfile` (`tools/w3_total_extract.py`, never executed): all 28 exon and intron matrices plus the 14
`cell_calls.tsv` into `corpus_w3_total/GSE145370/<gsm>/`. `w3_total_v1` on all 28: `matrix_ok = 1` on every field
(`results/w3_total_bundle.tsv`). v2 was not run on the bundle. The called set of each sample is a separate stream
(`w3_total_calls`: one flag per column, context = quarter-octave depth bucket, round trip verified; 140-405 B per sample,
3,180 B for the 14) so that `cell_calls.tsv` can be regenerated from it and the exon column sums (E4 verified that regeneration byte for
byte on 21/21 files; it was not re-run here). The new archive was written with `zipfile` and its size is the measured number.

{{BUNDLE}}

Not removed although E4 lists parts of them as derivable, to keep the number conservative: feature_vocab.json (293,660 B), manifest.json and the
numeric fields of summary.json / pileup_stats.json / provenance.json / study_meta.json (all JSON members together 324,471 B, 0.17% of today's bundle).
The tumor_3p sample GSM4317429 is also corpus tumor_3p: its two streams are byte-for-byte the same size in both runs (392,898 and 247,389 B).

## 6. Bytes per component of the final stream (the format-spec table)

Columns: row-name id (vocabulary reference), barcodes (whitelist index), user kv, row counts (the permutation replacement), column-nnz
stream (the depth side information: nnz of every column, which also yields the column pointers), rANS coder tables, chunk streams
(the symbols), header fixed fields. The vocabulary / whitelist model files are not counted. Hints: the row-count stream is
0.9-5.1% of a stream, and 8-10% on the two tumor_3p matrices, because the support (which rows a sample touches) is the
information; the barcode list is 2-4% of a stream on the dense matrices and 11-25% on the sparse ones.

{{T2}}

{{T3}}

## 7. Timing (item 5)

Method: `bin/w3_total_bench.exe`, single thread, HIGH priority, min of 5 reps, output vectors of the payload-only closures reused
across reps (as Track B's hooks), `decode()` timings include name materialisation (310k-386k `std::string` rownames and the
barcodes), the working-buffer copy and output allocation. The clock was calibrated per codec with Track B's 1-cycle dependency
chain: the core ran at **3.6-3.9 GHz on the first file of the session and then fell to 1.2-2.9 GHz** as the laptop heated and
other jobs started, so wall-clock Mnnz/s differ between rows for the clock alone. Rows with a calibrated clock of at least
2.5 GHz are the clean ones (marked *); the others are real measurements at a low clock. Nothing is converted to an estimate here.

{{T4}}

Cleanest rows (*, calibrated clock 2.57-2.89 GHz), Mnnz/s of the whole matrix:

| file (nnz) | today's reader | Track A decode | b_rans16 full | **w3_total_v1 full, canonical** | w3_total_v1 full, native | payload only, canonical | payload only, native |
|---|---|---|---|---|---|---|---|
| brain_sn_v3_deep exon (6.35 M) | 21.1 | 13.8 | 44.6 | **49.8** | 39.2 | 118.9 | 244.9 |
| brain_sn_v3_intronic exon (1.38 M) | 8.1 | 8.7 | 13.9 | **17.5** | 16.5 | 91.9 | 195.4 |
| pbmc_v2_shallow exon (5.30 M) | 16.8 | 15.4 | 37.8 | **57.0** | 62.2 | 158.7 | 282.3 |
| pbmc_v2_typical exon (2.57 M) | 8.2 | 5.8 | 12.5 | **30.0** | 33.9 | 110.7 | 214.2 |
| tcell_5p exon (10.5 M) | 11.5 | 14.3 | 30.8 | **72.4** | 60.9 | 116.7 | 307.7 |

Phase times of the full canonical decode on those rows (ms): names 32-50 (vocabulary split into strings and 15-96k barcode
strings), row counts 9-16, payload 18-98, buffer copy 0.5-2.2. **Name materialisation is 22-63% of the whole decode on these files**, which is why full decode (17-72) is so much slower than the payload alone (92-159); a lazy name view would
remove it. In native order the full decode is not faster than canonical on several rows because it also permutes the 310k row
names (a string move per row), so the payload-only columns are the fair native-versus-canonical comparison.

A second 3-rep run (`results/w3_total_timing.tsv`) was taken at 07:17, after the bundle run, when the machine was no longer idle
(other agents' jobs; 100% CPU before the run started, core 1.2-1.7 GHz). It is useful only as a ratio check: today's reader
4.0-4.7 Mnnz/s, b_rans16 9.5-19.3, w3_total_v1 canonical 11.5-19.2, payload only canonical 35-40 vs native 81-96; the ordering
of the codecs and the 2x native/canonical ratio are the same. Encode of v1 takes 3-16 s per file, 88 s for the 14 (the names-model verification and the
per-column `std::sort` into rank space dominate); it was not optimised.

## 8. Second pass (item 6)

{{T5}}

Expected >= 0.99; measured 0.9991-1.0002. xz and zstd are applied to the complete stream including the barcode indices; neither
finds anything because the payload is near entropy and the rest is already entropy coded or incompressible indices. The
context-mixing second pass (`d_cm_bytes`) was not run (see section 9); Track D measured 0.987-0.996 for it on earlier outputs.

## 9. What did not compose, caveats, and unfinished items

Composition notes:
- `b_rans16`'s own encoder writes its names (zstd) and row counts (LEB128 + zstd-3) itself, so the codec cannot be called
  as a black box. `w3_total` calls the functions it is built from and replaces those two sections; the payload bytes are
  Track B's (checked: 8.06 bits/nnz on brain_sn_v3_deep exon). The `b_rans16` rows in the logs are the unmodified Track B codec
  with its zstd names (a yardstick, not a component).
- Track B limits remain: m <= 2^19 rows, value - 2 < 2^19 (none of the 14 + 28 matrices hit them). Row counts are the only
  permutation information; `b_rans16` could not drop them (reference-order permutations cost 1.3-4.7 bits/nnz).
- Idea 2's coder needs the called set at encode time (read from `cell_calls.tsv`); decode needs nothing external.
- The called-set stream for v1 is only needed to regenerate `cell_calls.tsv`; the matrix itself does not use it.
- Barcodes are the largest remaining non-payload item (2.32 MB, 5.0% of v1; 15-25% on the sparse matrices): they stay in file
  order at 2.5-2.75 B each. Writing columns in whitelist order (E4/Track A) would cut them to 0.5-0.9 B; not done because it
  changes the matrix.

Unfinished (not attempted, per the instruction to stop at 07:40 EDT):
1. v2 (ambient empties) on the 28 bundle matrices.
2. A fully idle timing run: the machine became busy at about 07:17 (other agents' jobs); the clock also varied during the 14-file run.
   Section 7's clean rows are the best available; a rerun of `w3_total_bench.exe --reps 5` on an idle machine would replace Table 4.
3. The context-mixing second pass (`d_cm_bytes` / `dcm::compress`); only zstd-19 and xz -9e were run, on the 14 `w3_total_v1` streams.
4. Recoding `sj_counts`, `vdj_gene_usage`, `mt_heteroplasmy` (E5 was not run); building `splice_psi` back from `sj_counts` (E4 showed it is
   derivable; it is simply absent from the new archive); re-deriving `cell_calls.tsv` and `summary.json` numerics on the bundle.
5. The sort-free fast decoder for the class order (merge of pre-sorted runs) and a lazy name view.
6. Whitelist-order columns, a decoded-matrix digest, and the chunk-index-first layout of container v2.

## 10. Reproduce

```bash
cd bench/compression
bash build.sh harness/w3_total_bench.cpp harness/w3_total_calls.cpp harness/w3_total_second.cpp harness/w3_total_cnt.cpp
./bin/w3_total_bench.exe --reps 5 --out results/w3_total_14.tsv --detail results/w3_total_14_detail.tsv \
    --dump results/w3_total_streams corpus/*/exon_counts.1pz corpus/*/intron_counts.1pz
python -I tools/w3_total_extract.py corpus/bundles/GSE145370.singlet corpus_w3_total/GSE145370
./bin/w3_total_bench.exe --codec w3_total_v1 --reps 3 --no-native --out results/w3_total_bundle.tsv \
    --detail results/w3_total_bundle_detail.tsv --dump results/w3_total_bundle_streams \
    corpus_w3_total/GSE145370/GSM*/exon_counts.1pz corpus_w3_total/GSE145370/GSM*/intron_counts.1pz
./bin/w3_total_calls.exe --out results/w3_total_calls_bundle.tsv --dump results/w3_total_bundle_streams corpus_w3_total/GSE145370/GSM*/exon_counts.1pz
./bin/w3_total_second.exe results/w3_total_streams/*__w3_total_v1.bin
python -P tools/w3_total_tables.py > results/w3_total_tables.md
python -I tools/w3_total_bundle.py > results/w3_total_bundle_tables.md
```
