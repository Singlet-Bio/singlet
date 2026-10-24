# w3_total: one measured codec for the combined container, payload and empties design

Status: complete for the items below (2026-10-22); unfinished items are listed in section 9. Owner: the `w3_total` agent.
Files: `kernels/w3_total.h` (codec), `harness/w3_total_bench.cpp` (driver, `bin/w3_total_bench.exe`), `harness/w3_total_calls.cpp`
(called-set stream), `harness/w3_total_second.cpp` (second pass), `harness/w3_total_cnt.cpp` (row-count coder experiment),
`tools/w3_total_extract.py`, `tools/w3_total_bundle.py`, `tools/w3_total_tables.py`, `corpus_w3_total/`, `results/w3_total_*`.
This file is assembled from `results/w3_total_notes_template.md` + the generated tables (`results/w3_total_tables.md`, `results/w3_total_bundle_tables.md`); numbers are copied from `results/`.

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

### Table 1. 14 matrices, bytes of the whole stream (model files not counted)

| file | nnz | today (original .1pz) | Track A a_names_perm_model | w3_total_v1 | w3_total_v1c (class order) | w3_total_v2 (ambient empties) | w3_total_v2r (empties via b_rans16) | w3_total_v2c | v1 bits/nnz | matrix_ok |
|---|---|---|---|---|---|---|---|---|---|---|
| brain_sn_v3_deep exon | 6,349,311 | 13,268,199 | 7,803,364 | 6,762,639 | 6,745,046 | 6,768,187 | 6,797,027 | 6,755,379 | 8.52 | 1 |
| brain_sn_v3_deep intron | 6,678,099 | 11,685,933 | 7,274,229 | 6,431,491 | 6,410,585 | 6,420,938 | 6,480,109 | 6,405,116 | 7.70 | 1 |
| brain_sn_v3_intronic exon | 1,380,834 | 7,759,162 | 2,292,404 | 1,909,787 | 1,904,506 | 1,886,154 | 1,924,113 | 1,882,122 | 11.06 | 1 |
| brain_sn_v3_intronic intron | 8,144,777 | 13,300,669 | 8,887,826 | 7,988,031 | 7,969,323 | 7,941,133 | 8,018,880 | 7,925,275 | 7.85 | 1 |
| pbmc_v2_shallow exon | 5,304,754 | 10,539,256 | 6,076,758 | 4,979,147 | 4,973,974 | 4,937,686 | 4,999,170 | 4,933,335 | 7.51 | 1 |
| pbmc_v2_shallow intron | 1,170,031 | 6,306,035 | 2,344,425 | 2,065,880 | 2,059,384 | 2,051,153 | 2,095,771 | 2,046,967 | 14.13 | 1 |
| pbmc_v2_typical exon | 2,569,228 | 6,451,648 | 2,287,890 | 1,939,351 | 1,937,621 | 1,927,876 | 1,939,852 | 1,931,327 | 6.04 | 1 |
| pbmc_v2_typical intron | 1,202,661 | 5,156,912 | 1,474,146 | 1,335,913 | 1,329,345 | 1,329,984 | 1,346,933 | 1,323,769 | 8.89 | 1 |
| pbmc_v3_lowgenes exon | 1,759,828 | 6,180,180 | 1,784,425 | 1,430,173 | 1,429,950 | 1,426,012 | 1,430,462 | 1,431,703 | 6.50 | 1 |
| pbmc_v3_lowgenes intron | 1,076,297 | 5,263,802 | 1,366,058 | 1,164,742 | 1,163,169 | 1,163,216 | 1,167,787 | 1,161,539 | 8.66 | 1 |
| tcell_5p exon | 10,529,513 | 13,757,617 | 9,391,220 | 8,208,424 | 8,205,113 | 8,209,660 | 8,232,446 | 8,227,321 | 6.24 | 1 |
| tcell_5p intron | 1,098,311 | 5,676,401 | 1,803,670 | 1,582,545 | 1,577,082 | 1,572,364 | 1,598,796 | 1,567,387 | 11.53 | 1 |
| tumor_3p exon | 245,117 | 4,704,028 | 463,963 | 392,898 | 392,114 | 392,270 | 401,302 | 391,690 | 12.82 | 1 |
| tumor_3p intron | 107,364 | 4,040,778 | 283,775 | 247,389 | 247,005 | 248,236 | 254,731 | 248,040 | 18.43 | 1 |
| **total (14)** | 47,616,125 | **114,090,620** | **53,534,153** | **46,438,410** | **46,344,217** | **46,274,869** | **46,687,379** | **46,230,970** | 7.80 | all 1 |
| vs today (original) | | 1.000 | 0.469 | 0.407 | 0.406 | 0.406 | 0.409 | 0.405 | | |
| vs re-emitted today ref_1pz_v1_L3 (114,382,406) | | | 0.468 | 0.406 | 0.405 | 0.405 | 0.408 | 0.404 | | |

bits/nnz overall: today 19.17, Track A 8.99, w3_total_v1 7.80, w3_total_v2 7.77

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

### Table B1. GSE145370 bundle, today vs w3_total_v1 (bytes)

| member kind | today | w3_total (measured) | treatment |
|---|---|---|---|
| exon_counts.1pz (14) | 67,060,711 | 6,485,170 | w3_total_v1, bit-exact matrix on all 14 |
| intron_counts.1pz (14) | 58,705,745 | 5,266,761 | w3_total_v1, bit-exact matrix on all 14 |
| sj_counts.1pz (14) | 31,810,097 | 31,810,097 | kept as today: not recoded (E5 not run) |
| vdj_gene_usage.1pz (14) | 4,254,148 | 4,254,148 | kept as today: not recoded |
| mt_heteroplasmy.1pz (14) | 2,114 | 2,114 | kept as today (0x0 stubs) |
| splice_psi.1pz (14) | 24,575,390 | 0 | derived from sj_counts (E4: 14/14 byte-identical); not rebuilt here |
| cell_calls.tsv (14) | 313,060 | 3,180 | called-set stream (round trip verified); the TSV regenerates from it + exon column sums (E4) |
| JSON members | 324,471 | 324,471 | kept as today (E4 lists derivable numeric fields, not removed here) |
| zip framing (central directory + local headers) | 20,614 | 18,514 | measured: 129 entries now, 143 before |
| **total** | **187,066,350** | **48,164,455** | ratio 0.2575 = 3.88x smaller; file `corpus_w3_total/GSE145370.w3.zip` |

Share of today's bundle: exon_counts.1pz 35.8%, intron_counts.1pz 31.4%, sj_counts.1pz 17.0%, splice_psi.1pz 13.1%, vdj_gene_usage.1pz 2.3%, cell_calls.tsv 0.2%

New bundle: exon+intron 11,751,931 (24.4%), sj+vdj+mt unchanged 36,066,359 (74.9%), everything else 346,165

### Table B2. per sample (bytes)

| sample | called | exon today | exon w3 | intron today | intron w3 | calls stream |
|---|---|---|---|---|---|---|
| GSM4317423 | 2,722 | 5,027,865 | 639,858 | 4,307,655 | 449,121 | 233 |
| GSM4317424 | 1,727 | 4,639,118 | 337,236 | 4,018,334 | 226,721 | 183 |
| GSM4317425 | 1,207 | 4,726,504 | 402,983 | 4,130,724 | 313,880 | 151 |
| GSM4317426 | 2,928 | 5,454,394 | 975,281 | 5,069,379 | 1,083,533 | 211 |
| GSM4317427 | 3,576 | 4,952,111 | 566,994 | 4,271,075 | 411,030 | 289 |
| GSM4317428 | 2,473 | 4,736,261 | 437,280 | 3,997,696 | 232,202 | 405 |
| GSM4317429 | 3,820 | 4,704,028 | 392,898 | 4,040,778 | 247,389 | 316 |
| GSM4317430 | 3,732 | 4,568,381 | 296,282 | 3,991,858 | 221,051 | 203 |
| GSM4317431 | 3,371 | 4,504,589 | 245,565 | 3,947,508 | 187,476 | 174 |
| GSM4317432 | 4,289 | 4,983,448 | 615,731 | 4,378,531 | 528,116 | 276 |
| GSM4317433 | 2,028 | 4,913,525 | 567,004 | 4,485,853 | 629,456 | 165 |
| GSM4317434 | 2,106 | 4,621,610 | 343,801 | 4,041,051 | 259,800 | 140 |
| GSM4317435 | 2,977 | 4,648,199 | 359,875 | 4,047,987 | 267,133 | 210 |
| GSM4317436 | 3,658 | 4,580,678 | 304,382 | 3,977,316 | 209,853 | 224 |

### Table B3. Same bundle with Track A's container treatment applied to sj_counts and vdj (PROJECTION, not a measured codec)

sj_counts 31,810,097 -> about 14,347,675 (chunks 13,172,766 + other 48,920 + row counts 555,130 + study junction vocabulary subset coding 570,859); vdj_gene_usage 4,254,148 -> about 1,780,740. New bundle would be about 28,228,625 B (these use Track A's results/track_a_bundle_whatif.md.txt numbers, payload unchanged).

Not removed although E4 lists parts of them as derivable, to keep the number conservative: feature_vocab.json (293,660 B), manifest.json and the
numeric fields of summary.json / pileup_stats.json / provenance.json / study_meta.json (all JSON members together 324,471 B, 0.17% of today's bundle).
The tumor_3p sample GSM4317429 is also corpus tumor_3p: its two streams are byte-for-byte the same size in both runs (392,898 and 247,389 B).

## 6. Bytes per component of the final stream (the format-spec table)

Columns: row-name id (vocabulary reference), barcodes (whitelist index), user kv, row counts (the permutation replacement), column-nnz
stream (the depth side information: nnz of every column, which also yields the column pointers), rANS coder tables, chunk streams
(the symbols), header fixed fields. The vocabulary / whitelist model files are not counted. Hints: the row-count stream is
0.9-5.1% of a stream, and 8-10% on the two tumor_3p matrices, because the support (which rows a sample touches) is the
information; the barcode list is 2-4% of a stream on the dense matrices and 11-25% on the sparse ones.

### Table 2. Components of the final w3_total_v1 stream, bytes (hdr = magic, flags, m, vt, n, nnz, chunk_cols, order, length fields)

| file | row-name id | barcodes (whitelist index) | user kv | row counts | column-nnz stream (depth side info) | coder tables | chunk streams (the payload symbols) | header | total | b_rans16 payload bits/nnz | side info % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| brain_sn_v3_deep exon | 9 | 232,192 | 4 | 136,121 | 68,300 | 14,449 | 6,311,529 | 35 | 6,762,639 | 8.06 | 6.7 |
| brain_sn_v3_deep intron | 9 | 232,192 | 4 | 137,308 | 61,006 | 15,942 | 5,984,995 | 35 | 6,431,491 | 7.26 | 6.9 |
| brain_sn_v3_intronic exon | 9 | 216,985 | 4 | 72,756 | 46,810 | 12,436 | 1,560,752 | 35 | 1,909,787 | 9.39 | 18.3 |
| brain_sn_v3_intronic intron | 9 | 216,985 | 4 | 128,036 | 62,785 | 14,430 | 7,565,747 | 35 | 7,988,031 | 7.51 | 5.3 |
| pbmc_v2_shallow exon | 9 | 241,141 | 4 | 62,975 | 78,882 | 9,412 | 4,586,689 | 35 | 4,979,147 | 7.05 | 7.9 |
| pbmc_v2_shallow intron | 9 | 241,141 | 4 | 68,351 | 58,091 | 10,614 | 1,687,635 | 35 | 2,065,880 | 12.01 | 18.3 |
| pbmc_v2_typical exon | 9 | 37,989 | 4 | 57,583 | 14,104 | 12,433 | 1,817,194 | 35 | 1,939,351 | 5.74 | 6.3 |
| pbmc_v2_typical intron | 9 | 37,989 | 4 | 68,130 | 10,641 | 10,554 | 1,208,551 | 35 | 1,335,913 | 8.18 | 9.5 |
| pbmc_v3_lowgenes exon | 9 | 195,444 | 4 | 12,158 | 33,045 | 8,546 | 1,180,932 | 35 | 1,430,173 | 5.56 | 17.4 |
| pbmc_v3_lowgenes intron | 9 | 195,444 | 4 | 18,500 | 26,258 | 10,022 | 914,470 | 35 | 1,164,742 | 7.07 | 21.5 |
| tcell_5p exon | 9 | 173,149 | 4 | 106,847 | 49,566 | 13,724 | 7,865,090 | 35 | 8,208,424 | 6.02 | 4.2 |
| tcell_5p intron | 9 | 173,149 | 4 | 70,399 | 28,652 | 10,223 | 1,300,074 | 35 | 1,582,545 | 9.75 | 17.8 |
| tumor_3p exon | 9 | 62,694 | 4 | 31,588 | 14,443 | 4,002 | 280,123 | 35 | 392,898 | 9.74 | 28.7 |
| tumor_3p intron | 9 | 62,694 | 4 | 24,795 | 11,224 | 3,700 | 144,928 | 35 | 247,389 | 11.91 | 41.4 |
| **total** | 126 | 2,319,188 | 56 | 995,547 | 563,807 | 150,487 | 42,408,709 | 490 | 46,438,410 | | |
| share | 0.0% | 5.0% | 0.0% | 2.1% | 1.2% | 0.3% | 91.3% | 0.0% | 100% | | |

Model bytes (read-only side files, NOT counted above): GRCh38 vocabulary 3,569,124 (exon + intron lists), GRCm39 vocabulary 4,470,944, 3M-february-2018 whitelist 6,465, 737K-august-2016 whitelist 3,283.

### Table 3. w3_total_v2 components, bytes

| file | names (row id + barcodes + kv) | cells: row counts | cells: column-nnz | cells: tables | cells: chunk streams | empties: k per droplet | empties: ambient multiset | split bitmap | header | total | empties nnz | empties bits/nnz (k + multiset) | same empties via b_rans16 (v2r: counts + payload) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| brain_sn_v3_deep exon | 232,205 | 116,919 | 4,692 | 7,730 | 2,642,889 | 46,784 | 3,716,648 | 261 | 59 | 6,768,187 | 3,065,880 | 9.82 | 3,792,267 (9.90 b/nnz) |
| brain_sn_v3_deep intron | 232,205 | 123,563 | 4,965 | 8,586 | 3,626,129 | 44,292 | 2,380,867 | 272 | 59 | 6,420,938 | 1,982,231 | 9.79 | 2,484,325 (10.03 b/nnz) |
| brain_sn_v3_intronic exon | 216,998 | 64,183 | 8,405 | 8,284 | 910,805 | 32,186 | 644,830 | 404 | 59 | 1,886,154 | 517,586 | 10.46 | 714,970 (11.05 b/nnz) |
| brain_sn_v3_intronic intron | 216,998 | 118,925 | 11,562 | 7,297 | 5,165,095 | 38,446 | 2,382,359 | 392 | 59 | 7,941,133 | 1,820,227 | 10.64 | 2,498,547 (10.98 b/nnz) |
| pbmc_v2_shallow exon | 241,154 | 54,489 | 32,946 | 5,504 | 2,667,959 | 26,924 | 1,906,222 | 2,429 | 59 | 4,937,686 | 2,086,742 | 7.41 | 1,994,625 (7.65 b/nnz) |
| pbmc_v2_shallow intron | 241,154 | 58,671 | 26,649 | 8,944 | 1,049,471 | 23,861 | 639,862 | 2,482 | 59 | 2,051,153 | 413,432 | 12.84 | 708,336 (13.71 b/nnz) |
| pbmc_v2_typical exon | 38,002 | 56,578 | 6,493 | 4,607 | 1,662,096 | 4,101 | 155,803 | 137 | 59 | 1,927,876 | 166,823 | 7.67 | 171,875 (8.24 b/nnz) |
| pbmc_v2_typical intron | 38,002 | 67,013 | 6,070 | 7,353 | 1,141,795 | 2,365 | 67,182 | 145 | 59 | 1,329,984 | 52,289 | 10.64 | 86,491 (13.23 b/nnz) |
| pbmc_v3_lowgenes exon | 195,457 | 11,911 | 10,800 | 3,890 | 1,050,976 | 16,419 | 136,107 | 393 | 59 | 1,426,012 | 150,251 | 8.12 | 156,971 (8.36 b/nnz) |
| pbmc_v3_lowgenes intron | 195,457 | 18,002 | 10,918 | 8,124 | 824,115 | 10,530 | 95,582 | 429 | 59 | 1,163,216 | 83,505 | 10.17 | 110,678 (10.60 b/nnz) |
| tcell_5p exon | 173,162 | 103,972 | 15,669 | 5,575 | 6,783,623 | 21,304 | 1,105,690 | 606 | 59 | 8,209,660 | 1,102,416 | 8.18 | 1,149,775 (8.34 b/nnz) |
| tcell_5p intron | 173,162 | 67,362 | 11,051 | 7,416 | 1,131,423 | 11,066 | 170,189 | 636 | 59 | 1,572,364 | 120,263 | 12.06 | 207,682 (13.82 b/nnz) |
| tumor_3p exon | 62,707 | 25,445 | 2,979 | 2,704 | 152,502 | 7,994 | 136,619 | 1,261 | 59 | 392,270 | 102,677 | 11.27 | 153,640 (11.97 b/nnz) |
| tumor_3p intron | 62,707 | 19,197 | 2,672 | 3,071 | 75,668 | 6,714 | 76,869 | 1,279 | 59 | 248,236 | 46,847 | 14.27 | 90,073 (15.38 b/nnz) |

## 7. Timing (item 5)

Method: `bin/w3_total_bench.exe`, single thread, HIGH priority, min of 5 reps, output vectors of the payload-only closures reused
across reps (as Track B's hooks), `decode()` timings include name materialisation (310k-386k `std::string` rownames and the
barcodes), the working-buffer copy and output allocation. The clock was calibrated per codec with Track B's 1-cycle dependency
chain: the core ran at **3.6-3.9 GHz on the first file of the session and then fell to 1.2-2.9 GHz** as the laptop heated and
other jobs started, so wall-clock Mnnz/s differ between rows for the clock alone. Rows with a calibrated clock of at least
2.5 GHz are the clean ones (marked *); the others are real measurements at a low clock. Nothing is converted to an estimate here.

### Table 4. Decode time, single thread, min of 5, nearly idle machine with a varying core clock (ms); Mnnz/s in brackets

| file | nnz | today ref_1pz_v1_L3 | Track A a_names_perm_model | b_rans16 full (its own zstd names) | w3_total_v1 full, canonical | w3_total_v1 full, native | payload only canonical | payload only native | b_rans16 payload canonical | b_rans16 payload native | core GHz (calibrated) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| brain_sn_v3_deep exon* | 6,349,311 | 300.2 (21.1) | 461.3 (13.8) | 142.5 (44.6) | 127.5 (49.8) | 162.1 (39.2) | 53.4 (118.9) | 25.9 (244.9) | 52.1 (121.8) | 25.3 (250.9) | 2.77 |
| brain_sn_v3_deep intron | 6,678,099 | 1328.3 (5.0) | 1264.9 (5.3) | 397.1 (16.8) | 392.1 (17.0) | 347.6 (19.2) | 159.6 (41.8) | 79.5 (84.0) | 124.1 (53.8) | 68.6 (97.3) | 1.35 |
| brain_sn_v3_intronic exon* | 1,380,834 | 170.0 (8.1) | 159.0 (8.7) | 99.1 (13.9) | 79.1 (17.5) | 83.5 (16.5) | 15.0 (91.9) | 7.1 (195.4) | 14.2 (97.5) | 6.9 (199.5) | 2.78 |
| brain_sn_v3_intronic intron | 8,144,777 | 1505.7 (5.4) | 940.3 (8.7) | 325.3 (25.0) | 231.0 (35.3) | 357.0 (22.8) | 187.7 (43.4) | 89.3 (91.3) | 107.4 (75.8) | 58.8 (138.6) | 1.22 |
| pbmc_v2_shallow exon* | 5,304,754 | 315.1 (16.8) | 344.6 (15.4) | 140.3 (37.8) | 93.1 (57.0) | 85.3 (62.2) | 33.4 (158.7) | 18.8 (282.3) | 44.5 (119.2) | 23.6 (224.5) | 2.78 |
| pbmc_v2_shallow intron | 1,170,031 | 297.7 (3.9) | 175.8 (6.7) | 117.1 (10.0) | 106.2 (11.0) | 155.8 (7.5) | 34.6 (33.8) | 12.4 (94.0) | 18.4 (63.4) | 6.8 (171.5) | 1.61 |
| pbmc_v2_typical exon* | 2,569,228 | 314.1 (8.2) | 442.6 (5.8) | 205.9 (12.5) | 85.7 (30.0) | 75.8 (33.9) | 23.2 (110.7) | 12.0 (214.2) | 39.1 (65.6) | 21.1 (121.8) | 2.57 |
| pbmc_v2_typical intron | 1,202,661 | 230.6 (5.2) | 184.4 (6.5) | 102.9 (11.7) | 80.2 (15.0) | 103.4 (11.6) | 15.0 (80.1) | 7.6 (158.7) | 16.0 (75.0) | 8.8 (136.1) | 1.94 |
| pbmc_v3_lowgenes exon | 1,759,828 | 347.9 (5.1) | 220.2 (8.0) | 149.0 (11.8) | 118.0 (14.9) | 170.3 (10.3) | 26.0 (67.6) | 17.3 (101.6) | 31.9 (55.2) | 17.9 (98.5) | 1.83 |
| pbmc_v3_lowgenes intron | 1,076,297 | 262.1 (4.1) | 161.3 (6.7) | 100.1 (10.8) | 74.9 (14.4) | 85.5 (12.6) | 14.5 (74.5) | 8.8 (123.0) | 21.2 (50.7) | 10.4 (103.9) | 2.07 |
| tcell_5p exon* | 10,529,513 | 912.2 (11.5) | 734.9 (14.3) | 341.4 (30.8) | 145.5 (72.4) | 172.9 (60.9) | 90.2 (116.7) | 34.2 (307.7) | 130.2 (80.9) | 61.9 (170.1) | 2.89 |
| tcell_5p intron | 1,098,311 | 182.5 (6.0) | 196.6 (5.6) | 106.1 (10.4) | 96.0 (11.4) | 86.6 (12.7) | 13.7 (80.1) | 8.7 (125.8) | 14.6 (75.3) | 8.2 (133.3) | 1.93 |
| tumor_3p exon | 245,117 | 131.6 (1.9) | 70.1 (3.5) | 95.9 (2.6) | 71.9 (3.4) | 113.6 (2.2) | 5.1 (48.2) | 2.0 (120.6) | 4.1 (60.2) | 1.6 (154.3) | 1.66 |
| tumor_3p intron | 107,364 | 153.1 (0.7) | 92.0 (1.2) | 129.3 (0.8) | 61.8 (1.7) | 109.9 (1.0) | 3.4 (31.8) | 1.6 (67.3) | 6.1 (17.7) | 1.6 (65.1) | 1.56 |
| **sum of 14** | 47,616,125 | 6451 (7.4) | 5448 (8.7) | 2452 (19.4) | 1763 (27.0) | 2109 (22.6) | 675 (70.6) | 325 (146.4) | 624 (76.3) | 322 (148.1) | |

`*` = calibrated core clock >= 2.5 GHz (clean row). Other rows ran at 1.2-2.0 GHz.

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

### Table 5. Second pass over the w3_total_v1 streams (bytes; ratio = recompressed / input)

| file | stream | zstd -19 | ratio | xz -9e | ratio |
|---|---|---|---|---|---|
| brain_sn_v3_deep exon | 6,762,639 | 6,762,351 | 1.0000 | 6,762,476 | 1.0000 |
| brain_sn_v3_deep intron | 6,431,491 | 6,430,921 | 0.9999 | 6,431,220 | 1.0000 |
| brain_sn_v3_intronic exon | 1,909,787 | 1,909,844 | 1.0000 | 1,909,600 | 0.9999 |
| brain_sn_v3_intronic intron | 7,988,031 | 7,987,502 | 0.9999 | 7,987,928 | 1.0000 |
| pbmc_v2_shallow exon | 4,979,147 | 4,979,270 | 1.0000 | 4,978,340 | 0.9998 |
| pbmc_v2_shallow intron | 2,065,880 | 2,065,937 | 1.0000 | 2,065,792 | 1.0000 |
| pbmc_v2_typical exon | 1,939,351 | 1,938,984 | 0.9998 | 1,939,496 | 1.0001 |
| pbmc_v2_typical intron | 1,335,913 | 1,335,058 | 0.9994 | 1,335,688 | 0.9998 |
| pbmc_v3_lowgenes exon | 1,430,173 | 1,430,215 | 1.0000 | 1,430,168 | 1.0000 |
| pbmc_v3_lowgenes intron | 1,164,742 | 1,164,778 | 1.0000 | 1,164,832 | 1.0001 |
| tcell_5p exon | 8,208,424 | 8,201,384 | 0.9991 | 8,205,692 | 0.9997 |
| tcell_5p intron | 1,582,545 | 1,581,168 | 0.9991 | 1,581,972 | 0.9996 |
| tumor_3p exon | 392,898 | 392,916 | 1.0001 | 392,972 | 1.0002 |
| tumor_3p intron | 247,389 | 247,404 | 1.0001 | 247,312 | 0.9997 |

lowest ratio over all: 0.9991

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
