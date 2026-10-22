# w4_corpus: do the study's findings survive a diverse corpus?

Status: PRELIMINARY (coordinator hard stop 07:40 EDT, 2026-10-22). Owner: the `w4_corpus` agent. Everything below is measured on
the files that were downloaded and run before the stop; the sections "Not finished" lists what was skipped on purpose or ran out of time.
Files: `corpus_w4_corpus/` (downloads, git-ignored size ~950 MB), `tools/w4_corpus_*.py`, `harness/w4_corpus_*.cpp`,
`results/w4_corpus_*`, this note. Regenerate the tables with
`python -P tools/w4_corpus_report.py > results/w4_corpus_tables.md; python -P tools/w4_corpus_report2.py > results/w4_corpus_tables2.md`.

## 1. What was done

| step | tool | output |
|---|---|---|
| central directories (37 bundles of `corpus_w4.tsv`, about 100 more bundles of `w4_atlas_ids.txt` for the large-sample search) and full member inventory | `tools/w4_corpus_fetch.py inventory\|scan` | `results/w4_corpus_members.tsv`, `w4_corpus_bundles.tsv`, `w4_corpus_scan.tsv` |
| download by HTTP range of exon, intron, sj, splice_psi, cell_calls, summary, pileup_stats, provenance (+ manifest, study_meta) for 37 samples + 2 large + 1 stub | `tools/w4_corpus_fetch.py fetch` | `corpus_w4_corpus/<id>/`, `results/w4_corpus_requests.tsv` (231 requests in total) |
| sample master table, row-name hash per (build, member), barcode statistics, whitelist membership | `tools/w4_corpus_names.py` | `w4_corpus_samples.tsv`, `w4_corpus_rownames.tsv`, `w4_corpus_variants.tsv`, `w4_corpus_barcodes.tsv`, `w4_corpus_barcode_reuse.tsv` |
| vocabulary models for the 6 builds (the "vocabulary id replaces names" premise needs one list per build) | `harness/w4_corpus_models.cpp` | `results/w4_corpus_models/track_a_vocab_<build>.bin` (zstd-19 lists; Track A's xz would be ~25% smaller) |
| codec runs: `ref_1pz_v1_L3`, `a_names_perm_model`, `b_rans16` with payload / side / names accounting | `harness/w4_corpus_bench.cpp` | `results/w4_corpus_bench_p*.tsv`, `w4_corpus_perfile.tsv` |
| E4 derivability checks (read-only reuse of `tools/e4_audit.py`) | `tools/w4_corpus_e4.py` | `results/w4_corpus_e4.tsv` |
| called cells versus other columns | `tools/w4_corpus_empties.py` | `results/w4_corpus_empties.tsv` |

Sample selection. The task listed 36 samples; `corpus_w4.tsv` has 37 rows (16 human, 10 mouse, 4 barnyard, 3 zebrafish, 2 macaque, 2 rat).
Two large samples were found by reading central directories (`cell_calls.tsv` size / 50 B per line): `L_a` = GSE309702/GSM9277685
(67,271 columns; mouse 10xv3) and `L_b` = GSE264667/GSM8225104 (70,726 columns; human 10xv3); both have about 30-32k called cells by
the size estimate. Only these two of about 100 scanned bundles qualified (most atlas samples have 1-15k cells). `x_stub` = GSE214532/GSM6610876
is an edge case, not an aggregate member: its `cell_calls.tsv` lists 33,091 cells but all four matrix members are 151-byte empty
`.1pz` files (m = n = 0). `a_names_perm_model` cannot round-trip a 0x0 matrix ("lzma decode failed"): the codec needs an
empty-matrix special case.

Request count: 231 GET range requests to data.singlet.bio (rule: a few hundred at most). No other service was contacted.

## 2. Layout of the bundles (what "older layout" contains)

All 37 samples have `exon_counts.1pz` and `intron_counts.1pz`; none uses `matrix.mtx` as its only matrix. Two layouts exist:
a 10-member layout (exon, intron, sj, splice_psi, vdj, mt_heteroplasmy `.1pz`, summary / pileup_stats / provenance JSON,
`cell_calls.tsv`; 17 samples, the same as the original corpus) and a 34-46-member layout (20 samples) that adds `ambient_profile.tsv`
(3.6 MB), `auto_barcodes.tsv`, `cell_qc_metrics.tsv`, `cell_cycle_scores.tsv`, `doublet_scores.tsv`, `gene_expression.tsv`,
`star_SJ.out.tab`, `splice_events.tsv`, mt / donor / nonhost tables and, for seven samples, the raw and filtered STARsolo
`matrix.mtx` + `barcodes.tsv` + `features.tsv` (raw `barcodes.tsv` is 10.5 MB for v3 and 2.2 MB for v2 samples, i.e. by size the whole whitelist; not opened). The exon+intron `.1pz`
pair is 47% of the bytes of all sample members of the 37 samples, so the bundle-level headline (4.5x) cannot be transferred from the
10-member layout to the new layout without looking at those extra text members; that is the open question this inventory exists for.
Full per-member sizes: `results/w4_corpus_members.tsv`.

## 3. Verdicts (one line per claim; n = matrices unless stated)

Counted: 77 matrices (exon + intron) from 39 samples through ref / a_names_perm_model; b_rans16 on 69 of them (the barnyard matrices have m > 2^19 rows and the lane codec refuses them). Round trips: a_names_perm_model 77/77.

| claim | verdict | n | evidence |
|---|---|---|---|
| (a) container fixes (names to vocabulary id, permutation to row counts, no ptr / colsums, barcodes to whitelist index where a whitelist exists): 2.14x on the original 14 | **holds with caveat** | 77 matrices, 39 samples | pooled byte-weighted 0.465 of today = 2.15x (original 0.468 = 2.14x). Per matrix 0.012 to 0.740; worst h_microwell/exon 0.740 (payload-dominated: the names are a small share of these files), best m_mars/intron 0.012 (sparse plate-based samples: 88-99% of the bytes are removable). The pooled factor is set by which samples carry the bytes; for non-whitelist protocols the barcodes stay inline (zstd-19), so it is lower than a generic 2-bit packing would give (that was not run for lengths other than 16). |
| (b) b_rans16 payload is 15.1% below today's payload (11-24% per file) | **holds (pooled); per-file spread is wider** | 69 matrices | pooled -14.8% (-12.0% if the row-count side stream is charged); per matrix -35.9% to -6.9%; 13 of 69 fall outside the original -11..-24% band (least: h_mars/intron -6.9%; most: h_dropseq/intron -35.9%). Caveat: the lane codec does not support m > 2^19 rows (barnyard builds, 8 matrices). |
| (c) per-feature x depth model within ~0.1 bits/nnz of richer in-file models | **not tested** | 0 | `c_r2` skipped by coordinator instruction; no richer model was run on the new corpus. |
| (d) empties versus called cells split of the payload | **differs by protocol (nnz proxy only)** | section 6 table | share of nnz in called cells ranges from 1.6-1.8% (Quartz-Seq2), 13-17% (SPLiT-seq), 16% (Microwell-seq, Drop-seq 16-21%) through 64-74% (10x v2/v3 means) to 94% (Visium); payload-bit split not computed. 'Empties are a thin tail' holds for Visium and the best 10x samples and fails for plate-based and non-10x droplet protocols, where the non-called columns hold most of the nnz. |
| (e) row names identical within a reference build, so a vocabulary id replaces them | **holds** | 78 matrices, 12 (build, member) pairs | exactly one list per (build, member) in every pair; builds GRCh38, GRCm39, barnyard, GRCz11, Mmul_10, mRatBN7.2. One 0x0 stub matrix (x_stub) has no names. Vocabulary files cost 3.0-8.9 MB per build (zstd-19; xz about 25% less); the barnyard build alone is 8.9 MB. |
| (f) barcodes are 10x whitelist members | **holds for 10x v2/v3 only; fails elsewhere** | 39 samples | 10xv3 1.0000 in the 3M list (n=14); 10xv2 1.0000 in the 737K list (n=9); 10x multiome GEX 0.0071 in either (n=2: a third whitelist is needed); Visium 0.0014 in either (n=2), but the two spot lists are identical (cross-sample reuse 1.000), so a 4,991-entry list would serve; non-10x protocols 0.0002 (n=12; lengths 6, 7, 12, 15, 18, 24, 27; Cel-seq2 and MARS-seq2 enumerate the whole 4^6 / 4^7 space). |

E4 derivability checks on the new samples (compared / equal):

| E4 claim | verdict | compared | equal | failing |
|---|---|---|---|---|
| colsums block equals value recomputed from the matrix (exon, intron, sj, psi members) | holds | 152 | 152 |  |
| ptr block equals value recomputed from the matrix (exon, intron, sj, psi members) | holds | 152 | 152 |  |
| perm block equals value recomputed from the matrix (exon, intron, sj, psi members) | holds | 152 | 152 |  |
| colsums zstd frame bytes equal production zstd 1.5.5 re-compression | build-dependent | 152 | 97 | 24 samples differ: frames are not reproducible across zstd builds (as E4 found) |
| splice_psi structure (indptr, indices) equals sj_counts-derived events | holds | 35 | 35 |  |
| splice_psi values equal rint(PSI) | **fails as a universal rule** | 35 | 8 | stored values match rint for 8 samples and trunc(PSI) for 30; neither rule matches 0 (). The rounding rule differs across pipeline versions, so a derivation recipe must be pinned per version. |
| splice_psi member present when sj_counts has events | fails | 4 | 0 | L_b, h_v2_c, h_v3_b, h_v3_c: psi member is an empty stub although sj_counts has events (0-byte-sized psi members also appear in the inventory for h_v3_b, h_v3_c, h_v2_c) |
| cell_calls.tsv regenerated byte-for-byte from called set + exon colsums | fails for emptydrops samples | 39 | 35 | h_bd, h_indrop, h_microwell, m_dropseq: these use call_method=emptydrops, list non-called rows (is_cell FALSE) and carry real p-values, i.e. independent information; E4's finding that cell_calls.tsv never contains an EmptyDrops result is specific to the star_solo-only pipeline version |
| summary.json numeric fields reproduced | fails | 663 | 631 | h_bd, h_indrop, h_microwell, m_dropseq: same emptydrops samples; the test counts every cell_calls row as called (h_bd: 4541 rows vs n_cells_called 3367), so these mismatches are partly an artefact of the unadapted test, not evidence about summary.json |
| pileup_stats.json matrix-sum fields (exon_hits, intron_hits, sj_hits, umi_unique, n_barcodes) | holds with caveat | 195 | 135 | L_a, h_dropseq, h_visium, m_arc, m_celseq2, m_dropseq, m_mars, m_v2_a, m_v2_b, m_v3_a, m_v3_b, m_v3_c: stored counters differ from the matrix sums in these samples |
| provenance.json output fields | fails | 117 | 113 | h_bd, h_indrop, h_microwell, m_dropseq (same artefact: output.cells compared with all rows) |

Per-sample total ratio (exon + intron together, combined projection / today): m_mars 0.05, m_celseq2 0.09, h_visium 0.10, L_a 0.10, h_split 0.18, bn_dropseq 0.18, h_celseq2 0.24, m_visium 0.24, h_indrop 0.27, m_v2_a 0.29, h_dropseq 0.30, r_v3 0.31, h_v2_a 0.32, bn_v2 0.36, m_arc 0.37, m_v3_a 0.38, h_v2_c 0.38, bn_v3_b 0.38, h_v2_b 0.38, h_arc 0.41, h_bd 0.41, bn_v3_a 0.43, m_v2_b 0.43, h_v3_a 0.44, z_v3_a 0.45, h_v3_b 0.48, m_v3_c 0.49, z_v2 0.49, mq_v2 0.50, h_mars 0.51, m_v3_b 0.51, r_v2 0.52, h_quartz 0.53, z_v3_b 0.54, h_microwell 0.55, h_v3_c 0.55, m_dropseq 0.57, L_b 0.60, mq_v3 0.60.

Pooled byte-weighted combined projection (A names + A row counts + B payload; A payload where B is unsupported): 282.3 MB of 679.8 MB = 0.415 (2.41x), versus the original corpus 46 / 114.4 = 0.40 (2.5x). Per-sample distribution: median 0.41, min 0.05, max 0.60.

## 4. Not finished (and how to resume)

- Stopped at 07:38 EDT: 77 of 78 matrices have ref / a / b rows; `h_dropseq/exon_counts.1pz` was still running when the drivers were killed (re-run: `TRACK_A_MODELS=results/w4_corpus_models ./bin/w4_corpus_bench.exe --out results/w4_corpus_bench_p9.tsv corpus_w4_corpus/h_dropseq/exon_counts.1pz`, then the two report scripts and `tools/w4_corpus_verdicts.py`, `tools/w4_corpus_notes.py`).
- `x_stub` (empty matrices) is excluded from every aggregate; E4 rows for it were not produced.

- `c_r2` / any richer in-file model: skipped by the coordinator's instruction, so claim (c) is untested on the new corpus.
  Resume: `TRACK_A_MODELS=results/w4_corpus_models ./bin/<driver> --codec c_r2_nomodel,c_r2` on 4 small GRCh38/GRCm39 samples
  (h_v3_a, m_v2_a, h_visium, h_celseq2); the shipped c_r2 model only exists for GRCh38 / GRCm39 row-name lists.
- `w3_total_v1` (the real combined codec) was not run: the combined ratios here are *projections* made by adding measured
  components (A names + A row counts + B payload + 64 B), exactly the method the original 2.5x used. They are not a single codec.
- b_rans16 refuses m > 2^19 rows (`track_b lane codec: m > 2^19 unsupported`): the 8 barnyard matrices (696,825 and 579,921 rows)
  have no b_rans16 number; the combined projection uses the A payload for them (conservative, about 15% too large on the payload).
- Payload-bit split between called cells and empties (claim (d)) needs per-column code lengths from a codec; only the nnz split is given.
- Barcodes of non-16-mer protocols are stored inline (zstd-19) by `a_names_perm_model`; a 2k-bit packing or a whitelist for
  Visium / multiome would shrink the container ratio for those protocols (projection not computed).
- Model costs: the vocabulary files (3.0-8.9 MB per build) and whitelists are not charged to the ratios, as in Track A.
  The `results/.ref_identity.tmp.1pz` scratch file of `ref_1pz_v1_L3` is shared between concurrent drivers, so the re-emission size
  and round-trip flag of the ref rows are not trusted; all ref columns used here come from the original file's 96-byte header.
- The E4 test for cell_calls / summary / provenance assumes every cell_calls row is a called cell; it must be adapted for emptydrops samples.
- Scope of the E4 psi test: Python re-implementation versus stored values only (no production `compute_psi()` byte comparison).
- Timings were not taken (shared machine, 5 concurrent processes).


## 5. Generated tables (tools/w4_corpus_report.py)

files with ref+a rows: 77; a_names_perm_model matrix roundtrip OK on 77
files with b_rans16 rows: 69; roundtrip OK on 69; unsupported on 8

POOLED all files: today 679.8 MB; a_names_perm_model 316.2 MB = 0.465 (2.15x) [original corpus 0.468 = 2.14x]
POOLED combined projection (A names + A row counts + B payload): 282.3 MB = 0.415 (2.41x)
POOLED b_rans16 payload vs today's payload (chunks+ptr+chunk table), n=69 files: -14.8%; incl. row-count side stream: -12.0% [original corpus -15.1%]
  exon: n=34 payload -15.6%
  intron: n=35 payload -13.0%

#### (a) container ratio = a_names_perm_model bytes / today's file bytes, by protocol (original corpus pooled 0.468)

| protocol | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| 10x-arc-gex | 4 | 2 | 0.429 | 0.320 | 0.537 | h_arc/intron (0.537) | 0.434 |
| 10x-visium | 4 | 2 | 0.176 | 0.083 | 0.362 | m_visium/exon (0.362) | 0.207 |
| 10xv2 | 18 | 9 | 0.430 | 0.187 | 0.691 | z_v2/exon (0.691) | 0.459 |
| 10xv3 | 28 | 14 | 0.487 | 0.076 | 0.738 | L_b/exon (0.738) | 0.518 |
| bd-rhapsody | 2 | 1 | 0.431 | 0.298 | 0.565 | h_bd/exon (0.565) | 0.470 |
| celseq2 | 4 | 2 | 0.186 | 0.088 | 0.318 | h_celseq2/intron (0.318) | 0.188 |
| dropseq | 5 | 3 | 0.358 | 0.101 | 0.730 | m_dropseq/exon (0.730) | 0.429 |
| indrop | 2 | 1 | 0.323 | 0.252 | 0.394 | h_indrop/exon (0.394) | 0.335 |
| marsseq2 | 4 | 2 | 0.260 | 0.012 | 0.688 | h_mars/intron (0.688) | 0.364 |
| microwell-seq | 2 | 1 | 0.523 | 0.305 | 0.740 | h_microwell/exon (0.740) | 0.633 |
| quartzseq2 | 2 | 1 | 0.560 | 0.437 | 0.683 | h_quartz/exon (0.683) | 0.602 |
| splitseq | 2 | 1 | 0.195 | 0.129 | 0.262 | h_split/intron (0.262) | 0.197 |
| **all** | 77 | 39 | 0.408 | 0.012 | 0.740 | | 0.465 |

#### (a) container ratio = a_names_perm_model bytes / today's file bytes, by species (original corpus pooled 0.468)

| species | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| barnyard | 8 | 4 | 0.331 | 0.101 | 0.457 | bn_v3_a/intron (0.457) | 0.351 |
| human | 33 | 17 | 0.417 | 0.083 | 0.740 | h_microwell/exon (0.740) | 0.490 |
| macaque | 4 | 2 | 0.607 | 0.451 | 0.718 | mq_v3/exon (0.718) | 0.634 |
| mouse | 22 | 11 | 0.342 | 0.012 | 0.730 | m_dropseq/exon (0.730) | 0.427 |
| rat | 4 | 2 | 0.455 | 0.210 | 0.641 | r_v2/exon (0.641) | 0.501 |
| zebrafish | 6 | 3 | 0.542 | 0.316 | 0.709 | z_v3_b/exon (0.709) | 0.585 |
| **all** | 77 | 39 | 0.408 | 0.012 | 0.740 | | 0.465 |

#### (b) b_rans16 payload vs today's payload, by protocol (negative = smaller; original corpus -15.1% pooled, -11..-24% per file)

| protocol | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| 10x-arc-gex | 4 | 2 | -0.145 | -0.238 | -0.086 | h_arc/intron (-0.086) | -0.142 |
| 10x-visium | 4 | 2 | -0.169 | -0.215 | -0.117 | m_visium/exon (-0.117) | -0.138 |
| 10xv2 | 16 | 8 | -0.151 | -0.205 | -0.109 | r_v2/intron (-0.109) | -0.151 |
| 10xv3 | 24 | 12 | -0.151 | -0.325 | -0.085 | h_v3_a/intron (-0.085) | -0.140 |
| bd-rhapsody | 2 | 1 | -0.157 | -0.180 | -0.133 | h_bd/exon (-0.133) | -0.143 |
| celseq2 | 4 | 2 | -0.138 | -0.160 | -0.118 | h_celseq2/intron (-0.118) | -0.129 |
| dropseq | 3 | 2 | -0.230 | -0.359 | -0.137 | m_dropseq/exon (-0.137) | -0.153 |
| indrop | 2 | 1 | -0.233 | -0.281 | -0.185 | h_indrop/intron (-0.185) | -0.255 |
| marsseq2 | 4 | 2 | -0.151 | -0.273 | -0.069 | h_mars/intron (-0.069) | -0.080 |
| microwell-seq | 2 | 1 | -0.191 | -0.234 | -0.149 | h_microwell/exon (-0.149) | -0.157 |
| quartzseq2 | 2 | 1 | -0.250 | -0.286 | -0.214 | h_quartz/exon (-0.214) | -0.223 |
| splitseq | 2 | 1 | -0.246 | -0.322 | -0.170 | h_split/intron (-0.170) | -0.207 |
| **all** | 69 | 35 | -0.164 | -0.359 | -0.069 | | -0.148 |

#### (b) b_rans16 payload vs today's payload, by species (negative = smaller; original corpus -15.1% pooled, -11..-24% per file)

| species | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| human | 33 | 17 | -0.163 | -0.359 | -0.069 | h_mars/intron (-0.069) | -0.145 |
| macaque | 4 | 2 | -0.139 | -0.159 | -0.128 | mq_v3/intron (-0.128) | -0.136 |
| mouse | 22 | 11 | -0.165 | -0.325 | -0.091 | m_v3_a/intron (-0.091) | -0.145 |
| rat | 4 | 2 | -0.176 | -0.228 | -0.109 | r_v2/intron (-0.109) | -0.158 |
| zebrafish | 6 | 3 | -0.171 | -0.205 | -0.139 | z_v3_a/intron (-0.139) | -0.172 |
| **all** | 69 | 35 | -0.164 | -0.359 | -0.069 | | -0.148 |

#### (c) derivable share of today's file bytes = (names + permutation + ptr + colsums) / file, by protocol (original corpus: 77.9% per sample)

| protocol | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| 10x-arc-gex | 4 | 2 | 0.622 | 0.503 | 0.725 | h_arc/intron (0.503) | 0.617 |
| 10x-visium | 4 | 2 | 0.837 | 0.652 | 0.927 | m_visium/exon (0.652) | 0.807 |
| 10xv2 | 18 | 9 | 0.607 | 0.334 | 0.840 | z_v2/exon (0.334) | 0.575 |
| 10xv3 | 28 | 14 | 0.550 | 0.280 | 0.967 | L_b/exon (0.280) | 0.514 |
| bd-rhapsody | 2 | 1 | 0.622 | 0.476 | 0.768 | h_bd/exon (0.476) | 0.580 |
| celseq2 | 4 | 2 | 0.824 | 0.693 | 0.924 | h_celseq2/intron (0.693) | 0.822 |
| dropseq | 5 | 3 | 0.731 | 0.319 | 0.918 | m_dropseq/exon (0.319) | 0.640 |
| indrop | 2 | 1 | 0.743 | 0.660 | 0.825 | h_indrop/exon (0.660) | 0.729 |
| marsseq2 | 4 | 2 | 0.756 | 0.333 | 0.994 | h_mars/intron (0.333) | 0.653 |
| microwell-seq | 2 | 1 | 0.536 | 0.290 | 0.782 | h_microwell/exon (0.290) | 0.412 |
| quartzseq2 | 2 | 1 | 0.666 | 0.468 | 0.864 | h_quartz/exon (0.468) | 0.599 |
| splitseq | 2 | 1 | 0.874 | 0.809 | 0.939 | h_split/intron (0.809) | 0.873 |
| **all** | 77 | 39 | 0.636 | 0.280 | 0.994 | | 0.578 |

#### (c) derivable share of today's file bytes = (names + permutation + ptr + colsums) / file, by species (original corpus: 77.9% per sample)

| species | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| barnyard | 8 | 4 | 0.690 | 0.568 | 0.915 | bn_v3_a/intron (0.568) | 0.670 |
| human | 33 | 17 | 0.640 | 0.280 | 0.939 | L_b/exon (0.280) | 0.569 |
| macaque | 4 | 2 | 0.431 | 0.310 | 0.598 | mq_v3/exon (0.310) | 0.402 |
| mouse | 22 | 11 | 0.694 | 0.319 | 0.994 | m_dropseq/exon (0.319) | 0.610 |
| rat | 4 | 2 | 0.594 | 0.393 | 0.854 | r_v2/exon (0.393) | 0.544 |
| zebrafish | 6 | 3 | 0.501 | 0.319 | 0.741 | z_v3_b/exon (0.319) | 0.454 |
| **all** | 77 | 39 | 0.636 | 0.280 | 0.994 | | 0.578 |

#### (c2) names (row + column) share of today's file bytes, by protocol

| protocol | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| 10x-arc-gex | 4 | 2 | 0.497 | 0.403 | 0.579 | h_arc/exon (0.579) | 0.493 |
| 10x-visium | 4 | 2 | 0.671 | 0.522 | 0.742 | h_visium/exon (0.742) | 0.646 |
| 10xv2 | 18 | 9 | 0.485 | 0.263 | 0.671 | m_v2_a/intron (0.671) | 0.459 |
| 10xv3 | 28 | 14 | 0.439 | 0.220 | 0.783 | L_a/intron (0.783) | 0.411 |
| bd-rhapsody | 2 | 1 | 0.497 | 0.379 | 0.616 | h_bd/intron (0.616) | 0.463 |
| celseq2 | 4 | 2 | 0.659 | 0.552 | 0.742 | m_celseq2/intron (0.742) | 0.657 |
| dropseq | 5 | 3 | 0.594 | 0.254 | 0.768 | h_dropseq/intron (0.768) | 0.518 |
| indrop | 2 | 1 | 0.596 | 0.526 | 0.666 | h_indrop/intron (0.666) | 0.584 |
| marsseq2 | 4 | 2 | 0.604 | 0.262 | 0.801 | m_mars/intron (0.801) | 0.521 |
| microwell-seq | 2 | 1 | 0.429 | 0.229 | 0.629 | h_microwell/intron (0.629) | 0.328 |
| quartzseq2 | 2 | 1 | 0.557 | 0.382 | 0.732 | h_quartz/intron (0.732) | 0.497 |
| splitseq | 2 | 1 | 0.705 | 0.653 | 0.758 | h_split/exon (0.758) | 0.704 |
| **all** | 77 | 39 | 0.510 | 0.220 | 0.801 | | 0.463 |

#### (c2) names (row + column) share of today's file bytes, by species

| species | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| barnyard | 8 | 4 | 0.555 | 0.454 | 0.735 | bn_dropseq/intron (0.735) | 0.538 |
| human | 33 | 17 | 0.514 | 0.220 | 0.768 | h_dropseq/intron (0.768) | 0.456 |
| macaque | 4 | 2 | 0.341 | 0.244 | 0.476 | mq_v2/intron (0.476) | 0.318 |
| mouse | 22 | 11 | 0.558 | 0.254 | 0.801 | m_mars/intron (0.801) | 0.489 |
| rat | 4 | 2 | 0.474 | 0.312 | 0.684 | r_v3/intron (0.684) | 0.434 |
| zebrafish | 6 | 3 | 0.398 | 0.251 | 0.593 | z_v2/intron (0.593) | 0.360 |
| **all** | 77 | 39 | 0.510 | 0.220 | 0.801 | | 0.463 |

#### Total file ratio, combined projection (vocab id + whitelist/inline barcodes + row counts + b_rans16 payload) / today, by protocol (original corpus about 0.40)

| protocol | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| 10x-arc-gex | 4 | 2 | 0.380 | 0.290 | 0.496 | h_arc/intron (0.496) | 0.384 |
| 10x-visium | 4 | 2 | 0.153 | 0.069 | 0.321 | m_visium/exon (0.321) | 0.181 |
| 10xv2 | 18 | 9 | 0.381 | 0.164 | 0.579 | z_v2/exon (0.579) | 0.408 |
| 10xv3 | 28 | 14 | 0.436 | 0.069 | 0.636 | L_b/exon (0.636) | 0.465 |
| bd-rhapsody | 2 | 1 | 0.381 | 0.263 | 0.499 | h_bd/exon (0.499) | 0.415 |
| celseq2 | 4 | 2 | 0.164 | 0.077 | 0.283 | h_celseq2/intron (0.283) | 0.166 |
| dropseq | 5 | 3 | 0.329 | 0.101 | 0.643 | m_dropseq/exon (0.643) | 0.390 |
| indrop | 2 | 1 | 0.267 | 0.227 | 0.306 | h_indrop/exon (0.306) | 0.273 |
| marsseq2 | 4 | 2 | 0.239 | 0.011 | 0.642 | h_mars/intron (0.642) | 0.337 |
| microwell-seq | 2 | 1 | 0.452 | 0.263 | 0.641 | h_microwell/exon (0.641) | 0.547 |
| quartzseq2 | 2 | 1 | 0.499 | 0.414 | 0.584 | h_quartz/exon (0.584) | 0.528 |
| splitseq | 2 | 1 | 0.175 | 0.114 | 0.236 | h_split/intron (0.236) | 0.176 |
| **all** | 77 | 39 | 0.363 | 0.011 | 0.643 | | 0.415 |

#### Total file ratio, combined projection (vocab id + whitelist/inline barcodes + row counts + b_rans16 payload) / today, by species (original corpus about 0.40)

| species | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| barnyard | 8 | 4 | 0.331 | 0.101 | 0.457 | bn_v3_a/intron (0.457) | 0.351 |
| human | 33 | 17 | 0.370 | 0.069 | 0.642 | h_mars/intron (0.642) | 0.433 |
| macaque | 4 | 2 | 0.534 | 0.394 | 0.631 | mq_v3/exon (0.631) | 0.558 |
| mouse | 22 | 11 | 0.301 | 0.011 | 0.643 | m_dropseq/exon (0.643) | 0.375 |
| rat | 4 | 2 | 0.395 | 0.185 | 0.559 | r_v2/exon (0.559) | 0.435 |
| zebrafish | 6 | 3 | 0.463 | 0.270 | 0.587 | z_v3_b/exon (0.587) | 0.496 |
| **all** | 77 | 39 | 0.363 | 0.011 | 0.643 | | 0.415 |

#### per-file detail

| sample | member | protocol | build | m | n | nnz | file MB | derivable % | names % | container ratio | b payload vs today | combined ratio | A row/bc mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L_a | exon | 10xv3 | GRCm39-2024-A | 386028 | 67271 | 429257 | 6.39 | 88.6 | 71.3 | 0.150 | -23.6% | 0.127 | 1/2 |
| L_a | intron | 10xv3 | GRCm39-2024-A | 307730 | 67271 | 74209 | 4.76 | 96.7 | 78.3 | 0.076 | -32.5% | 0.069 | 1/2 |
| L_b | exon | 10xv3 | GRCh38-2024-A | 310797 | 70726 | 12784194 | 16.84 | 28.0 | 22.0 | 0.738 | -14.6% | 0.636 | 1/2 |
| L_b | intron | 10xv3 | GRCh38-2024-A | 272191 | 70726 | 3784856 | 9.61 | 43.8 | 34.6 | 0.593 | -13.3% | 0.525 | 1/2 |
| bn_dropseq | exon | dropseq | GRCh38-GRCm39-barnyard | 696825 | 25840 | 4120360 | 12.61 | 77.3 | 62.2 | 0.238 | n/a | 0.238 | 1/0 |
| bn_dropseq | intron | dropseq | GRCh38-GRCm39-barnyard | 579921 | 25840 | 668236 | 8.84 | 91.5 | 73.5 | 0.101 | n/a | 0.101 | 1/0 |
| bn_v2 | exon | 10xv2 | GRCh38-GRCm39-barnyard | 696825 | 83275 | 7389676 | 16.49 | 61.5 | 49.4 | 0.404 | n/a | 0.404 | 1/2 |
| bn_v2 | intron | 10xv2 | GRCh38-GRCm39-barnyard | 579921 | 83275 | 2675308 | 11.64 | 72.8 | 58.5 | 0.299 | n/a | 0.299 | 1/2 |
| bn_v3_a | exon | 10xv3 | GRCh38-GRCm39-barnyard | 696825 | 85824 | 4892188 | 16.65 | 61.3 | 49.2 | 0.408 | n/a | 0.408 | 1/2 |
| bn_v3_a | intron | 10xv3 | GRCh38-GRCm39-barnyard | 579921 | 85824 | 6138866 | 15.08 | 56.8 | 45.4 | 0.457 | n/a | 0.457 | 1/2 |
| bn_v3_b | exon | 10xv3 | GRCh38-GRCm39-barnyard | 696825 | 75366 | 8031264 | 16.97 | 59.7 | 47.9 | 0.423 | n/a | 0.423 | 1/2 |
| bn_v3_b | intron | 10xv3 | GRCh38-GRCm39-barnyard | 579921 | 75366 | 3279094 | 11.84 | 71.3 | 57.4 | 0.316 | n/a | 0.316 | 1/2 |
| h_arc | exon | 10x-arc-gex | GRCh38-2024-A | 310797 | 46324 | 1500040 | 6.14 | 72.5 | 57.9 | 0.320 | -12.2% | 0.290 | 1/1 |
| h_arc | intron | 10x-arc-gex | GRCh38-2024-A | 272191 | 46324 | 4173348 | 7.90 | 50.3 | 40.3 | 0.537 | -8.6% | 0.496 | 1/1 |
| h_bd | exon | bd-rhapsody | GRCh38-2024-A | 310797 | 60922 | 4900306 | 9.81 | 47.6 | 37.9 | 0.565 | -13.3% | 0.499 | 1/0 |
| h_bd | intron | bd-rhapsody | GRCh38-2024-A | 272191 | 60922 | 722361 | 5.42 | 76.8 | 61.6 | 0.298 | -18.0% | 0.263 | 1/0 |
| h_celseq2 | exon | celseq2 | GRCh38-2024-A | 310797 | 4096 | 1333900 | 5.31 | 79.1 | 63.0 | 0.215 | -12.5% | 0.190 | 1/0 |
| h_celseq2 | intron | celseq2 | GRCh38-2024-A | 272191 | 4096 | 2242285 | 5.38 | 69.3 | 55.2 | 0.318 | -11.8% | 0.283 | 1/0 |
| h_dropseq | intron | dropseq | GRCh38-2024-A | 272191 | 435682 | 213808 | 6.79 | 91.8 | 76.8 | 0.320 | -35.9% | 0.303 | 1/0 |
| h_indrop | exon | indrop | GRCh38-2024-A | 310797 | 95760 | 2079158 | 7.25 | 66.0 | 52.6 | 0.394 | -28.1% | 0.306 | 1/0 |
| h_indrop | intron | indrop | GRCh38-2024-A | 272191 | 95760 | 599993 | 5.16 | 82.5 | 66.6 | 0.252 | -18.5% | 0.227 | 1/0 |
| h_mars | exon | marsseq2 | GRCh38-2024-A | 310797 | 16383 | 741536 | 5.52 | 76.9 | 61.0 | 0.257 | -12.1% | 0.231 | 1/0 |
| h_mars | intron | marsseq2 | GRCh38-2024-A | 272191 | 16383 | 6732602 | 11.43 | 33.3 | 26.2 | 0.688 | -6.9% | 0.642 | 1/0 |
| h_microwell | exon | microwell-seq | GRCh38-2024-A | 310797 | 117485 | 12227148 | 17.15 | 29.0 | 22.9 | 0.740 | -14.9% | 0.641 | 1/0 |
| h_microwell | intron | microwell-seq | GRCh38-2024-A | 272191 | 117485 | 718911 | 5.65 | 78.2 | 62.9 | 0.305 | -23.4% | 0.263 | 1/0 |
| h_quartz | exon | quartzseq2 | GRCh38-2024-A | 310797 | 692824 | 6597852 | 18.15 | 46.8 | 38.2 | 0.683 | -21.4% | 0.584 | 1/0 |
| h_quartz | intron | quartzseq2 | GRCh38-2024-A | 272191 | 692824 | 509641 | 8.96 | 86.4 | 73.2 | 0.437 | -28.6% | 0.414 | 1/0 |
| h_split | exon | splitseq | GRCh38-2024-A | 310797 | 85694 | 159444 | 5.00 | 93.9 | 75.8 | 0.129 | -32.2% | 0.114 | 1/0 |
| h_split | intron | splitseq | GRCh38-2024-A | 272191 | 85694 | 627825 | 5.23 | 80.9 | 65.3 | 0.262 | -17.0% | 0.236 | 1/0 |
| h_v2_a | exon | 10xv2 | GRCh38-2024-A | 310797 | 87935 | 3077129 | 7.82 | 60.6 | 48.2 | 0.432 | -18.6% | 0.365 | 1/2 |
| h_v2_a | intron | 10xv2 | GRCh38-2024-A | 272191 | 87935 | 1014814 | 5.39 | 78.2 | 62.9 | 0.273 | -12.3% | 0.252 | 1/2 |
| h_v2_b | exon | 10xv2 | GRCh38-2024-A | 310797 | 76221 | 4413066 | 9.38 | 49.9 | 39.5 | 0.533 | -14.6% | 0.464 | 1/2 |
| h_v2_b | intron | 10xv2 | GRCh38-2024-A | 272191 | 76221 | 774980 | 5.23 | 79.5 | 63.7 | 0.255 | -14.1% | 0.232 | 1/2 |
| h_v2_c | exon | 10xv2 | GRCh38-2024-A | 310797 | 24807 | 5775529 | 9.44 | 46.4 | 36.6 | 0.554 | -14.4% | 0.479 | 1/2 |
| h_v2_c | intron | 10xv2 | GRCh38-2024-A | 272191 | 24807 | 489204 | 4.60 | 84.0 | 67.0 | 0.187 | -16.4% | 0.164 | 1/2 |
| h_v3_a | exon | 10xv3 | GRCh38-2024-A | 310797 | 98824 | 3979688 | 8.56 | 56.4 | 45.0 | 0.481 | -12.4% | 0.433 | 1/2 |
| h_v3_a | intron | 10xv3 | GRCh38-2024-A | 272191 | 98824 | 3552487 | 7.64 | 56.7 | 45.5 | 0.484 | -8.5% | 0.453 | 1/2 |
| h_v3_b | exon | 10xv3 | GRCh38-2024-A | 310797 | 77618 | 7798444 | 10.86 | 43.4 | 34.4 | 0.597 | -13.0% | 0.527 | 1/2 |
| h_v3_b | intron | 10xv3 | GRCh38-2024-A | 272191 | 77618 | 2562317 | 6.86 | 61.2 | 49.0 | 0.435 | -10.2% | 0.400 | 1/2 |
| h_v3_c | exon | 10xv3 | GRCh38-2024-A | 310797 | 77373 | 7690782 | 11.75 | 40.2 | 31.8 | 0.626 | -13.8% | 0.547 | 1/2 |
| h_v3_c | intron | 10xv3 | GRCh38-2024-A | 272191 | 77373 | 5350939 | 9.92 | 42.6 | 33.9 | 0.608 | -9.2% | 0.560 | 1/2 |
| h_visium | exon | 10x-visium | GRCh38-2024-A | 310797 | 4991 | 301870 | 4.53 | 92.7 | 74.2 | 0.083 | -21.5% | 0.069 | 1/1 |
| h_visium | intron | 10x-visium | GRCh38-2024-A | 272191 | 4991 | 520401 | 4.30 | 86.7 | 69.4 | 0.149 | -16.2% | 0.128 | 1/1 |
| m_arc | exon | 10x-arc-gex | GRCm39-2024-A | 386028 | 96281 | 2101051 | 8.74 | 67.2 | 53.6 | 0.383 | -23.8% | 0.311 | 1/1 |
| m_arc | intron | 10x-arc-gex | GRCm39-2024-A | 307730 | 96281 | 2819316 | 8.20 | 58.6 | 47.0 | 0.474 | -13.4% | 0.424 | 1/1 |
| m_celseq2 | exon | celseq2 | GRCm39-2024-A | 386028 | 4096 | 629605 | 5.94 | 88.7 | 71.1 | 0.122 | -14.7% | 0.107 | 1/0 |
| m_celseq2 | intron | celseq2 | GRCm39-2024-A | 307730 | 4096 | 250260 | 4.57 | 92.4 | 74.2 | 0.088 | -16.0% | 0.077 | 1/0 |
| m_dropseq | exon | dropseq | GRCm39-2024-A | 386028 | 241982 | 12620291 | 20.85 | 31.9 | 25.4 | 0.730 | -13.7% | 0.643 | 1/0 |
| m_dropseq | intron | dropseq | GRCm39-2024-A | 307730 | 241982 | 1205421 | 7.54 | 73.0 | 59.2 | 0.402 | -19.3% | 0.360 | 1/0 |
| m_mars | exon | marsseq2 | GRCm39-2024-A | 386028 | 9411 | 384352 | 5.69 | 92.9 | 74.5 | 0.081 | -14.0% | 0.072 | 1/0 |
| m_mars | intron | marsseq2 | GRCm39-2024-A | 307730 | 9411 | 14833 | 4.26 | 99.4 | 80.1 | 0.012 | -27.3% | 0.011 | 1/0 |
| m_v2_a | exon | 10xv2 | GRCm39-2024-A | 386028 | 81247 | 3903581 | 9.27 | 62.0 | 49.7 | 0.411 | -17.1% | 0.351 | 1/2 |
| m_v2_a | intron | 10xv2 | GRCm39-2024-A | 307730 | 81247 | 906856 | 5.63 | 83.0 | 67.1 | 0.219 | -14.8% | 0.199 | 1/2 |
| m_v2_b | exon | 10xv2 | GRCm39-2024-A | 386028 | 95392 | 7533447 | 12.40 | 47.2 | 37.7 | 0.556 | -13.9% | 0.486 | 1/2 |
| m_v2_b | intron | 10xv2 | GRCm39-2024-A | 307730 | 95392 | 2097474 | 7.14 | 66.9 | 53.9 | 0.376 | -13.7% | 0.337 | 1/2 |
| m_v3_a | exon | 10xv3 | GRCm39-2024-A | 386028 | 73747 | 4544589 | 9.75 | 58.9 | 47.1 | 0.444 | -12.7% | 0.396 | 1/2 |
| m_v3_a | intron | 10xv3 | GRCm39-2024-A | 307730 | 73747 | 2419391 | 6.94 | 67.2 | 54.2 | 0.373 | -9.1% | 0.347 | 1/2 |
| m_v3_b | exon | 10xv3 | GRCm39-2024-A | 386028 | 79736 | 9784223 | 15.82 | 36.7 | 29.2 | 0.655 | -14.4% | 0.567 | 1/2 |
| m_v3_b | intron | 10xv3 | GRCm39-2024-A | 307730 | 79736 | 2990373 | 8.05 | 58.8 | 47.1 | 0.454 | -12.1% | 0.409 | 1/2 |
| m_v3_c | exon | 10xv3 | GRCm39-2024-A | 386028 | 78712 | 9241054 | 14.79 | 39.2 | 31.2 | 0.631 | -14.6% | 0.546 | 1/2 |
| m_v3_c | intron | 10xv3 | GRCm39-2024-A | 307730 | 78712 | 2287781 | 7.64 | 61.8 | 49.6 | 0.422 | -13.3% | 0.378 | 1/2 |
| m_visium | exon | 10x-visium | GRCm39-2024-A | 386028 | 4991 | 3610087 | 8.13 | 65.2 | 52.2 | 0.362 | -11.7% | 0.321 | 1/1 |
| m_visium | intron | 10x-visium | GRCm39-2024-A | 307730 | 4991 | 355403 | 4.70 | 90.4 | 72.6 | 0.111 | -18.2% | 0.094 | 1/1 |
| mq_v2 | exon | 10xv2 | Mmul_10 | 244671 | 74351 | 6536918 | 9.53 | 38.6 | 30.5 | 0.644 | -13.6% | 0.566 | 1/2 |
| mq_v2 | intron | 10xv2 | Mmul_10 | 209239 | 74351 | 1776687 | 5.36 | 59.8 | 47.6 | 0.451 | -15.9% | 0.394 | 1/2 |
| mq_v3 | exon | 10xv3 | Mmul_10 | 244671 | 81323 | 8455875 | 12.16 | 31.0 | 24.4 | 0.718 | -13.3% | 0.631 | 1/2 |
| mq_v3 | intron | 10xv3 | Mmul_10 | 209239 | 81323 | 3471705 | 7.68 | 43.0 | 34.0 | 0.613 | -12.8% | 0.547 | 1/2 |
| r_v2 | exon | 10xv2 | mRatBN7.2 | 243603 | 93227 | 6707987 | 9.53 | 39.3 | 31.2 | 0.641 | -14.2% | 0.559 | 1/2 |
| r_v2 | intron | 10xv2 | mRatBN7.2 | 213041 | 93227 | 2531804 | 6.06 | 55.5 | 44.5 | 0.499 | -10.9% | 0.455 | 1/2 |
| r_v3 | exon | 10xv3 | mRatBN7.2 | 243603 | 74304 | 2096414 | 6.40 | 57.2 | 45.4 | 0.470 | -22.7% | 0.380 | 1/2 |
| r_v3 | intron | 10xv3 | mRatBN7.2 | 213041 | 74304 | 334866 | 3.83 | 85.4 | 68.4 | 0.210 | -22.8% | 0.185 | 1/2 |
| z_v2 | exon | 10xv2 | GRCz11 | 272316 | 87347 | 8765227 | 12.44 | 33.4 | 26.3 | 0.691 | -17.5% | 0.579 | 1/2 |
| z_v2 | intron | 10xv2 | GRCz11 | 239796 | 87347 | 957140 | 5.06 | 74.1 | 59.3 | 0.316 | -20.5% | 0.270 | 1/2 |
| z_v3_a | exon | 10xv3 | GRCz11 | 272316 | 98959 | 3329249 | 8.24 | 51.5 | 40.8 | 0.530 | -17.9% | 0.450 | 1/2 |
| z_v3_a | intron | 10xv3 | GRCz11 | 239796 | 98959 | 2330634 | 6.99 | 55.3 | 44.1 | 0.500 | -13.9% | 0.446 | 1/2 |
| z_v3_b | exon | 10xv3 | GRCz11 | 272316 | 88399 | 8087519 | 13.19 | 31.9 | 25.1 | 0.709 | -18.5% | 0.587 | 1/2 |
| z_v3_b | intron | 10xv3 | GRCz11 | 239796 | 88399 | 2438187 | 7.01 | 54.3 | 43.2 | 0.505 | -14.1% | 0.448 | 1/2 |


## 6. Generated tables (tools/w4_corpus_report2.py)

### Layout of the bundles (what the members of each sample are)

| members per sample | samples | ids |
|---|---|---|
| 10 | 17 | h_v3_a, h_v3_b, h_v3_c, h_v2_a, h_v2_b, h_v2_c, h_arc, h_bd, h_celseq2, h_mars, h_split, h_indrop, h_quartz, bn_v3_a, bn_v3_b, bn_v2, bn_dropseq |
| 34 | 7 | z_v3_a, z_v3_b, z_v2, mq_v2, mq_v3, r_v2, r_v3 |
| 35 | 3 | m_dropseq, m_mars, m_celseq2 |
| 36 | 2 | h_dropseq, h_microwell |
| 40 | 1 | h_visium |
| 44 | 6 | m_v3_a, m_v3_b, m_v3_c, m_v2_a, m_v2_b, m_arc |
| 46 | 1 | m_visium |

New-layout samples (34-46 members) carry extra text members; mean compressed bytes per sample where present:

| member | samples with it | mean bytes |
|---|---|---|
| ambient_profile.tsv | 20 | 3,581,858 |
| star_Solo.out/Gene/raw/barcodes.tsv | 7 | 5,454,772 |
| star_Solo.out/Gene/raw/matrix.mtx | 7 | 19,124,532 |
| star_Solo.out/Gene/filtered/matrix.mtx | 7 | 11,271,604 |
| cell_qc_metrics.tsv | 20 | 1,211,659 |
| auto_barcodes.tsv | 20 | 475,921 |
| gene_expression.tsv | 20 | 811,306 |
| star_SJ.out.tab | 20 | 1,024,288 |
| splice_events.tsv | 20 | 242,382 |

Across the 37 samples: exon+intron .1pz are 654 MB of 1387 MB of sample members (47.1%).


#### (d) Row-name invariance per (reference build, member)

| build | member | distinct lists | rows | samples per list | note |
|---|---|---|---|---|---|
| GRCm39-2024-A | exon | 1 | 386028 | 11 |  |
| GRCm39-2024-A | intron | 1 | 307730 | 11 |  |
| GRCh38-2024-A | exon | 1 | 310797 | 17 | plus an empty-matrix stub (0 rows, x_stub) that has no names |
| GRCh38-2024-A | intron | 1 | 272191 | 17 | plus an empty-matrix stub (0 rows, x_stub) that has no names |
| GRCh38-GRCm39-barnyard | exon | 1 | 696825 | 4 |  |
| GRCh38-GRCm39-barnyard | intron | 1 | 579921 | 4 |  |
| Mmul_10 | exon | 1 | 244671 | 2 |  |
| Mmul_10 | intron | 1 | 209239 | 2 |  |
| mRatBN7.2 | exon | 1 | 243603 | 2 |  |
| mRatBN7.2 | intron | 1 | 213041 | 2 |  |
| GRCz11 | exon | 1 | 272316 | 3 |  |
| GRCz11 | intron | 1 | 239796 | 3 |  |

#### (f) Barcodes: whitelist membership by protocol (exon member; fractions of columns)

| protocol | samples | barcode length | columns | in 3M-feb-2018 (v3) | in 737K-aug-2016 (v2/5') | in either | worst sample (in either) |
|---|---|---|---|---|---|---|---|
| 10xv3 | 14 | 16 | 1128182 | 1.0000 | 0.0152 | 1.0000 | L_a (1.0000) |
| dropseq | 3 | 12/16 | 703504 | 0.0005 | 0.0001 | 0.0005 | bn_dropseq (0.0000) |
| 10xv2 | 9 | 16 | 703802 | 0.0763 | 1.0000 | 1.0000 | bn_v2 (1.0000) |
| 10x-arc-gex | 2 | 16 | 142605 | 0.0059 | 0.0012 | 0.0071 | h_arc (0.0069) |
| bd-rhapsody | 1 | 27 | 60922 | 0.0000 | 0.0000 | 0.0000 | h_bd (0.0000) |
| celseq2 | 2 | 6 | 8192 | 0.0000 | 0.0000 | 0.0000 | h_celseq2 (0.0000) |
| indrop | 1 | 18 | 95760 | 0.0000 | 0.0000 | 0.0000 | h_indrop (0.0000) |
| marsseq2 | 2 | 7 | 25794 | 0.0000 | 0.0000 | 0.0000 | h_mars (0.0000) |
| microwell-seq | 1 | 18 | 117485 | 0.0000 | 0.0000 | 0.0000 | h_microwell (0.0000) |
| quartzseq2 | 1 | 15 | 692824 | 0.0000 | 0.0000 | 0.0000 | h_quartz (0.0000) |
| splitseq | 1 | 24 | 85694 | 0.0000 | 0.0000 | 0.0000 | h_split (0.0000) |
| 10x-visium | 2 | 16 | 9982 | 0.0014 | 0.0000 | 0.0014 | h_visium (0.0014) |

Cross-sample reuse (fraction of a sample's barcodes that also occur in another sample of the same protocol in this corpus), only protocols with >= 2 samples:

| protocol | samples | mean fraction | min | max |
|---|---|---|---|---|
| 10xv3 | 14 | 0.251 | 0.246 | 0.254 |
| 10xv2 | 9 | 0.593 | 0.582 | 0.626 |
| 10x-arc-gex | 2 | 0.096 | 0.062 | 0.130 |
| 10x-visium | 2 | 1.000 | 1.000 | 1.000 |

#### E4 derivability claims on the new samples (pass / compared)

| claim | item | compared | equal | samples | failing |
|---|---|---|---|---|---|
| colsums | exon_counts | 39 | 39 | 39 |  |
| colsums_frame_bytes_zstd155 | exon_counts | 39 | 24 | 39 | bn_v2:exon_counts, bn_v3_a:exon_counts, bn_v3_b:exon_counts, h_bd:exon_counts, h_microwell:exon_counts, m_arc:exon_counts, m_v2_b:exon_counts, m_v3_a:exon_counts, m_v3_b:exon_counts, m_v3_c:exon_counts, mq_v2:exon_counts, mq_v3:exon_counts |
| ptr | exon_counts | 39 | 39 | 39 |  |
| perm | exon_counts | 39 | 39 | 39 |  |
| colsums | intron_counts | 39 | 39 | 39 |  |
| colsums_frame_bytes_zstd155 | intron_counts | 39 | 19 | 39 | bn_dropseq:intron_counts, bn_v2:intron_counts, bn_v3_b:intron_counts, h_arc:intron_counts, h_bd:intron_counts, h_indrop:intron_counts, h_mars:intron_counts, h_microwell:intron_counts, m_arc:intron_counts, m_v2_a:intron_counts, m_v2_b:intron_counts, m_v3_b:intron_counts |
| ptr | intron_counts | 39 | 39 | 39 |  |
| perm | intron_counts | 39 | 39 | 39 |  |
| colsums | sj_counts | 39 | 39 | 39 |  |
| colsums_frame_bytes_zstd155 | sj_counts | 39 | 23 | 39 | L_b:sj_counts, bn_v2:sj_counts, bn_v3_b:sj_counts, h_bd:sj_counts, h_microwell:sj_counts, m_v2_b:sj_counts, m_v3_a:sj_counts, m_v3_b:sj_counts, m_v3_c:sj_counts, m_visium:sj_counts, mq_v2:sj_counts, mq_v3:sj_counts |
| ptr | sj_counts | 39 | 39 | 39 |  |
| perm | sj_counts | 39 | 39 | 39 |  |
| colsums | splice_psi | 35 | 35 | 35 |  |
| colsums_frame_bytes_zstd155 | splice_psi | 35 | 31 | 35 | h_bd:splice_psi, h_microwell:splice_psi, m_arc:splice_psi, m_v2_b:splice_psi |
| ptr | splice_psi | 35 | 35 | 35 |  |
| perm | splice_psi | 35 | 35 | 35 |  |
| psi_structure |  | 35 | 35 | 35 |  |
| psi_values_rint |  | 35 | 8 | 35 | bn_dropseq:splice_psi, bn_v2:splice_psi, bn_v3_a:splice_psi, bn_v3_b:splice_psi, h_arc:splice_psi, h_bd:splice_psi, h_dropseq:splice_psi, h_indrop:splice_psi, h_mars:splice_psi, h_microwell:splice_psi, h_visium:splice_psi, m_arc:splice_psi |
| psi_values_trunc |  | 35 | 30 | 35 | h_celseq2:splice_psi, h_quartz:splice_psi, h_v2_a:splice_psi, h_v2_b:splice_psi, h_v3_a:splice_psi |
| cell_calls_file_regen |  | 39 | 35 | 39 | h_bd:cell_calls.tsv, h_indrop:cell_calls.tsv, h_microwell:cell_calls.tsv, m_dropseq:cell_calls.tsv |
| cell_calls_log10 |  | 202625 | 202625 | 39 |  |
| summary_field |  | 663 | 631 | 39 | h_bd:estimated_cells, h_bd:fraction_reads_in_cells, h_bd:mean_reads_per_cell, h_bd:median_genes_per_cell, h_bd:median_umi_per_cell, h_bd:median_umis_per_cell, h_bd:n_cells_called, h_bd:total_genes_detected, h_indrop:estimated_cells, h_indrop:fraction_reads_in_cells, h_indrop:mean_reads_per_cell, h_indrop:median_genes_per_cell |
| pileup_stats_field |  | 195 | 135 | 39 | L_a:exon_hits, L_a:intron_hits, L_a:umi_unique, h_dropseq:exon_hits, h_dropseq:intron_hits, h_dropseq:umi_unique, h_visium:exon_hits, h_visium:intron_hits, h_visium:umi_unique, m_arc:exon_hits, m_arc:intron_hits, m_arc:umi_unique |
| provenance_field |  | 117 | 113 | 39 | h_bd:output.cells, h_indrop:output.cells, h_microwell:output.cells, m_dropseq:output.cells |
| psi_stub |  | 4 | 0 | 4 | L_b:splice_psi, h_v2_c:splice_psi, h_v3_b:splice_psi, h_v3_c:splice_psi |

#### (4) Called cells vs the other columns (exon and intron), nnz share

| protocol | member | samples | columns | called | frac of columns called | frac of nnz in called (mean / min / max) | nnz per called col | nnz per other col |
|---|---|---|---|---|---|---|---|---|
| 10x-arc-gex | exon | 2 | 142605 | 11682 | 0.082 | 0.746 / 0.623 / 0.870 | 224 | 7.5 |
| 10x-arc-gex | intron | 2 | 142605 | 11682 | 0.082 | 0.883 / 0.853 / 0.913 | 532 | 5.9 |
| 10x-visium | exon | 2 | 9982 | 7851 | 0.787 | 0.942 / 0.938 / 0.945 | 468 | 112.2 |
| 10x-visium | intron | 2 | 9982 | 7851 | 0.787 | 0.923 / 0.898 / 0.949 | 104 | 29.6 |
| 10xv2 | exon | 9 | 703802 | 36746 | 0.052 | 0.662 / 0.291 / 0.887 | 990 | 26.6 |
| 10xv2 | intron | 9 | 703802 | 36746 | 0.052 | 0.738 / 0.446 / 0.909 | 279 | 4.4 |
| 10xv3 | exon | 14 | 1128182 | 130733 | 0.116 | 0.637 / 0.287 / 0.905 | 456 | 31.6 |
| 10xv3 | intron | 14 | 1128182 | 130733 | 0.116 | 0.732 / 0.313 / 0.916 | 241 | 9.5 |
| bd-rhapsody | exon | 1 | 60922 | 4541 | 0.075 | 0.710 / 0.710 / 0.710 | 766 | 25.2 |
| bd-rhapsody | intron | 1 | 60922 | 4541 | 0.075 | 0.521 / 0.521 / 0.521 | 83 | 6.1 |
| celseq2 | exon | 2 | 8192 | 1000 | 0.122 | 0.263 / 0.236 / 0.290 | 536 | 198.5 |
| celseq2 | intron | 2 | 8192 | 1000 | 0.122 | 0.280 / 0.244 / 0.316 | 770 | 239.5 |
| dropseq | exon | 3 | 703504 | 2674 | 0.004 | 0.161 / 0.016 / 0.286 | 1305 | 23.2 |
| dropseq | intron | 3 | 703504 | 2674 | 0.004 | 0.214 / 0.017 / 0.326 | 217 | 2.1 |
| indrop | exon | 1 | 95760 | 1572 | 0.016 | 0.227 / 0.227 / 0.227 | 301 | 17.1 |
| indrop | intron | 1 | 95760 | 1572 | 0.016 | 0.368 / 0.368 / 0.368 | 141 | 4.0 |
| marsseq2 | exon | 2 | 25794 | 1000 | 0.039 | 0.210 / 0.127 / 0.293 | 207 | 37.1 |
| marsseq2 | intron | 2 | 25794 | 1000 | 0.039 | 0.212 / 0.124 / 0.299 | 841 | 238.2 |
| microwell-seq | exon | 1 | 117485 | 3902 | 0.033 | 0.165 / 0.165 / 0.165 | 518 | 89.9 |
| microwell-seq | intron | 1 | 117485 | 3902 | 0.033 | 0.161 / 0.161 / 0.161 | 30 | 5.3 |
| quartzseq2 | exon | 1 | 692824 | 500 | 0.001 | 0.018 / 0.018 / 0.018 | 235 | 9.4 |
| quartzseq2 | intron | 1 | 692824 | 500 | 0.001 | 0.016 / 0.016 / 0.016 | 16 | 0.7 |
| splitseq | exon | 1 | 85694 | 424 | 0.005 | 0.133 / 0.133 / 0.133 | 50 | 1.6 |
| splitseq | intron | 1 | 85694 | 424 | 0.005 | 0.170 / 0.170 / 0.170 | 251 | 6.1 |
