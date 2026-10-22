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
