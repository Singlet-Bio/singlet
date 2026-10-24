# w4_bugs: defects surfaced by the compression study, root-caused and reproduced

Written 2026-10-22 (hard stop 07:40 EDT). Production sources were only read; patches are files in `results/w4_bugs_patches/`
(unapplied; each was tested against a scratch copy or a driver, see the test line under each). Scratch data and extracted bundle
members are in `corpus_w4_bugs/`. Tools: `tools/w4_bugs_*.py`; C++ drivers `harness/w4_bugs_*.cpp` (build:
`bash build.sh harness/w4_bugs_psi.cpp`; `w4_bugs_fuzz` and `w4_bugs_edge` were built by hand with `-D_GLIBCXX_ASSERTIONS`).
Nothing was sent to any production service. "Verified" = I ran it here; "derived" = follows from code reading or from the
coordinator's production observations; "not done" is stated plainly in section 9.

## 0. Prioritised list (severity = user impact x breadth)

| # | defect | severity | state | fix |
|---|---|---|---|---|
| 1 | Migration rewrites bundles but never rebuilds the website's member index (stale offsets, size, ETag) | **critical** (half the atlas; partial downloads return wrong bytes / 502 / 416) | confirmed in production by the coordinator; code cause located here | `results/w4_bugs_patches/restrict_published_bundles_index_refresh.patch` + website guard (section 1) |
| 2 | Every shipped `splice_psi.1pz` is `rint(PSI)` in {0,1}; in-tree writer would store `trunc(PSI)` | **high** (a whole modality is a binary indicator; 7.8% of GSM4317429 entries are explicit stored zeros) | verified, 14/14 samples | `results/w4_bugs_patches/psi_writer_and_donor_priority.patch`; regenerate from `sj_counts.1pz` (section 2) |
| 3 | `summary.json` exonic/intronic/intergenic fractions divide UMIs by reads: "intergenic" is 94.8-99.0% in all 14 GSE145370 samples | **medium** (misleading QC on every sample) | verified from the files + `export.h:1336-1343` | section 5 |
| 4 | GSE145370: 13 of 14 samples have 50-77% of reads flagged `wrong_strand` and a called-cell median of 33-114 exon UMIs (GSM4317429: 273,090 exon UMIs for 3,820 called cells from 20.3 M barcoded reads) | **possibly high for that study** | **unverified hypothesis** (needs the BAM / STAR log); the numbers are verified | section 6 |
| 5 | `docs/format-1pz.md` and `docs/format-spec.md` describe a different file format than the code writes (20 discrepancies, section 7) | **medium** (an implementer following the docs cannot read a single file) | verified against writer, reader and shipped headers | section 7 |
| 6 | `cell_calls.tsv` never holds an EmptyDrops result; column `empty_drops_pvalue` holds the BH FDR (or a sentinel 0); docs say otherwise | **low-medium** (semantics/docs) | verified 21/21 | docs fix (section 3) |
| 7 | `pz_writer.h` is memory-unsafe on invalid input (`max_val + 1` wraps at 2^32-1, unchecked row indices, float NaN) | **low** today (the pipeline passes uint16, or float in [0,1]); medium for a user-facing `write_1pz` | verified with assertions on | `pz_writer_input_validation.patch` (section 8) |
| 8 | `compute_psi()` comment says donors win; code takes the first event in bytewise string order (differs on 0-8 junctions per sample) | **low** | verified | in the PSI patch |
| 9 | `splice_events.tsv` is advertised as a modality (`bundle.py:213`) but the packer never includes it; `bundle.splice_psi()` docstring says "cells x events" but the columns are junctions | **low** | verified (143 members, none) | section 2 |
| 10 | Reader: file CRC and every chunk CRC are verified; 13,812 mutations: no crash, no hang | **no defect found** | verified | section 4 |

## 1. Item 5b: restricted-sidecar migration vs the website's cached index

### What `scripts/restrict_published_bundles.py` updates, and what it does not
* Updates: the public R2 object `data/<GSE>/<GSE>.singlet` (`upload`, line 278) and the private sidecar object (line 258); a local
  JSONL ledger (lines 188-191; records at 264, 266, 286, 288).
* Never updates: the D1 `bundle_index` row (offset, size, method per member), the per-sample chunk-offset index, any catalog
  size/ETag column, any CDN/Cache-API entry. The script has no HTTP client and no website token; its docstring (lines 6-14) lists four
  steps and none is "refresh the index". The sbatch wrapper only runs the Python script. So "done" in the ledger (lines 288-303) means
  "object verified in R2", not "servable".
* Why every offset moves, not only those after the removed members: `_restricted.py:127-130` rewrites each kept member with a fresh
  `zipfile.ZipInfo(name, date_time=...)` and `zout.open(zi, "w", force_zip64=True)`, i.e. a new local header with a ZIP64 extra field, and
  the removed members shrink the front of the archive. The manifest is rewritten too (profile `public-1`). Size, ETag (a new multipart
  ETag), Last-Modified, central directory and every data offset change.
* What a client sees with stale offsets (observed by the coordinator for GSE111113: index says 324,676,242 B / 215 entries, the object is
  170,777,036 B / 179 members): `/entry` 502; the range recipe for `exon_counts.1pz` returns bytes `06 d5 3d 23`, not `TP1Z` (wrong bytes with
  HTTP 206). Derived, not observed: a recipe whose start is >= the new size gets 416; a range straddling the new end returns a short 206 body;
  a client that checks the magic (`pz_reader.h:282`) or the file CRC (`pz_reader.h:345`) fails there, one that does not gets garbage.

### Code review (file:line)
| where | finding | risk |
|---|---|---|
| `restrict_published_bundles.py:278` | the overwrite of the public key has no precondition: the ETag was compared once (255), before the sidecar upload and its SHA-256 read-back, which can take tens of minutes for a 30 GB study. A concurrent re-pack of the same GSE in that window is silently replaced by a redaction of the older bytes | lost update |
| `:278-304` | after the overwrite there is no index step, so the study is "done" while stale | critical (this finding) |
| `:283` | the rollback `store.upload(src, ...)` is neither verified nor guarded: if it raises, the outer handler (381) logs a generic FAILED and its comment "nothing public was changed" (383) is false; the public object may be the new file or a half-restored one | integrity |
| `:264`, `:286` | ledger stage names do not say which of {old, new} the public object is after a failure | operability |
| `:322-324` (`restore`) | the rebuilt complete bundle is uploaded over the public key and checked by size only (migrate reads back a SHA-256); restore also leaves the index stale and re-publishes the unredacted members | integrity + same bug |
| `_restricted.py:223`, `:288-289` | members absent from the source manifest's `checksums` are not verified (`if name in vouched`), and `_verify_pair` returns early when the original manifest has no checksums: such a bundle is split without verifying anything | silent acceptance |
| `_restricted.py:198-226` | duplicate member names in a zip are not detected | edge |
| keep | sidecar uploaded and SHA-256 verified before the public object is touched (258-265); `.part` + `shutil.move` locally (`_restricted.py:173,252`); idempotent re-run (a split bundle reports 0 restricted members, line 236); shard by accession order (344) | |

### Safe procedure (minimal)
1. Per study, in the same job right after the verified overwrite: call the website's index refresh for that GSE (needs the ingest token, so
   pass it to the job and give the script `--refresh-cmd`), then read one member back **through the index** and compare its SHA-256 with
   `manifest.json`, and compare the index's `bytes` with the object's `ContentLength`. The patch does exactly this (`refresh_and_verify_index`),
   marks the study `INDEX-STALE` (exit 1, ledger stage `index-stale`) when either check fails, and never writes `done` before it passes.
2. Repair what is already rewritten: list `data/*/*.singlet` with `LastModified` and rebuild the index for every object whose `LastModified` is
   later than its `indexed_at` row (the coordinator's 6 of 6 rewritten vs 2 of 2 untouched suggests this is the 2026-10-20/21 set).
3. Website side (`functions/_shared/bundle-reader.ts`, not in this repo, not read): `getBundleIndex` should compare the cached row's `bytes` and
   ETag with a HEAD of the live object, or store the ETag in the row and send `If-Match` on every ranged GET so R2 answers 412 instead of other
   bytes; on mismatch rebuild or return 503, never serve offsets of a different object.
4. Every further bundle rewrite (section 2's PSI regeneration, a container-v2 migration) must use the same hook; do them in one pass.

Test (scratch copy, local store, stub HTTP API: `corpus_w4_bugs/patch_test/make_and_test.py scripts/restrict_published_bundles.py`): no
`--refresh-cmd` -> `INDEX-STALE`, exit 1; a refresh that leaves the index at the old size -> `INDEX-STALE`; a refresh that works ->
`split 46128 -> public 5898 + sidecar 40721 bytes`, exit 0, ledger has `done`. The unpatched script reports success in the first scenario by
construction (line 304).

## 2. Item 1: `splice_psi.1pz` holds `rint(PSI)` in {0,1}  [verified 14/14]

Commands: `bash build.sh harness/w4_bugs_psi.cpp; python -P tools/w4_bugs_psi_all.py; python -P tools/w4_bugs_psi_recipe.py`
(`results/w4_bugs_psi_all.tsv`, `results/w4_bugs_psi_recipe.json`).

* Writer path: `export.h:1120-1127` `compute_psi()` (float32 PSI in (0,1]) -> `atomic_write_1pz` (`atomic_io.h:92`) -> `write_1pz<float>`;
  `pz_writer.h:615` `u_data[k] = static_cast<uint32_t>(data[k])` truncates, so only PSI == 1.0 exactly would survive as 1 (the value type
  is then `uint8`, vt_code 1, `pz_writer.h:618`). The line has been identical since 7e0a1ee (2026-04-28).
* What is shipped (GSE145370, 14 samples, 4,664,595 entries): values are only 0 or 1; PSI is exactly 1.0 in 89.3-97.3% of entries; the
  float PSI has 237-606 distinct values per sample, the member keeps one bit. Stored zeros are explicit entries: 28,883 of 372,571 (7.8%) in
  GSM4317429, which `eliminate_zeros()` (scanpy, scipy) turns into "junction not observed", losing the difference between "observed, PSI <= 0.5"
  and "absent".
* Shipped vs what the in-tree writer produces from the same sj matrix, measured with my driver on the production `compute_psi` and `write_1pz`:
  they differ in **56,947 of 4,664,595 entries (1.22%; 0.45-2.99% per sample)**; every difference is shipped = 1, in-tree = 0, PSI in
  (0.5, 1). Shipped equals `nearbyint(PSI)` (round half to even: PSI = 0.5 is stored as 0) on 4,664,595 / 4,664,595; `floor(x+.5)` is wrong
  on the 0.5 cases (3.08% of GSM4317429), `ceil` on 7.75%.
* Which writer produced it: the shipped bytes equal `write_1pz<float>(nearbyint(PSI))` **byte for byte on 14/14 files with the headers of this
  tree and 14/14 with the headers of git 8e1803b** (the `git_sha` in all 14 `summary.json`, pipeline 0.3.0, run 2026-06-05; `provenance.json`
  has `singlet_git_sha: unknown`, built under `/mnt/home/debruinz/Singlet-AI/singlet/build`); the truncating variant matches 0/14. There is
  no rounding step anywhere in the repository history (checked with `git log -S` on `nearbyint`/`lround` and by reading the cast at every
  commit), so the production binary was built from a working tree with a local change that was never committed. The fix must therefore be
  committed, not assumed present.
* Consumers: `bundle.splice_psi()` (`bundle.py:1747`) returns the member as uint32 with `adata.X` in {0,1}; R `singlet_read(..., "splice_psi")`
  likewise. Nothing in the docs states the scale (`format-singlet.md:186` says "percent-spliced-in"; `derive/psi.py` is a skeleton whose
  docstring defines PSI differently, inclusion / (inclusion + exclusion)).
* Recomputation is exact (independent of E4): my numpy implementation of `splice_psi.h` (events = donor `chrom:start-D` / acceptor `chrom:end-A`
  groups with >= 2 junctions, sorted bytewise, a junction takes the first event, PSI = float32(count) / float32(event sum)) is **bit-identical
  to the C++ `compute_psi` on 4,664,595 entries, 14/14 samples** (`numpy_vs_cpp_bit_identical`). Event sums are integers < 2^24.
* Fix (`psi_writer_and_donor_priority.patch`, `git apply --check` passes on HEAD): store `lround(PSI * 10000)` as uint16 (basis points, the
  convention `mt_heteroplasmy.1pz` already uses at `export.h:1178`) and write `value_scale=10000` and `value_kind=psi` into the member's
  key/values. Test: driver `harness/w4_bugs_psi_fix.cpp` + `tools/w4_bugs_psi_fixcheck.py` on GSM4317429, GSM4317432, GSM4317427: vt_code 2,
  max |stored/10000 - PSI| = 5.0e-05, 237-606 distinct stored values, no explicit zeros at these samples, member 1.69 MB vs 1.64 MB shipped
  (+3%).
* Migration of published files: regenerate `splice_psi.1pz` from the bundle's own `sj_counts.1pz` with the fixed writer. It is lossless
  because the recipe is pinned and exact: the shipped files are reproduced byte for byte from `sj_counts.1pz` alone (14/14, above), and the
  float PSI is reproduced to 0 ulp. Caveats: (a) the donor-priority change (section 8, item 8) changes the stored value of 0-8 junctions
  per sample (<= 58 entries; 8 entries changed value in GSM4317432); decide on the rule and record it in the member's key/values, (b) the
  bundle rewrite changes all offsets, so it needs the index hook of section 1, (c) alternative: stop shipping the member and derive it on read
  (E4: 13% of a sample), which needs the recipe in the client; either way the 14 published `splice_psi` members are wrong today.
* Related (item 3 of the brief): `compute_psi()` `splice_psi.h:105-106` says "donor events take priority over acceptor events due to sorting
  order". Event ids are compared as strings, so `"chr1:100007033-A" < "chr1:99993741-D"`: an acceptor event wins whenever a junction's end
  has more digits than its start and sorts earlier (junctions that cross a power of ten: chr1:99993741-100007033, chr12:991322-1027747,
  chr10:99972134-100009837, chr4:987949-1000611, ...). In GSM4317429: 9,293 junctions belong to both a donor and an acceptor event, **1 differs**
  (`chr1:99993741-100007033`, 1 PSI entry, stored value unchanged); over the 14 samples 0-8 junctions, 0-58 entries, and the stored value
  changes for 0-8 entries (GSM4317427: 2, GSM4317432: 8, GSM4317433: 2). The shipped files have the string-order behaviour (they match the
  string-order rule on all 14; the donor-first rule fails on the three samples where a stored value changes). Separately, for the 9,293 + ...
  junctions of GSM4317429 that sit in both a donor and an acceptor event, only one PSI is ever computed (the donor side, for all but the exceptions
  above), while `splice_events.tsv` would list the junction under both events.
  Patch: explicit two-pass assignment (donor events first). Test: patched C++ == my donor-first numpy recipe bitwise on 3 samples.
* `splice_events.tsv` is not packed (`bundle.py` `_TSV_SIDECARS` has only `cell_calls.tsv`; the 143 members of GSE145370 contain none) although
  `MODALITIES["splice_events"]` (`bundle.py:213-217`) and the R `modalities.R:70` advertise it, so a PSI row cannot be mapped to its event
  without re-deriving it from the names.

## 3. Item 2: `cell_calls.tsv` never carries an EmptyDrops result  [verified 21/21]

`python -P tools/w4_bugs_cellcalls.py` (`results/w4_bugs_cellcalls.json`): 7 corpus + 14 bundle samples. Header always
`barcode, is_cell, log10_umi, empty_drops_pvalue, call_method`; `call_method` = `star_solo` 21/21; `is_cell` = TRUE on every row; the p-value
column is `0.0000e+00` on every row; rows == `summary.json:n_cells_called` 21/21; all called barcodes are columns of the exon matrix (0 missing),
and the last called column is at 1.05-1.54x n_called (tumor_3p 3.9x; `called_pos_max` in the json, same as E4).
* Code: `export.h:456-513` prefers STAR's `Solo.out/Gene/filtered/barcodes.tsv` whenever it exists and then skips EmptyDrops entirely
  (`used_star_solo`); it fills the "sentinel" `deviance = 0`, `fdr = 0`, `tested_indices = cell_indices` (493-496). `write_cell_calls`
  (`cell_calling.h:1044`) iterates `tested_indices`, so only called cells appear; and the column named `empty_drops_pvalue` is written from
  `result.fdr[i]` (1051), i.e. a BH FDR even when EmptyDrops does run. Its doc comment (`cell_calling.h:1023`) lists different columns
  (`barcode, total_umi, deviance, fdr, is_cell`). `top_n_fallback` and the knee fallback use the same sentinel. For every 10x-style
  protocol STAR writes `Solo.out`, so the repository's headline EmptyDrops caller (N5, README notebook `cell_calling`) never runs in the
  atlas; the call is STARsolo's filter, recorded only as the string `star_solo`. **Which STARsolo filter is not recorded anywhere in the
  files, and E4's "EmptyDrops_CR" is an assumption from code comments (`export.h:181`, `cell_calling.h:551`)**: nothing in `src/pipeline` or
  `include/singlet` passes `--soloCellFilter`, and the vendored STAR's default is `CellRanger2.2 3000 0.99 10`
  (`src/star/parametersDefault:898`), a UMI-knee threshold, not EmptyDrops. The 1.05-1.54x depth ordering and E4's observation that a single UMI
  threshold predicts the called set to 0.07-3.8% are what a knee filter looks like. The `.1fq` metadata could override the parameter
  (not checked), so this is "probably CellRanger2.2", not proven.
* Docs: `format-singlet.md:29` ("called-cell barcode list (is_cell=True rows)") is right for star_solo but implies FALSE rows exist;
  `format-singlet.md:56` says the file is used "to filter called cells from the full whitelist in .1pz": the .1pz columns are the barcodes the
  pipeline discovered (15k-96k columns, `provenance` whitelist has 737k), not the whitelist.
* Interaction with the compression study's "empties": empties = matrix columns not in the called set = discovered-but-uncalled barcodes
  (barcodes with enough reads to be discovered, `export.h:555`), not the ambient pool; the set is a truncated tail of the ambient
  distribution whose size depends on the discovery threshold, and "called" is a STARsolo decision that cannot be recomputed from the bundle
  (E4: only the column order and log10_umi are derivable). Any codec that treats "empty" as a statistically defined class should define it as
  "not in `cell_calls.tsv`" and record `call_method`.
* Recommendation: documentation fix now (state that `cell_calls.tsv` lists called barcodes only, method `star_solo` = STARsolo's default
  filter, p-value column is a sentinel, columns are discovered barcodes), plus a small code fix: write `NA` instead of `0.0000e+00` when no
  test was run, rename the column `fdr`, and write the method as `star_solo:<soloCellFilter string actually used>` (read it from STAR's
  `Log.out`/parameters; today even the filter type is unrecorded). No migration of data is needed for the docs
  option; a code fix needs new `cell_calls.tsv` members only for studies re-run.

## 4. Item 5c: reader hardening and the CRC claims  [verified]

Driver `harness/w4_bugs_fuzz.cpp` (production `read_1pz_buffer`, `-D_GLIBCXX_ASSERTIONS`, 480 s, seed 1) on `sj_counts.1pz` of GSM4317428
(1.3 MB); results `results/w4_bugs_fuzz_sj428.tsv`, `.summary.txt`. **13,812 mutations: 0 crashes, 0 hangs (>3 s), 0 non-`PZReadError`
exceptions.**
* Raw mutations (CRC left stale): 2,569 bit flips and 1,275 truncations, all rejected. The reader verifies the **file CRC32 before
  parsing** (`pz_reader.h:345`, "file CRC32 mismatch"), the **footer magic and chunk count** (290-293), and **every chunk's CRC32** over the
  packed bytes after zstd decompression (574, "chunk CRC32 mismatch"); zstd itself also verifies the frame checksum the writer enables
  (`ZSTD_c_checksumFlag`, so each chunk is checked twice) and the reader checks the decompressed length. Every failure is a `PZReadError`
  with the message in quotes; nothing is partially returned. 1,240 truncations with a freshly valid footer were also rejected.
* Structural mutations with the file CRC recomputed (so only the structural checks stand): 8 classes x ~1.2k; all rejected except
  (i) 125 header mutations, 14 zeroed ranges, 9 splices, 6 byte flips, 1 metadata flip that were **accepted and decode to the identical
  matrix**: the reader ignores bytes in `_pad0` (27), `transpose_*` (64-79) and `reserved` (80-95) although format-1pz.md says "must be 0",
  and the entire colsums block (checked: the accepted byte/range positions 627,285-640,123 lie inside the colsums block 626,834-641,703;
  the reader never decodes it, only the file CRC protects it); (ii) **425 accepted-but-different, all names**: with a valid
  zstd frame and CRC the metadata TLV can be rewritten, so row/column names (and key/values) are unauthenticated; `indptr`, `indices` and `data`
  never differed in an accepted file. This is the documented limit ("the CRC32 only detects corruption", `pz_reader.h:19-21`), not a bug, but the
  bundle `manifest.json` SHA-256s are the only authentication and are inside the same archive.
* Gaps worth a test: `ReadLimits.max_nnz = 2^29` (`pz_reader.h:72`), `max_rows = 2^22`, `max_cols = 2^24`: the writer has no such caps (indptr
  int32 allows 2^31-1), so a legitimately written file with 2^29 < nnz < 2^31 (or > 4.19 M rows, e.g. 500-bp ATAC tiles over a human genome)
  is unreadable by this reader (derived from code, not tested).

## 5. Item 5d (found while reading): `summary.json` fractions are UMI/read ratios  [verified]

`export.h:1336-1343`: `exonic_fraction = exon_hits / mapped_reads`, `intergenic_fraction = 1 - (exon_hits + intron_hits) / mapped_reads`, where
`exon_hits` is the UMI-deduplicated molecule count (equals `umi_unique` and the exon matrix sum, all 14 samples) and `mapped_reads` is the BAM record
count of `pileup_stats.json` (includes secondary alignments, e.g. 2,131,388 in GSM4317429; and it is a different number from
`summary.json:mapped_reads`, the STAR unique count, 21,510,922). Result: GSM4317429 `exonic_fraction 0.0113`, `intergenic_fraction 0.9838`
(273,090 / 24,205,492); across the 14 samples intergenic is 94.8-99.0%. For comparison `pbmc_v2_typical` (same code, deeper library) shows exonic
0.456 and intergenic 0.421 for the same formula. "Intergenic" here is "not a unique exonic or intronic molecule": duplicates, multimappers, unassigned
and unbarcoded reads are all in it. Also `fraction_reads_in_cells` is a UMI fraction (E4), `mean_reads_per_cell` uses the pileup read count
(24,205,492 / 3,820 = 6,336.5) while `total_reads` says 30,000,000, and `n_input_reads` is exactly 30,000,000 in 10 of 14 samples while
`provenance.input.reads` is 24.2-32.4 M (GSM4317432: 32.4 M > 30 M). Fix: compute fractions from read-level classification counters or relabel
them (`exon_umis_per_mapped_read`); no data migration beyond regenerating `summary.json` (E4: the fields are derivable).

## 6. Hypothesis to check (not verified): strand handling in GSE145370

`pileup_stats.json` of GSE145370 (14 samples): `strand_auto_flipped: true`, `reverse_strand: false`, and `wrong_strand` / `total_reads` =
0.50-0.77 in 13 samples (0.01 in GSM4317428); exon UMIs / barcoded reads = 0.8-3.6%; called-cell median exon UMIs 33-114, whereas
STARsolo's filter called 1,207-4,289 cells per sample. A cell caller that keeps cells at 50-100 UMI is not EmptyDrops_CR's behaviour for a
10x v2 library of 24-32 M reads (about 6,300 reads per cell in GSM4317429, of which 84% are barcoded), while the pipeline's own tcell_5p
  (median 1,149) and pbmc_v2_typical (716) are normal; a knee filter (section 3) would not keep barcodes at 33-114 exon UMIs unless STAR's own
  UMI counts are tens of times higher than the pileup's. Either the pileup
counts are 10-20x too low for this study (strand auto-detection picking the wrong orientation for most reads, so ~60% of reads are discarded
as `wrong_strand`), or the study is genuinely shallow. This needs the STAR `Solo.out` gene counts or the BAM for one sample; I did not
have them. Severity if confirmed: every count matrix of the study (and its PSI/sj/vdj) is depleted, a re-run would be needed.

## 7. Item 5a: documentation vs code for `.1pz` (the code side is the writer `pz_writer.h`, reader `pz_reader.h`, and the shipped headers)

| # | doc location | doc says | code / data says | right side |
|---|---|---|---|---|
| 1 | format-1pz.md L61-73 `vt_code` | 0 float, 1 double, 2 int32, 3 uint32, 4 int16, 5 uint16, 6 uint8 | 1 uint8, 2 uint16, 3 uint32 (`pz_writer.h:618`; reader accepts 1..3 only, `pz_reader.h:298`; R docs `r/src/pz_io.cpp:42`) | code |
| 2 | L73 | "UMI count matrices use uint16 (code 5)" | code is chosen from the file's max value: shipped exon/splice_psi/sj members have vt_code 1; no code 5 exists | code |
| 3 | L149-156 value types | exon/intron uint16, sj uint32, snp uint32, mt float | in-memory type is `SparseAccumulator<uint16_t>` for all (`export.h:247`, values clamp at 65,535, `sparse_accumulator.h:154-156`); file type is by max value; mt_heteroplasmy is uint16 = VAF x 10000 (`export.h:1173-1178`); splice_psi is missing from the table | code |
| 4 | L17-28 layout | header, optional perm, optional metadata, optional colsums, chunks (ptr_chunk + val_chunk), optional transpose, footer | header, perm_z, ptr_z, chunk table (4 B per chunk), chunks, colsums, metadata, footer (`pz_writer.h:694-701`); metadata and colsums come after the chunks | code |
| 5 | L20, L129-141, flag bit 2 | metadata is zstd-compressed JSON with `singlet_version`, `pipeline_mode`, `created` | TLV records (tag 1 rownames, 2 colnames, 3 user key/values; NUL-separated strings; tag 0 end), zstd-compressed (`pz_writer.h:551-581`); shipped `user_kv` is `{}` (verified), so no version stamp exists | code |
| 6 | L24, L43 `ptr_width` | 1, 2 or 4 bytes; pointer block holds offsets into the data block | 2 or 4 only (`pz_reader.h:302`); the block holds per-column nnz counts, not offsets (`pz_writer.h:416-420`) | code |
| 7 | L47-48 | `perm_z_sz`, `ptr_z_sz` may be 0 = absent | both must be non-zero and `FLAG_HAS_PERM` is required (`pz_reader.h:300,330`) | code |
| 8 | L79, L123-125 permutation | maps output column order to input barcode order (cells reordered by density) | a **row** (feature) frequency permutation of length m (descending row nnz, ties by row id); columns keep their order (`pz_writer.h:367-392`) | code |
| 9 | L106-119 VOCSC | per chunk: pointer block + (gap-encoded rows, byte-split, plus byte-split values) | per chunk blob `[ng, msz, zsz, crc, psz][zstd(bitmap(bitplane(gap plane 0) + upper gap planes + varint metadata))]`; rows are grouped by **value** per column (value-ordered CSC), values live as varints in the metadata stream, gaps reset to 0 per value group (`pz_writer.h:276-342, 466-520`) | code |
| 10 | L89 `FEAT_BITPLANE_BITMAP` | "values bitplane-encoded" | bit-planes apply to the low byte of the **gaps**; the bitmap zero-filter applies to the whole pre-filtered buffer (`pz_writer.h:488-502`) | code |
| 11 | L26, L54-56, L64-77 header | optional transpose section | writer never writes one; reader ignores `transpose_*`, `_pad0`, `reserved` (verified by fuzz) although the doc says "must be 0" | code (doc should say "ignored") |
| 12 | L5-7 | "one column decompresses in ~10 us", random column access | the chunk table stores sizes only (no offsets) and a chunk of 1,024 columns must be fully decoded; whole-file CRC is checked first | code |
| 13 | format-spec.md L11-22 (whole page) | 64-byte header, msgpack metadata, separate zstd sections for gene names, barcodes, indptr, indices, data, "layers" (spliced/unspliced, ADT, ATAC), "cell barcodes with QC annotations", Ensembl IDs | a different format entirely: 96-byte header, VOCSC chunks, TLV names; no layers, no QC, no Ensembl IDs in the file | code (page is stale; delete or point to format-1pz.md) |
| 14 | format-singlet.md L177-178 | "pipeline v1 uses TP1Z, the v2 format uses 1PZ02" | 1PZ02 reader removed (`CHANGELOG.md:34`); reader rejects everything but TP1Z version 1 (`pz_reader.h:282-285`) | code |
| 15 | format-singlet.md L29, L56 | `cell_calls.tsv` is used to filter called cells "from the full whitelist in .1pz" | columns are the discovered barcodes, not the whitelist (section 3) | code |
| 16 | format-singlet.md L186; `bundle.py:1747`; `bundle.py:213` | splice_psi is "per-junction percent-spliced-in"; "cells x events"; `splice_events` modality present | values are rint(PSI) in {0,1}; columns are junctions; `splice_events.tsv` never packed (section 2) | code |
| 17 | pipeline.md "What gets written" | `counts.1pz`, `snp.1pz`, `mt.1pz`, `cell_meta.parquet`, `saturation_curve.tsv`; links `CANONICAL_OUTPUT_FORMAT.md` | writer emits `exon_counts.1pz`, `intron_counts.1pz`, `sj_counts.1pz`, `splice_psi.1pz`, `cell_calls.tsv`, ... (`export.h`); `CANONICAL_OUTPUT_FORMAT.md` does not exist anywhere in the repo | code |
| 18 | format-1pz.md L99 footer | CRC32 of the file before the footer | correct (`pz_reader.h:345`); not a discrepancy | -- |
| 19 | format-1pz.md L100-103 footer `_reserved` "must be 0" | reader does not check | code (ignored) | -- |
| 20 | format-1pz.md L83-88 flags | `FLAG_HAS_METADATA` "Metadata JSON block present" | TLV, see 5 | code |
| 21 | `summary.json` / provenance | n/a | see section 5 (semantic mismatch of fractions) | code |
| 22 | `splice_psi.h` header comment L11 "uint16_t counts, PSI float [0,1]" and `export.h` | float PSI is written | file holds uint8 {0,1} | data |

(Rows 18-19 are listed to be explicit about what was checked and found consistent or ignored.) No patch: the docs need a rewrite from the
writer; `format-spec.md` should be deleted or replaced by a pointer.

## 8. Item 5d: silent-wrong-data and memory-safety hunt (reproductions with `harness/w4_bugs_edge.cpp`, assertions on)

| finding | evidence | impact |
|---|---|---|
| `val_rows.resize(max_val + 1)` wraps when a value is 0xFFFFFFFF (`pz_writer.h:309`, `max_val` is `uint32_t`), then `val_rows[v]` writes out of bounds | writing int32 `{-1, 1}` aborts on the `operator[]` assertion; in a release build this is a heap write out of bounds | unreachable from the pipeline (uint16 / float in [0,1]); reachable by any caller of `write_1pz` with negative int32 (the Python binding states it rejects negatives for signed input, `_pz.cpp` ~L49; not re-tested) |
| memory is O(max value) per chunk thread (`val_rows` is a vector of vectors indexed by value): a value of 7e7 allocates 1.7 GB, 4e9 would need 96 GB | value 7e7 round-trips but takes ~9 s and 1.7 GB; value 4e9 aborts | any non-count matrix written with large integers |
| row indices are not validated against `m` (`pz_writer.h:368-370` `rnnz[i_ptr[k]]++`) | index 7 with m = 2 hits the vector assertion | heap out of bounds for malformed CSC |
| `static_cast<uint32_t>(float)` of NaN/negative/>2^32 is undefined behaviour (`pz_writer.h:615`) | `write_1pz<float>` of NaN throws `std::bad_alloc` (the cast gives a huge value); 0.9999 -> 0, 2.5 -> 2 silently | the PSI bug is this cast |
| silent clamp at 65,535 and indptr clamp at INT32_MAX in the accumulator (`sparse_accumulator.h:154-156`, `123-127`) | by reading; no counter or warning | a feature with > 65,535 UMIs in one cell (hemoglobin in erythroid cells, chrM allele depth) is stored as 65,535; derived |
| names with an embedded NUL: the writer emits them, the reader sees extra names and rejects the whole file ("more names than rows/columns"); key/value with NUL in a value: "unterminated value" | reproduced | file unreadable; names from the pipeline do not contain NUL |
| a trailing empty name round-trips; a TLV payload without its final NUL is accepted (`pz_reader.h:240`) | reproduced | benign |
| key/value duplicates: `user_kv[key] = value` keeps the last (`pz_reader.h:257`); a writer from another tool could repeat a key | by reading | benign |
| `ReadResult.indptr` is `uint32` and `nnz` is capped at 2^32 (`pz_reader.h:307`); Python returns `int32` `indptr`/`indices` (`_pz` dtype check above): a matrix with nnz >= 2^31 would wrap in the Python layer | derived (reader caps at 2^29 first) | none today |
Patch `results/w4_bugs_patches/pz_writer_input_validation.patch` (`git apply --check` passes on HEAD): `write_1pz` throws `std::invalid_argument` for an inconsistent `indptr`, a row index >= m, a negative value, and a float that is not a non-negative integer (this last check would have stopped the PSI bug at write time: `atomic_write_1pz` catches it, logs, and `record_failed_write` marks the sample failed), and sizes `val_rows` with `size_t`. Test: `harness/w4_bugs_edge.cpp` built against the patched header (`bin/w4_bugs_edge_fixed.exe x`): int32 -1 -> "negative value", index 7 with m = 2 -> "row index out of range", float 0.9999 -> "float values must be non-negative integers" (unpatched: assertion abort / silent truncation). One more writer/reader asymmetry: duplicate (row, column) entries are written without complaint and produce a file the reader rejects ("decoded row index out of bounds", reproduced; `to_csc` de-duplicates, so the pipeline cannot hit it).

## 9. What I did not finish
* Item 4 (reader profile) was not done: Track A's 52% (sort, dense `tcell_5p` exon) and 86% (names, sparse `tumor_3p` intron) are
  unverified by me. Safe optimisations by code reading (no format change): the per-column `std::sort` on (row, value) pairs at
  `pz_reader.h:648` can be replaced by a merge of the value-group runs (each run is already sorted by permuted rank; mapping through `perm` is
  not monotone, so a counting/radix pass keyed by original row is needed), `bit_planes_decode` is a bit-at-a-time loop over 8 planes per gap
  (`pz_reader.h:183-193`, vectorisable), the CRC is computed twice over the data (file CRC + chunk CRCs), and names are parsed into
  `std::string` one by one (`pz_reader.h:239`) where the Python/R binding then copies them again.
* The 5d hunt was limited to the cases above; the Python and R loaders (`bundle.py` partial-download paths, `_loader.py`) were not read for
  narrowing bugs. The fuzz ran on one file (sj of GSM4317428), not three.
* Nothing was run against production; the coordinator's index observations are quoted, not re-measured.

## 10. Files
`results/w4_bugs_patches/{restrict_published_bundles_index_refresh.patch, psi_writer_and_donor_priority.patch, pz_writer_input_validation.patch}`; `results/w4_bugs_psi_all.tsv`,
`results/w4_bugs_psi_recipe.json`, `results/w4_bugs_cellcalls.json`, `results/w4_bugs_fuzz_sj428.{tsv,summary.txt}`; `tools/w4_bugs_psi_all.py`,
`w4_bugs_psi_recipe.py`, `w4_bugs_psi_check.py`, `w4_bugs_psi_fixcheck.py`, `w4_bugs_cellcalls.py`; `harness/w4_bugs_{psi,psi_fix,fuzz,edge}.cpp`;
scratch `corpus_w4_bugs/` (121 MB of extracted members; safe to delete).
