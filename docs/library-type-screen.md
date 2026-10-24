# Which atlas samples are not gene-expression libraries?

Screen for the v2 migration (2026-10-23). The empty-droplet study (`docs/empty-droplets.md`, draft PR #17) found that the v2 pilot
GSE145370 holds TCR-enriched V(D)J libraries processed as gene expression. This screen looks for the same mix-up across the whole
atlas: V(D)J (TCR/BCR), hashtag (HTO), antibody (ADT/CITE-seq), CellPlex/MULTI-seq (CMO), CRISPR guide libraries, and also ATAC and
other DNA libraries, deposited next to the mRNA libraries and processed as RNA.

Code and outputs: `bench/libtype_screen/`. Read-only: D1 and R2 were only read; nothing in the pipeline changed.

* `libtype_suspects.tsv`: one row per flagged sample (10,906 rows): class, confidence, recommendation, the matching mRNA GSM and
  whether it is in the atlas, GEO title and library strategy, the atlas QC numbers, and the text evidence (`field:CLASS[words]`).
* `libtype_by_study.tsv`: counts per study (atlas samples, flagged by confidence, classes, mRNA siblings in/not in the atlas).

## Result

**8,824 of the atlas's 71,356 samples (12%), in 982 of 8,236 studies, are flagged at high or medium confidence.** The four known
cases are all caught: GSE145370 GSM4317423-36 (14/14 VDJ, high), GSE108313 GSM2895283/84 and GSM3501447 (HTO/ADT, high),
GSE256089's 16 odd GSMs (HTO, high), GSE213957 GSM6596914/15/16/18/19/20 (ADT/HTO/guide, medium), plus 12 more ADT/HTO/GDO lanes in
GSE213957 (GSM6596925-36). None of the GEX siblings in those studies is flagged.

| class | high | medium | studies | pass atlas QC (tier 0) | mRNA sibling in atlas | no sibling found | mRNA sibling not in atlas |
|---|---|---|---|---|---|---|---|
| VDJ (TCR/BCR) | 3,636 | 271 | 369 | **3,059** | 2,427 | 1,121 | 359 |
| ADT / CITE-seq | 1,935 | 401 | 261 | 92 | 1,487 | 572 | 277 |
| HTO | 1,298 | 161 | 316 | 48 | 764 | 524 | 171 |
| guide (sgRNA/GDO) | 517 | 99 | 71 | 37 | 343 | 242 | 31 |
| CMO / MULTI-seq | 197 | 63 | 75 | 6 | 106 | 141 | 13 |
| ATAC | 125 | 14 | 38 | 98 | 72 | 32 | 35 |
| other DNA (ChIP, CUT&Tag, WGS ...) | 83 | 24 | 10 | 60 | 20 | 81 | 6 |
| **total** | **7,791** | **1,033** | **982** | **3,400** | **5,219** | **2,713** | **892** |

What the table says:

1. **Feature-barcode libraries are mostly already hidden; V(D)J libraries are not.** HTO/ADT/CMO/guide reads do not land in
   transcripts, so 4,488 of 4,671 such samples already fail the atlas QC (tier 1/2: median genes < 30 or mapping < 20%). V(D)J
   amplicons map well (median 69% of reads) to the TR/IG loci and give a median of 57 genes and 98 UMIs per called cell, above the
   tier-1 cut of 30 genes, so **3,059 of 3,907 V(D)J libraries are tier 0**: they are in the downloadable set, the search counts and
   the headline cell numbers. Over all classes, 3,400 flagged samples are tier 0 and hold 9.0 M called cells, 3.3% of the 274.8 M
   cells in tier-0 samples. GSE145370 is one of 369 such studies, not an oddity.
2. **126 studies have every atlas sample flagged** (795 samples; 38 HTO-only, 29 VDJ-only, 21 ADT-only, ...): the bundle holds no
   gene-expression library at all.
3. **892 flagged samples have an mRNA sibling that is not in the atlas** (701 distinct mRNA GSMs in 203 studies; the pilot's
   GSM4317409-22 are among them). 5,219 have their mRNA sibling in the atlas already.

## Recommendations for the v2 migration

1. **Exclude high and medium samples from the gene-expression migration** (8,824). Do not write them as GEX sample objects; list
   them in the study manifest as excluded with the class, so the counts stay explainable. Where the mRNA sibling is in the atlas
   (5,219) nothing is lost.
2. **Replace where the mRNA library exists but was not processed**: queue the 701 mRNA GSMs (`recommendation` = `replace with mRNA
   library GSMxxx`) for the pipeline before or with the migration of their studies. Withhold the 126 all-flagged studies until then
   (most of them have no mRNA sibling in the same GEO series; their GEX data may sit in a sister series, which this screen does not
   follow).
3. **Pick a gene-expression pilot.** Numbers measured on GSE145370 (sizes, entries in uncalled barcodes) describe amplicon libraries.
   Re-run the pilot on the mRNA libraries GSM4317409-22 or on another study.
4. **Keep, but check before migrating, the `+GEX` and `cleared` rows** (86 low-confidence `+GEX` rows and 456 `cleared` rows):
   titles such as "GEX and ADT" or "RNA (D1) + HTO (D701)" mean one GSM may list several libraries, and the pipeline would pool their
   runs into one sample. `cleared` rows have library words in GEO but QC numbers of a gene-expression library (>= 200 genes, >= 10%
   exonic, >= 50% mapped): most are GEX libraries of hashed/CRISPR/CITE-seq experiments; a few may be pooled GEX+feature runs. The
   SRA run library names would settle both; that is not done here.
5. **Low confidence is review-only.** 1,323 of the 1,626 low rows are `UNSPECIFIED_FB`: a sample with feature-barcode-like QC in a
   series that names non-GEX libraries, without words of its own. Most are failed samples rather than a library-type question; 733
   are GeoMx DSP regions of GSE163530 (spatial profiling, not single-cell). Leave them to the QC tiers.
6. **Gate this at ingest, not after.** The pipeline already refuses feature-barcode-only input (`src/pipeline/singlet.cpp`
   `is_feature_barcode_only`), but only when the `.1fq` header says `CITE_SEQ_ADT`, which evidently was not set for these samples.
   A metadata check (GEO title words and `!Sample_library_strategy`, as in `screen.py`) before download would have stopped most of
   them. QC alone is not enough for V(D)J: 7,284 tier-0 samples have the V(D)J-like QC pattern (>= 35% mapped, < 300 genes, not
   GEX-like) without any library words, most of them shallow or low-quality GEX libraries. Text and QC together are what separate
   them.

## Method

Inputs, all read-only:

* **Atlas QC** for every sample with a `sample_qc` row in D1 (71,356 samples, 8,236 studies), joined to `gsm`
  (`wrangler d1 execute singlet-catalog --remote`, paged by `gsm_id`, 142,664 rows read in total). D1 `gsm.title` holds the series
  title, so sample titles come from GEO.
* **GEO sample records**: the family SOFT file of every atlas series from `ftp.ncbi.nlm.nih.gov/geo/series/` (8,234 of 8,236; GSE293426
  and GSE301966 return 404), parsed into 155,093 sample records (every sample of each series, in the atlas or not): title, source,
  description, library strategy/source/selection, molecule, characteristics, supplementary file names, SRA links.
* **Bundle member sizes** (`vdj_gene_usage.1pz` vs `exon_counts.1pz` compressed bytes) from the ZIP central directories of the 395
  studies with a V(D)J flag plus 80 random studies (range reads on data.singlet.bio). This signal does not separate V(D)J libraries:
  median vdj/exon byte ratio 0.060 in high-confidence V(D)J samples against 0.041 in unflagged GEX-like samples. It is reported in
  the TSV (`vdj_exon_ratio`) and not used. The fraction of UMIs on TR/IG genes inside `exon_counts` would be the content-level
  signal; it needs the matrices and was not computed.

Text evidence (`screen.py: classify`). Library words per class are matched on letter boundaries in the sample title (weight 3),
`!Sample_library_strategy` (ATAC-seq, ChIP-Seq, WGS ... weight 3), library selection/source/molecule (2), description lines that
are not file names (2), characteristics whose key names a library/assay (2) and supplementary file names (1, e.g.
`filtered_contig_annotations`, `HTO`, `ADT`). Words that only describe biology are removed before matching (anti-TCR, TCR-transgenic,
TCR KO, OT-I ...). The title decides first: a title that names a non-GEX library and no GEX library (GEX, mRNA, RNA, gene expression,
scRNA-seq ...) is the strongest evidence; when a title names both, the sample counts only if the QC agrees, or as `+GEX` when the
title joins them ("GEX and ADT", "RNA + HTO").

QC fingerprint (`screen.py: qc_pattern`), from the atlas numbers:

| pattern | rule | what it is |
|---|---|---|
| `gex` | >= 200 genes per cell, >= 10% exonic reads, >= 50% mapped | a gene-expression library (5th percentile of exonic fraction over samples with no library words and >= 300 genes: 0.075) |
| `fb` | mapping < 20% and < 30 genes, or < 5% exonic and < 50 genes | feature barcodes: reads do not land in transcripts (median 1 gene per cell for title-HTO/ADT/CMO samples) |
| `vdj` | >= 35% mapped, < 300 genes, not `gex`/`atac` | amplicon on the TR/IG loci; the constant-region exons make 10-40% of reads exonic, so exonic fraction alone does not separate it |
| `atac` | >= 50% mapped, >= 300 genes, < 7% exonic | genomic fragments over gene bodies (median 724 "genes" in title-ATAC samples) |

Confidence: **high** = title (or library strategy, or description/characteristics with no GEX word in the title) names the class
and the QC pattern agrees; **medium** = the title names the class and the QC neither agrees nor looks like GEX (often intermediate
numbers: HTO libraries with some spurious mapping, V(D)J with more genes), or the title names both and QC agrees; **low** = weaker
combinations (description-only without QC agreement, file names only, `+GEX`, QC-only `UNSPECIFIED_FB`); **cleared** = library
words but `gex` QC (kept, listed for review).

Matching mRNA GSM: in the same series, the sample whose title is the same after removing library and GEX words ("S133_TCR Sample 1" /
"S133_mRNA Sample 1"; "... ART, HTO" / "... ART, GEX"), nearest GSM number first; otherwise the closest GEX-worded title by
string similarity (>= 0.75 and clearly ahead of the next). Matching does not follow sister series of a SuperSeries, so `exclude`
without a sibling can still have an mRNA library elsewhere in GEO.

Checks: all 39 known suspect GSMs flagged, none of their 23 GEX siblings; 60 randomly drawn high-confidence rows all named a non-GEX
library in their GEO title or description on reading; medium rows were read by class (V(D)J, ADT, HTO, guide) and were mostly correct,
with the errors being GEX libraries of perturbation or hashing experiments whose QC sits just below the `gex` cut.

Reproduce (laptop is enough, ~40 min, mostly downloads): export D1 to `d1_qc.parquet` (columns of `sample_qc` plus `gsm.source,
characteristics, modality, organism`), then `python -P geo_fetch.py`, `python -P parse_soft.py`, optionally
`python -P cd_scan.py <gse list>`, and `python -P screen.py`; set `LIBTYPE_WORK` to keep caches out of the checkout.
