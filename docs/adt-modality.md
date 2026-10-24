# ADT (CITE-seq) and HTO (hashtag) libraries as a Singlet modality

Status 2026-10-24, ADT session (branch `claude/adt-modality-0637a6`, draft PR into `feat/singlet-v2`). Code: `python/singlet/adt/`
(package), `bench/adt/` (inventory, driver, benchmark, Clipper jobs), tests `tests/python/test_adt_*.py`. Nothing has been written
to R2 or D1; the upload plan for Zach is at the end.

<!-- RESULTS -->

## 1. What was wrong

The pipeline processes every library as RNA. Antibody-capture libraries (ADT / CITE-seq / TotalSeq / feature-barcoding antibody
capture, and HTO cell hashing) deposited next to the gene-expression (GEX) libraries were therefore aligned to the genome. Their
reads do not land in transcripts, so 4,488 of 4,671 such samples already fail atlas QC (`docs/library-type-screen.md`, PR #21).

The pipeline does have antibody code, `include/singlet/pileup/adt_matcher.h` + `adt_counter.h`, wired into `src/pipeline/singlet.cpp`
(T4 pass after the GEX export). It has never run on the atlas: it runs only with `--feature-ref FILE`, which no atlas job passes,
and only for `.1fq` input whose header says `CITE_SEQ_ADT`/`GEX`. Run as wired on a 10x reference library it counts nothing
(benchmark below): it matches the cell barcode exactly against GEX barcodes, without whitelist correction and without the
3' v3 feature-barcode translation, and it looks for the tag only at R2 offsets 0-5, while TotalSeq-B/C barcodes start at offset 10.
UMIs are deduplicated exactly, with no correction.

## 2. Inventory (`bench/adt/inventory.py` -> `bench/adt/inventory/adt_hto_inventory.tsv`)

One row per flagged library (screen classes ADT and HTO, high or medium confidence), with the paired GEX GSM, the GEX protocol, run
accessions and FASTQ sizes from ENA, chemistry words from GEO (TotalSeq-A/B/C, 5', 3' v3, BD AbSeq, hashtag, CITE-seq), candidate
feature-reference files and the authors' own antibody count files.

| class | reason | libraries | studies | FASTQ (TB) | reads (G) |
|---|---|---:|---:|---:|---:|
| ADT | **processable** (10x 3' v2/v3, 5'; GEX sample in the atlas) | **1,340** | 156 | 9.23 | 142.4 |
| ADT | no paired GEX library found by title (440 in studies with 10x GEX samples: paired by barcodes, section 3.4) | 572 | 87 | 3.53 | 81.7 |
| ADT | paired GEX library not in the atlas | 277 | 53 | 2.38 | 47.0 |
| ADT | chemistry not supported yet (BD Rhapsody AbSeq 79, Drop-seq 60, SPLiT-seq 5, MARS-seq 2, Microwell 1) | 147 | 11 | 1.08 | 14.1 |
| HTO | **processable** | **738** | 187 | 3.32 | 64.1 |
| HTO | no paired GEX library found by title (372 in studies with 10x GEX samples) | 524 | 87 | 3.01 | 57.8 |
| HTO | paired GEX library not in the atlas | 171 | 67 | 0.79 | 12.8 |
| HTO | chemistry not supported yet | 26 | 10 | 0.05 | 1.0 |

Human 3,011 libraries, mouse 763, macaque 9. In GEO text, ADT libraries name TotalSeq-C in 371, TotalSeq-B in 209, TotalSeq-A in 202, BD
AbSeq in 189 and 5' chemistry in 918; 721 ADT libraries mention hashtags too. 632 ADT libraries (79 studies) have a supplementary
file that may be a feature reference, 764 have the authors' antibody counts.

Not done here: the 277 + 171 libraries whose GEX library is not in the atlas need that library processed first (the screen's
`replace with mRNA library` list); BD Rhapsody AbSeq needs its own barcode logic and panel list.

## 3. Method

### 3.1 Counting (`singlet.adt.count`): Cell Ranger's feature-barcoding logic, ported

Per library: reads collapse to unique `(cell barcode, UMI, feature window)` keys (a sort, not a hash); the feature window is R2 at
the library's offset, measured from the reads (`panel.detect_offset`: the offset where most windows are catalog barcodes; 0 for
TotalSeq-A, 10 for TotalSeq-B/C). Then, as Cell Ranger does:

* **feature barcode**: exact, else a unique Hamming-1 neighbour (one N tried as each base); ambiguous neighbours are dropped;
* **cell barcode**: exact whitelist match, else the Hamming-1 neighbours weighted by how often each was seen exactly (+0.5), taken when
  its posterior is >= 0.975 (Cell Ranger also weights by base quality; not kept). On 3' v3 TotalSeq-B libraries the corrected
  feature-side barcode is translated to its GEX barcode with 10x's `translation/3M-february-2018.txt.gz`. Whether a library needs the
  translation (and in which orientation) is decided from the reads: the share of sampled barcodes that are GEX columns with and
  without it (0.4% -> 95.7% on GSE248289);
* **UMI**: Cell Ranger's `correct_umis` (`lib/rust/tx_annotation/src/mark_dups.rs`): a UMI moves to its Hamming-1 neighbour with
  the most reads when that has more reads, or as many and is lexicographically larger; one step; UMIs with N are dropped;
* **chemistry**: the whitelist (737K for 3' v2 and 5', 3M-february-2018 for 3' v3) is the one that holds the GEX sample's barcodes
  (100% vs ~1.6%), and the UMI length comes from the R1 length (26 nt = 16+10, 28 = 16+12). The atlas protocol is only a fallback:
  it calls GSE100866's 2017 v2 libraries `10xv3`.

The matrix has the GEX sample's columns, in `cells` table order (`col`): molecules on other whitelist barcodes go to a per-feature
ambient profile.

### 3.2 Benchmark against Cell Ranger (`bench/adt/benchmark.py`)

10x `pbmc_1k_protein_v3` (3' v3, TotalSeq-B, 17 antibodies, 12.6 M read pairs), Cell Ranger 3.0.0 filtered matrix as truth (713
cells, 3,686,238 UMIs), every method run on the same antibody FASTQs and scored on Cell Ranger's barcodes:

| method | UMIs / Cell Ranger | Pearson r, log1p entries | entries identical | per-cell Pearson (median) | time (8 CPU) |
|---|---:|---:|---:|---:|---:|
| **singlet.adt** (panel from the 10x reference) | **1.0002** | **0.999995** | 95.5% | 1.000 | 104 s |
| singlet.adt, panel identified from the reads | 1.0002 (14 of 17 antibodies) | 0.999993 | 95.7% | 1.000 | 92 s |
| kallisto \| bustools (`kb count --workflow kite:10xFB`, kb 0.29.1) | 1.029 | 0.99956 | | 0.9997 | 100 s |
| `adt_counter.h`, offsets 0-5 as wired in `singlet.cpp` | 0 | - | | - | 20 s |
| `adt_counter.h`, offsets 0-10 | 0.00002 | 0.046 | | | 18 s |

alevin-fry was planned as a third standard: the conda env resolved salmon 2.0 (the Rust rewrite), which no longer has
`salmon index --features`; it was not rerun with salmon 1.10 or piscem. kite counts 2.9% more UMIs because bustools does not
correct UMIs by default. The read-identified panel misses three antibodies that are not in the vendor catalog (2020 BioLegend
list); the library is then flagged `incomplete` (29.7% of feature reads on unknown barcodes) rather than counted under guessed names.

**Choice: singlet.adt.** It reproduces Cell Ranger to 0.02% of UMIs, it is pure numpy (no binary to pin), it aligns the matrix to the
GEX sample's `cells` table directly, and it gives the read-level QC numbers Cell Ranger reports. `adt_counter.h` should not be used
for these libraries as it stands (follow-up: port the translation, Hamming-1 cell barcode correction, measured offset and UMI
correction into it, or retire the T4 pass).

### 3.3 Panels (`singlet.adt.panel`)

A panel row is `(id, name, sequence, kind, barcode_id, ensembl, source)`, `kind` in antibody / isotype / hashtag. Sources, in order:

1. **study**: the authors' feature reference, curated under `bench/adt/panels/<GSM or GSE>.csv` (pilot: GSE248289, GSE299416 and the
   six GSE227505 samples), or harvested from GEO supplementary files (`bench/adt/harvest_panels.py`: every small text file the inventory
   names as a possible reference; parsed when it has a barcode column).
2. **catalog**: identified from the reads. 15-mers at the library's offset in reads of GEX-called cells are matched (exact or Hamming
   1) to the TotalSeq catalog (`python/singlet/adt/data/totalseq_catalog.tsv`, 647 sequences: 600 antibodies, 38 hashtags, 9 isotype
   controls; built by `bench/adt/build_catalog.py` from BioLegend's 2020 export and four public feature references). Catalog barcodes
   with >= 50 reads and >= 0.02% of matched reads are the panel. TotalSeq barcode numbers share one sequence across A/B/C, so the
   catalog is keyed by sequence; every sequence names one antigen (its spellings are in `aliases`).

Reads that match nothing are classified: low complexity (NovaSeq poly-G of empty clusters, poly-A), **shifted** (a panel barcode one
base early or late: an indel in the 10-nt handle; Cell Ranger's fixed pattern drops them too) and **other**. Only "other" counts as
evidence of an incomplete panel. A library is **unresolved** (not counted) when no catalog barcode reaches support, and
**incomplete** (counted, reported, `panel_unexplained` rule) when "other" sequences carry > 5% of feature-like reads. Names are never
inferred from anything but sequence identity with a study reference or a vendor/author list.

### 3.4 Pairing with the GEX sample

The screen pairs libraries by title (`S133_TCR Sample 1` / `S133_mRNA Sample 1`). Every library is also checked from the data: the
share of sampled read barcodes that are columns of the paired GEX sample (`barcode_overlap_head`) and the cells/empty-droplet ratio
(QC). Libraries without a titled sibling are paired by barcodes (`run_library.pair_by_barcodes`): the GEX sample of the study whose
called barcodes hold >= 30% of sampled reads and at least 3x the runner-up (`cell_calls.tsv` of each candidate, one small range read
each).

## 4. QC (`singlet.adt.qc`, required before anything is published)

Library rules (`fail` blocks publication; `warn` is published with the flag):

| metric | meaning | fail | warn |
|---|---|---|---|
| `frac_valid_barcode` | reads with a whitelist cell barcode after correction | < 0.50 | < 0.75 |
| `frac_feature` | reads with a panel feature barcode | < 0.20 | < 0.50 |
| `frac_usable_in_cells` | reads in called cells with valid barcode and feature (Cell Ranger "usable") | < 0.05 | < 0.25 |
| `cell_empty_ratio` | median ADT UMIs of called cells / of empty droplets: the pairing check | < 3 | < 10 |
| `frac_cells_with_adt` | called cells with >= 10 ADT UMIs | < 0.50 | < 0.80 |
| `median_umi_cells` | median ADT UMIs per called cell | < 20 | < 100 |
| `rna_concordance` | median Spearman, over antibodies whose gene is measured in >= 5% of cells, of CLR(ADT) vs log RNA (exon rows summed per gene); assessed with >= 3 pairs | < 0.05 | < 0.15 |
| `panel_unexplained` | feature-like reads on unknown barcodes | > 0.20 | > 0.05 |
| `saturation` | 1 - molecules / usable reads | - | < 0.20 |
| `isotype_share_median` | per-cell share of isotype-control UMIs | - | > 0.10 |
| `hto_singlet_share` | hashtag libraries: called cells with one hashtag | < 0.30 | < 0.50 |

Also reported: panel antibodies detected above background (mean in cells >= 2x mean in empty droplets), the permutation null of the
RNA concordance, the canonical markers (CD3E, CD4, CD8A, MS4A1, CD14, CD19, NCAM1, FCGR3A) separately, uncalled columns as ADT-rich
as a median called cell (cells the GEX calling may have missed), offset shifts and low-complexity reads.

Per cell (`cells.parquet`, called cells): `col`, `adt_total`, `n_adt_detected`, `isotype_share`, and for hashtags `hto_call`,
`hto_lfc_top`, `hto_lfc_second`.

Derived layers (recipes, not stored; raw counts are the truth): `adt-clr-v1` (Seurat CLR, margin 2) and `adt-dsb-v1` (DSB, Mulè et al.
2022: `log(x+10)` standardised by the empty droplets, then the per-cell technical component, the background mean of a 2-component
mixture with the isotype controls, regressed out). Empty droplets: uncalled columns with ADT total between 1 and the lower of the
uncalled 99th percentile and the called 5th percentile.

Hashtag calls (`singlet.adt.hto`): `DropletUtils::hashedDrops` rule. Counts are divided by ambient proportions from the empty
droplets. Confident singlet when log2(top/second) >= 2; doublet when log2(second/ambient level) is > median + 3 MAD and >= 2. They are
published as a feature sidecar `<GSM>.features.hto.parquet` (`hto/call` category, `hto/margin` score), not in the matrices.

<!-- PILOT -->

## 6. Storage as a v2 modality (`singlet.adt.store`, `docs/format-v2.md` section 13)

Members of the **GEX** sample (the antibody library has no sample object of its own):

* `adt_counts.1pz`: pz2, antibodies + isotype controls x the sample's columns (every column of the `cells` table, called or not),
  u32, rows `Inline`, columns `Cells`;
* `hto_counts.1pz`: hashtags, same layout;
* `adt_panel.json`: stored JSON, one entry per row (name, sequence, kind, TotalSeq id, Ensembl gene, source) plus counting facts and
  the QC verdict.

Several libraries of one GEX sample (ADT + HTO, or several sequencing libraries) merge into one member per kind (`assemble.py`).
Attachment, since objects are immutable: **at migration** as extra members of the GEX `.s2` (preferred: the campaign is on hold), or
**after** as a content-addressed sidecar `v2/x/<GSM>/adt.<d16>.s2` with the same members, referenced from the manifest as
`samples[i].modalities.adt = {key, bytes, sha256, members}`. Excluded libraries are listed as `samples[i].modalities_excluded.adt =
{reason, failed}`.

`modalities=` vocabulary (proposed to the v2-modalities session): `adt` -> `adt_counts.1pz` (`obsm["adt"]`, `altExp("adt")`), `hto` ->
`hto_counts.1pz`; aliases protein, cite, antibody, hashtag. Proposed note for the website's `get_modalities` / catalog: *"adt: antibody
UMIs per cell (CITE-seq / TotalSeq), raw counts aligned to the gene matrix columns; panel with TotalSeq ids and genes in
adt_panel.json; DSB/CLR computed by the client. hto: hashtag UMIs; per-cell hashtag calls in the hto feature set."* D1 would
gain per-sample facts only (has_adt, n_antibodies, panel source, QC verdict) in `sample_attr`, never cells.

Proposed addition to `docs/CELL-ARCHITECTURE.md` (canonical copy in singlet-website; not edited here): in section 1, Raw data:
"count matrices (exon, intron), junctions, heteroplasmy, VDJ, **antibody and hashtag UMIs (ADT/HTO, members of the GEX sample, or
a manifest-referenced sidecar `v2/x/<GSM>/adt.<d16>.s2`)**"; in section 5: "hashtag call (donor/condition pool) | cell | feature sidecar
`hto`".

<!-- FULL -->
