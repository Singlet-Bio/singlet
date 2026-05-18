# Singlet Canonical Output Format

> Status: **Proposal v2** — May 2026
> Scope: per-sample pipeline outputs and per-reference side-information
> Supersedes: v1 (single-tensor design)

This document specifies a unified, redundancy-free data layout for all singlet
pipeline outputs at the scale of **10,000+ GEO studies** (~90,000 samples). It
replaces today's 30+ heterogeneous per-sample files with three layers:

1. A **reference bundle** (built once per genome+GTF, shared by all samples).
2. A small set of **canonical per-sample matrices** (`.1pz` files), each
   tuned to its own row-axis sparsity profile.
3. **Per-sample sidecars** (one Parquet for cell metadata, one JSON for
   sample-level scalars, a handful of small auxiliaries).

---

## 1. Design principles

1. **Single source of truth.** No derived view is stored. Anything that can be
   computed from canonical files in <200 ms is computed on demand.
2. **Reference-side annotations are shared.** Feature vocabularies (exons,
   introns, junctions, SNP sites) live in the reference bundle, not in
   per-sample files. Per-sample matrices reference them by integer ID.
3. **One `.1pz` per row-axis class.** Matrices that share a row-axis and
   sparsity profile go in the same file. Matrices with fundamentally
   different row dimensions or gap distributions stay separate, because the
   `.1pz` codec is tuned to a single gap-size regime per file.
4. **Universal vs. optional separation.** Matrices that exist for every
   sample go in **canonical** files with fixed names. Library-type-specific
   matrices (guides, antibodies, V(D)J) go in **optional** files whose
   presence is recorded in `summary.json`.
5. **Partition invariant for UMI counts.** Every UMI-deduplicated transcript
   molecule appears in exactly one row of `counts.1pz` (exon-body,
   intron-body, or junction). Gene totals are a derived projection.
6. **Reproducibility by reference ID.** The sample records the
   `reference_id` (e.g. `GRCh38-2024-A@sha256:...`). All annotation lookups
   are keyed by that ID; mismatched references fail loudly.

---

## 2. Why splitting matrices by row-axis matters

`.1pz` (VOCSC) compresses CSC indices as gaps between consecutive nonzero
row indices within each column, then zstd over the codec output. Gap-size
distribution governs compression ratio:

| Matrix | Row dim | Typical nnz / cell | Median gap | Codec regime |
|---|---|---|---|---|
| `exon_body` | ~300 K | 1–5 K | ~100 | dense / short-gap |
| `intron_body` | ~280 K | 0.5–3 K | ~100–500 | dense / short-gap |
| `junctions` | ~500 K | 50–500 | ~1 K | dense / short-gap |
| `snp_*` | ~8 M | 50–500 | ~10 K–100 K | sparse / long-gap |
| `mt_*` | ~16.5 K | 100–5 K | ~5–50 | tiny / dense |
| `guides` | 10–10 K | 0–10 | varies | tiny / dense |
| `antibodies` | 10–500 | 0–100 | varies | tiny / dense |
| `vdj_gene_usage` | ~200 | 0–4 | varies | tiny / dense |

Combining `exon_body` (dense, short-gap) with `snp_ad` (sparse, long-gap)
in one file would force a shared codec parameter set and waste 10–20%
compression on one of them. Splitting by row-axis class preserves
codec-level locality at the cost of a few extra small files.

**Pairing rule**: matrices that share both row-axis and per-cell sparsity
mask (`snp_ad` + `snp_dp`, `mt_alt_ad` + `mt_dp`) go in the *same* file as
a **two-data-layer CSC** (one `indptr`/`indices`, two `data` arrays). This
deduplicates the structural arrays and keeps the codec regime uniform.

---

## 3. Reference bundle (`reference/{build_id}/`)

Built **once per genome+GTF** by the reference-prep tool. Mmap-able binary
files, immutable, content-addressed by SHA-256 of the GTF + FASTA.

```
reference/GRCh38-2024-A/
├── manifest.json          # build_id, sha256, GTF source, build date
├── genome.fa.gz           # FASTA (existing)
├── star_2.7.11b/          # STAR index (existing)
├── features.fbin          # feature vocabulary (NEW, packed binary)
├── snp_sites.fbin         # SNP panel used for ASE/donor demux (NEW)
└── chrom_map.tsv          # tid → chrom name, with offsets
```

The Kraken2 + Bracken database (Standard + Viral RefSeq supplement,
~17.5 GB) lives **outside** the per-genome reference bundle because it
is host-species-agnostic and shared across all samples regardless of
host. Path: `reference/kraken2/k2_standard_viral_{date}/`. It is
versioned and content-addressed independently; `nonhost.json` records
the `db_id` used per sample for reproducibility.
```

### 3.1 `features.fbin` — the feature vocabulary

A single packed binary describing every gene, exon interval, intron
interval, and annotated junction. Loaded once at process start via mmap;
per-sample matrices reference by integer ID.

```c
struct FeatureHeader {
    char     magic[8];          // "SLFEAT01"
    uint32_t version;
    char     build_id[32];      // "GRCh38-2024-A"
    char     gtf_sha256[32];
    uint32_t n_genes, n_exon_intervals, n_intron_intervals, n_junctions;
    uint64_t offset_genes, offset_exons, offset_introns, offset_junctions;
    uint64_t offset_strings;
};

struct GeneRec {                // ~32 B/gene
    uint32_t name_offset;       // into string pool ("ENSG00000167286")
    uint32_t symbol_offset;     // ("CD3D"), 0 if absent
    uint8_t  chrom, strand, biotype, reserved;
    uint32_t tx_start, tx_end;
    uint32_t exon_lo, exon_hi;
    uint32_t intron_lo, intron_hi;
    uint32_t junction_lo, junction_hi;
};

struct ExonRec   { uint32_t gene_id; uint8_t chrom; uint32_t start, end; };
struct IntronRec { uint32_t gene_id; uint8_t chrom; uint32_t start, end;
                   uint32_t flank_exon_lo, flank_exon_hi; };
struct JunctionRec {
    uint32_t gene_id;           // UINT32_MAX if intergenic
    uint32_t donor_feat_id, acceptor_feat_id;
    uint8_t  donor_kind:1, accept_kind:1, type:2, canonical:1, motif:3;
    uint8_t  reserved;
    uint32_t donor_pos, acceptor_pos;
};
```

**Size (human GRCh38-2024-A)**: ~23 MB raw, ~6 MB zstd.

### 3.2 `snp_sites.fbin` — population SNP panel

Today's pipeline ships `genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.vcf.gz`
into every job. It belongs in the reference.

```c
struct SnpHeader {
    char     magic[8];          // "SLSNP01\0"
    char     build_id[32];
    char     panel_id[32];      // "1KGP_AF5e2"
    uint32_t n_sites;
};
struct SnpRec {                 // 14 B
    uint8_t  chrom;
    uint32_t pos;
    char     ref, alt;
    float    af_pop;
    uint32_t rsid;
};
```

~8 M sites × 14 B = ~110 MB raw, ~40 MB zstd, mmap-able.

Per-sample SNP matrices use the row index of `snp_sites.fbin` as their
row dimension, eliminating chrom/pos/ref/alt from every sample.

---

## 4. Per-sample files

### 4.1 Canonical (always present)

| File | Row axis | Cols | What |
|---|---|---|---|
| `counts.1pz` | features (exons ∥ introns ∥ junctions) | barcodes | UMI counts, three contiguous row blocks satisfying the partition invariant |
| `snp.1pz` | sites from `snp_sites.fbin` | barcodes | two-layer CSC: `alt_ad` and `dp`; shared sparsity mask |
| `mt.1pz` | mt positions (1..n) | barcodes | two-layer CSC: `alt_ad` and `dp`; small dense-ish matrix over chrM |
| `cell_meta.parquet` | n/a | one row per barcode | all cell-level scalars |
| `summary.json` | n/a | n/a | sample-level metrics + provenance |
| `saturation_curve.tsv` | n/a | n/a | 6-row downsampling table (tiny) |
| `star_Log.final.out` | n/a | n/a | STAR's native log, verbatim |

### 4.2 Optional (library-type or analysis-specific)

| File | Present when | Notes |
|---|---|---|
| `donor_consensus.fa` | donor demux ran | per-donor mt consensus |
| `donor_variants.vcf.gz` | donor demux ran | per-donor genotype calls |
| `ambient_profile.npy` | ambient correction ran | dense gene vector (~240 KB) |
| `splice_events.tsv` | splice event clustering ran | sample-specific AS event clusters |
| `guides.1pz` | CRISPR library | rows = `guide_panel.fbin` (separate panel ref) |
| `antibodies.1pz` | CITE-seq / hashing | rows = `feature_ref.fbin` |
| `vdj_gene_usage.1pz` | TCR/BCR data | rows = `vdj_panel.fbin` |
| `nonhost.json` | non-host screening ran | raw Kraken2 + Bracken per-species summary (§4.6) |
| `nonhost_species.1pz` | non-host screening ran | per-cell × NCBI-taxid read counts; rows match `nonhost.json["species"]` |

`summary.json` declares which optional files exist (`"outputs": {...}`).

### 4.3 `counts.1pz` — gene-expression UMI partition

Single `.1pz` with **three row blocks**, contiguous in row order:

```
counts.1pz
├── magic            "1PZ02\0\0\0"
├── header (CBOR)
│   ├── reference_id "GRCh38-2024-A@sha256:..."
│   ├── n_cells
│   ├── cell_barcodes  (deduped, sorted; shared across all .1pz files)
│   └── blocks[] = { name: "exon_body" | "intron_body" | "junctions",
│                    row_offset_in_global_axis, row_dim, nnz,
│                    indptr_size, indices_size, data_size, crc }
└── per-block CSC payloads (one zstd stream per block)
```

**Partition invariant**: for every cell `c`,
`sum(exon_body[:,c]) + sum(intron_body[:,c]) + sum(junctions[:,c]) ==
total_umis(c)`. Classifier rule:

1. Read pair contains any `N` cigar op (intron skip) → `junctions`.
2. Else, footprint entirely within one annotated exon → `exon_body`.
3. Else, footprint entirely within one annotated intron → `intron_body`.
4. Else (exon-intron straddle, no splice) → `junctions` with EI/IE type.

All three blocks share `cell_barcodes` and codec regime; each stores its
own CSC arrays because `nnz` differs.

### 4.4 `snp.1pz` — population-SNP read counts (two-layer CSC)

Rows = `snp_sites.fbin` order (~8 M sites). Per-cell sparsity is governed
by which sites the cell's reads cover. `alt_ad ≤ dp` everywhere.

```
snp.1pz
├── magic, header (references snp_sites.fbin sha256)
├── shared: indptr, indices  (one set, defined by dp>0)
└── two zstd-compressed data arrays: data_dp, data_alt_ad
```

Why a separate file:
- Row gap distribution is 100×–1000× larger than `counts.1pz`.
- Many samples (no SNP genotyping requested) skip this entirely.
- Two-layer CSC halves the structural overhead vs. two independent files.

Replaces today's `donor/snp_ad.1pz`, `donor/snp_dp.1pz`, and the
redundant 17 MB `donor/donor0_coverage.tsv`.

### 4.5 `mt.1pz` — mitochondrial coverage (two-layer CSC)

Rows = chrM positions 1..N (~16.5 K for human). Same two-layer scheme as
`snp.1pz`. Codec regime is "tiny / dense": median gap ~5–50, very
different from both `counts.1pz` and `snp.1pz`, which is why it lives in
its own file.

Replaces today's `mt/mt_heteroplasmy.1pz` plus the redundant
`mt/mt_variants.tsv`.

### 4.6 Non-host: Kraken2 + Bracken (bulk summary + per-cell taxid matrix)

Non-host detection is **Kraken2 + Bracken only** — no secondary
alignment, no curated pathogen panel, no per-pathogen-gene matrix. The
13,458-sample EDA showed that everything biologically real that
appeared in our cohort (bacterial pathogens, mouse retroviruses, HIV in
COVID BAL) is captured by Kraken2's k-mer classification at species
resolution; the things Kraken2 misses (pathogen-gene-level expression,
strain-level variants) are out of scope for a pipeline run across 10K+
heterogeneous GEO studies.

#### Database choice: Standard + Viral RefSeq supplement

| Component | Size | Justification |
|---|---|---|
| **Kraken2 Standard** | ~16 GB | RefSeq complete bacteria + archaea + human. Covers every real bacterial hit from the 13,458-sample EDA (*S. aureus*, *K. pneumoniae*, *Stenotrophomonas*, *Pseudomonas*) and all top contaminants for cohort-level background detection. |
| **Viral RefSeq supplement** | ~1.5 GB | All complete viral genomes. Validated on COVID BAL (SRR11537951) recovering HIV; mouse retroviruses (MLV, MMTV, endogenous proviruses) appear in 20–50% of mouse samples and are biologically informative. |
| **Bracken** weights for the above | ~50 MB | Abundance reweighting from k-mer counts to read counts using genome length + ambiguity model. Cheap (~5 s, near-zero RAM). |
| **Total** | **~17.5 GB** | Fits in `/dev/shm` on every node, mmap-shared across jobs. |

Excluded from the database, with justification:

| Excluded | EDA finding | Why skip |
|---|---|---|
| Protozoa (PlusPF +15 GB) | *Toxoplasma gondii* 42.7% prevalence at **0 reads**, *Plasmodium vivax* 48.6% prevalence at ~3,900 reads — both pure database artifacts | Zero real signal, two prominent false-positive hotspots |
| Fungi (PlusPF +50 GB) | *Puccinia striiformis* (wheat rust) 80.6% in human at **0 reads** — top artifact; only *Malassezia* (skin commensal) was real | One organism's worth of signal in skin samples only; not worth a 3× database size increase |
| Plant (PlusPFP +100 GB) | *Curtobacterium flaccumfaciens* 41.1% prevalence at **0 reads** — pure artifact | No real plant-derived signal in human/mouse samples |

If a downstream use case genuinely needs *Malassezia* (skin
microbiome) or fungal lung infections (e.g. *Aspergillus* in CF
patients), run a one-off PlusPF pass on the relevant subset — don't bake
the larger database into the universal pipeline.

#### Pipeline steps

1. **Capture unmapped reads** from STAR (already in pipeline).
2. **Kraken2** classifies unmapped reads against the Standard+Viral
   database loaded once per node from `/dev/shm`.
3. **Bracken** reweights Kraken2 counts to abundance estimates using
   the precomputed kmer-length distribution file. ~5 s per sample.
4. **Per-cell join** (`NonHostCellMatrix`, already implemented in cycle
   151): each Kraken2-classified read has a cell barcode from its FASTQ
   header; the join produces `(barcode, taxid) → read_count`. Output:
   `nonhost_species.1pz`.

No filtering or background subtraction is performed at processing time.
Every sample stores **raw Kraken2+Bracken outputs** verbatim, so
downstream analysis (cohort-level z-scores, tissue stratification, novel
contaminant detection) can be done offline against whatever cohort
definition the analyst chooses, without reprocessing.

#### `nonhost.json` — bulk Kraken2 + Bracken summary

```json
{
  "kraken2": {
    "db_id": "k2_standard_viral_2025-04-15@sha256:...",
    "db_size_gb": 17.5,
    "total_unmapped_reads": 452636,
    "classified_reads": 78631,
    "unclassified_reads": 374005,
    "classified_fraction": 0.1737
  },
  "bracken": {
    "version": "3.0",
    "read_length": 90,
    "level": "S",
    "threshold": 10
  },
  "species": [
    {
      "row": 0,
      "taxid": 11676,
      "name": "Human immunodeficiency virus 1",
      "rank": "S",
      "kraken_reads": 2791,
      "kraken_kmer_hits": 18722,
      "bracken_reads": 2734,
      "bracken_abundance": 0.0348,
      "lineage": "Viruses;Riboviria;...;Lentivirus"
    },
    {
      "row": 1,
      "taxid": 1280,
      "name": "Staphylococcus aureus",
      "rank": "S",
      "kraken_reads": 4102,
      "kraken_kmer_hits": 39511,
      "bracken_reads": 4287,
      "bracken_abundance": 0.0545,
      "lineage": "Bacteria;Firmicutes;Bacilli;...;Staphylococcus"
    }
  ]
}
```

Every species detected at species rank (`S`) above Bracken's threshold
is recorded with its **raw Kraken2 reads, raw Bracken-reweighted reads,
and abundance**. No filtering. `species[].row` is the row index in
`nonhost_species.1pz`. Stable NCBI `taxid` is the canonical identifier
for cross-sample joins.

#### `nonhost_species.1pz` — per-cell × taxid (sample-specific rows)

Sparse uint16 CSC. Rows: NCBI taxids detected in this sample, ordered
to match `nonhost.json["species"][].row`. Cols: barcodes (shared cell
axis with `counts.1pz`, `snp.1pz`, `mt.1pz`).

Values are **raw per-cell read counts** assigned to each taxid by
Kraken2 (post-Bracken reweighting at the bulk level; per-cell counts are
just the per-barcode tally of reads with that taxid). No filtering.

Replaces today's `nonhost_per_cell.tsv` (3-column text triplet — a few
MB per sample, occasionally tens of MB on infected samples like COVID
BAL). The `.1pz` form is 5–20× smaller and matches the codec regime of
`mt.1pz` (tiny row dim, dense-ish per cell).

Why sample-specific rows: the species universe across 10K GEO studies
is unbounded. A global panel keyed by all of RefSeq would make every
per-sample matrix 99.999% empty. The stable NCBI taxid lookup is one
JSON read away in `nonhost.json`. Cross-sample analyses join on `taxid`
through the catalog.

#### What this enables (today, with no extra processing)

- Viral / bacterial load per cell and per cell-type (join
  `nonhost_species.1pz` × `cell_meta.parquet`).
- Co-infection patterns (multiple non-zero rows per cell).
- Cell-tropism of infection (cross-tabulate taxid presence with
  host cell-type annotations).
- Cohort-level outlier detection (offline z-score from raw
  `bracken_reads` across all samples — no pipeline change required).
- Novel pathogen surveillance: a new outbreak organism appears in the
  output the moment Kraken2's database covers it; the next monthly
  database refresh brings it in automatically.


### 4.7 `cell_meta.parquet`

One Parquet file with one row per **barcode** (every barcode seen, not
just called cells). Columns:

| Column | Type | Source |
|---|---|---|
| `barcode` | string | primary key; sort order matches `.1pz` columns |
| `is_cell` | bool | cell calling |
| `cell_call_method` | uint8 (enum) | |
| `cell_call_pvalue` | float32 | emptyDrops |
| `n_umi` | uint32 | from `counts.1pz` |
| `n_genes_detected` | uint32 | derived |
| `pct_mt` | float32 | from `mt.1pz` (dp) |
| `pct_ribo` | float32 | |
| `intronic_fraction` | float32 | sum(intron_body)/n_umi |
| `spliced_fraction` | float32 | sum(EE junctions)/n_umi |
| `saturation` | float32 | duplication rate |
| `doublet_score` | float32 | |
| `cell_cycle_phase` | uint8 (G1/S/G2M) | |
| `cell_cycle_s_score` | float32 | |
| `cell_cycle_g2m_score` | float32 | |
| `donor_id` | uint16 | UINT16_MAX = unassigned |
| `donor_prob` | float32 | |
| `donor_is_doublet` | bool | |
| `ambient_contamination` | float32 | per-cell ρ |
| `sex_call` | uint8 (M/F/unknown) | |

Replaces 7 separate TSVs (`cell_calls`, `cell_qc_metrics`,
`cell_cycle_scores`, `doublet_scores`, `auto_barcodes`,
`donor_assignments`, `ambient_contamination`). ~3 MB raw → ~500 KB
column-compressed; column-projectable.

### 4.8 `summary.json`

Sample-level scalars + provenance. Merges today's `summary.json`,
`pileup_stats.json`, `rrna_report.json`, `sex_call.json`,
`ancestry_call.json`, `ambient_contamination.json`, `provenance.json`,
`metrics_summary.csv`:

```json
{
  "schema_version": "2.0",
  "sample_id": "GSM8225039",
  "reference_id": "GRCh38-2024-A@sha256:...",
  "singlet_version": "0.4.0",
  "provenance": { "run_id": "...", "host": "c008", "wall_s": 193.3 },
  "input":      { "n_reads": 18440162, "protocol": "10xv3" },
  "alignment":  { "mapped": 18440162, "uniquely_mapped": 15154913,
                  "mapping_rate": 0.907, "mismatch_rate": 0.0023 },
  "umi":        { "unique": 10164823, "duplicate": 142917,
                  "saturation": 0.014 },
  "feature_hits": { "exon_body": 7453020, "intron_body": 2711803,
                    "junctions": 412910 },
  "cells":      { "called": 12055, "median_umi": 524,
                  "median_genes": 408 },
  "mt":         { "events": 55049, "donors_with_consensus": 1 },
  "donors":     { "n_inferred": 1, "method": "vb_binomial" },
  "ambient":    { "rho_median": 0.012 },
  "rrna":       { "fraction": 0.0034 },
  "sex":        { "call": "F", "prob": 0.99 },
  "ancestry":   { "top": "EUR", "prob": 0.82 },
  "outputs":    { "snp": true, "mt": true, "nonhost": true,
                  "guides": false, "antibodies": false, "vdj": false,
                  "donor_demux": true, "splice_events": true,
                  "ambient_profile": true },
  "qc_flags":   { "low_mapping": false, "low_cells": false }
}
```

### 4.9 Eliminated outputs

| Removed | Reason |
|---|---|
| `gene_counts.1pz` | derived; compute from `counts.1pz` + features.fbin |
| `gene_counts_em.1pz` | derived; recompute from junctions + multimapper EM state |
| `spliced.1pz`, `unspliced.1pz`, `ambiguous.1pz` | derived USA decomposition |
| `splice_psi.1pz` | derived from `junctions` block |
| `gene_expression.tsv` | derived; bulk metric |
| `mt_variants.tsv` | redundant with `mt.1pz` |
| `donor0_coverage.tsv` (17 MB!) | redundant with `snp.1pz` |
| `metrics_summary.csv`, `read_stats.tsv` | redundant with `summary.json` |

---

## 5. Final canonical layout (per sample)

```
{quant_root}/scrna/{gse_shard}/{gse_id}/{gsm_id}/
├── counts.1pz                  # always: exon_body + intron_body + junctions
├── snp.1pz                     # always (canonical SNP panel): alt_ad + dp
├── mt.1pz                      # always: alt_ad + dp on chrM
├── cell_meta.parquet           # always
├── summary.json                # always
├── saturation_curve.tsv        # always (tiny)
├── star_Log.final.out          # always
├── nonhost.json                # if non-host screening ran
├── donor_consensus.fa          # if donor demux ran
├── donor_variants.vcf.gz       # if donor demux ran
├── ambient_profile.npy         # if ambient correction ran
├── splice_events.tsv           # if splice event clustering ran
├── nonhost.json                # if Kraken2 non-host screening ran
├── nonhost_species.1pz         # if Kraken2 non-host screening ran (per-cell × taxid)
├── guides.1pz                  # CRISPR libraries only
├── antibodies.1pz              # CITE-seq libraries only
└── vdj_gene_usage.1pz          # TCR/BCR data only
```

**Typical 10x v3 sample (12 K cells, 18 M reads), human, no
guides/antibodies/V(D)J:**

| Group | Today | Proposed |
|---|---|---|
| Gene-expression matrices (9 files) | 34 MB | **counts.1pz: ~30 MB** |
| SNP matrices (`snp_ad`, `snp_dp`, `donor0_coverage.tsv`) | 130 MB | **snp.1pz: ~80 MB** (two-layer dedup of indptr/indices, plus removal of 17 MB redundant TSV) |
| Mt (`mt_heteroplasmy.1pz` + `mt_variants.tsv`) | ~1.7 MB | **mt.1pz: ~600 KB** |
| Cell-level TSVs (7 files) | ~6 MB | **cell_meta.parquet: ~500 KB** |
| Summary JSONs (8 files) | ~10 KB | **summary.json: ~3 KB** |
| Non-host (1 TSV + 1 JSON + per-cell TSV) | ~3–10 MB | **nonhost.json (~5 KB) + nonhost_species.1pz (~200 KB–2 MB)** |
| **Total per sample** | **~170 MB across 31 files** | **~111 MB across 8–11 files** |

**Across 10,000 GEO studies (~90 K samples)**:
- ~5 TB storage saved (~35% reduction).
- ~2 million fewer files on NFS (per-sample file count drops from 31 to
  8–11). Inode pressure drops by an order of magnitude.

---

## 6. Migration

1. **Add `features.fbin` and `snp_sites.fbin` builders** to reference-prep.
   Validate that recomputing today's gene_counts from the new
   exon+intron+junction layers gives bit-identical results on a regression
   set of 100 samples.
2. **Refactor `pileup_engine.h`** to classify each deduplicated molecule
   into exactly one of `exon_body`, `intron_body`, `junctions` (fixing the
   double-counting noted earlier). Splice junction extraction stays, but
   its UMI dedup unifies with the gene-level dedup.
3. **Extend `pz/writer.h`** to support (a) multi-row-block layout for
   `counts.1pz`, and (b) two-data-layer CSC for `snp.1pz` and `mt.1pz`.
4. **Implement Python reader views** in `python/singlet/io/`:
   `SingletCounts.exons()`, `.introns()`, `.junctions()`, `.gene_counts()`,
   `.usa()`, `.psi()`; `SingletSnp.ad()`, `.dp()`, `.vaf()`; `SingletMt`
   analogous.
5. **Cut-over**: write both old and new layouts for one cycle, compare
   bit-identically against derived projections, then drop the legacy
   writer in the following release.

---

## 7. Open questions

- **Cross-block UMI dedup**: a molecule with reads supporting both an
  exon-body and an intron-body of the same gene must land in exactly one
  block. The proposed classifier (§4.3) prefers `junctions` (most
  informative) → `intron_body` → `exon_body`. Needs a regression test.
- **Multi-sample concatenation**: do we ever stack `counts.1pz` files for
  joint analysis? If so, a higher-level container (e.g. a parquet
  manifest, or zarr) should sit above this layout, not replace it.
- **Kraken2 database refresh cadence**: monthly refresh of Standard +
  Viral RefSeq supplement is sufficient for novel-pathogen surveillance.
  Each refresh gets a new `db_id` (date + sha256) recorded in
  `nonhost.json`; samples processed against different `db_id`s remain
  joinable by NCBI `taxid`.
- **GPU loading**: the proposed CSC layout is GPU-friendly (contiguous
  arrays, single decompress per block). Confirm cupy/cuSPARSE
  compatibility.
