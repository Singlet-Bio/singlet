# Per-sample outputs added or changed for the atlas re-run (prod/modality-fixes)

The atlas re-run processes about 70,000 10x samples from all reads and deletes each BAM after its sample, so anything not
computed in the run is lost. This page lists what `singlet ... --pipeline` writes beyond the count matrices since
`prod/modality-fixes`, which outputs are public and which are private, and what the v2 packer needs. Benchmarks and
before/after numbers: `docs/modality-benchmarks/` (branch `plan/atlas-rerun`) and `bench/modalities/*_fix/`,
`bench/modalities/prod_modfix/`.

## Public and private outputs

Genotypes are never stored in public objects or served (`docs/CELL-ARCHITECTURE.md`). Every genotype-bearing output now
goes to **`<out>/private/`**:

| Private file | Content |
|---|---|
| `private/snp_ad.*`, `private/snp_dp.*` | per-cell alt / total depth over the SNP panel (were `donor/snp_*`) |
| `private/ase_counts.tsv` | allele-specific counts per cell and SNP (was top level) |
| `private/donor/donor*.vcf`, coverage maps | per-donor genotypes from demux (were in `donor/`) |
| `private/mt_allele_counts.*`, `private/mt_allele_counts_rev.*` | raw per-cell chrM allele counts, all reads and the reverse-strand subset (forward = total - reverse); features `chrM:<pos>:<A/C/G/T>`; columns of non-called barcodes empty |
| `private/mt/donorN_mt_variants.vcf`, `donorN_mt_consensus.fa`, `het_site_donor_af.tsv` | mt haplotypes per donor (homoplasmic calls with DP/AD/AF/SB/NC) |
| `private/mt_sites.tsv` | every chrM site the caller looked at, with its class and the filters it failed |

`python/singlet/_restricted.py` treats everything under `private/` as restricted (as it already did for `donor/` except
`donor_assignments.tsv`, `mt/` except `mt_summary.tsv`, `snp_*`, VCFs), so the legacy packer (`bundle.py`) puts these in
the restricted sidecar. `run_provenance.h::is_private_output` mirrors the rule and `pileup_stats.json` ->
`provenance.outputs` lists every file with its size and visibility. Donor labels stay public: `donor/donor_assignments.tsv`
and the top-level `donor_assignments.tsv` (no longer a header-only stub).

## New and changed members

| Output | Kind | What | Notes |
|---|---|---|---|
| `ambient_corrected_counts` (`.1pz` or `.mtx.gz` + features/barcodes) | matrix genes x barcodes | counts with each called cell's estimated ambient molecules removed (integer, totals preserved by largest-remainder rounding); other columns empty; raw counts untouched | `ambient_correction.h::correct_ambient_gene_level` (cluster-aware off-gene estimate, Gamma prior per cell) |
| `ambient_contamination.tsv` | table | barcode, rho, n_genes_used, cluster, n_umi, n_ambient_umi_removed (called cells) | rho was depth-only |
| `ambient_contamination.json` | json | mean / median rho, method, profile stats | `estimated_contamination_fraction` was always 0 |
| `ambient_profile.tsv` | table | now gene level (`gene_id_gene_name`) | was exon level |
| `doublet_scores.tsv` | table | same columns; scrublet-style score (simulated doublets in PCA kNN + logistic on kNN fraction and library size), call = top `min(0.8%/1000 cells, 25%) x n x detectable fraction` | called 20-25% of cells before |
| `donor_assignments.tsv`, `donor/donor_assignments.tsv` | table | `prob_doublet`, `best_singlet`, `best_doublet` filled; label `doublet`; auto-K by held-out likelihood | `prob_doublet` was 0, auto-K picked 2 of 8 |
| `mt_variants.tsv`, `mt_heteroplasmy` | table, matrix | heteroplasmic sites only, VAF against the genome's chrM (rCRS for GRCh38), mgatk-like filters (strand concordance, masks, per-cell significance, MAPQ gate on chrM reads) | 2,396 -> 64 sites on Kang; `mt_heteroplasmy.mtx.gz` and `_features/_barcodes.tsv.gz` now carry `.gz` |
| `cell_qc_metrics.tsv` | table | new columns after `doublet_score`: `is_doublet`, `multimapped_reads`, `mismatch_rate` (STAR nM over aligned bases of unique reads), `star_cb_reads`, `unmapped_reads`, `star_multimapped_reads`, `exonic_reads`, `intronic_reads`, `antisense_reads`, `mito_reads`, `intergenic_reads` | STARsolo `--soloCellReadStats Standard` (CB_UMI_Simple runs), `star_cell_read_stats.h`; the raw `CellReads.stats` is removed after parsing |
| `read_qc_summary.json` | json | reads whose barcode matched no passlist entry (`CBnotInPasslist`), with their genome-unique / multi counts | sample level |
| `polya_counts` (`.1pz` or `.mtx.gz`) | matrix sites x barcodes | molecules per poly(A) site cluster and cell, from unique reads whose transcript 3' end carries a soft-clipped A tail (>= 6 nt, >= 80% A); 3' protocols only | `polya_sites.h` |
| `polya_sites.tsv.gz` | table | one row per `polya_counts` row: gene, chrom, strand, mode position, span, molecules, reads, cells, `internal_priming` (>= 12 A in the 20 nt downstream or an A run >= 8, from the STAR Genome binary), `downstream_a`, `pas_motif` | flagged, not removed |
| `sj_features.tsv.gz` | table | one row per `sj_counts` junction: STAR intron coordinates, strand, motif, annotated, unique / multi reads, max overhang, `star_sj_pass` (junction in STAR's filtered SJ.out.tab = the STARsolo-comparable set), molecules, cells | all junctions stay in `sj_counts` |
| `filtered_feature_bc_matrix/` (mtx) | matrix | called cells only (held every barcode) | `raw_feature_bc_matrix/` with `--raw-matrix` (was a no-op) |
| `splice_psi.mtx.gz`, `splice_psi_{features,barcodes}.tsv.gz` | | gzip files now carry `.gz` | |
| `barnyard_classification.tsv` | table | called cells only (covered every barcode: 69k "doublets" on hgmm1k) | |
| `nonhost/nonhost_per_cell.*` | table | written on a fresh output directory; only EM-confirmed species are counted per cell | `nonhost_summary.json` adds `per_cell_species` and `missing_reference_fasta` |

## Pipeline defaults that changed

| Default | Before | Now | Why |
|---|---|---|---|
| 5' constant-prefix clip | probe on the first 25 reads of a barcode-sorted file (PBMC 1k v3 lost 11 bp from every read) | >= 10k reads spread through the file, >= 80% one base per position in both halves, > 10 bp, not poly-A/T; `--no-clip5p` | `prefix_probe.h`; decision in `provenance.clip5p` |
| `--bam-sort-mode` | auto (in-RAM sort OOM at 80 GB on 182 M reads) | disk | identical output, no measured time penalty |
| SNP panel in `--pipeline` | none unless `--snps` | `<ref-base>/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz` (human), `mus_musculus.vcf.gz` (mouse), none for mixed or other species; `--no-snps` | donor demux and genotype doublets need it; `provenance.snp_panel` |
| `--no-adapter-clip` | n/a | new flag: skip STAR `--clip3pAdapterSeq` (the protocol TSO on 5' kits) | see the 5' timing note in the PR |

## What the v2 packer needs (python/singlet/v2 is unchanged)

* **Skip `private/`** (and every path `_restricted.is_restricted_member` flags) when building an `.s2` from a pipeline
  output directory; route those files to private storage. The v2 migration reads public legacy bundles, which already
  exclude them.
* `ambient_corrected_counts`, `polya_counts`: pz2 matrices like `ambiguous_counts` / `sj_counts`. `ambient_corrected_counts`
  rows are the gene vocabulary (`gene_id_gene_name`, the vocabulary-pack order); `polya_counts` rows have inline names
  (`gene:chrom:strand:pos`), as `sj_counts`. Both have the sample's barcode columns; only called cells are nonzero.
* `polya_sites.tsv.gz`, `sj_features.tsv.gz`, `read_qc_summary.json`: stored blobs (kind 3). `sj_features` is a candidate
  for the shared junction table (`docs/format-v2.md` section 12): `star_sj_pass`, motif and annotation are per junction,
  not per sample, except the read counts.
* `cell_qc_metrics.tsv` has 11 more columns; the `cells`-table derivation does not cover them yet, so it falls back to
  stored until the cells table gains them (they are per-cell QC counts: tier-2 columns in `docs/CELL-ARCHITECTURE.md`).
* `doublet_scores.tsv`, `ambient_contamination.tsv`, `donor_assignments.tsv`: per-cell labels and scores keyed by
  barcode; candidates for the per-sample label sidecar (`v2/a/<GSM>/<d16>.parquet`), not D1.
