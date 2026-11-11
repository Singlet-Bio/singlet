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
| `private/mt_allele_counts.*`, `private/mt_allele_counts_rev.*` | raw per-cell chrM allele counts, all reads and the reverse-strand subset (forward = total - reverse); features `chrM:<pos>:<A/C/G/T>`; columns of non-called barcodes empty (uncalled barcodes: see below) |
| `private/mt_allele_counts_uncalled.*`, `_rev.*` | the same layout for the uncalled barcodes; written only with `SINGLET_UNCALLED_MODE=3` (off by default: the pooled file below covers the ambient mitochondrial share) |
| `private/mt_allele_counts_uncalled.tsv.gz` | the chrM allele counts of all uncalled barcodes pooled: one row per covered position (A C G T, then the reverse-strand subset), for the ambient share of mitochondrial alleles |
| `private/allsite/allele_delta_uncalled.bin`, `cells_uncalled.tsv.gz` | the all-site delta store of the uncalled barcodes (stored against the reference genome), same format as `allele_delta.bin` / `cells.tsv.gz` |
| `private/donor/donorN_callable.rle.gz` | run-length coded intervals where donor N's summed depth gives a confident genotype (panel and all-site sites): per contig a `>name` line, then `gap<TAB>length` lines (start = previous end + gap); a donor VCF can tell reference from not covered |
| `private/mt/donorN_mt_variants.vcf`, `donorN_mt_consensus.fa`, `het_site_donor_af.tsv` | mt haplotypes per donor (homoplasmic calls with DP/AD/AF/SB/NC) |
| `private/mt_sites.tsv` | every chrM site the caller looked at, with its class and the filters it failed |
| `private/editing/editing_alt.*`, `private/editing/editing_dp.*` | per-cell edited / total UMIs at the REDIportal A-to-I sites covered in called cells (rows `chrom:pos:A>G` or `chrom:pos:T>C`, genome strand); same pileup kernels and filters as the SNP panel |
| `private/editing/editing_sites.tsv.gz` | per site: strand, ref / edited UMIs and editing level over called cells, cells covered and edited, UMIs over all barcodes |
| `private/hla/hla_genotype.tsv`, `hla_allele.tsv`, `hla_allele.vcf` | sample HLA genotype from T1K (class I and II genes, allele, abundance, quality), T1K's own formats |
| `private/hla/donors/donorK_genotype.tsv` | the same per assigned donor (cells with label `donorK` and prob_max >= 0.9) |
| `private/hla/hla_candidate_reads.fq.gz` | the reads that reached T1K (k-mer prefilter on the IPD-IMGT/HLA sequences), named `idx_CB`, kept so HLA can be typed again with a newer index |
| `private/hla/hla_summary.json` | reads scanned and kept, alleles in the index and typed against (`typed_alleles`), donors typed, timings, T1K and IPD-IMGT/HLA versions |
| `private/hla/pending/` | with `--hla-defer`, and by default on the GPU engine for 5' libraries (`--hla-gpu-3p`, `--hla-gpu-5p` choose `type` or `defer`), until the CPU step has run: the candidate reads and `manifest.json` for `singlet hla --pending` (`docs/gpu-engine/hla_pending_batch.py` runs a batch). Deleted once `hla_summary.json` has landed; holds reads, owner-only, never public |

### HLA frequency prior (K=2)

T1K types against the alleles recorded in at least 2 cells of IPD-IMGT/HLA (`hlaidx/Allele_status.txt`, the release of the index): 9,881 of 30,322 alleles in release 3.66.0. 3' reads do not cover the exons that separate alleles of one group, so unrestricted T1K picks rare, mostly unconfirmed alleles over the common one. The prefilter still uses the full index. The prior is applied when `Allele_status.txt` sits in the T1K `hlaidx` directory (otherwise the full index is used and the log says so); `hla_summary.json` reports `typed_alleles` and `min_cells`. Cost: 34 of about 26,000 truth alleles in the validation sets belong to a group the prior removes, so those can no longer be called.

Validation (owner-only data): Geuvadis RNA-seq against Abi-Rached 2018, 29 samples typed by all four methods, 2-field accuracy over class I and DRB1/DQB1: Singlet K=2 268/268, stock T1K 99.3%, arcasHLA 89.2%. HLA-A: K=2 58/58, T1K 58/58, arcasHLA 51/58. 10x 3' libraries from donors of known type: K=2 0.914 against 0.686 for the unrestricted index (T1K's genotyper unchanged) and for arcasHLA. See `bench/d101`.


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
| `ambient_contamination.json` | json | `estimated_contamination_fraction` = UMI-weighted ambient fraction the correction removes over the called cells that are not called doublets (`estimated_contamination_basis`, `n_cells_in_estimate`, `n_doublets_excluded`); the per-cell mean it used to report is `mean_rho_cells` (= `mean_rho`); also median rho, method, profile stats. Show the per-cell `rho_total` column of the TSV, not the mean | the headline was the mean of per-cell exon-basis rho: 2.1% on hgmm1k_v3 (truth 1.14%), 4.7% on Kang ctrl (truth 1.48%) |
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
| `nonhost/nonhost_confirmed_reads.bin` | binary (`SNCR` v2: species keys carry bit 31 for microbial species, D-100; `include/singlet/pileup/nonhost/nonhost_confirmed_reads.h`) | the cDNA reads (2-bit) with cell barcode, UMI and species id of EM-confirmed non-host species only; written only when `em_species_count > 0` and the host Bloom filter is loaded, and never a read with a host k-mer; absent on uninfected samples | the one place sequenced reads are kept |
| `editing_cells.tsv` | table | per called cell: edited UMIs, total UMIs and sites covered over the REDIportal panel, and `editing_index` = edited / total (an Alu editing index over known sites) | public: a per-cell aggregate, no site-level allele counts; `rna_editing.h` |
| `editing_summary.json` | json | panel name and size, sites covered, sites with >= 10 UMIs in called cells, sample editing index and its definition | `provenance.editing_panel` records the panel path and its manifest (REDIportal version, sha256) |
| `multimap_counts` (`.1pz` or `.mtx.gz` + features/barcodes) | matrix gene sets x barcodes | molecules whose reads fit two genes, per gene set and cell; the gene counts never include them | see "Multi-mapping record" below; `multimap_record.h` |

## Reads-per-UMI (Beta freeze)

Reads are deleted after the Beta campaign, so the number of reads behind each molecule is kept with the counts. The engine writes
three members whenever UMI collapse and a gene model are on (`--no-reads-matrix` turns them off). CPU and GPU outputs are byte
identical (frontier gate 803937, pbmc1k_v3).

| Output | Kind | What | In the `.s1` |
|---|---|---|---|
| `exon_reads.1pz`, `intron_reads.1pz` | matrix, same rows, columns and nonzeros as `exon_counts` / `intron_counts` | reads merged into each counted entry (uint32) | as `exon_reads_delta.bin` / `intron_reads_delta.bin`: reads minus UMIs on the called cells' nonzeros (`singlet.s1.readsvals.reads_matrix` rebuilds the matrix); the full matrices stay in the run directory |
| `reads_per_umi_hist.tsv.gz` | table, one row per barcode | molecules with 1 to 15 reads and with 16 or more (`n1` .. `n15`, `n16plus`), over every molecule the counts emit (exon and intron) | member `reads_per_umi_hist`, called cells only (`bundle._reads_member`, `scripts/s1_campaign.py` `MODALITY_MEMBERS`) |

Validation against Cell Ranger 3.0's `molecule_info.h5` read counts on pbmc1k_v3, over the 1,217 cells both callers keep
(`bench/readlevel/rpu_vs_molinfo.py`, `bench/readlevel/README.md`):

| Metric | Engine | Cell Ranger |
|---|---|---|
| Median of the per-cell mean reads per UMI | 3.59 | 3.65 |
| Molecules with one read | 21.6% | 20.9% |
| Sequencing saturation over these cells | 71.6% | 72.2% |
| Per-cell mean reads per UMI, Pearson / Spearman | 0.967 / 0.961 | |
| Pooled histogram, total variation distance | 0.014 | |

The engine counts 1.48 times as many molecules per cell because its histogram includes intronic molecules, which Cell Ranger
3.0 leaves out.


## Multi-mapping record

The gene count matrices count a molecule only when its reads are assigned to one gene. STARsolo's Unique counts make the same choice, so
reads that fit two genes go uncounted: multimappers whose alignments touch two genes, uniquely mapped reads on the shared exon of two
overlapping genes, and reads on one gene's exon inside another gene's intron (STARsolo GeneFull's gene-body set, the engine's default
`kGeneFullSet` intron rule). Gene families with near-identical paralogs lose most of their molecules this way. `multimap_counts` keeps those reads as
molecules so that a user can add them back when the data are pulled, or leave them out. The stored gene counts do not change.

Each row is a gene set, named `<gene_id>_<gene_name>|<gene_id>_<gene_name>` for two genes. Columns are the sample's barcodes, as in every
other count matrix, and values are molecule counts. Molecules are built the way STARsolo builds them in `collapseUMIperCB`: reads are
grouped by cell and exact UMI, and the molecule keeps the genes present in every one of its reads. A UMI whose reads share one gene becomes
a single-gene row (`<gene_id>_<gene_name>`), and a UMI whose reads share no gene is dropped. Reads with three or more candidate genes are
left out, as are multimappers with a secondary alignment that itself spans two genes, which keeps the record small. A multimapper is
judged on the gene set the counts use: under the default `kExonPrecedence` rule, its exonic alignments when it has any. A read that rule
counts for one gene (a retrocopy in another gene's intron, for example) is in the gene counts and never in the record. A UMI that also
has a read in the gene counts is left out as well, because STARsolo pools all of a UMI's reads and one uniquely assigned read makes the
whole UMI a unique molecule. No molecule is counted twice; the run log reports reads, molecules, gene sets, dropped UMIs and UMIs already
in the gene counts on its `[multimap]` line.

The CPU pileup collects the reads in its multimapper buffer and on unique two-gene primaries. The GPU pileup returns the same reads from
`k_resolve` and the exact (cell, UMI) keys of its gene store, and one host function (`multimap::build`) turns either into the matrix, so
the two engines write identical bytes.

Retrieval distributes the molecules over their genes in `singlet.multimap.distribute`, a port of STARsolo's `--soloMultiMappers`:

```python
b.raw_counts(gsm)                                   # unique reads only (default)
b.raw_counts(gsm, include_multimappers=True)        # + multimappers by EM
b.raw_counts(gsm, include_multimappers=True, multimappers="Uniform")
```

EM uses the cell's unique counts as the prior, with STARsolo's constants, so a molecule shared by a highly and a lowly expressed gene goes
mostly to the first. Uniform splits each molecule evenly. With multimappers included, `X` becomes float, the added part is in
`layers["multimapped"]` and `uns["multimappers"]` names the method. Samples processed before the record existed keep their unique counts and
raise a warning.

## Pipeline defaults that changed

| Default | Before | Now | Why |
|---|---|---|---|
| 5' constant-prefix clip | probe on the first 25 reads of a barcode-sorted file (PBMC 1k v3 lost 11 bp from every read) | >= 10k reads spread through the file, >= 80% one base per position in both halves, > 10 bp, not poly-A/T; `--no-clip5p` | `prefix_probe.h`; decision in `provenance.clip5p` |
| `--bam-sort-mode` | auto (in-RAM sort OOM at 80 GB on 182 M reads) | disk | identical output, no measured time penalty |
| SNP panel in `--pipeline` | none unless `--snps` | `<ref-base>/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz` (human), `mus_musculus.vcf.gz` (mouse), none for mixed or other species; `--no-snps` | donor demux and genotype doublets need it; `provenance.snp_panel` |
| `--no-adapter-clip` | n/a | new flag: skip STAR `--clip3pAdapterSeq` (the protocol TSO on 5' kits) | see the 5' timing note in the PR |
| RNA editing panel in `--pipeline` | none | `<ref-base>/pileup_indices/rna_editing_GRCh38.tsv.gz` (REDIportal v3, 15.6 M sites) for human, `rna_editing_GRCm39.tsv.gz` (REDIportal v3 mm10 lifted to GRCm39, 107 k sites) for mouse, none for mixed species; `--editing-sites FILE`, `--no-editing` | D-101; sites are appended to the SNP panel rows and split off before every SNP consumer, so SNP outputs are byte-identical; panel built by `scripts/editing/build_rediportal_panel.py` |
| HLA typing in `--pipeline` | none | the CPU engine types by default; the GPU engine by default writes only the candidate reads (`private/hla/pending/`, typed later on a CPU node with `singlet hla --pending`; `--no-hla`: none). When typing, human non-barnyard runs type HLA with T1K v1.0.10 (MIT) and the index at `<ref-base>/hla/t1k` (IPD-IMGT/HLA, `manifest.json`); `--hla-t1k DIR`; also `singlet hla` and `singlet hla-prefilter` on a finished run | D-101; reads only the input .1fq, so CPU and GPU runs give the same calls; all outputs owner-only |

## What the v2 packer needs (python/singlet/v2 is unchanged)

* **Skip `private/`** (and every path `_restricted.is_restricted_member` flags) when building an `.s2` from a pipeline
  output directory; route those files to private storage. The v2 migration reads public legacy bundles, which already
  exclude them.
* `ambient_corrected_counts`, `polya_counts`: pz2 matrices like `ambiguous_counts` / `sj_counts`. `ambient_corrected_counts`
  rows are the gene vocabulary (`gene_id_gene_name`, the vocabulary-pack order); `polya_counts` rows have inline names
  (`gene:chrom:strand:pos`), as `sj_counts`. Both have the sample's barcode columns; only called cells are nonzero.
* `polya_sites.tsv.gz`, `sj_features.tsv.gz`, `read_qc_summary.json`: stored blobs (kind 3). `sj_features` is a candidate
  for the shared junction table (`docs/format-s1.md` section 12): `star_sj_pass`, motif and annotation are per junction,
  not per sample, except the read counts.
* `cell_qc_metrics.tsv` has 11 more columns; the `cells`-table derivation does not cover them yet, so it falls back to
  stored until the cells table gains them (they are per-cell QC counts: tier-2 columns in `docs/CELL-ARCHITECTURE.md`).
* `doublet_scores.tsv`, `ambient_contamination.tsv`, `donor_assignments.tsv`: per-cell labels and scores keyed by
  barcode; candidates for the per-sample label sidecar (`v2/a/<GSM>/<d16>.parquet`), not D1.
