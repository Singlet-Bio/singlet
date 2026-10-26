# mtDNA for clonality and lineage: what Singlet stores and how it is used

Status: design + benchmark, 2026-10-26. Code: `bench/mtdna_lineage/`. Raw results: `bench/mtdna_lineage/results/*.json`.
Pipeline under test: the `prod/modality-fixes` mt caller (`include/singlet/pileup/mt_heteroplasmy.h`, `mt_event_caller.h`), binary
`singlet-modfix`. Every number below comes from real libraries with lineage truth. Nothing reported is simulated. The one derived
experiment, thinning real MAESTER molecules, is labelled as such and checked against the real 3' library of the same cells.

## Answers

| Question | Answer |
|---|---|
| 1. Representation | **Both, in different places.** Private: a molecule-level per-cell x position x base x strand tensor, stored as depth by strand plus non-reference counts (13 MB for a 3,532-cell full-depth sample, against 200 MB for what Singlet writes today). Public: a cell x site AD/DP matrix of molecule counts for the somatic sites that pass the privacy rule, plus a per-site QC table (about 1 MB). The public VAF-only `mt_heteroplasmy` matrix should be retired. |
| 2. Whole-genome deconvolution | **Useful for donor and haplogroup identity, not for somatic lineage.** Per-cell mixtures of reference haplotypes assign K562 vs BT142 cells with 99.8% accuracy from 3' data alone, and pseudobulk haplogroups from 3' match MAESTER for all three genomes. Somatic structure is not a mixture of a few whole-genome haplotypes: two variants that define one clone sit at median VAF 0.17 and 0.013 in the same cells. 3' molecules read 90 bp, so variants more than about 100 bp apart are never co-observed. Site-level molecule counts are the right unit. |
| 3. Clonal inference on 3' | **Calling known clone markers works; discovering them from 3' alone mostly does not.** At the 26 MAESTER clone variants, 3' VAFs agree with MAESTER at r = 0.98, and 64% of the 1,397 clone cells carry a 3' molecule with their own variant. Discovery from 3' alone is poor: Singlet's caller finds 5 of 26, and mgatk's default rules find 0, because 3' reads come from one strand and mgatk's strand-correlation test is undefined. At full SRA depth, about half of atlas samples have the chrM depth of this benchmark. At the 30M-read cap the campaign used, 4% do. |
| 4. Privacy | **Public if the AF is below 0.9 in every donor's pseudobulk and the variant is not a common homoplasmic population variant (gnomAD AF_hom >= 0.001); everything else is private.** A sample-level "pseudobulk VAF < 0.9" rule leaks 29 of 40 donor-defining variants in a two-donor mix. Singlet's public list today carries 6 (K562 mix) and 52 (BPDCN) common population variants. |

## 1. Data and truth

MAESTER (Miller et al. 2022, Nat Biotechnol 40:1030, GEO GSE182685): the same 10x 3' v3 cDNA was sequenced twice, as a
standard 3' library (what the atlas has) and as a mtDNA-enriched MAESTER library (long reads, about 260 bp). All reads in SRA
were used. Nothing was capped.

| Sample | 3' library | MAESTER library | Lineage truth |
|---|---|---|---|
| K562 + BT142 cell-line mix | SRR15598773, 260 M reads | SRR15598774, 683 M reads | RNA cell-line label for 2,401 cells (authors' `TenX_CellLineMix_Cells.rds`); K562 subclonal variants by the authors' rule (`3.4_TenX_K562_clones.R`) on our MAESTER pileup: 27 variants |
| BPDCN712 bone marrow (clonal hematopoiesis) | SRR15598776, 539 M reads | SRR15598777, 281 M reads | 23 clones defined by 26 MAESTER variants, 1,397 cells (authors' `4.4_positive_cells.txt`, `4.2_vois.txt`) |

Processing:
- Singlet: the production binary (`singlet-modfix --pipeline --tagged-bam`) ran on each 3' library.
- MAESTER: STARsolo on the same GRCh38-2024-A index, keeping only the chrM BAM.
- Per-cell chrM allele counts: `mtpile.cpp` read both BAMs with Singlet's counting rules (MAPQ >= 20, base quality >= 10), at read
  level and at molecule level (UMI consensus; for MAESTER, molecules with >= 3 reads, as maegatk `-mr 3`).
- Reference tools on the same reads and cells: mgatk 0.9.0 (`bcall -ub UB`) and maegatk (`-mr 3`), MQuad 0.1.8 and vireoSNP 0.5.9.
- Cells:
  - BPDCN: Singlet's `called_v2` cells (15,495), evaluated where MAESTER saw >= 100 molecules (13,345).
  - K562 mix: the knee call (2,854) plus the truth cells it missed (3,532 in all), because `called_v2` fails on this sample (section 8).

**Singlet's counts are exact.** Singlet's private `mt_allele_counts.1pz` equals mtpile's read-level counts at every one of
27.8 M (K562) and 131 M (BPDCN) cell x position entries. mgatk's UMI-deduplicated counts on the same cells agree with
mtpile's molecule counts at r = 0.9997 for depth and 0.994 for non-reference counts (K562). Everything that follows therefore
scores what Singlet keeps, not a re-implementation.

## 2. Coverage: what 3' 10x data contains

| | K562 mix 3' | BPDCN 3' | K562 MAESTER | BPDCN MAESTER |
|---|---|---|---|---|
| Molecule depth per cell, median (mean) | 9.8 (17.3) | 5.3 (6.9) | 62 (110) | 14.7 (19.1) |
| chrM positions with >= 1 molecule, median cell | 70% | 63% | 80% | 72% |
| chrM positions with >= 5 molecules, median cell | 32% | 27% | 70% | 50% |
| Reads per molecule (chrM) | 1.2 | 2.9 | 4.3 | 13.3 |
| Reference bases read per molecule, median | 90 (one read) | 90 | 250 | 260 (span incl. N skips 580) |

- **Strongly position-biased.** 3' reads pile up at the 3' ends of the mt transcripts: 100-bp bins range from 0.04 to 230
  molecules per cell (`results/lines.json`, `pooled_profile_100bp`). Clone variant 1171A>G (12S rRNA) is covered in 0.9% of BPDCN
  cells, while 9753G>A is covered in 95%.
- **Far from saturated.** The K562 3' library has 1.2 reads per chrM molecule, so more sequencing adds molecules almost one for
  one. This is why the atlas read cap matters (section 5).

## 3. Representation and size (question 1)

Measured at full depth, cells as above, zstd-19 over varint fields (`analyze.py size`, `encode_sizes`):

| Encoding | K562 3' (3,532 cells) | BPDCN 3' (15,495 cells) | BPDCN MAESTER |
|---|---|---|---|
| Singlet today, private `mt_allele_counts.1pz` + `_rev` (read level, **all** 204k / 944k barcodes) | 199.5 MB | @@B_SING@@ | n/a |
| E1 full tensor, cell x pos x base x strand, molecule level | 29.6 MB | @@B_E1@@ | @@M_E1@@ |
| E3 depth by strand + non-reference counts only, molecule level (**proposed private**) | 13.1 MB | @@B_E3@@ | @@M_E3@@ |
| E4 as E3 without strand | 12.4 MB | @@B_E4@@ | @@M_E4@@ |
| E5 cell x site AD/DP, 500 most variable sites (**proposed public**) | 1.2 MB | @@B_E5@@ | @@M_E5@@ |
| Singlet today, public `mt_heteroplasmy.1pz` (VAF x 10^4, "het" cells only) | 1.1 MB | | |
| For scale: `exon_counts.1pz` | 48.5 MB | | |

Why both:
- **The private tensor is what a re-analysis needs.** Anything not stored is lost when the BAM is deleted. Examples: a new
  caller, the authors' variant list for one study, a new mask, haplogroup calls, donor demux by mtDNA.
- **Strand is cheap, so keep it, but it tells nothing in RNA libraries.** E3 with strand is 5% larger than without. At each
  chrM position, 98.6-98.9% of 3' molecules and 99.96% of MAESTER molecules come from one strand (`results/facts.json`). In these
  libraries strand cannot separate artefacts from variants. It does in libraries read on both strands: ATAC, full-length plate
  protocols, and 5' libraries with paired reads.
- **Molecule level, not read level.** PCR duplicates make read counts overdispersed for binomial clone models. At about 3 reads
  per molecule for full-depth BPDCN they also inflate apparent depth. MAESTER needs the UMI consensus (13 reads per molecule).
- **The public AD/DP matrix is what clone tools consume.** MQuad, vireoSNP, cardelino and Signac all take alt and total
  molecule counts per cell and site. The current public matrix cannot be used for this:
  - It stores a VAF only for cells Singlet calls "het": 3.4% (K562) and 0.5% (BPDCN) of the covered cell-site pairs, never a
    zero and never a depth. A reference-only covered cell and an uncovered cell look the same.
  - Its stored values are mostly wrong: 35% (K562) and 83% (BPDCN) of entries have MAESTER VAF < 0.01 in the same cell.
  - Against MAESTER its values correlate at r = 0.28 (K562) and 0.86 (BPDCN), while the raw molecule counts at the same
    places agree at r = 0.98 (BPDCN, 26 clone sites).

## 4. Whole-genome deconvolution (question 2)

**Donor / haplogroup identity: feasible from 3' data.**
- K562 and BT142 differ at 22 homoplasmic sites (MAESTER pseudobulk, AF >= 0.9 in one line and <= 0.1 in the other).
- A per-cell maximum-likelihood mixture of the two reference haplotypes, from 3' molecules alone:
  - assigns 99.83% of the 2,401 cells to their RNA cell line;
  - every cell has informative molecules (median 164);
  - MAESTER gives 99.79%;
  - 7.3% of cells have a minor-line fraction between 0.1 and 0.9 (ambient RNA and doublets).
- Singlet's own donor demux (no SNP panel) separates the two lines with ARI 0.998 against RNA.
- Haplogroups (Haplogrep 3.2.2, PhyloTree 17) from pseudobulk homoplasmic calls are identical between 3' and MAESTER for all three
  genomes. They are private (section 6), so they are not printed here; they are in `$WORK/out/mtdna_lineage/private_vcf/*.hg.txt`
  on Clipper.

| Genome | 3' (Haplogrep quality, homoplasmic variants) | MAESTER (quality, variants) | Same haplogroup |
|---|---|---|---|
| BPDCN712 donor | 0.95, 42 | 0.85, 35 | yes |
| K562 | 0.91, 39 | 0.93, 42 | yes |
| BT142 | 0.71, 10 | 0.85, 15 | yes |

**Somatic lineage: whole-genome haplotypes are the wrong unit.**
- **One clone, two VAF levels.** A clone is not one new genome that a cell either carries or lacks. In BPDCN MAESTER molecules
  (cells with >= 5 molecules at both sites), the three clones defined by two variants each behave differently:
  - 10158T>A + 6293T>C (32 cells): equal VAF in every cell (both near 1), consistent with one haplotype.
  - 6205G>A + 9164T>C (118 cells): both near-homoplasmic, but 20% of cells differ (p < 0.001 per cell, pooled p ~ 1e-270).
  - 1415G>A + 9753G>A (28 cells): median VAF 0.17 vs 0.013 in the same cells, and 18% of cells differ significantly.

  A model of K whole-genome haplotypes per cell would need a new haplotype for every distinct VAF level of every variant. Per-site
  VAFs (AD/DP) represent this directly.
- **Short reads cannot phase.** 3' molecules read 90 bp: 12.9 of 14.3 M K562 molecules cover 90-99 reference bases. Of the 3' molecules
  at a BPDCN clone site, 10% also cover a second clone site, and almost all such pairs are 13-116 bp apart. MAESTER molecules
  span a median 580 bp, and 58% cover two clone sites. Variants kilobases apart, like 6205 and 9164, are never seen on one 3' molecule.
- **Where the models agree.** When a clone's variants are near-homoplasmic in its cells (15 of 23 BPDCN clones have median VAF >= 0.9 in MAESTER), a
  per-cell "which haplotype" call and a per-site call give the same answer. Storing site-level counts loses nothing there either.

## 5. Clonal inference from 3' data (question 3)

**Known markers can be read in 3' data.** For the 26 BPDCN clone variants (`results/bpdcn.json`, `per_variant_3p`):
- **Accuracy:** 3' molecule VAF agrees with MAESTER at r = 0.98 (mean absolute difference 0.0005; 99.97% of 3' counts are
  binomially consistent with the MAESTER VAF; MAESTER depth >= 20).
- **Sensitivity:** 64% of the 1,397 clone cells carry >= 1 3' molecule with their own variant (82% have their site covered).
  Per variant it ranges from 94% (683G>A) and 86% (1323G>A, 1222A>G) down to 26% (15140G>A, 3628A>G) and 0% (1171A>G, 4104A>G),
  following the 3' coverage profile.
- **Specificity:** among cells MAESTER calls reference (>= 20 molecules, 0 alt), <= 0.7% show a 3' alt molecule at 25 of 26 sites
  (3.7% at 10158T>A).
- **K562 subclonal variants:** the 27 low-heteroplasmy variants give r = 0.69 against MAESTER; 99.9% of counts are consistent.

**Per-cell depth matters less than position.** Real data, no thinning (`results/depthcurve.json`): splitting the clone cells
into deciles of their own 3' chrM depth (214 to 40,600 reads per cell), the fraction showing their marker stays at 0.56-0.70
up to 4,200 reads per cell. It reaches 0.78 only in the top decile. Most of the loss is variants in poorly covered transcript
regions, not shallow cells.

**Discovery from 3' alone is the weak step:**

| Caller on the 3' library | K562: 27 subclonal truth variants | BPDCN: 26 clone truth variants |
|---|---|---|
| Singlet public list (`mt_variants.tsv`) | 11 found / 68 calls | 5 found / 694 calls |
| mgatk rules (n_cells_conf_detected >= 5, strand r >= 0.65, log10 VMR > -2) | 0 / 0 | 0 / 0 |
| MQuad (deltaBIC knee; candidates with >= 5 carrier cells) | 7 / 32 (20 of the 32 are donor germline) | @@MQ_B@@ |

- **mgatk calls nothing on 3' data.** Its "confidently detected" cells need alt reads on both strands. 3' reads come from one
  strand, so the strand correlation is undefined or near 0 at almost every site.
- **Singlet's list is mostly noise.** Only 7 of the 68 K562 public sites reach pooled AF >= 0.5% in the MAESTER library of the
  same cDNA, which averages 444 molecules per position per cell. The true sites Singlet misses fail its over-dispersion test or its
  minimum carrier-cell count (`singlet_truth_sites_evaluated`).
- **MQuad's top calls in a pool are the donors' haplotype variants.** In the cell-line mix 20 of its 32 calls are line-defining
  homoplasmic sites, so MQuad output cannot be published as-is.

**Clone assignment once markers are known.** vireoSNP binomial mixture over the 26 truth variants, scored on the 1,397 clone
cells:

| Counts given to vireoSNP | ARI vs truth | Clones with F1 >= 0.5 |
|---|---|---|
| MAESTER molecules | 0.61 | 18 of 23 |
| 3' molecules | 0.35 | 15 of 23 |

K was chosen by the ELBO elbow on a 10-30 grid, and both runs took K = 30, so both over-split. The F1 for clones whose variants
3' covers well (683G>A, 6205G>A-9164T>C, 10158T>A-6293T>C, 11864T>C) is 0.84-0.95 from 3' alone.

**Enrichment vs depth (derived, real molecules).** Thinning the real BPDCN MAESTER molecules: at 19x mean depth 91% of clone
cells show their marker, at 5.7x 74%, at 1.9x 50%, at 0.57x 29%, at 0.19x 14%. The real 3' library at 6.9x mean depth reaches only
64%, below thinned MAESTER at 5.7x. Thinning keeps MAESTER's coverage profile and misses the 3' bias, so thinned curves overstate
what 3' data can do. The real-library numbers above are the ones to use.

**Atlas.** Survey of 747 human samples in 400 random atlas studies; 630 samples (339 studies) with >= 100 cells and an SRA read
count are tabulated (`atlas_survey.py`, `atlas_depth.py`, `results/atlas_survey_depth.csv`):
- **Metric.** chrM reads per called cell, from each sample's `pileup_stats.json` (`chrm_reads` x fraction of reads in cells / cells).
  On this scale the BPDCN 3' library is about 2,400.
- **The cap.** The campaign read at most 30 M reads per sample (88% of surveyed samples hit it). SRA holds a median of 256 M reads per
  sample, so the campaign used a median 11% of what exists.

| chrM reads per cell | >= 200 | >= 500 | >= 1,000 | >= 2,000 (about BPDCN) |
|---|---|---|---|---|
| Atlas as processed (30 M cap) | 49% | 23% | 11% | 4% |
| Atlas at full SRA depth (projected) | 81% | 75% | 67% | 49% |

The projection scales chrM reads linearly with reads used. It is an upper bound: GSMs whose extra runs are other library types
(VDJ, feature barcodes) inflate the SRA count.

What the atlas can support:

| Tier (chrM reads per cell, full depth) | What it supports | Share of atlas samples |
|---|---|---|
| A (>= 1,000) | Donor and haplogroup identity, plus somatic clone calls for clones whose markers are known or are high-VAF and in well-covered regions (about 60% of clone cells, as BPDCN) | 67% |
| B (200-1,000) | Donor identity; somatic calls only for the best-covered sites, with sensitivity not benchmarked below about 200 | 14% |
| C (< 200) | Donor and haplogroup identity from the pseudobulk; no per-cell somatic lineage | 19% |

MAESTER-enriched libraries are a different class (median 15-60 molecules per position per cell, over 72-80% of chrM). The atlas holds them only where a
study deposited them. They are worth ingesting as a separate modality when present (GSE182685 is one).

## 6. Privacy split (question 4)

Evaluated on the variable sites a caller could publish: Singlet's public list, MQuad's calls, and candidates with pooled AF
>= 0.5% in >= 5 cells. Donor groups are Singlet's own demux (2 donors in the K562 mix, 1 in BPDCN).

| Public if ... | K562 mix: donor germline leaked (of 40) | common homoplasmic leaked | somatic truth kept | BPDCN: germline leaked (of 42) | common homoplasmic leaked | somatic truth kept |
|---|---|---|---|---|---|---|
| R1 sample pseudobulk AF < 0.9 | 29 | 44 | 13 / 13 | 0 | 64 | 9 / 9 |
| R2 AF < 0.9 in every donor pseudobulk | 0 | 19 | 13 / 13 | 0 | 64 | 9 / 9 |
| **R3 R2 and gnomAD AF_hom < 0.001** | **0** | **0** | **11 / 13** | **0** | **0** | **9 / 9** |
| R4 R2 and absent from gnomAD (AF_hom = 0, AF_het <= 0.001) | 0 | 0 | 8 / 13 | 0 | 0 | 6 / 9 |

"Common homoplasmic" means gnomAD v3.1 AF_hom >= 0.001: a haplogroup marker. "Somatic truth kept" counts the truth variants
present among the plausible sites.

Rule (recommended): **public = AF < 0.9 in every donor pseudobulk AND gnomAD v3.1 AF_hom < 0.001. Everything else is private.**
- **Why per donor.** A pooled sample dilutes a donor's homoplasmic variants to that donor's share of the pool, so a sample-level
  0.9 cut publishes them (R1 leaks 29 of 40 in a two-donor mix). Demux comes first, and the cut is applied per donor.
- **Why gnomAD.** It is a backstop when demux misses a donor; auto-K was broken in the earlier genotype-pool benchmark. It also
  removes population variants seen at partial AF, which are either ambient RNA from another donor or germline heteroplasmy. The
  cost is 2 of 13 K562 somatic variants. R4, which also drops anything gnomAD saw as heteroplasmic, removes recurrent somatic
  hotspots and costs too much (8 of 13 kept).
- **What stays private.** The haplogroup, the per-donor homoplasmic calls and the whole count tensor (it contains the haplotype at
  every covered position).
- **Today's outputs violate this.** Singlet's public `mt_variants.tsv` carries 6 (K562) and 52 (BPDCN) common homoplasmic
  population variants. MQuad's calls in a pool are mostly donor germline (20 of 32).

## 7. Proposed output schema change (not implemented; `python/singlet/v2` unchanged)

Per sample, replacing the present mt members:

| Member | Visibility | Content | Replaces |
|---|---|---|---|
| `mt_counts` | private (restricted sidecar, like `snp_ad`/`snp_dp`) | Molecule-level (UMI consensus, >= 75% of reads; read count >= 1) tensor for called cells: per (cell, position) depth on each strand plus non-reference base counts on each strand (encoding E3). Feature space: rCRS 16,569 positions. About 13 MB for 3,532 full-depth cells. | `private/mt_allele_counts.1pz`, `private/mt_allele_counts_rev.1pz` (read level, all barcodes, 200 MB) |
| `mt_sites_private` | private | Every evaluated site with class and rejection bits (today's `private/mt_sites.tsv`), plus per-donor homoplasmic calls and the Haplogrep haplogroup | `private/mt_sites.tsv`, `private/mt/donor*_mt_variants.vcf` |
| `mt_sites` | public | Sites passing rule R3, with QC: pos, ref, alt, pooled depth and AF, n cells covered / with alt, alt fwd/rev, over-dispersion, VMR, MQuad deltaBIC | `mt_variants.tsv` |
| `mt_ad`, `mt_dp` | public | cells x `mt_sites` molecule counts (alt, total), every covered cell, zeros implied by `mt_dp` | `mt_heteroplasmy.1pz` |
| cells table columns | public | `mt_molecules`, `mt_mean_depth`, `mt_frac_covered` per cell | none |

Catalog (D1, per `docs/CELL-ARCHITECTURE.md`: per-sample summaries only):
- `sample_property` `mt_depth_tier` (A/B/C above), `n_public_mt_sites`;
- `sample_cell_summary` rows for `mt_mean_depth` quantiles.

Clone labels, once computed in batch (vireoSNP or MQuad, versioned), are a per-cell label sidecar joined by `(sample ordinal, col)`
and registered as a `cell_feature`.

This fits the architecture as written:
- raw counts stay in immutable per-sample objects;
- genotype-like data stays out of public objects;
- D1 holds only per-sample summaries.

One addition is worth recording in the canonical `CELL-ARCHITECTURE.md` (singlet-website first): **somatic mtDNA sites that pass
rule R3 are not genotypes and may be served; the haplogroup and homoplasmic calls are genotypes and may not.**

Pipeline changes this implies (`mt_heteroplasmy.h` / `export.h`):
1. Count UMI-consensus molecules for chrM, as for genes, not reads.
2. Restrict the private tensor to called cells.
3. Write `mt_ad`/`mt_dp` for every covered cell, not VAF for "het" cells.
4. Apply R3 (needs demux first, plus a gnomAD AF_hom >= 0.001 site list shipped with the reference: 1,579 of gnomAD v3.1's 15,770 chrM SNVs).
5. Make the strand-correlation and both-strand tests depend on library type. They cannot pass on 3' or MAESTER data, which are
   98.6-99.96% single-strand per position. Keep them only for libraries read on both strands (ATAC, full-length plate protocols).

## 8. Defects found

1. Public `mt_heteroplasmy.1pz` stores no depth and no zeros, covers 0.5-3.4% of covered cell-site pairs, and is mostly
   unsupported by MAESTER (section 3).
2. Public `mt_variants.tsv` is mostly noise (7 of 68 K562 sites supported; BPDCN recall 5 of 26 with 694 calls) and contains
   common population variants (section 6).
3. `private/mt_allele_counts*.1pz` covers every barcode (204k and 944k columns), at read level: 15x the proposed size.
4. `called_v2` overcalls the K562/BT142 cell-line mix: 126,598 cells, of which 123,744 are added by the EmptyDrops step with median
   312 UMIs, while the knee calls start at 8,134 UMIs. On BPDCN the 13,612 added calls have median 6,340 UMIs and look real, so
   the knee is too strict there. Owner: the called_v2 work (PR #23 lineage).
5. mgatk 0.9.0 from bioconda imports `bam-dedup`, which the package omits (`setup_env.sbatch` installs it).

## 9. Caveats

- Two samples and one chemistry (10x 3' v3). The clone truth is the authors' MAESTER calls, so the 3' sensitivity is measured
  against clones MAESTER can see. Clones MAESTER misses are not counted.
- K562 subclone truth uses the authors' variant rule on our MAESTER pileup (27 variants). The paper validated its variants with
  bulk ATAC; we did not repeat that.
- Molecule counts are Singlet-rule counts (MAPQ >= 20, BQ >= 10). maegatk on K562 MAESTER (3,532 cells) was still running when
  this was written. mgatk and maegatk on BPDCN ran on a 3,000-cell subset (all clone cells plus random others), because their
  per-cell steps take more than 10 h on 15k cells.
- The vireoSNP K grid (10-30) was too narrow: both runs took K = 30. ARI therefore understates assignment quality; per-clone F1 is
  the better summary.
- The atlas full-depth projection is linear in reads and an upper bound (section 5).

## 10. Reproduce

`bench/mtdna_lineage/README.md` lists the jobs in order. Clipper paths: data `$WORK/data/maester/`, outputs
`$WORK/out/mtdna_lineage/`, private haplogroup VCFs `$WORK/out/mtdna_lineage/private_vcf/` (not committed).
