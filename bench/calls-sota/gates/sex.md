# Gate: chromosomal sex call (`sex_chrom`)

Fixed on 2026-11-11, before any held-out scoring. Owner decisions D-138, D-102, D-109. The engine is frozen (v1.3.7); the call is
computed after the freeze from stored counts by `python/singlet/celllabels/sex.py`.

## Classes

Output vocabulary of the labeller: `female`, `male`, `ambiguous`.

- `female`, `male`: scored against truth.
- `ambiguous` is the abstain class. It covers three situations: low evidence (no XIST and no chrY UMI in a cell whose group has no
  confident sex), mixed signal (XIST and chrY UMIs both present in one cell, the signature of a cross-sex doublet or of ambient
  contamination), and a group whose male fraction is not decided. An abstention counts as wrong in accuracy and is reported
  separately as the abstention rate.
- XXY, X0, mosaic loss of chrY and X-inactivation escape cannot be scored here. No donor in the truth sets carries a known
  karyotype other than XX or XY. They are listed as untested, and no claim is made about them.
- Cross-sex doublets have no truth in these sets and are not scored. Only the false-flag rate (abstentions in single-sex donors) is
  reported.

## Truth sets

Tier 1, independent of expression. Donor sex is the 1000 Genomes sex of the individual and every cell is mapped to its donor by
library of origin or by genotype demultiplexing.

- Development, tier 1: the in-silico pool of two 1000 Genomes lines of opposite sex, HG00362 (female) and HG01277 (male), GSE230239.
  Truth for each cell is the library its barcode's reads came from (barcodes found in both libraries are dropped).
- Held-out, tier 1: the uninfected library of GSE205796 (scHi-HOST), a pool of 1000 Genomes lymphoblastoid lines of both sexes. Truth
  for each cell is the 1000 Genomes sex of the line assigned by the authors' genotype demultiplexing (`Sex` and `LCL` columns of
  `GSE205796_Uninfected_SeuratMetadata.csv.gz`). Never used for any threshold, tuning or rule choice. A repository-wide search of
  earlier sex scripts and results found no use of it. It is scored once, after this file is committed and the dev scores are final.
  The infected library of the same series (87 percent viral reads) is not scored.
- Single-sex negative control (not gated): osorio2019 LCL mix, two female 1000 Genomes lines.

Tier 2, weaker, labelled distinctly in every table as "GEO metadata": 171 samples (18 studies, 472,363 cells) whose GEO sample
characteristics give the donor's sex. The depositor's record is not independent of a donor's expression in every case (three
samples were adjudicated with expression in the earlier validation), so tier 2 never decides the gate. It is reported to show
behavior across tissues, chemistries and depth. Samples with recorded sex contradicted by all tools are scored as recorded.

## Methods

- Singlet: `singlet.celllabels.sex.call_sex`, in the deployed mode (donor groups where demultiplexing exists, otherwise the whole
  sample as one group) and in the ungrouped mode (the whole library as one group, the like-for-like case for per-cell tools).
- Two-gene rule: female if XIST has at least one UMI and chrY (non-PAR genes used by Singlet) has none; male if chrY has at least one
  UMI and XIST has none; abstain otherwise. Implemented here from the usual description of the rule.
- cellXY (phipsonlab/cellXY 0.99.0, GPL-3.0, confirmed from the repository license): comparator only, run as an installed R
  package with its defaults (`classifySex`, qc = TRUE). No cellXY code is copied or ported.
- Counts: the stored Singlet counts (exon plus intron) wherever they exist. For the held-out library the author's deposited
  Cell Ranger counts are used, because the library has no Singlet run; the call reads only XIST and non-PAR chrY genes.

## Metrics

1. Per-cell accuracy among cells of tier-1 donors (abstention counts as wrong), macro-averaged over donors, and the call rate.
2. Per-donor accuracy: a donor is correct when its call matches. For every method the donor call is the majority of that donor's
   called cells (a tie or no called cell is an abstention and counts as wrong). Singlet's own group call is reported beside it.
   Report the abstention rate and the number of donors. Donors with at least 200 cells only.
3. Per-cell precision (accuracy among called cells).

## Bar

- Per-donor accuracy at least 0.99 on donors with 200 or more cells (tier 1, development and held-out each).
- Singlet per-cell accuracy, deployed mode, at or above the best comparator on tier 1 development and on the held-out library.
  "At or above" means the point estimate is not below the best comparator's.
- Singlet per-cell accuracy, ungrouped mode, at or above the best comparator's per-cell accuracy among cells the comparator calls
  (precision), reported as the like-for-like row.
- Tier 2 is reported without a bar.

A Singlet number that misses the bar triggers one fix route (a permissive-license port or a model adjustment), done on development
data only. The held-out library is then scored once with the final call.

## Verdict rule

- Both bars met on development and held-out: enforce as an annotation (never a filter).
- Donor bar met, per-cell bar missed against a comparator: ship as metadata, hidden by default.
- Donor bar missed: withheld.
- Runtime ceiling: all calls together at most 2 core-minutes per ten thousand cells; the sex call's share is measured and reported.
