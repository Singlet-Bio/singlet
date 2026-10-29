# E3: One gene-level code for exon and intron: the chain-rule hierarchy made concrete

> **Status 2026-10-21 23:30: downgraded.** Idea 3 measured the exon ↔ intron conditional gain at 0.06-0.17 bits per
> intron gene-nonzero for PBMC and nuclei (0.42 for 5' T cells), so the "anticipated result" below is falsified as a
> bits lever. What survives: the gene-level hierarchy for model size and out-of-sample transfer (gene rates transfer
> between studies, interval tables do not), and one code path for both matrices. Run E3 only as a container-layout
> experiment (one gene-interval member per sample, the two matrices as views) after E2 lands; expect ≤ 5% bits.

Context: `notes/theory_fable.md` sections 2, 4 (corollary) and 7 (T4). Today a sample stores exon_counts and
intron_counts as two independent interval × barcode matrices. Both are views of one object: per cell, per gene,
a multiset of UMIs each landing in one interval of that gene (exonic or intronic). Coding the gene total first,
then the exon/intron/interval split given the total, captures I(exon; intron) (m: ~10% of intron presence bits in
a 3' tumour sample; expected much larger for nuclei) and the interval entropy exactly once.

## Design

Per cell c: gene totals T_{c,g} = exon + intron UMIs of gene g (a sparse vector over ~38k genes), coded under the
adaptive per-gene × depth model (E2's coder, feature space = genes). Then for each (c, g) with T > 0:
1. the exonic fraction: number of exonic UMIs out of T, a binomial with gene-specific, block-adaptive probability
   (nuclear fraction is cell-type dependent; test conditioning the probability on depth bucket and on the
   cell's running intron fraction, which is known from already-decoded genes of the same cell);
2. the exon interval split (multinomial over the gene's exon intervals, adaptive per-gene table) and the intron
   interval split likewise.
Both matrices are reconstructed exactly from (totals, fractions, splits). Column alignment: the two files' barcode
sets may differ (verify; if so code the union with presence flags).

## Measurements

Per sample: bits for (gene totals) + (exon fraction) + (splits) versus (Track/Idea best separate exon) + (best
separate intron); the exon-fraction entropy per (c, g) with and without the running-intron-fraction context;
I(exon; intron) at gene level per sample (Idea 3 B measurements, extended to all 7 samples); decode time.

## Deliverables

- `harness/e3_joint.cpp` (loads both matrices, encodes jointly, decodes both, verifies both exactly), `kernels/e3*.h`,
  `results/e3_joint.tsv`, `notes/e3.md` with the per-sample saving table and the recommended container layout
  (one gene-interval member per sample instead of two interval matrices, with the two classic matrices as views).

## Anticipated result

Nuclei (brain_sn_*): 15-30% fewer bits for exon+intron together than the best separate coding; 3' cells: 5-10%;
5' T cells: ~5%. Decode cost: same order as E2 (one extra binomial symbol per (cell, gene) with T > 0).

## Depends on

Idea 3 results (measurements and first codec) and E2's coder. Start after Idea 3's notes exist.
