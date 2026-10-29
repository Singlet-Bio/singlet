# E5: Does the theory transfer to other single-cell omics members? (sj counts, PSI, ATAC, VDJ, heteroplasmy)

> **Status 2026-10-22: not run.** Partly answered by E4 (splice_psi is derived from sj_counts; heteroplasmy members in
> the corpus are empty stubs). Remaining: junctions given gene counts, ATAC.

Context: `notes/theory_fable.md` section 1 (the decomposition is generic: any sparse count matrix from a sampling
process has a sampling floor plus a state term plus redundancy). The `.singlet` bundle carries, beyond exon/intron
counts: sj_counts (splice junctions × cells, uint32), splice_psi (float), vdj_gene_usage, mt_heteroplasmy (float),
and for other assays ATAC peak/bin matrices and ADT counts. This plan maps each onto the framework and finds which
are (a) derived (zero bits), (b) count matrices with a conditional structure (code given gene counts), or (c)
genuinely new information.

## Design

For the whole-bundle sample set (GSE145370, 14 samples) and the corpus samples:
1. sj_counts: junction counts are a sub-sampling of the gene's UMIs: a junction UMI is also a gene UMI. Measure
   H(sj counts | gene exon counts of the junction's gene, depth) versus H(sj counts | depth): per sample bits per
   junction nonzero. If the conditional cost is < 50% of the unconditional, sj belongs in the gene-level
   hierarchy (E3) as another split level.
2. splice_psi: confirm it is a pure function of sj_counts (Part A of E4); if yes, zero bits.
3. vdj_gene_usage: tiny; measure order-0 vs per-cell-type; likely leave as is.
4. mt_heteroplasmy: float matrix; is it (alt, depth) pairs or a ratio? If a ratio of integers, store the integers.
5. ATAC (if any sample in the atlas has it; else synthesise from a public fragment file): peaks × cells binary-ish
   matrix; the framework predicts per-peak × depth adaptive coding near optimal with state term small (most
   cells have 1-2 reads per open peak). Measure order-0 vs per-peak × depth cross-entropy on one sample.

## Deliverables

- `tools/e5_*.py`, `results/e5_*.tsv`, `notes/e5.md` with one table: member type, bytes today, derivable?,
  conditional structure available?, predicted compressed share, and the recommended treatment.

## Anticipated result

sj_counts conditional on gene counts loses 40-60% of its bits; splice_psi is derived; heteroplasmy is better as
integer pairs; ATAC behaves like a shallow count matrix (per-peak × depth adaptive is within 5% of anything
fancier). Net: the hierarchy gene → (exon|intron) → (intervals, junctions) → PSI covers the transcriptomic members.

## Depends on

E4 Part A for derivability; E3 for the hierarchy coder. Measurements can start now.
