# Gate: single-donor samples are not split by the donor demultiplexer

Fixed 2026-11-11 before any scoring. Decisions D-138, D-102, D-109. Engine frozen (v1.3.7); nothing here changes the engine.

## Question
When a library holds one donor (or one clonal line), does the auto-K demultiplexer report one donor?

## Truth
Single-donor by design: the depositor describes one individual or one clonal cell line per library. This is design truth, not genotype-confirmed truth. Libraries are labeled as such in every table.

## Libraries (fixed list, all pilot libraries with stored donor output; marked held out in benchmark-store, never used to tune the demultiplexer)
GSM5130155 (T cells, 5p, donor F), GSM5130163 (T cells, 5p, donor M), GSM5171610 (T cells, 5p TCR), GSM7064234 (retina snMultiome, M), GSM7884577 (retina snMultiome, F), GSM8183378 (gastric epithelium), GSM8164511 (HIO organoid, one line), GSM9038586 (MDA-MB-231 xenograft, one human line plus mouse stroma).
Excluded because multi-donor by design: GSM8085472 (10-donor pool), GSM4987144 and GSM4987148 (cell-line mixes), GSM5682562 (hashed pool).
Source: pilot/<GSM>/s1cpu_r/out/donor_assignments.tsv where present, else pilot/<GSM>/s1cpu_0ea48783/out/donor_assignments.tsv. Both are reported when both exist; the pre-freeze build is named in the results.

## Definitions (per library)
- Labeled cells: rows with cell_status "cell" and a donor label donor<k> (posterior prob_max >= 0.9 is already applied by the demultiplexer; unassigned and doublet rows are not labeled).
- Dominant donor: the label with the most labeled cells.
- False-split rate = labeled cells carried by non-dominant donor labels / labeled cells.
- Split declared (product rule from the call-layer design): a non-dominant donor with at least 50 cells and at least 1% of the sample, which is the condition under which a donor label would be published.
- Genotype doublet fraction is reported separately (expected near 0 for K=1); it is not part of the pass rule.

## Pass rule
1. Pooled false-split rate over all libraries below 0.5%.
2. No library has a split declared under the product rule.
3. Every library has a per-library rate below 0.5%; any exception is listed with its cause.
Verdict: all three pass means the donor label is enforced for K>=2 samples; a split declared in any library withholds the donor label until fixed.

## Not covered
Single-donor libraries with genotype truth (Osorio single lines, GSE230239 single lines) have no stored demultiplexer output; see results.md for the cheapest route.

## Amendment A (2026-11-11, before the in silico run; the pilot-library results above were already seen when this was written)
Second test with genotype truth: the Kang 2018 batch 2 control lane (GSE96583), eight donors. For each donor, the demultiplexer is run alone on the truth singlets of that donor (demuxlet identity from the authors' table), using the stored allele counts of the pool (private snp_ad and snp_dp, all barcodes) and the standalone harness bench/modalities/demux_fix/demux_harness.cpp compiled against the frozen headers (tag v1.3.7, donor_demux.h blob aeef964e), K auto (-1), default settings, seed 42. Same definitions and pass rule as above. Eight subsets are eight tests of K=1 on real data with known single donor; each is a library for the pooled rate. Caveat: the subset is a computational restriction of a pool, so ambient and doublet structure of a true single-donor library are absent.
