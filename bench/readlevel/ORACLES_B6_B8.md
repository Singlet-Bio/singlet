# Launch oracles B6, B7 and B8: read-level outputs against open tools

Pass rules fixed on 2026-11-07 at 20:45 EST, before any score was computed. Bar (D-102): meets or exceeds the best open tool on that
tool's benchmark data. "Meets" allows a tie band of 0.005 on correlation and AUROC scores; nothing else is rounded in the engine's favor.

## Build under test

The frozen engine b0a96f49 (gpu/frontier) carries neither `--chimeric` nor TE counts: both live on unmerged feature branches. The oracles
therefore run on `bench/readlevel-oracles`, which is b0a96f49 with feat/chimeric-junctions (99e4ea69) and feat/te-counts (46b808ba)
merged on top. Both merges are conflict-free. Engine source is identical at 13ee63ac and the later bench-only commits.

## Pass rules

**B6, chimeric junctions** (Penter et al. 2024, GSE243227: K562 and Kasumi-1 mixed 1:1.5 and 1:15, 10x 5'). Truth: BCR–ABL1 and
NUP214–XKR3 in K562, RUNX1–RUNX1T1 in Kasumi-1. Cells are labelled by genotype (engine donor demultiplexing) and named by globin
expression, with no fusion information. Comparators: STAR-Fusion 1.15.1 and Arriba 2.5.1 at their README defaults on the cDNA read.
B6 passes when, on both mixes,
1. sample level: the engine's recall over labelled cells, read at each comparator's false-positive count, is at least that comparator's
   recall; and
2. cell level: for every truth fusion, the engine's fraction of positive-line cells with a supporting molecule is at least the best
   comparator's, while its fraction of other-line cells stays at or below the larger of that comparator's rate and 0.5%.

**B7, TE counts.** Comparators: scTE and SoloTE (GitHub heads recorded in the job log) on a STARsolo BAM, with the engine's TE
annotation. B7 passes when
1. A375 (GSE212945, the same library on nanopore as truth): the engine's Spearman correlation of log CPM with the long-read subfamily
   abundance is within 0.005 of the best comparator or above it, both over all subfamilies and over the young subfamilies; and
2. mESC 2C-like cells (E-MTAB-8869, positives = Zscan4-on cells from STARsolo gene counts): the engine's AUROC of the MERVL fraction is
   within 0.005 of the best comparator or above it.

**B8, reads-per-UMI histogram** (pbmc_1k_v3; Cell Ranger 3.0 molecule_info is the reference, as no open tool writes this histogram).
B8 passes when, over the cells both callers keep,
1. at least 90% of Cell Ranger's molecules appear in the engine's molecule set (same barcode, gene and UMI);
2. on those matched molecules the pooled 16-bin histograms differ by a total variation distance of at most 0.02; and
3. the saturation the two matched histograms imply differs by at most 0.01.
