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

## Results

Jobs: part 1 is 804363 (build, every engine run, comparators, B7 fetch and STARsolo, plus the B6/B8 evaluations as steps inside
it); part 2 is 804377 (after 804363: scTE, SoloTE, nanopore truth, te_eval). Every number below is in `results/`.

| Oracle | Engine | Best comparator | Verdict |
|---|---|---|---|
| B6 chimeric junctions | 0 of 3 truth fusions at matched false positives on both mixes; cell recall at most 1 cell per fusion | Arriba 3 of 3 (1:1.5, 44 false positives); STAR-Fusion and Arriba 1 of 3 (1:15); 8 RUNX1–RUNX1T1 cells | **fail** |
| B7 TE counts | pending (804377) | pending | pending |
| B8 reads per UMI | 96.9% of Cell Ranger molecules matched; matched TVD 0.0153; saturation 0.7288 vs 0.7248 | Cell Ranger 3.0 is the reference | **pass** |

### B8: the 1.48 molecule ratio is intron-only molecules

The engine's histogram counts every molecule it emits, and Cell Ranger 3.0's molecule_info counts only gene-assigned exonic ones.
The engine's per-molecule dump (`SINGLET_MOLECULE_DUMP`, which reproduces the histogram file exactly) splits the 13.96 M molecules
in the 1,218 shared cells into 3.07 M exon-only (S), 6.45 M exon-and-intron (A) and 4.44 M intron-only (U). Counting S and A alone
gives a per-cell median ratio of 1.03 to Cell Ranger, while the intron-only molecules add the rest (median ratio 0.48). Joined on
barcode, gene and UMI, 96.9% of Cell Ranger's molecules are in the engine's set; 95% of S and 92% of A molecules have a Cell Ranger
twin, against 1.8% of U molecules.

On the 8.94 M matched molecules the pooled histograms differ by a TVD of 0.0153 and the implied saturation by 0.004, and 89.7% of
molecules carry the same read count (median difference 0). The engine has slightly fewer one-read molecules (18.9% against 20.5%):
it keeps the intronic reads of a molecule that Cell Ranger, counting only transcriptome-compatible reads, leaves out. The full-set
comparison on this build gives TVD 0.0084 and saturation 0.718 against 0.722 (`b8_full.json`), in line with job 803959.

### B6: the fusion reads arrive, then get the wrong gene names and drown in artifacts

Cell labels: genotype demultiplexing finds three clusters per mix (Kasumi-1 splits into two), named by globin expression. Line counts
are 5,136 K562 and 4,587 Kasumi-1 in the 1:1.5 mix and 737 and 6,832 in the 1:15 mix, with 85% and 97% agreement between
genotype and globin call.

The 1:15 mix shows the mechanism. The engine's chimeric pass finds the same 12 RUNX1–RUNX1T1 reads that STAR-Fusion reports, at
the same breakpoint (chr21:34,859,474 to chr8:92,017,364), in 7 cells. Only one reaches the table as RUNX1–RUNX1T1, because
`chimeric_table.h` `gene_at` names each side by the gene on the strand STAR reports, which is the read's strand. 10x 5' R2 reads
are antisense, so on chr21 the lookup prefers the plus-strand lncRNA ENSG00000286153 inside RUNX1. Renaming by transcript strand
(`fusion/reannotate_strand.py`: motif strand for canonical junctions, library strand for the rest) recovers 6 cells
(`b6_*_strandfix.json`).

The ranked list is the second problem. The engine's strongest pairs are recurrent non-canonical artifacts present in both mixes
(ETFA with ENSG00000242593, GPC5, LINC02479 and NDUFAF6; LINC00486–NUCKS1; CTNNA3–SLBP), each with 100 to 500 molecules, and the
table applies no blacklist. Restricted to canonical-motif junctions, and with the strand fix, the engine matches both comparators
on the 1:15 mix (recall 1 of 3 at 6 and at 39 false positives). On the 1:1.5 mix it still finds none of the three: its raw
junction file holds 4 RUNX1–RUNX1T1 reads and no BCR–ABL1 read, against Arriba's 5 and 1.

Every caller is weak on this data. No tool finds a truth fusion in more than 0.12% of its line's cells, which matches Penter et
al.'s reason for adding targeted long reads. The rule is still unmet.

Proposed fix (engine code left untouched here):
1. `gene_at`: pick the gene on the transcript strand, which is the junction strand for type 1, the opposite strand for type 2, and
   the library strand from the auto-strand probe for type 0.
2. A default call set that drops non-canonical junctions, or carries Arriba's blacklist, and flags pairs recurring at high
   support across unrelated samples.
3. Check whether main-pass reads with a long soft clip should also go to the chimeric pass. Today only unmapped reads do, which may
   explain the missing 1:1.5 support. This one is a hypothesis.
