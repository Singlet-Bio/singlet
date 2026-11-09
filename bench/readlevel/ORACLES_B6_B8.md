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
it); part 2 is 804425 (scTE, SoloTE, nanopore truth, te_eval, and the B6 fix builds and reruns as steps). Every number below is
in `results/`.

| Oracle | Engine | Best comparator | Verdict |
|---|---|---|---|
| B6 chimeric junctions | Arriba chimeric settings (#220, e967c1e7) + soft-clip 15 + combined score: both clauses met on both mixes. Without the Mitelman prior, 1:1.5 clause 1 reaches 2 of 3 against Arriba 3 of 3 | Arriba 3 of 3 (1:1.5, 44 false positives); STAR-Fusion and Arriba 1 of 3 (1:15); 8 RUNX1–RUNX1T1 cells | **pass, prior-dependent** |
| B7 TE counts | pending: the A375 comparators finish in 804425 or 804512; mESC comparators rerun in 804512 after a mate swap | pending | pending |
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

### B6 rescore after fixes 1 and 2 (Singlet-Bio/singlet#220)

Fixes 1 and 2 are in `fix/chimeric-strand-calls` at a704996d: transcript-strand gene names, canonical-motif junctions only by
default, Arriba v2.5.1's GRCh38 blacklist, and a `filter=recurrent` flag for the 149 pairs that recur with at least two molecules
in two of three PBMC samples (`results/chimeric_recurrent_GRCh38.tsv`). Scoring keeps only PASS rows and uses the committed rule
unchanged. All runs were steps inside job 804425 on one node.

| Run | 1:1.5 recall at Arriba's 44 FP | 1:1.5 cells (BCR–ABL1, NUP214–XKR3, RUNX1–RUNX1T1) | 1:15 recall at 6 / 39 FP | 1:15 RUNX1–RUNX1T1 cells | Verdict |
|---|---|---|---|---|---|
| Arriba | 3 of 3 | 1, 2, 3 | 1 of 3 | 8 | |
| STAR-Fusion | 0 of 3 at 9 FP | 0, 0, 0 | 1 of 3 | 8 | |
| Fixes 1–2, blacklist (`b6_*_fix12bl.json`) | 0 of 3 | 0, 1, 1 | 1 of 3 / 1 of 3 | 6 | fail |
| Fixes 1–2, blacklist and recurrent flag (`b6_*_fix123.json`) | 0 of 3 | 0, 1, 1 | 1 of 3 / 1 of 3 | 6 | fail |
| Plus soft-clip prototype, 20 bases (`b6_*_sc20.json`) | 0 of 3 | 1, 2, 2 | 1 of 3 / 1 of 3 | 7 | fail |

No caller calls a truth fusion in a cell of the other line, so every false-positive rate is 0. The fixes close the 1:15 ranking
gap and keep it closed. B6 still fails on two counts: the 1:1.5 mix ranks none of the truth fusions above the artifacts, and the
1:15 mix reaches 6 RUNX1–RUNX1T1 cells against the comparators' 8. The recurrent flag removes few calls (49 to 46 pairs in two or
more cells on 1:1.5), because the artifacts that now lead are canonical-motif pairs specific to these cell lines.

Fix 3 was prototyped because 1:1.5 stayed short (`proto/chimeric-softclip` at a8d58f06, CPU path only). Sending reads that leave
at least 20 cDNA bases unaligned to the chimeric pass raises second-pass input by 4.0% on both mixes (6.41 M to 6.67 M reads on
1:1.5, 5.79 M to 6.02 M on 1:15). It recovers most of the missing support: the engine now ties Arriba on BCR–ABL1 and NUP214–XKR3
cells in the 1:1.5 mix and sits one cell short on RUNX1–RUNX1T1 in each mix. The cost lands in precision. Non-canonical junction
lines rise by about half, and pairs called in two or more cells grow from 46 to 88 on 1:1.5, so the truth pairs, at two cells each,
still rank below 44 false positives.

With one or two supporting cells per truth fusion, the rule's first clause on the 1:1.5 mix needs a ranking that places
two-cell pairs above every artifact. More reads will not supply that. The ranking needs evidence the table does not yet use, such
as breakpoint consistency across cells or the known-fusion catalogue.

### B6 ranking evidence: pre-registered score

Written and committed before either mix is scored with it. The B6 rule (6de5821a) stays as it is. Only the order in which the
engine's gene pairs are read changes, which enters clause 1 (recall at each comparator's false-positive count). Clause 2 counts
the cells of each line with at least one supporting molecule, and no ranking changes that count. With the PR build at 6 of 8
RUNX1–RUNX1T1 cells on 1:15, clause 2 will still fail whatever clause 1 does.

For each unordered gene pair in the PASS rows of labelled cells, after the generic filters of `fusion_eval.engine_calls`
(different named genes, no chrM, no same-chromosome pair within 1 Mb, unique placement), `fusion/fusion_rank.py` computes:

- M, the molecules summed over cells and junctions, which is the current ranking.
- Breakpoint clusters. Rows are sorted by donor and acceptor position, and a row joins the first cluster whose first row lies
  within 5 bp at both ends. The top cluster has the most distinct cells.
- C, the distinct cells in the top cluster. Within a cell, molecules are distinct barcode and UMI pairs.
- f, the share of M in the top cluster, which measures agreement of the split-read position.
- s, splice-site use at both ends: 0.5 for each end of the top cluster that lies within 2 bp of an annotated exon boundary in
  the run's GRCh38-2024-A GTF. Every PASS row already carries a GT/AG motif.
- K = 1 when the pair appears as a fusion (A::B, either order) in the Mitelman Database MCGENE table, else 0.

| Variant | Score |
|---|---|
| baseline | M |
| cells | C |
| agreement | M · f |
| splice | M · (1 + s) |
| breakpoint (feature 1) | C · f · (1 + s) |
| known (feature 2) | M · (1 + 4K) |
| combined | C · f · (1 + s) · (1 + 4K) |

Ties are broken by M, and pairs still tied are counted together, which is the conservative reading of the curve. The weight of
4 lets a known fusion outrank an artifact with up to five times its support.

The known-fusion prior uses the Mitelman Database of Chromosome Aberrations and Gene Fusions in Cancer (Mitelman F, Johansson B,
Mertens F; mitelmandatabase.isb-cgc.org), licensed CC BY 4.0. The release of 2026-10-05 was used, with zip sha256 `e63ff330aa237db7291cac9efdb9e0573e9ea35888768db4c5e282ef258f9244`.
ChimerDB 4.0 publishes no license terms, so it is not used, and COSMIC is excluded. Mitelman curates published fusions,
including those reported in K562 and Kasumi-1, so the prior favors the truth pairs by construction. The table therefore reports
every variant with and without it.

Both mixes are scored for the PR build (run tag `chimfix-a704996d-r2`: blacklist and recurrent flag) and the soft-clip prototype
(`chimproto-a8d58f06-sc20`). The PR build sets the verdict.

Addendum, written before the run it describes. The 20-base prototype still misses one comparator cell per mix, because the
supporting read never reaches the chimeric pass. The second pass runs with `--chimSegmentMin 15`, so a read whose unaligned tail
is 15 to 19 bases can form a chimera but is never sent there. One more prototype run sets the soft-clip threshold to that same
15 bases (`chimproto-a8d58f06-sc15`). It is scored with the formula above, unchanged, and reported beside the others.

### B6 ranking evidence: results

Scored with `fusion_rank.py` as pre-registered (3f35b10a, plus the 15-base run declared at e7f77465), in job 804425. Each cell
of the table gives the engine's recall read at STAR-Fusion's false-positive count and at Arriba's (9 and 44 on 1:1.5, 6 and 39 on
1:15). The comparators' own recall at those points is 0 and 1 on 1:1.5, and 1/3 and 1/3 on 1:15. Results are in
`results/b6_*_rank_{fix,sc20,sc15}.json`.

| Build | Variant | 1:1.5 recall (SF / Arriba) | 1:15 recall (SF / Arriba) | Clause 1 |
|---|---|---|---|---|
| PR #220 (a704996d) | baseline, cells, agreement, splice, breakpoint | 0 / 0 | 1/3 / 1/3 | fail |
| PR #220 | known | 2/3 / 2/3 | 1/3 / 2/3 | fail |
| PR #220 | combined | 2/3 / 2/3 | 2/3 / 2/3 | fail |
| soft-clip 20 | baseline, cells, agreement | 0 / 0 | 1/3 / 1/3 | fail |
| soft-clip 20 | splice, breakpoint | 0 / 2/3 | 1/3 / 1/3 | fail |
| soft-clip 20 | known | 2/3 / 1 | 1/3 / 2/3 | pass |
| soft-clip 20 | combined | 1 / 1 | 1/3 / 2/3 | pass |
| soft-clip 15 | baseline, cells, agreement | 0 / 0 | 0 / 1/3 | fail |
| soft-clip 15 | splice, breakpoint | 0 / 2/3 | 1/3 / 1/3 | fail |
| soft-clip 15 | known | 2/3 / 1 | 1/3 / 2/3 | pass |
| soft-clip 15 | combined | 1 / 1 | 1/3 / 2/3 | pass |

Clause 2 does not depend on the ranking, and it fails for every build. With the PR build the engine reaches 0, 1 and 1 cells for
BCR–ABL1, NUP214–XKR3 and RUNX1–RUNX1T1 on 1:1.5 (Arriba 1, 2 and 3) and 6 RUNX1–RUNX1T1 cells on 1:15 (comparators 8). With
either soft-clip build it ties on BCR–ABL1 and NUP214–XKR3 but stays one RUNX1–RUNX1T1 cell short on each mix, at 2 against 3 and
7 against 8. No caller marks a cell of the other line.

**Verdict: B6 fails.** The PR build fails both clauses on 1:1.5 and clause 2 on 1:15. The best prototype, the soft-clip route with
the combined score, passes clause 1 on both mixes and misses clause 2 by one cell per mix.

Breakpoint consistency alone barely reorders the list. The truth fusions have one or two cells each, while the strongest
artifacts are themselves consistent multi-cell events at a single breakpoint, such as CCDC144NL-AS1–UBBP4 in 18 cells and
ENSG00000227240–TNR in 11. Of the breakpoint features, the exon-boundary term carries the signal, lifting the truth pairs from
about rank 90 to rank 25 on the soft-clip 1:1.5 run. The Mitelman prior does most of the work. Its 33,965 pairs include all three
truth fusions by construction, and between 4 and 11 of each run's other pairs as well (AFF1–KMT2A on 1:15, for one), so it
admits known-fusion artifacts too.

The missing cell has the same cause on both mixes. The supporting read (barcode CCTTACGAGCCCGAAA on 1:15, ACTGCTCCATTGTGCA on
1:1.5) never reaches the engine's chimeric pass, with a soft-clip threshold of 20 or 15 bases. The soft-clip route recovers
the other missed cells (GTACTTTCATACTACG, GGTGTTAGTCCCTACT). The 15-base route costs 5.8% more second-pass reads than the PR build
(6.78 M against 6.41 M on 1:1.5) and adds nothing over 20 bases. Two explanations remain open. The main pass may align the read
with a shorter clip, or the engine may route it elsewhere before alignment. The comparators' copy of the 1:15 read carries
barcode CCTTACGATCCCGAAA, one mismatch from the cell it was assigned to, which points at barcode correction for that read.

### B6 circularity check: declared before scoring

Mitelman lists all three truth fusions, so clause 1 passing with the prior can be read as circular. Two checks follow, written
before the runs they describe, and neither changes the rule or the formula.

1. **Breakpoint features alone.** The soft-clip prototype is reported on the `cells`, `agreement`, `splice` and `breakpoint`
   scores at both comparators' false-positive counts. These numbers already appear in the results table above, since those
   variants were scored at 3f35b10a.
2. **Leave-truth-out prior.** The `known` and `combined` scores are recomputed with the three truth pairs removed from the
   Mitelman set (`fusion_rank.py --leave-truth-out`), so the truth fusions get K = 0 while every other pair keeps its K. This
   runs on the PR build and both soft-clip builds, and results go to `results/b6_*_rank_*_lto.json`. Whatever drop in recall it
   shows is the share of the clause-1 pass that depends on the prior having seen the answer.

### B6 missing cell: trace, and the run it leads to (declared before scoring)

Both reads behind the missing RUNX1–RUNX1T1 cell were traced from the raw FASTQ (`fusion/trace/`).

| | 1:15 read SRR26063325.13217221 | 1:1.5 read SRR26063321.364249 |
|---|---|---|
| raw CB (qualities) | CCTTACGATCCCGAAA (`,:,:FFFF,,,:FFFF`) | ACTGCTCCATTGTGCA (all F) |
| whitelist candidates within 1 mismatch | CCTTACGAGCCCGAAA only (position 9, Q11; 4,240 exact reads) | exact hit |
| engine correction (counts only, quality F) | CCTTACGAGCCCGAAA, single candidate | exact |
| STARsolo 1MM_multi (read qualities) | CCTTACGAGCCCGAAA, single candidate | exact |
| comparators' cell | CCTTACGAGCCCGAAA | ACTGCTCCATTGTGCA |
| engine main pass (STAR 2.7.11b replay) | mapped uniquely, 74M16S on RUNX1T1 | unmapped, too short |
| engine chimeric pass settings | no junction | no junction |
| STAR-Fusion settings | junction 67M23S / 67S23M | no junction |
| Arriba settings | junction | junction 22M68S / 22S49M19S |

Barcode correction is not the cause. With a single whitelist candidate, the engine and STARsolo agree on both reads, and the
engine took in all 78,348,197 reads of the 1:15 mix. The engine's corrector does ignore barcode quality (the `.1fq` stores
R1 as all F). That only matters when a barcode has several candidates, and it is a separate question for the gate.

The reads are lost in the chimeric pass. Its settings (`--chimSegmentMin 15 --chimJunctionOverhangMin 15`, default
`--chimScoreDropMax 20`) reject the 23-base segment of the 1:15 read and the 19-base unaligned tail of the 1:1.5 read. The
1:15 read is also mapped in the main pass with 16 clipped bases, so only a soft-clip route at 16 bases or less sends it on.

The run declared here sets the second pass to Arriba's published chimeric settings (`--chimSegmentMin 10
--chimJunctionOverhangMin 10 --chimScoreMin 1 --chimScoreDropMax 30 --chimScoreJunctionNonGTAG 0 --chimScoreSeparation 1
--alignSJstitchMismatchNmax 5 -1 5 5 --chimSegmentReadGapMax 3 --chimMultimapNmax 50`), the set that finds both reads above.
Both mixes are rerun without soft-clip and with soft-clip at 20 and at 15 bases, keeping the blacklist and the existing
149-pair recurrent list. They are scored with the committed rule and the pre-registered scores, unchanged, including the
leave-truth-out prior. The run tags end in `-arriba`.

### B6 results with Arriba's chimeric settings

Prototype build chimproto-68d44117 carries PR #220 at e967c1e7 (where the second pass now uses Arriba's settings) plus the
soft-clip route, which runs only when its environment variable is set. All runs were in job 804425, with the committed rule and
the pre-registered scores unchanged. Recall is read at STAR-Fusion's and Arriba's false-positive counts (9 and 44 on 1:1.5, 6
and 39 on 1:15). Results are in `results/b6_*_rank_{nosc,sc20,sc15}_arriba[_lto].json`.

| Soft-clip | Score | 1:1.5 recall | 1:15 recall | Clause 1 | 1:1.5 cells (BCR–ABL1, NUP214–XKR3, RUNX1–RUNX1T1) | 1:15 RUNX1–RUNX1T1 cells | Clause 2 |
|---|---|---|---|---|---|---|---|
| none | breakpoint features only | 0 / 1/3 | 1/3 / 1/3 | fail | 0, 1, 2 | 6 | fail |
| none | combined | 2/3 / 2/3 | 2/3 / 2/3 | fail | 0, 1, 2 | 6 | fail |
| none | combined, leave-truth-out | 0 / 1/3 | 1/3 / 1/3 | fail | | | |
| 20 bases | breakpoint features only | 0 / 2/3 | 1/3 / 1/3 | fail | 1, 2, 3 | 7 | fail |
| 20 bases | combined | 1 / 1 | 1/3 / 2/3 | pass | 1, 2, 3 | 7 | fail |
| 20 bases | combined, leave-truth-out | 0 / 2/3 | 1/3 / 1/3 | fail | | | |
| 15 bases | breakpoint features only | 0 / 2/3 | 1/3 / 1/3 | fail | 1, 2, 3 | 8 | **pass** |
| 15 bases | Mitelman prior only | 2/3 / 1 | 1/3 / 2/3 | pass | 1, 2, 3 | 8 | **pass** |
| 15 bases | combined | 1 / 1 | 1/3 / 2/3 | **pass** | 1, 2, 3 | 8 | **pass** |
| 15 bases | combined, leave-truth-out | 0 / 2/3 | 1/3 / 1/3 | fail | | | |

The comparators reach 1, 2 and 3 cells on 1:1.5 (Arriba) and 8 on 1:15, and no caller marks a cell of the other line.

**Verdict: B6 passes the committed rule** with Arriba's chimeric settings, the soft-clip route at 15 bases and the combined
score. On both mixes the engine matches or beats both comparators at their false-positive counts, and it matches the best
comparator's cell recall for every truth fusion with a false-positive rate of 0.

The pass rests on the known-fusion prior. On breakpoint features alone, or with the truth pairs removed from Mitelman, the engine
reaches 2 of 3 truth fusions within Arriba's 44 false positives on 1:1.5, against Arriba's 3 of 3. On 1:15 it ties both
comparators without the prior. Clause 2 does not depend on the ranking, and it is met because Arriba's settings and the 15-base
route recover the two traced reads. Two further caveats belong with the verdict. The 15-base threshold and the change of settings
were chosen after the first misses on these mixes, although each was declared before its run was scored. The soft-clip route is
also a CPU-path prototype.

## B6 held-out run: registration

The coordinator records B6 as a development pass on the Penter mixes. The held-out run decides it. This registration is
committed before any held-out data are downloaded or any tool is run on them.

**Library.** GEO GSE329598 (SRA SRP696266; bioRxiv 10.64898/2026.05.06.723232): Ku812, a CML line carrying BCR::ABL1, mixed
with HMC1.2, a mast-cell line with no BCR::ABL1, and labelled with hashtags. The chemistry is 10x 3' v3.1, with reverse
transcription and PCR extension lengthened to enrich long molecules. The authors confirmed BCR::ABL1 by Sanger sequencing and by
targeted PacBio long reads from the same 10x library. Runs: gene expression SRR38330751 and SRR38330752, hashtags SRR38330749
and SRR38330750. The PacBio per-cell count table is `GSM9707175_Leukemic_cell_lines_long_read_count_table.csv.gz`. No setting
was tuned on this library. Its main weakness is that 3' chemistry places few reads across the breakpoint (the authors report
about six Ku812 cells with junction reads), so recall will be low for every caller and the run tests false positives as much
as sensitivity.

**Cell labels.** Hashtag reads (TotalSeq-B0253 `TTCCGCCTCTCTTTG`, B0254 `AGTAAGTTCAGCGTA`, searched in the first 40 bases of
the feature read) are counted per corrected cell barcode as distinct UMIs. A cell takes a hashtag when that hashtag holds at
least 70% of its hashtag UMIs and at least 10 UMIs, and other cells are left out. The hashtag whose cells carry BCR::ABL1 long
reads more often (PacBio table, at least one read) is named Ku812 and the other HMC1.2. If fewer than half of the feature reads
contain either hashtag, the run stops before any caller runs.

**Truth.** BCR–ABL1 in Ku812 cells. HMC1.2 cells are the negative population.

**Rule.** The B6 rule (6de5821a) on this single library. Clause 1: the engine's recall over labelled cells, read at each
comparator's false-positive count, is at least that comparator's recall. Clause 2: the engine's BCR–ABL1 cell recall in Ku812
cells is at least the best comparator's, with a false-positive rate in HMC1.2 cells at most max(that comparator's rate, 0.5%).

**Frozen configuration.** PR #220 at e967c1e7 (Arriba chimeric settings) plus the soft-clip route at 15 bases, as built in
chimproto-68d44117 with `SINGLET_CHIM_SOFTCLIP_MIN=15`. It runs with `--chimeric`, Arriba's v2.5.1 blacklist and the 149-pair
PBMC recurrent list, and is ranked by the combined score of 3f35b10a. The verdict uses this configuration only.

**Reported beside the verdict, all from the same single engine run:** the combined score with the full Mitelman prior;
breakpoint features alone (`breakpoint`, `cells`, `agreement`, `splice`); the Mitelman prior alone; and the leave-truth-out
`known` and `combined` scores. STAR-Fusion 1.15.1 and Arriba 2.5.1 run on the same reads (R2) with the commands used for
Penter. As a secondary check outside the verdict, each caller's BCR–ABL1 cells are compared with the cells that have PacBio
BCR::ABL1 reads. There is one engine run and one scoring, with no reruns after seeing results.

### Label correction (before any scoring)

The first label pass (job 804554) counted hashtag UMIs under the feature library's raw cell barcodes. On 3' v3.1 beads the
capture oligo that primes TotalSeq-B carries a different barcode from the poly-dT oligo, so these barcodes are not the cells'
gene-expression barcodes. Only 10 of the 4,400 PacBio barcodes, which are in gene-expression space, matched a labelled cell, and
the hashtag split (10,416 against 309 cells) was not a cell-line mix. The registered rule asks for corrected cell barcodes, so
`hto_labels.py --translate` now maps each feature barcode to its gene-expression barcode through Cell Ranger 9.0.1's
translation table (`barcodes/translation/3M-february-2018_NXT.txt.gz`, column 2 to column 1) before the whitelist check. Every
other label rule is unchanged. The labels are rebuilt while the engine and comparators are still running, and no caller output
has been read. The uncorrected summary is kept as `results/b6_holdout_labels_untranslated.json`.

### Label amendment (before any scoring)

The translated hashtag labels fail an orthogonal check that uses no caller output. Of the 602 labelled cells in the PacBio
table, 135 of 201 "Ku812" cells carry a KIT mutant long read (D816V or V560G, the HMC1.2 markers), and 104 of 401 "HMC1.2" cells
carry a BCR::ABL1 long read. The hashtag split (308 against 10,362 barcodes) also matches neither the PacBio mix (934 BCR::ABL1
cells, 2,373 KIT-mutant cells among 4,400) nor a plausible cell count, which points to ambient hashtag swamping the 70% rule.
Hashtag labels this noisy would make clause 2 measure the labels, not the callers.

The verdict therefore uses labels from the authors' orthogonal genotyping, declared here before any caller output is read.
A cell is Ku812 when its PacBio row has at least one BCR::ABL1 read and no KIT mutant read at either site; it is HMC1.2 when it
has at least one KIT mutant read (`KIT_c2447_MT` or `KIT_c1679_MT`) and no BCR::ABL1 read. Cells with both or neither are left
out. Truth, rule, frozen configuration and the reported variants are unchanged. Because Ku812 cells are now defined by an
expressed BCR::ABL1 molecule, recall here measures detection in cells known to express the fusion, so it will read higher than
on hashtag-labelled cells. The translated hashtag labels are scored as well and reported as secondary.

### Fusion-independent check (coordinator ruling, declared before scoring)

The coordinator accepted the label amendment, and the PacBio-labelled run stays the deciding run. Recall on those labels is
partly circular, because a Ku812 cell is defined there by a BCR::ABL1 read. It is therefore reported beside a check that uses no
fusion information. A cell is Ku812 by KIT genotype when its PacBio row has at least one KIT wild-type read (`KIT_c2447_WT` or
`KIT_c1679_WT`) and no KIT mutant read; it is HMC1.2 when it has at least one KIT mutant read, whatever its BCR::ABL1 count.
For each caller this check reports the share of KIT-genotype Ku812 cells in which it finds BCR–ABL1 and its BCR–ABL1 calls
in HMC1.2 cells. HMC1.2 is heterozygous for its KIT mutations, so a few HMC1.2 cells with only wild-type reads will sit in the
Ku812 group and pull every caller's share down by the same amount.

## B6 held-out result: pass on the registered rule, tied with Arriba

One frozen engine run (chimproto-68d44117, soft-clip at 15 bases, job 804554) sent 13.6 million of the library's 71.3 million
read pairs into the chimeric pass. STAR-Fusion and Arriba ran on the same reads. Scoring used the PacBio genotype labels
(547 Ku812, 2,111 HMC1.2 cells) with the committed rule 6de5821a and the combined score of 3f35b10a (`results/b6_holdout_rank_pacbio*.json`,
`results/b6_holdout_cells.json`).

**Clause 1 passes.** STAR-Fusion makes two calls and misses BCR–ABL1, so at two false positives both callers have zero recall.
Arriba finds BCR–ABL1 among 610 calls. The engine's combined score ranks BCR–ABL1 seventh, so the engine reaches full recall
after six false positives, long before Arriba's 609.

**Clause 2 passes, as a tie.** The engine and Arriba each find BCR–ABL1 in 2 of 547 Ku812 cells (0.37%) and in 3 of 2,111
HMC1.2 cells (0.14%), well under the 0.5% ceiling. STAR-Fusion finds it in no cell. As the registration expected, 3' chemistry
puts very few reads across the breakpoint, so every caller's cell recall is low.

| Truth rank of BCR–ABL1 (engine pairs in labelled cells) | full Mitelman prior | leave-truth-out prior |
|---|---|---|
| combined (verdict) | 7 | 610 |
| known (prior alone) | 2 | 11 |
| breakpoint (C·f·(1+s)) | 579 | 579 |
| cells / agreement / splice | 27 / 32 / 6 | same |
| baseline (molecules) | 10 | 10 |

Every variant passes clause 1, because Arriba's 609 false positives set a deep bar. The ranking itself reproduces the dev
finding: breakpoint features alone place BCR–ABL1 near the bottom, and the leave-truth-out prior moves the combined score to
rank 610. The good rank of the verdict score therefore comes from the Mitelman prior knowing BCR–ABL1.

**Fusion-independent check (KIT genotype only; `results/b6_holdout_rank_kit*.json`).** Among 1,104 KIT wild-type cells the
engine and Arriba each find BCR–ABL1 in one cell (0.09%), and STAR-Fusion in none. Both make 7 calls among 2,498 KIT-mutant
cells (0.28%), and STAR-Fusion makes none. The KIT-mutant group includes 387 cells that also carry a BCR::ABL1 long read
(ambient molecules or doublets), so some of these "false" calls may sit in Ku812 material. Clause 2 passes on these labels too,
again as a tie with Arriba.

**Secondary checks.** Across all 4,400 PacBio-genotyped cells the engine calls BCR–ABL1 in 9 cells and Arriba in 10, and 6
of each set carry a PacBio BCR::ABL1 read. This matches the authors' report of about six Ku812 cells with junction reads. On the
translated hashtag labels no caller finds BCR–ABL1 in a "Ku812" cell, and the engine makes one call in the other group against
Arriba's four.

**Verdict.** B6 passes the registered rule on held-out data. The pass rests on parity with Arriba at the cell level and on
the known-fusion prior for ranking, with no evidence of better sensitivity than Arriba.

## Soft-clip route in PR #220: CPU = GPU

The route went into PR #220 (fix/chimeric-strand-calls, 79961da4): 200af961 adds it, 928144a6 fixes a literal, 601eb0ca adds
the GPU device path, and 79961da4 fixes another literal. Reads with a GPU record skip STAR's loop at E10 levels 4 and 5, so the
device record kernel flags the unique mappers with at least 15 unaligned bases and the prefill writes them out. Without that
step the GPU path routed 43 reads against the CPU's 19,237 on pbmc2m.

Gate job 804598 ran on one RTX 8000 (g008) with `SINGLET_ENGINE_GPU_MIN_CC=70`. CPU references came from 928144a6, whose CPU
code is the same as 79961da4's. Host ctest passes, with test 33 skipped as before. Both 8-sample gates are IDENTICAL CPU
against GPU on every sample: the defaults (gate A) and `--chimeric` (gate B). On all 8 samples the chimeric table has the same
sorted rows, and both paths send the same reads into the second pass (`results/chimgate_804598_parity.tsv`).

| Sample | Unmapped reads | Soft-clip reads | Added by soft-clip | GPU wall, defaults → --chimeric (s) |
|---|---|---|---|---|
| pbmc2m | 123,500 | 19,237 | 15.6% | 40 → 70 |
| pbmc10m | 582,190 | 95,470 | 16.4% | 63 → 95 |
| pbmc1k_v3 | 3,582,744 | 624,844 | 17.4% | 204 → 271 |
| pbmc1k_v2 | 6,372,909 | 553,605 | 8.7% | 211 → 313 |
| pbmc1k_cite_v3 | 2,479,234 | 447,809 | 18.1% | 163 → 215 |
| pbmc5p_v1 | 4,818,388 | 125,983 | 2.6% | 201 → 278 |
| neuron1k_v3 | 4,071,521 | 450,382 | 11.1% | 167 → 221 |
| nuclei2k_v2 | 22,941,319 | 1,868,692 | 8.1% | 415 → 648 |

The soft-clip route adds 3% to 18% to the second pass's reads. The wall-time column covers the whole chimeric pass, unmapped
and soft-clip reads together, which runs as a CPU STAR job after the GPU main pass.

## B6 ranking without the prior: Arriba's cascade (D-109, declared before scoring)

Owner decision D-109 asks for parity by porting the competitor's code. Arriba 2.5.1 is MIT licensed, so its filter cascade and
its confidence scoring are ported into the engine's chimeric caller (`fusion/arriba_cascade.py` here, `chimeric_table.h` on
Singlet-Bio/singlet branch `feat/arriba-cascade`). The new ranking must not use the Mitelman prior. This section is committed
before either dev mix is scored with it. The committed rule 6de5821a and the scores of 3f35b10a stay unchanged.

**Input.** The frozen configuration's raw STAR junction file (chimproto-68d44117, soft-clip at 15 bases, Arriba's chimeric
settings) for each mix. Only reads whose raw barcode matches a labelled cell, exactly or as the single one-mismatch neighbour,
enter the cascade, which is the same cell set the existing variants use. All junction types enter, since Arriba applies no
motif filter, and the 149-pair recurrent list is not used. Arriba's own GRCh38 v2.5.1 blacklist is applied by the ported
`filter_blacklisted_ranges`, keywords included.

**Variant `arriba`.** Arriba's passing calls (high, medium and low confidence) are ranked as Arriba sorts its output
(`sort_fusions_by_support` in `output_fusions.cpp`): confidence first, then supporting reads, then e-value, lower first. Here a
supporting read is a molecule, because duplicates are removed by barcode, UMI and breakpoint. Each unordered gene pair takes the
rank of its best breakpoint. Pairs with the same confidence, support and e-value are counted together, as in the existing
curves. Pairs the cascade discards are absent from the ranked list, and their filter is reported. No term uses Mitelman or any
other fusion catalogue, and Arriba's known-fusions file is not loaded either.

**Reported.** For both mixes: truth ranks, clause 1 against STAR-Fusion and Arriba, and clause 2, for the `arriba` variant
alone, next to the `combined` score with and without `--leave-truth-out` and next to `breakpoint`. Clause 2 does not depend on
the ranking and is restated from the same run. Each ported filter's effect is reported as reads or calls removed per filter on
each mix, in the order of Arriba's own log. Inputs Arriba needs that this data lacks are substituted, and each substitution is
listed with the results.

## B6 ranking without the prior: result (D-109)

The port is C++ in the engine: `include/singlet/pileup/arriba_cascade.h` with a one-sample CLI
(`src/pipeline/arriba_cascade_cli.cpp`) and the unit test `tests/cpp/test_arriba_cascade.cpp` on Singlet-Bio/singlet branch
`feat/arriba-cascade` (c3c5c6e7, b13e8a7c, fbeaa915). No Python port was needed. The scored binary was built from b13e8a7c, and
fusion_rank.py from d4b45ff6 adds the `arriba` variant. Job 804706 on Clipper wrote the outputs to `$J/arriba-cascade/`. It took
about 3 minutes and 10 GB per mix.

**Verdict.** Without the prior, the cascade passes clause 1 on mix1to15 and fails it on mix1to1p5. RUNX1–RUNX1T1 ranks first
on mix1to15. On mix1to1p5 the cascade calls NUP214–XKR3 (rank 9) and RUNX1–RUNX1T1 (rank 7) but discards BCR–ABL1, so at
Arriba's 44 false positives it recovers two of three truth pairs where Arriba recovers all three. Clause 2 passes on both mixes,
as before. The prior-bearing `combined` score still passes both mixes. With the truth pairs left out of the prior, it fails
mix1to1p5 in the same way the cascade does.

| mix | variant | BCR–ABL1 | NUP214–XKR3 | RUNX1–RUNX1T1 | recall at STAR-Fusion's FP (engine / STAR-Fusion) | recall at Arriba's FP (engine / Arriba) | clause 1 |
|---|---|---|---|---|---|---|---|
| mix1to15 | arriba | discarded | absent | 1 | 0.33 / 0.33 (6 FP) | 0.33 / 0.33 (39 FP) | pass |
| mix1to15 | combined | 11 | absent | 1 | 0.33 / 0.33 | 0.67 / 0.33 | pass |
| mix1to15 | combined, leave-truth-out | 755 | absent | 1 | 0.33 / 0.33 | 0.33 / 0.33 | pass |
| mix1to15 | breakpoint | 753 | absent | 1 | 0.33 / 0.33 | 0.33 / 0.33 | pass |
| mix1to1p5 | arriba | discarded | 9 | 7 | 0.67 / 0.00 (9 FP) | 0.67 / 1.00 (44 FP) | fail |
| mix1to1p5 | combined | 5 | 3 | 2 | 1.00 / 0.00 | 1.00 / 1.00 | pass |
| mix1to1p5 | combined, leave-truth-out | 69 | 27 | 18 | 0.00 / 0.00 | 0.67 / 1.00 | fail |
| mix1to1p5 | breakpoint | 62 | 20 | 12 | 0.00 / 0.00 | 0.67 / 1.00 | fail |

Ranks count tied pairs together. "Absent" means the engine holds no NUP214–XKR3 chimeric read from a labelled cell. As a call set,
the cascade passes 81 gene pairs on mix1to15 and 102 on mix1to1p5, against Arriba's 40 and 47.

**Why BCR–ABL1 is lost.** On mix1to1p5 the engine holds the e14a2 junction (chr22:23290413 to chr9:130854064, splice sites at both
ends) as one molecule plus a UMI duplicate, and a second breakpoint as one more molecule. `min_support` (two reads) discards both.
Arriba's own call on this mix is a single read at low confidence, and Arriba keeps such calls only through its known-fusions
recovery, which the registration excludes. The gap is therefore the catalogue term, and without it the cascade behaves as Arriba
would. Arriba's call on mix1to15 is RUNX1–RUNX1T1 alone, and the engine sees one BCR–ABL1 molecule there.

**Per-filter effect.** Reads remaining after each read-level filter and candidate breakpoints remaining after each fusion-level
step, in Arriba's log order (job 804706, `mix*.seq.log.tsv`).

| step | mix1to15 | mix1to1p5 |
|---|---|---|
| junction lines | 9,186,887 | 10,503,246 |
| reads from labelled cells | 6,731,047 | 7,505,378 |
| duplicates | 6,423,708 | 7,208,755 |
| uninteresting_contigs | 6,322,973 | 7,102,254 |
| read_through | 6,321,361 | 7,100,569 |
| inconsistently_clipped | 6,321,361 | 7,100,569 |
| homopolymer | 1,896,714 | 1,975,034 |
| small_insert_size, long_gap | 1,896,714 | 1,975,034 |
| same_gene | 1,896,703 | 1,975,030 |
| hairpin | 1,847,849 | 1,921,261 |
| mismatches | 122,463 | 131,176 |
| low_entropy | 98,731 | 105,665 |
| candidate breakpoints (find_fusions) | 56,460 | 57,174 |
| merge_adjacent | 56,300 | 56,980 |
| multimappers | 24,521 | 24,093 |
| non_coding_neighbors | 24,521 | 24,093 |
| intragenic_exonic | 21,239 | 20,772 |
| min_support | 1,345 | 1,487 |
| relative_support | 821 | 936 |
| intronic | 619 | 688 |
| in_vitro | 618 | 688 |
| spliced (recovery) | 625 | 693 |
| select_best | 452 | 474 |
| marginal_read_through | 452 | 474 |
| many_spliced (recovery) | 465 | 496 |
| blacklist | 118 | 159 |
| short_anchor | 110 | 146 |
| end_to_end | 107 | 140 |
| no_coverage | 85 | 109 |
| select_best | 82 | 102 |
| calls: high / medium / low | 10 / 36 / 36 | 9 / 41 / 52 |

Before the cascade, 2,113,693 and 2,621,493 lines carry a barcode outside the labelled cells, 825 and 962 match two cells, and
341,322 and 375,413 fail Arriba's segment disjoining. The read-sequence filters do most of the work, as in Arriba's own log on
mix1to15, where homopolymer leaves 1,380,543 reads and mismatches 159,437. A run without read sequences (those three filters
skipped, multimapper scores from STAR) passes 192 and 231 pairs and ranks the truth pairs lower: RUNX1–RUNX1T1 4th on mix1to15,
NUP214–XKR3 20th and RUNX1–RUNX1T1 15th on mix1to1p5.

**Substitutions.** Arriba reads a BAM file. The engine keeps STAR's junction file, so these inputs stand in.

- Each junction line becomes Arriba's single-end split-read record (the longer segment is the split read, mate 1 a copy of it),
  as `remove_malformed_alignments` builds it for single-end BAM records.
- Duplicates use Arriba's external marking: a read is a duplicate when an earlier read has the same cell, UMI and breakpoints.
- Read sequences come from the sample's .1fq, decoded with `singlet decode`, whose read names match the engine's (r<n>). Every
  loaded read found its sequence at the expected length.
- Coverage, which Arriba counts from all alignments in 20-base windows, is the mean read depth of the most-counted gene
  overlapping the position: count × 110 / effective length from `gene_expression.tsv`. A fragment starts or ends in a window when
  a counted gene overlaps it. This touches in_vitro, the spliced recovery, marginal_read_through, no_coverage and confidence.
- The e-value's library size is STAR's uniquely plus multi-mapped reads of the main pass.
- Strandedness comes from Arriba's own `detect_strandedness`, which finds "no" as the comparator run did. The fragment length
  falls back to Arriba's 200 because single-end data give no estimate.
- Gene sets are ordered by gene ID where Arriba orders them by pointer, and fusions sit in an ordered map, so ties resolve the
  same way on every run.

**Not ported.** Viral-contig filters, since GRCh38-2024-A has no viral contigs. Known fusions and tags, excluded by the
registration. Internal tandem duplication detection and read-through extraction from ordinary alignments, because no main-pass
BAM is kept. `homologs` and `mismappers`, which need Arriba's k-mer index of gene sequences and are a separate port. `isoforms`,
which adds breakpoints to pairs that already pass and so leaves pair ranks unchanged. Genomic support (no WGS). The transcript,
peptide and protein-domain output columns. `small_insert_size` and `inconsistently_clipped` are ported but have nothing to act on
in single-end data.

**Tests.** The unit test covers the annotation index, splice sites, contig patterns, k-mers, the split-read record, the whole
cascade on a synthetic GTF and junction file, and low_entropy. It passes 30 of 30 at fbeaa915 (job 804712). The branch's host
ctest at b13e8a7c (inside job 804706) passed 163 of 164. The one failure was this test expecting `min_support` for a single-read
spliced fusion, which ends in `in_vitro` as it does in Arriba; fbeaa915 corrects the expectation.

## B7: scTE and SoloTE against the engine's TE counter (D-109)

Owner decision D-109 asks for parity by porting the competitor's code. scTE (JiekaiLab/scTE at 566f6ab3) is MIT licensed, so
its counting rule is ported with credit in NOTICE. SoloTE (bvaldebenitom/SoloTE at 49a1d2b5) is GPL-3: it was read to learn
the method, and its rule is rewritten from the method alone, with no code or code structure carried over. Both live on
Singlet-Bio/singlet branch `feat/te-port` (gpu/frontier with feat/te-counts merged) as two new values of `--te-rule`. The
default rule and the committed B7 rule 6de5821a are unchanged. The engine column below is the feat/te-counts counter scored so far.

| | scTE (exclusive mode, the default) | SoloTE (method) | Engine (feat/te-counts) | Gap |
|---|---|---|---|---|
| Index | Genes are the merged exon and UTR bases of each gene_name. TE loci overlapping an exon of a protein-coding or lincRNA gene are dropped from the index at build time. Only contigs 1-49, X, Y and M count | Every TE locus in the BED, no gene filter | Every TE locus in the GTF | Index-time removal of TEs on coding exons |
| Read to TE | A fixed 101-base window from the read's leftmost position, CIGAR ignored, counts once for every overlapping feature, genes and TEs alike | Aligned blocks; one count for every TE locus overlapped by at least one base | Aligned blocks; one subfamily, the one covering most bases; ties dropped | Every-overlap assignment and its tie handling |
| Multimappers | The single STARsolo primary, wherever it lands, MAPQ ignored | The primary; MAPQ 255 counts at the locus, anything else at the subfamily | The primary; NH 1 counts at the locus, NH above 1 at the subfamily | None |
| Intronic reads | Counted for an overlapping TE (the nointron mode drops TEs inside gene spans) | Counted, since STARsolo assigns no gene to them | Counted unless a block touches any exon | None for wholly intronic reads |
| TEs overlapping exons | Locus removed when the exon belongs to a coding or lincRNA gene; reads on TEs inside other genes' exons count for both | A read stays with the gene only when STARsolo assigns it one gene (unique, transcript-compatible, sense); antisense, partly intronic and multi-mapped reads count for the TE | Any block touching any exon, either strand, any gene type, sends the read to the gene | Narrower exclusion in both tools |
| Cells and UMIs | One read per (barcode, UMI) across the whole BAM, the leftmost; cells by a minimum gene count and an expected-cell cap | Distinct UMIs per (feature, barcode), every STARsolo-corrected barcode | Distinct (cell, feature, UMI), the engine's cell calls | Genome-wide leftmost-read molecule |

**Ported from scTE** (`te_counts.h`, each function commented with its source): `TeModel::mask_scte_exclusive` and
`scte_main_contig` (bin/scTE_build `genomeIndex`, `readGtf`, `cleanexon`; scTE/annotation.py `annoGtf`, including the nointron
mode and scTE's exact boundary test), `TeModel::overlaps_all` with the scte branch of `TeModel::records` (scTE/base.py `align`)
and `scte_leftmost_per_molecule` (the first-read-per-barcode-and-UMI step of scTE/base.py `Bam2bed`). Two pieces are left out
on purpose. The fixed 101-base window replaces the aligned blocks only as an approximation, so the port keeps the blocks. scTE's
cell filter is not ported either, because B7 scores every tool on the engine's called cells.

**Reimplemented from the SoloTE method** (Rodriguez-Quiroz and Valdebenito-Maturana, Communications Biology 2022): reads that
the gene counter would give to exactly one gene stay with it (`te_gene_assigned`: unique, every aligned block inside an exon of
that gene on the sense strand); every other read counts for every TE locus it overlaps by at least one base, at the locus when
unique and at the subfamily when multi-mapped. SoloTE also writes family and class matrices; the engine's subfamily row names
already carry `subfamily:family:class`, so those sums need no new output. SoloTE's dual mode, which counts gene-assigned reads
for TEs as well, is what the scte rule does for reads on TEs outside coding exons.

`SINGLET_TE_ALL_RULES=1` writes `te_counts.scte.1pz` and `te_counts.solote.1pz` beside `te_counts.1pz`, so one engine run
scores all three rules against the committed rule.

### B7 port: first scores (provisional, scTE pending)

Jobs 804679 (build 349fd32f, host ctest 166 of 166, engine on both datasets) and 804708 (build 060b322c with the contig fix, host
ctest 166 of 166, mESC rerun) ran the engine with `SINGLET_TE_ALL_RULES=1`. scTE's output for this round was still being written
by 804624, so the numbers below come from te_eval with the scTE slot left out (`jobs/teport/prov/`). 804708 scores the full rule
with scTE as soon as `te/<dataset>/scte.h5ad` appears (`jobs/teport/results/b7_<dataset>_<variant>.json`). "Old" is the scored
feat/te-counts build (rlo-13ee63ac); the singlet rule on the new build gives identical numbers.

| Variant | A375 Spearman, all | A375 Spearman, young | mESC AUROC, MERVL fraction |
|---|---|---|---|
| Engine, old (singlet rule) | 0.9699 | 0.9791 | 0.6185 |
| Engine, scte rule | 0.9657 | 0.9727 | 0.6211 |
| Engine, solote rule | 0.9686 | 0.9767 | 0.6219 |
| SoloTE | 0.9688 | 0.9860 | 0.6695 |

The counting rule moves the scores by less than 0.01, so porting the rules does not close B7. Two findings outside the rule
explain more of the gap.

**mESC: the engine discards high-copy multimappers.** The engine's STAR pass reports 4.47 M reads (2.1%) as mapped to too many
loci, against 9,971 in the comparators' STARsolo run, which allows 100 loci (`--outFilterMultimapNmax 100 --winAnchorMultimapNmax
100`). MERVL-int and MT2_Mm are high-copy families, so these are the reads that carry the 2C signal. Raising the engine's
multimapper limit for TE counting is the next experiment; it changes the aligner, not the counter.

**A375: the comparators see 30 of the engine's cells.** The engine calls 4,309 cells. STARsolo in `te_bench.sbatch` used the
3M-february-2018 whitelist (R1 is 28 bases) and kept 163, of which 30 match the engine's barcodes, so every A375 score rests on
a 30-cell pseudobulk. The engine's barcodes do not look like 3M-february-2018 barcodes, so the comparators likely need another
whitelist (the engine's detected one). The committed rule is unchanged; the comparator run needs the matching whitelist before
the A375 clause can be read.

## B6 closed (D-111)

Under owner rule D-111, B6 is closed at parity with Arriba on the held-out run. The Arriba-cascade port stops where it stands:
branch feat/arriba-cascade (fbeaa915) stays unmerged, and its dev scores above are kept as a record. One observation is left
for later. The engine's fusion ranking depends on the Mitelman prior: without it, the held-out truth ranks 579th on breakpoint
features and 610th with the leave-truth-out prior.

## B7 A375 comparator correction (before the verdict)

A375 (GSE212945) is a Multiome gene-expression library. The engine resolved its whitelist as `gex_737K-arc-v1`, but the
comparators' STARsolo run chose `3M-february-2018` from the 28-base R1 and called 163 cells, of which only 30 match the
engine's 4,309. Most comparator reads therefore carried no valid cell barcode. The first A375 score (job 804624: engine
0.9699 and 0.9791, scTE 0.9664 and 0.9784, SoloTE 0.9688 and 0.9860, over all and young subfamilies) used that run and is kept
as a record only. STARsolo, scTE and SoloTE are rerun on A375 with `CBWL_OVERRIDE` set to the 737K-arc list, and the rule is
unchanged.

## B7 mESC result: fail

Job 805757 scored the 2C-like clause against 6de5821a on 4,161 shared cells. The engine's MERVL-fraction AUROC is 0.6268, scTE's
0.6328 and SoloTE's 0.6686, so the engine trails the best comparator by 0.0418 against a tie band of 0.005. B7 needs both clauses,
so it fails whatever A375 shows. On raw MERVL counts, which the rule does not score, the engine's AUROC is 0.8471, scTE's 0.8501
and SoloTE's 0.7316.

The suspected cause is the engine's too-many-loci cap: it drops 4.47M mESC reads (2.1%) as mapped to too many loci, while the
comparators' STARsolo run at 100 loci drops 9,971. These reads include high-copy MERVL and MT2 copies. This is untested, and the
aligner is unchanged.

Job 804624 ended early because `b7run.sbatch` did not export `PZ_SO`. `te_eval` stopped after mESC's scTE, the batch exited, and
Slurm cancelled the overlapping A375 step. A375 is rerunning as job 805756 with the Multiome whitelist, whose STARsolo now calls
4,356 cells.
