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
