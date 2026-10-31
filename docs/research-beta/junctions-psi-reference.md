# junctions-psi: the reference carries Singlet's own STAR options (amendment 2026-10-31)

**Decision.** K5, the Beta aligner profile (CellRanger4 clipping, `--alignSJoverhangMin 8`, `--outFilterIntronMotifs RemoveNoncanonicalUnannotated`,
`--outFilterMismatchNoverLmax 0.05`), stays as shipped. The `junctions-psi` reference becomes **STARsolo SJ with the STAR options Singlet itself runs for the
dataset**, and the primary evidence for the junction guard is long-read truth. STARsolo agreement is secondary.

**Why.** In the release gate (docs/research-beta/release-gate.md) `junctions-psi` went from 2 of 42 failing metrics (02aa9b9) to 4 of 42. The cause is the
junction guard of the Beta aligner profile (#96), measured by turning the profile off (`--cr4-probe-reads 0`): pbmc1k_v3 filtered-junction Jaccard 0.815 with
the profile, 0.897 without (0.891 at 02aa9b9). The guard removes about a third of the junction list (121,707 junctions against 181,164 on pbmc1k_v3). The
contract's reference, STARsolo at STAR defaults, keeps those junctions, so agreement with it falls. It is a reference mismatch, not wrong output: the removed
junctions are not in the long-read truth.

## 1. Primary evidence: long-read truth (PBMC-E pbmc1, dev split)

singlet-agents `bench/ground-truth` (frozen study; Illumina reads linked to PacBio and ONT molecules by cell barcode and UMI). Data read only from
`/mnt/projects/debruinz_project/debruinz/ground-truth`; only the dev split (the whole of PBMC-E) is used.

### 1a. STAR alone, per junction (the study's M2; reads with tier-1 truth)

| arm | junction recall | junction precision | F1 | read recall / precision |
|---|---|---|---|---|
| production STAR (p0) | 0.8715 | 0.7760 | 0.821 | 0.9294 / 0.9497 |
| + CellRanger4 clipping (a) | 0.8780 | 0.8117 | 0.844 | 0.9410 / 0.9509 |
| a + overhang 8 only | 0.8779 | 0.8165 | 0.846 | 0.941 / 0.951 |
| a + non-canonical filter only | 0.8781 | 0.8661 | 0.872 | 0.941 / 0.952 |
| a + both (J1, the junction guard) | 0.8779 | 0.8709 | 0.874 | 0.9410 / 0.9521 |
| K5 (J1 + mismatch cap 0.05; the shipped profile) | 0.8774 | 0.8820 | 0.880 | 0.9394 / 0.9522 |

The guard adds about 6 points of precision and loses no recall; the non-canonical filter does almost all of it (overhang 8 alone adds half a point). No single flag
gives better recall at the same precision, so K5 is unchanged.

### 1b. Singlet's own junction table, per junction (every read in the truth cells)

Cells carrying a tier-1 molecule (4,855). Truth: T1 = 132,228 junctions crossed by tier-1 reads, Tall = 288,581 by any linked read. recall = share of T1
recovered; precision = share of emitted junctions in Tall; k = minimum molecules per junction in those cells. Singlet is the tip with the stall-watchdog fix
(22b5782f); the protocol is named because this library's R1 is 27 nt. The `star_sj_pass` set is what the contract scores.

| table | k | junctions | precision | recall | F1 |
|---|---|---|---|---|---|
| Singlet K5, all primary alignments (before the unique-only rule) | 1 | 238,792 | 0.689 | 0.885 | 0.775 |
| Singlet K5, uniquely mapped alignments only | 1 | 219,394 | 0.740 | 0.877 | 0.803 |
| STARsolo with the same options | 1 | 219,478 | 0.740 | 0.877 | 0.803 |
| STARsolo at STAR defaults (the old reference) | 1 | 240,313 | 0.672 | 0.873 | 0.759 |
| Singlet K5, all primary alignments | 2+ | 121,774 | 0.920 | 0.724 | 0.811 |
| Singlet K5, uniquely mapped alignments only | 2+ | 117,229 | 0.941 | 0.716 | 0.813 |
| STARsolo with the same options | 2+ | 117,295 | 0.941 | 0.716 | 0.813 |
| STARsolo at STAR defaults | 2+ | 121,854 | 0.899 | 0.712 | 0.795 |
| Singlet profile off, all primary alignments | 2+ | 137,622 | 0.816 | 0.723 | 0.767 |

The absolute numbers of 1b are lower than those of 1a for every arm, including STARsolo, because every read in the cells counts in 1b and only reads with tier-1
truth in 1a. Compare rows against each other, not across the two tables, and not with the 0.88 / 0.88 of the earlier study.

## 2. Secondary evidence: agreement with STARsolo carrying Singlet's options

Singlet (pinned d9834b8a run, before the unique-only rule) against STARsolo SJ run with CellRanger4 clipping, overhang 8, the non-canonical filter, mismatch cap
0.05 and `--outFilterScoreMin 10`, with the Cell Ranger emulation as the competitor (it has no guard):

| metric | pbmc1k_v3 Singlet / competitor | hgmm1k_v3 Singlet / competitor |
|---|---|---|
| filtered-junction Jaccard | 0.913 / 0.770 | 0.959 / 0.902 |
| junction-count correlation | 0.966 / 0.908 | 0.973 / 0.961 |
| per-cell junction correlation | 0.948 / 0.842 | 0.923 / 0.903 |
| cassette-PSI pseudobulk r | 0.9955 / 0.9931 | 0.9908 / 0.9910 |
| per-cell total correlation | 1.000 / 0.998 | 0.9996 / 0.9999 (fails by 3e-4) |

An attempt with a guard-matched reference that kept the Cell Ranger emulation's `--outFilterScoreMin 30` did not recover agreement (Jaccard 0.772): Singlet runs
score minimum 10, and the reference has to carry every STAR option Singlet runs, not only the guard.

## 3. What the amendment changes

- `starsolo_public.sbatch` gains `MODE=reference`: `--outFilterScoreMin 10`, `--soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts`, and for the 10x 3' v2/v3 datasets
  the Beta aligner profile; SJ only. `MODE=default` and `MODE=cr` are unchanged.
- The extractor takes `starsolo-public-reference` as the reference when it exists and falls back to `starsolo-public-default`.
- The module's tool declaration, methods, caveats and figure captions name the new reference; the long-read evidence is the first caveat.
- Singlet counts a junction molecule only from uniquely mapped alignments (separate PR, `sj_unique_only`), the policy STARsolo's SJ feature has. Before it, Singlet
  emitted 74,131 more single-molecule junctions than STARsolo on pbmc1, with long-read precision 0.076.

**Not done in this amendment:** the 8-dataset re-run of the reference and the re-score of all gating metrics. Run `MODE=reference` over the cohort, then
`run_all.sh extract` and the module; the release gate re-run on the tip does it.
