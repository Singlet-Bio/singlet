# B4 observations (D-111 close-out)

Neutral-set scores, REDItools3 3.7 against Singlet, per donor (bulk STAR as the reference). Score files: `editing_neutral/<run>/score.json` on Clipper.

| Donor | Neutral sites | Singlet coverage | REDItools3 coverage | Matched-depth r, Singlet | Matched-depth r, REDItools3 | Difference [95% CI] |
|---|---|---|---|---|---|---|
| A | 187,148 | 0.846 | 0.928 | 0.602 | 0.559 | +0.043 [+0.035, +0.052] |
| B | 179,941 | 0.770 | 0.956 | 0.601 | 0.534 | +0.067 [+0.056, +0.077] |

Observation: Singlet covers fewer neutral sites than REDItools3 (coverage is a site with at least 5 counts in the tool's own unit). Singlet counts molecules and REDItools3 counts reads, so the same depth cutoff is stricter for Singlet. Sites are a uniform 200k sample of the union of the neutral and Singlet-selected sets; the verdict uses neutral sites only.

Phred 30 at the editing sites (`--editing-min-baseq 30`, feat/rna-editing 076da76d) was not needed and was not run. The option stays available and defaults to `--min-baseq`.
