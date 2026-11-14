# Chromosomal sex call: results against the gate

Gate: `bench/calls-sota/gates/sex.md`, committed before any scoring. Scripts: this directory. Outputs stay on Clipper under
`bench/calls-sota/sex/` (score_dev.md, score_heldout.md, units_*.tsv). Call scored: `python/singlet/celllabels/sex.py` unchanged
(origin/main 4227560a). Comparators: the two-gene rule (our implementation) and cellXY 0.99.0 (GPL-3.0, run as an installed
package, no code copied).

## Development

| Library | Truth | Method | Per-cell acc (macro) | Call rate | Donor acc | Donor abstain |
|---|---|---|---|---|---|---|
| Tier 1, 4 donors (in-silico HG00362 F plus HG01277 M pool, two-female real pool) | 1000 Genomes sex, library or genotype | Singlet, donor groups | 0.9805 | 0.9815 | 4 of 4 | 0 |
| | | Singlet, no groups | 0.7865 | 0.7869 | 4 of 4 | 0 |
| | | two-gene rule | 0.5752 | 0.5764 | 4 of 4 | 0 |
| | | cellXY | 0.8740 | 0.8905 | 4 of 4 | 0 |
| Tier 2, 170 samples (GEO metadata, weaker) | depositor | Singlet | 0.9644 | 0.9938 | 0.9706 | 0 |
| | | two-gene rule | 0.5270 | 0.5420 | 0.9706 | 0 |
| | | cellXY | 0.6865 | 0.8762 | 0.8176 | 0 |

Singlet with donor groups beats cellXY per cell on tier 1 by 0.107 (95% bootstrap CI 0.027 to 0.186) and on tier 2 by 0.278
(0.234 to 0.324). Without donor groups the call is far more cautious: precision among called cells is 0.9988 against 0.9799
for cellXY, but only 79 percent of cells in the mixed pool receive a call, because most female cells carry no XIST UMI. The
five tier 2 samples that every method calls opposite to GEO are the earlier suspected depositor errors.

## Held-out (scored once): GSE205796 uninfected pool, 48 lines of both sexes, 20,129 cells

| Method | Per-cell acc (macro) | Call rate | Precision | Donor acc | Donor abstain |
|---|---|---|---|---|---|
| Singlet, donor groups | 0.9742 | 0.9764 | 0.9792 | 0.9792 (47 of 48) | 0 |
| Singlet, no groups | 0.9103 | 0.9161 | 0.9792 | 0.9792 | 0 |
| two-gene rule | 0.8474 | 0.8530 | 0.9790 | 0.9792 | 0 |
| cellXY | 0.9113 | 0.9376 | 0.9718 | 1.0000 | 0 |

Per-cell bar met: Singlet with groups minus cellXY is +0.063 (CI +0.013 to +0.093). Like-for-like precision bar met.
Donor bar of 0.99 missed: one line, HG00154, labeled female, is called male by Singlet and by the two-gene rule. Its cells carry
13 XIST UMIs in total (0.02 per cell, against about 8 per cell in the other 23 female lines) and chrY at the ambient level
of the female lines (0.12 per cell, against 10 to 20 in the male lines). The line has lost XIST expression, which is known to
happen in lymphoblastoid lines, so the only sex-informative UMIs are a chrY trickle from the male lines in the pool, and any
XIST-against-chrY split reads that as male. cellXY calls the line female because it also uses X-linked escape genes.

## Fix route (not applied)

The tested idea, a floor on the fraction of cells with any XIST or chrY UMI, does not separate this line from honest low-evidence
samples: 43 of the 170 GEO samples have a fraction below 0.25. A fix has to compare a group's chrY rate with the other
groups of the same library (a pool-aware ambient reference) or add X-dose evidence from escape genes. Both need design on development
data and a second held-out mixed-sex pool, which this work did not have. The held-out library was not rescored, and no threshold was
tuned on it. Singlet's group model remains unchanged.

## Verdict

Under the pre-registered rule the donor bar is missed on held-out, so the call is withheld. Facts behind that rule: 4 of 4 tier 1
donors correct, 47 of 48 on held-out, and the one miss is an XIST-silent female line that the XIST and chrY evidence cannot resolve.
Per cell, Singlet is ahead of both comparators everywhere. If the owner prefers to relax the rule, the supportable form is metadata
only, hidden by default, as an annotation and never a filter, with abstention when a group's chrY rate is at pool ambient level
once that check exists.

## Runtime

Evidence extraction plus the call, one core, 25,899 cells by 38,606 genes: 0.26 s, which is 0.0017 core-minutes per 10,000 cells
(ceiling for all calls together: 2). For comparison, cellXY needed 70 s for the 20,769-cell pool and 102 s for the 20,129-cell held-out library, in dense blocks.

## Open problems

- Only 4 independent tier 1 donors in development (3 female, 1 male). Kang lanes and the Neavin village have donor assignments or arrays
  but no stored donor sex; genotype-derived sex from chrX heterozygosity was not built in the time available.
- The held-out lines carry no Singlet run; author Cell Ranger counts were used, and the call reads only XIST and chrY genes.
- XXY, X0 and mosaic loss of chrY remain untested.

## Addendum: pool check with a literature X-escape set (post-hoc, registered in the gate addendum)

cellXY is GPL-3.0, so nothing was ported. The escape set comes from the literature, and the rule is in `python/singlet/celllabels/sex.py`
(`_pool_check`, constants fixed by principle, not fitted). Development finding: the escape fraction separates the sexes only within one
library (about 2 times, development pools) and not across libraries (AUC 0.60 over 170 GEO samples), so the rule is pool-relative and
inactive wherever fewer than 4 donor groups exist.

| Library | Rule | Per-cell acc | Donor acc |
|---|---|---|---|
| Development (both pools, GEO samples) | check never fires | unchanged | unchanged |
| Held-out uninfected pool (post-hoc, seen) | before / after | 0.9742 / 0.9747 | 47 of 48 / 47 of 48 |
| Infected pool, same lines (not independent) | before / after | 0.9692 / 0.9692 | 46 of 47 / 46 of 47 |

The check fires on the XIST-silent line (chrY at ambient level, correctly identified) but turns it into an abstention, because its
escape-gene rate is not nearer the pool's female median than the male median. Escape-gene dose from 25 genes is too weak to rescue a
single group. The verdict is unchanged: withheld by the gate. The check is kept because it converts a confident wrong call into an
abstention, but it does not meet the donor bar. A clean test needs an unseen mixed-sex pool.

Real XXY, X0 and chrY-loss data found by search, not yet downloaded or scored: GSE136353 (PBMC of one 47,XXY patient with male and
female controls), GSE169062 (testis, Klinefelter cases, Sertoli cells lacking XIST), PRJNA908831 (isogenic iPSC lines 47,XXY, 46,XX,
46,XY, 45,X; single-cell content unconfirmed). No labelled Turner single-cell data was found.
