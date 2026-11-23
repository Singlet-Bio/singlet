# Protocol-detection truth panel: registration (2026-11-23)

Written before any detector run or threshold choice on this panel. Built by `make_panel.py` from the sources it names.

| file | rows | sha256 |
|---|---|---|
| panel.tsv | 223 | 9d193349559d022e509ac67f94c103ddddae7e35343136f8613a2f68e9278109 |
| dev.tsv | 114 | 1d7c26e269ce9ef917f01c023ef017cfb9bface4e93f0142b4199704eabf7ff9 |
| heldout.tsv | 109 | 256a38dea7778376b6ec9e4d1105df20a5853f41a6369cd84e7c4411740e7d12 |

## Scoring rules (fixed now)

- Unit: one row (one run or one library source). The detector sees only reads of that source (no metadata hint enters the
  family or version call; the stated chemistry may only order the trial-mapping candidates).
- Relabel rule: a row with `deposit_check = barcode_read` whose head holds no slot reaching 5 percent exact hits on any 10x
  list (3', 5', v1 to v4, ARC GEX, Visium) at any offset or strand is scored as family `no_cell_barcode`, version "".
- Family recall: share of rows of each family called that family; reported per family and pooled over all rows.
  Family precision: share of calls of each family that are right; pooled over all calls.
- Version recall: over rows with a non-empty version, share whose final call (detector, then trial mapping where it runs) names
  that engine tag. 5' v1 and v2 both count as 10x-5p-v2 (the engine's route for 5' v1).
- A call of "ambiguous" or "unknown" counts as a miss for recall and is not a call for precision; ambiguous calls are listed.
- Bars (task brief): held-out family recall and precision at least 0.98, version recall at least 0.95, trial mapping included.
- The held-out half is scored once with the frozen detector. Its numbers are reported whether or not they clear.

## Known limits of the panel

- The identify held-out rows (47) were scored once by identify a5c45523 on 2026-11-13; their aggregate misses are described in
  the phase-2 notes, which the designer of this detector has read. Row-level outputs of that run were not used here.
- The short-R1 Ding PBMC runs were described (layout and whitelist hit rate) in the S2 structure probe before this panel was built.
- 10x 3' rows dominate (131 of 223). Families with one or two rows (10x_flex, vdj, visium_probe) cannot meet a 0.98 bar alone;
  pooled recall carries the bar, and per-family rows are listed.

## Freeze before the held-out score (2026-11-23)

Tuned on the development half only. Development score with trial mapping: family recall 105/114 = 0.921, family precision
105/114 = 0.921, version recall 65/68 = 0.956. The held-out slot scan was started before this freeze, and no held-out output was
opened before these hashes were recorded. The held-out half is scored once with:

| file | sha256 |
|---|---|
| hpc/clipper/campaign/protocol_detect.py | d656770c7c65c2419c53be0237eba1b4998289ef484e9e520c87e0d1defee313 |
| hpc/clipper/campaign/trial_map.py | f26a553421b4bf3ba426661a66a9afc0246aae1f298ecdc5d5e562aba5eea751 |
| bench/protocol-detect/score.py | 4b4cb204bc1cc6cf6c4c003887c1cea3134d63ea02b69eb3821d8bddc5844359 |

score.py changed once after registration: the called version of a versioned family is the top candidate's protocol tag, not its
engine tag (the engine tag was empty for barcode reads on a third or fourth slot, which the engine handles).
