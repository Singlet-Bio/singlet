# Candidate U rules for the new label (binary 72ac6555), pre-registered 2026-10-31 BEFORE the molecule dump was read

Data: Singlet 72ac6555 on full pbmc1 with SINGLET_MOLECULE_DUMP; truth = `truth/molecules.parquet` (Truth-R2 `clsR2` primary, Truth-LR `clsLR`
secondary), same CB + 11-nt UMI (first 11 of the 12-nt padded UMI) + gene join as before. Dump columns used: label, flags (OR of kRead*),
diag (OR of kDiag*: Multi 1, Spliced 2, Flank 4, OtherGene 8, Antisense 16, Outside 32), reads (reads per molecule). Because flags/diag are ORs over
a molecule's reads, "flank only" is exact only for single-read molecules; multi-read molecules are handled by the stated approximation.

A rule only decides which molecules labelled U stay U; a molecule removed from U becomes S if diag has Spliced, else A (the new label_of).
Baseline C0: label as dumped (U = any read touches an intron).

| id | molecule stays U iff (it is U under C0 and ...) |
|---|---|
| C1 | NOT (reads == 1 and Flank): a single read whose intronic part is <= 5 bp is not intron evidence |
| C2 | NOT Flank (any Flank read in the molecule removes it; conservative approximation for multi-read molecules) |
| C3 | reads >= 2 |
| C4 | NOT (OtherGene or Antisense) |
| C5 | C1 and C4 |
| C6 | C2 and C4 |
| C7 | C5 and NOT Multi |
| C8 | C1 and reads >= 2 for molecules with OtherGene/Antisense/Multi, i.e. ambiguous-locus molecules need 2 reads; clean molecules unchanged |

(A "> N intronic bases" rule needs per-read intronic length, which the dump does not carry; not evaluated.)

## Metrics
Molecule level (molecules with a truth and the same gene): precision, recall, F1 of U vs Truth-R2 U (also vs Truth-LR U). Share of U calls
that have a truth at all. (cell, gene) level: `score_usa.score_tool` on the relabelled Singlet matrices, Truth-R2, q >= 0.9 stratum: pair r,
gene-pooled r, per-cell median r, has-unspliced precision / recall / F1, tool_U share.

## Selection without reusing the reported molecules
Cells sorted, index parity: even = selection half, odd = report half. Choice: among rules whose molecule-level recall on the even half is
>= 0.95 x C0's recall there, the one with the highest molecule-level U F1 vs Truth-R2 on the even half (ties: the simpler id). All rules and
the chosen rule are reported on the odd half; the even-half numbers are shown only for the choice.
