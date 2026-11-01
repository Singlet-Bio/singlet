# Ancestry label rule (celllabels/data/ancestry_label_model.json)

`anc_q.py`/`anc_q.sbatch` (Clipper): proportions q of the 1000 Genomes reference individuals that are not benchmark test individuals
(exact genotypes, leave-one-out frequencies, fixed random 300,000 panel sites) and of the 150 MAGE test individuals at 0.5/2/8M read
pairs. `anc_fit.py`: cross-validation on the reference and one scoring pass on the test individuals (coverage, named precision,
accuracy with abstention = wrong, agreement with peddy). `anc_export.py`: writes the shipped JSON. Only proportions leave the
genotype step; no genotype is written. The shipped file holds the reference proportions (3 decimals) for the out-of-distribution guard.
