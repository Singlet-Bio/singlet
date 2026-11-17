# Beta freeze counting gate (build 1917e0cc, rules 61837)

The gate record a2cca300877f5d06.json reports FAIL on hgmm1k_v3 (barnyard.X, barnyard.exon, chrY) and hgmm1k_v2 (chrY). pbmc1k_cite_v3 passes.
The owner accepted these misses for the freeze on 2026-11-11 (decision D-136 in Singlet-Bio/singlet-agents, sprint/DECISIONS.md).
The excess chrY UMIs sit at one locus, ENSG00000291032, and follow from dropping rule bit 512 (kGeneFullSet). Rules 62349 are a possible post-launch improvement.

## v1.3.7 (record 99f03cb79a05811c.json, commit f84a90a0)
The rel137 gate run on the CPU engine reproduces the same four misses (hgmm1k_v3 barnyard.X, barnyard.exon, chrY; hgmm1k_v2 chrY) and passes the rest. They stay owner-accepted under D-136. The record lists them in its `accepted` field, and gate_check.py treats only those listed axes as accepted: any other FAIL axis still fails the check.
