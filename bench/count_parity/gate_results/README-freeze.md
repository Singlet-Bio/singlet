# Beta freeze counting gate (build 1917e0cc, rules 61837)

The gate record a2cca300877f5d06.json reports FAIL on hgmm1k_v3 (barnyard.X, barnyard.exon, chrY) and hgmm1k_v2 (chrY). pbmc1k_cite_v3 passes.
The owner accepted these misses for the freeze on 2026-11-11 (decision D-136 in Singlet-Bio/singlet-agents, sprint/DECISIONS.md).
The excess chrY UMIs sit at one locus, ENSG00000291032, and follow from dropping rule bit 512 (kGeneFullSet). Rules 62349 are a possible post-launch improvement.
