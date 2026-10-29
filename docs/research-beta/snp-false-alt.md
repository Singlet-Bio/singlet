# SNP false-alt root cause (D6) and rule R12

Counts only; no site, allele or genotype is stored here. Scripts: `bench/snp_falsealt/` (run on Clipper next to the private donor VCFs; jobs 792789, 792790).

## Definition (same as the `donor-snps` capability, `bench/public/genotype/lib/snp_concordance.py`)
Truth = the replicate lane's pseudobulk per truth donor (demuxlet singlets; cellsnp-lite counts at the 1000G panel), genotyped with a binomial model (error 0.01, flat prior, posterior >= 0.99). Singlet's donor cluster is matched to a truth donor by cell overlap. **False alt** = truth 0/0 and Singlet's donor-VCF GT is 0/1 or 1/1, over sites where the donor's own DP >= 10. Reproduced: Kang stim (`snp/none`, R11 binary) 658 / 11,022 = **0.0597**; the campaign run of the same lane 555 / 10,298 = **0.0539**.

## Cause
The false alts are made by the **genotype call, not by the counts**. `write_donor_vcfs` / `write_donor_coverages` call GT from the VB posterior allele frequency with fixed cuts (`af < 0.1` -> 0/0, `af > 0.9` -> 1/1, else 0/1). Sequencing/mapping error puts about 0.5-1% alt molecules at hom-ref sites, so at DP 10-25 a single alt molecule pushes the fraction (and the VB AF) over 0.1.

| stratum (Kang stim, hom-ref truth, DP >= 10) | n | false-alt | share of all false alts |
|---|---|---|---|
| alt molecules = 0 | 7,812 | 0.0023 | 2.7% (VB prior) |
| alt molecules = 1 | 1,773 | 0.233 | 62.8% |
| alt molecules = 2 | 552 | 0.199 | 16.7% |
| alt molecules >= 3 | 885 | 0.12-0.17 | 17.8% |
| alt fraction 0.05-0.10 | 651 | 0.625 | |
| alt fraction 0.10-0.20 / >= 0.20 | 182 / 57 | 0.97 / 1.00 | |
| DP 10-15 / 15-25 / 25-50 / 50-100 / >= 100 | 2768 / 2916 / 2504 / 1174 / 1660 | 0.124 / 0.079 / 0.024 / 0.012 / 0.006 | |

Counting is not the problem: the same aggregated AD/DP genotyped with the benchmark's binomial model gives **0.0051** false-alt (cellsnp-lite + vireo class, 0.005-0.008). Nothing else shows up in the counts: substitution type is flat (A>G 0.053, T>C 0.066, others 0.035-0.084 against 0.060 overall; A>G + T>C are 30% of false alts and 29% of the hom-ref sites, so no RNA-editing signal), chromosomes are flat (0.03-0.09, chrX 0.10), all 658 false alts are 0/1 (none 1/1), and base quality does not matter (#82). Strand, read position, junction distance, MAPQ/NH and per-molecule evidence cannot be stratified from the aggregated VCF (needs the pileup's per-molecule records); they are not needed to explain the 0.06 and are listed as the next step if the residual 0.005-0.006 matters.

## Fix: R12 (below) and its measurement
Rule sweep on the five pools with Kang truth, Singlet donor VCFs of the campaign runs, hom-ref DP >= 10. "false-alt" counts uncalled `./.` as not-alt; "true alt kept" = truth 0/1 or 1/1 called alt.

| pool | current false-alt | R12 false-alt | R12 true alt kept | R12 conc. (called) | called |
|---|---|---|---|---|---|
| kang2018_stim | 0.0539 | 0.0058 | 0.972 | 0.9876 | 0.986 |
| kang2018_ctrl | 0.0442 | 0.0059 | 0.977 | 0.9883 | 0.988 |
| kang2018_b1_a | 0.0330 | 0.0119 | 0.972 | 0.9828 | 0.988 |
| kang2018_b1_b | 0.0261 | 0.0034 | 0.980 | 0.9890 | 0.990 |
| kang2018_b1_c | 0.0404 | 0.0053 | 0.974 | 0.9871 | 0.988 |
| pooled (32,597 hom-ref sites) | 0.0437 | 0.0061 (among called) | | | |

Current true-alt kept is 0.995-0.998, so R12 costs 2-3% of true alt calls (target: <= 10%), not 10%. On the exact 0.0597 run (`snp/none`) R12 gives 0.0059 among called sites (60 residual of 10,246). Of the 1,423 pooled old false alts, R12 turns 1,088 into 0/0, leaves 136 uncalled and keeps 199; the residual shows no editing enrichment (A>G + T>C 0.0060 vs 0.0062 others) and sits at DP 10-50 with 2-10 alt molecules. Concordance over called sites rises 0.951-0.969 -> 0.983-0.989.

Curve of alternatives (Kang stim, same sites): VB-AF hom-ref cut 0.15 / 0.2 / 0.25 -> 0.0213 / 0.0063 / 0.0031 false-alt with true alt kept 0.995 / 0.988 / 0.973 (but it keeps the shrinkage artefact at low DP and no `./.`); "alt needs >= 3 molecules" -> 0.0097 / 0.982. Binomial error 0.005 / 0.01 / 0.02 at posterior 0.9: 0.0083 / 0.0058 / 0.0036, true alt 0.977 / 0.972 / 0.962. Posterior 0.99 at error 0.01: 0.0036, 0.941. R12 uses error 0.01, posterior 0.9 (the knee).

Second dataset: the other four Kang pools above (b1 lanes are different libraries and donors from the b2 lanes). The 1000G LCL pools have no Singlet donor VCFs yet (no `singlet_auto/private/donor` under `bench-public/runs/lcl_*`), so they are not measured. Not run: a full engine re-run with the patched binary (the rule is post-hoc on the VCF writer and depends only on the `ad` and `dp` that are already written; the unit test pins the anchor cases and the Python sweep uses the identical formula).

## R12. Donor genotype call: binomial posterior on aggregated molecule counts
**Rule.** For each (donor, covered site), with `dp` = aggregated molecule depth and `ad` = alt molecules from that donor's assigned cells (the R1/R9 molecule counts, unchanged), set `r = dp - ad`, `e = 0.01`, and the three log-likelihoods `L0 = r*ln(1-e) + ad*ln(e)`, `L1 = dp*ln(0.5)`, `L2 = ad*ln(1-e) + r*ln(e)`. With a flat prior over {0/0, 0/1, 1/1}, GT is the genotype whose posterior is >= 0.9, else `./.`. `dp == 0` -> `./.`. The VB `af_vb` stays in the output for information but no longer decides GT.

**Order-free.** The inputs are integer sums over the donor's cells (commutative), the function is pure and stateless, and ties cannot occur at posterior >= 0.9 > 0.5. The donor assignment of cells is unchanged by R12. Computation in double; the log and exp of three terms. Any threshold flip at exactly 0.9 is the only place implementations can differ, so the GPU reducer should use the same double formula (or compare in log-odds: `Lbest - ln(sum exp(L)) >= ln(0.9)`).

**GPU SNP reducer needs.** Per (donor, site): the two uint32 sums `ad` and `dp` (already in the sparse donor depth tables), then a trivial epilogue kernel that applies the formula above and writes the GT code (0, 1, 2, -1). It does not need qualities, positions or strands. Gate: identical GT codes to the CPU function for every (ad, dp) with dp <= 4096 (exhaustive table), and identical `ad`/`dp` sums to the CPU path.

**Evidence.** See above: false-alt 0.054 / 0.044 / 0.033 / 0.026 / 0.040 -> 0.0058 / 0.0059 / 0.0119 / 0.0034 / 0.0053 on five Kang pools, true alt kept 0.97-0.98.
