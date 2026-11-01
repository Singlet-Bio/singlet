# Status of donor-demux, donor-snps, mtdna (2026-10-26)

Results: `bench/public/results/{donor-demux,donor-snps,mtdna}.json`. They validate against `cohort.tsv` plus the rows in
`cohort_additions.py` (kang2018_stim, maester_k562_bt142_3p, maester_bpdcn712_3p; not yet in the cohort). Artifacts and
private runs are on Clipper under `$WORK/bench-public/{artifacts,runs}`; rebuild with `genotype/build_results.py`.

Done: Singlet auto-K and K-given, vireo (3 seeds, N = 12), souporcell, cellsnp-pileup and mgatk on 5 Kang lanes and 2
MAESTER pools; donor genotype concordance against replicate-lane truth; mt haplotypes, clonal variants, line mixture.

Not done:
- Osorio GM12878 + GM18502 mixture with 1000 Genomes truth (donor-snps, donor-demux K = 2): FASTQ download by the cohort
  session was still running. Truth genotypes are staged privately (`bench-public/private/truth_gt`, 23 chromosomes).
- lcl_gse230239 cannot be used: its FASTQs hold two 101-bp cDNA reads and no barcode read.
- Merge fix for the auto-K over-split (`fix/demux-merge`): designed, not implemented. Calibration: same donor in two lanes
  agrees at >= 0.956 of shared sites, different donors <= 0.678, the duplicate component in kang2018_ctrl 0.883, so a
  merge threshold near 0.80 separates them using held-out lanes only.
- Stoeckius consensus labels for the hto session.
