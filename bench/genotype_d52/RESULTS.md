# Genotype outputs (D-52): checks and concordance

Branch `feat/genotype-outputs` off `perf/gpu-candidate-pr2`. Clipper jobs, 2026-11-06. Genotypes stay on Clipper under `$WORK/genotypes/`;
this file holds counts only.

## What a pipeline run now writes

| File | Content |
|---|---|
| `private/genotype/donors.vcf.gz` | one VCF per sample with a column per demux donor, GT:GQ:AD:DP:PL. UMI counts summed over the called cells assigned to the donor with prob_max >= 0.9. GT is R12 (binomial, error 0.01, flat prior, posterior >= 0.99), and GQ and PL come from the same likelihoods |
| `private/genotype/cellsnp/` | the cellsnp-lite layout of the called cells at sites with DP >= 20 over those cells. `vireo -c` reads it. OTH is empty because the engine keeps no other-allele count per cell |

`private/` is created owner-only (0700) before anything is written into it, and the export ends by setting every directory below it to
0700 and every file to 0600. `$WORK` on Clipper is group project space, so genotype files are never readable by the group, also after a
mode-keeping copy. The campaign's genotype keeper (`scripts/atlas_rerun/genotype_keep.py`) and the backfill write 0700 / 0600 as well.

The writers run after the shared export, so the CPU and GPU paths write the same bytes. The engine's earlier outputs are unchanged
(`private/donor/donor{k}.vcf`, `private/snp_{ad,dp}.1pz`, `private/ase_counts.tsv`).

## Checks

| Check | Job | Result |
|---|---|---|
| CPU = GPU, pbmc1k_v3 and neuron1k_v3, binary a9ecfff7 | 803210 | IDENTICAL on both. out_equiv 71 / 5 / 4, every difference in E11's run-dependent set, level 5 of 5 with ingest |
| This branch's CPU against the candidate's CPU (c8849bb2) | 803301 | the 65 earlier files unchanged. Added: the 6 files above, plus `n_private` 12 -> 18 in pileup_stats.json |
| ctest (host recipe of gpu-pr2's ctest-host2) | 803199 | 158 of 158, `test_gpu_pileup_labels` skipped (no GPU on the build node), as on the base. New: `test_genotype_export` |
| Backfill C++ (engine writers, from .1pz) against Python (from .mtx), pbmc1k_v3, kang2018_ctrl, pbmc5p_v1 | 803228 | every file identical (decompressed) |

## Concordance against the gate's genotype truth

Truth and scorer are S10's donor-snps row: Kang replicate-lane genotypes and the 1000 Genomes high-coverage genotypes of GM12878 + GM18502
(Osorio mix). Runs are `bench-public/runs_gate` and the scorer is `snp_concordance.py`, extended here by one tool (`score_d52.py`).
Pooled over 18 donors in kang2018_b1_a, b1_b, b1_c, stim and osorio2019_lcl_mix. paired10 keeps only the sites where every tool has
depth >= 10 for the matched donor.

| Tool | Stratum | Sites | Concordance | Non-ref concordance | False alt |
|---|---|---:|---:|---:|---:|
| singlet donors.vcf.gz (this branch) | paired10 | 87,610 | **0.9901** | **0.9881** | 0.0061 |
| cellsnp-lite pileup | paired10 | 87,402 | 0.9901 | 0.9881 | 0.0060 |
| singlet donor{k}.vcf (before) | paired10 | 89,319 | 0.9864 | 0.9831 | 0.0074 |
| vireo | paired10 | 90,342 | 0.9773 | 0.9698 | 0.0086 |
| singlet donors.vcf.gz (this branch) | dp10 | 144,937 | 0.9852 | 0.9746 | 0.0038 |
| singlet donor{k}.vcf (before) | dp10 | 153,195 | 0.9798 | 0.9652 | 0.0049 |

On the 1000 Genomes pool alone (paired10) the scores are 0.9874 for this branch, 0.9874 for the cellsnp pileup, 0.9831 for the earlier VCFs and
0.9753 for vireo. The gain over the earlier VCFs comes from the called-cell rule, which drops ambient barcodes back-assigned to a donor.
