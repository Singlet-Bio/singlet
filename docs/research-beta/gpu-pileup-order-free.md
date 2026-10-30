# GPU pileup on the order-free stream: gate (a) against the CPU engine (2026-10-30)

Branch `gpu/engine-gpu-order-free` (PR #95, replaces #69), built on `prod/campaign` with #63 (unsorted stream, order-free rules R1-R11),
#66 (`--engine gpu`), #80 and #86 merged. Contract: counting (pileup, dedup, UMI collapse, cell calls, reducers) is exactly equal to the CPU
engine; alignment is held to the M1-M7 gate separately.

## What the GPU path mirrors from #63

| Rule | GPU path |
|---|---|
| R9 per (cell, gene) | The junction and SNP reducers take a UMI's molecule from the gene reducer's collapse (`stage_c` keeps the component of every unique (cell, gene, UMI) entry). A reducer entry whose reads were assigned a gene (NH 1, smallest gene over the entry's reads, as `UmiFeatureCollapse::add`) and whose UMI the gene store holds is one molecule per component; the other entries collapse among themselves (`r9_molecules`). `k_r9` takes `rid`, `k_gene_rid` looks the component up by binary search. V(D)J (on the host) gets the roots from the GPU: `PileupOutput::geneMerges` lists the UMIs the collapse merged into another (`PileupInput::wantGeneMerges`, only when a V(D)J model is loaded). |
| R11 CB correction | The hand-off pass applies `CbCorrector::resolve` to CR as `stream_cb` does, and counts the same status line. The GPU collector takes the corrected CB for stream reads. File-mode `.1fq` decode (the only mode `--engine gpu` supports today) now builds STAR's exact CB histogram, as pass 1 of the FIFO path does; without it the corrector was silently off (`stream CB: CR exact (no pass-1 histogram)`). The `[cb-correct]` status counts of the GPU run equal the CPU run's, and the histograms of file mode and pass 1 agree to the read (64 216 700 on GSM4006843). |
| R10 / R10b / R10c ambient fit | Not on the GPU: it runs in `export.h` after the pileup on the accumulators, the same code for both engines. Gate (a) compares its outputs (`ambient_*`), which are identical. |
| R12 (binomial genotype call, #93) | **Not mirrored: #93 is not on `prod/campaign`.** TODO: the genotype call reads `snp_ad` / `snp_dp`, which the GPU produces identically, so it may need nothing on the device; confirm with gate (a) once #93 lands. If #93 changes the SNP molecule rule (R1), `red_snp` / `k_r9` must follow. |

## Not mirrored: the intron rules of #80

`prod/campaign` now defaults to `intron_rules = 389` (read-level kUnique, kNoSplicedIntron, kMtPrefer; molecule-level kUmiGeneCR). They are not
on the GPU (`k_annotate` / `k_resolve` / the gene reducer). With any rule on, `singlet` ignores `SINGLET_GPU_PILEUP`, says so, and runs the CPU
pileup on the GPU-mapped reads (counts correct, only the pileup saving is lost). `SINGLET_INTRON_RULES=0` restores the pre-#80 classification,
and gate (a) below runs with it. **The Beta cost saving needs #80 on the GPU first** (follow-up).

## Gate (a): CPU engine vs `--engine gpu` + `SINGLET_GPU_PILEUP=on`, `SINGLET_INTRON_RULES=0`

Same `.1fq`, 18 threads, one H100 NVL node; Clipper jobs 793026 (build 793025, commit 21da30b5, the head of this branch: #86 merged, SNP rows saturate at 16 bit).
"Identical" is `bench/star_accel/out_equiv.py`: sha256 of every output file.

| Dataset | Reads | Files identical | Differ | genes / introns / junctions / SNP / chrM / cell calls / ambient / V(D)J | Identical |
|---|---|---|---|---|---|
| pbmc1k_v3 (10x 3' v3) | 66.60 M | 57 | 7 (below) | all | yes |
| GSM4006843 (10x 3' v2) | 66.70 M | 59 | 7 (below) | all | yes |
| GSM6470208 (10x 5' v2, 411 V(D)J genes) | 121.77 M | 55 | 7 (below) | all | yes |

A first run after merging #86 differed in `snp_ad` / `snp_dp` (and what derives from them): #86 widened the CPU SNP counters to 16 bit and the GPU rows
still saturated at 255. Fixed in 21da30b5; the table is the rerun.

The differing files, the same on every sample, are not counts: `nonhost_reads.fq` and `star_Unmapped.out.mate1/2` (same bytes, other read order),
`star_Log.std.out`, `read_stats.tsv` (stage rows), `pileup_stats.json` (`clip5p/probe/n_reads` 12194 vs 12000 on the two 3' samples, 12201 vs 12000
on the 5', `bytes_public`), `summary.json` (`memory_tier`). Two CPU runs of the same sample differ in the same logs and orderings (pbmc1k_v3).
`SINGLET_GPU_PILEUP=compare` (GPU pileup against the CPU pileup on the same alignments) reports `RESULT IDENTICAL` for the 9 outputs
(exon, intron, ambiguous, snp_ad, snp_dp, mt, mt_rev, junctions, mt_indels) on pbmc1k_v3 and GSM6470208.
With the default rules (389) CPU engine vs `--engine gpu` + `SINGLET_GPU_PILEUP=on` (CPU pileup fallback) is also identical outside the same files
(pbmc1k_v3 and GSM6470208).

## Cost

| Dataset | CPU engine, core-s (wall s) | GPU engine + GPU pileup, core-s (wall s) | core-s per M reads, CPU / GPU | GPU pileup, GPU ms / total ms |
|---|---|---|---|---|
| pbmc1k_v3 | 2646 (448) | 812 (351) | 39.7 / 12.2 | 1715 / 8326 |
| GSM4006843 | 2786 (665) | 1011 (462) | 41.8 / 15.2 | 233 / 13450 |
| GSM6470208 | 4086 (746) | 1345 (537) | 33.6 / 11.0 | 857 / 18081 |

core-s is the job cgroup's CPU time for the whole run (reference loading, `.1fq` decode, R11 histogram and the post-pileup stages included);
one run per cell. W11's numbers were 10.7 (5'), 10.5-11.2 (3') and 37.5 for the CPU engine. The 5' sample is in W11's range, pbmc1k_v3 slightly above it, and the 3' v2
sample (15.2, 15.0 in an earlier run on the same code) well above; not investigated (the GPU pileup itself is 0.2 s of GPU time there).
GPU wall time is the whole run, mapping on the GPU included; the pileup column is the part W11 measured.

## Reproduce

`bench/engine_gpu_gate/{build,gate}.sbatch` (copied to `$W/jobs/w11/of/` on Clipper; the gate's `--dependency` line names the build job) (build from the branch, then the modes `cpu0` = CPU engine,
`cmp0` = `--engine gpu` with `SINGLET_GPU_PILEUP=compare`, `on0` = GPU pileup, `cpuD` / `onD` = default rules).
