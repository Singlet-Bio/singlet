# GPU pileup on the order-free stream: gate (a) against the CPU engine (2026-10-30)

Branch `gpu/engine-gpu-order-free` (PR #95, replaces #69), built on `prod/campaign` with #63 (unsorted stream, order-free rules R1-R11),
#66 (`--engine gpu`), #80, #82, #86, #93 merged (prod/campaign 89cb3d98). Contract: counting (pileup, dedup, UMI collapse, cell calls, reducers) is exactly equal to the CPU
engine; alignment is held to the M1-M7 gate separately.

## What the GPU path mirrors from #63

| Rule | GPU path |
|---|---|
| R9 per (cell, gene) | The junction and SNP reducers take a UMI's molecule from the gene reducer's collapse (`stage_c` keeps the component of every unique (cell, gene, UMI) entry). A reducer entry whose reads were assigned a gene (NH 1, smallest gene over the entry's reads, as `UmiFeatureCollapse::add`) and whose UMI the gene store holds is one molecule per component; the other entries collapse among themselves (`r9_molecules`). `k_r9` takes `rid`, `k_gene_rid` looks the component up by binary search. V(D)J (on the host) gets the roots from the GPU: `PileupOutput::geneMerges` lists the UMIs the collapse merged into another (`PileupInput::wantGeneMerges`, only when a V(D)J model is loaded). |
| R11 CB correction | The hand-off pass applies `CbCorrector::resolve` to CR as `stream_cb` does, and counts the same status line. The GPU collector takes the corrected CB for stream reads. File-mode `.1fq` decode (the only mode `--engine gpu` supports today) now builds STAR's exact CB histogram, as pass 1 of the FIFO path does; without it the corrector was silently off (`stream CB: CR exact (no pass-1 histogram)`). The `[cb-correct]` status counts of the GPU run equal the CPU run's, and the histograms of file mode and pass 1 agree to the read (64 216 700 on GSM4006843). |
| R10 / R10b / R10c ambient fit | Not on the GPU: it runs in `export.h` after the pileup on the accumulators, the same code for both engines. Gate (a) compares its outputs (`ambient_*`), which are identical. |
| R12 (binomial genotype call, #93) | Nothing on the device: it runs in `donor_demux.h` on `snp_ad` / `snp_dp`, which the GPU produces identically (gate below, run on the merged head). |
| #86 grouped nonhost EM | Host code after the pileup, shared; `nonhost_gpu.cu` only does the screen (phases A-C). Nonhost outputs are identical in the gate. |
| #82 binary read source | Note only (P3): `--engine gpu` still decodes the `.1fq` to FASTQ temp files (file mode); the GPU prefill could take `StarChunkSourceT` blocks instead, which would save the decode and the temp files. Not attempted. The R11 histogram is counted in the file-mode `process_block` on top of #82's `RunSet` decode. |

## Not mirrored: the intron rules of #80 (incl. D2, chrM/NUMT tie)

`prod/campaign` now defaults to `intron_rules = 389` (read-level kUnique, kNoSplicedIntron, kMtPrefer; molecule-level kUmiGeneCR). They are not
on the GPU (`k_annotate` / `k_resolve` / the gene reducer). With any rule on, `singlet` ignores `SINGLET_GPU_PILEUP`, says so, and runs the CPU
pileup on the GPU-mapped reads (counts correct, only the pileup saving is lost). `SINGLET_INTRON_RULES=0` restores the pre-#80 classification,
and gate (a) below runs with it. The GPU multimapper resolve (`k_resolve`) has the same gap D2 had on the serial path (kMtPrefer); on neuron1k_v3
(mouse) mt-Atp6 is 16 with rules 0 on both engines and 326 619 with the default rules on both engines (the GPU run is the CPU-pileup fallback). **The Beta cost saving needs #80 on the GPU first** (follow-up).

## Gate (a): CPU engine vs `--engine gpu` + `SINGLET_GPU_PILEUP=on`, `SINGLET_INTRON_RULES=0`

Same `.1fq`, 18 threads, one H100 NVL node. Clipper jobs 793026 (build 793025, commit 21da30b5: #86 merged, SNP rows saturate at 16 bit) for the three human samples and the cost table;
jobs 793137 (build 793136, commit 6b6ae50d, merged with 89cb3d98 = #80/#82/#86/#93) for pbmc1k_v3 again and neuron1k_v3.
"Identical" is `bench/star_accel/out_equiv.py`: sha256 of every output file.

| Dataset | Reads | Files identical | Differ | genes / introns / junctions / SNP / chrM / cell calls / ambient / V(D)J | Identical |
|---|---|---|---|---|---|
| pbmc1k_v3 (10x 3' v3) | 66.60 M | 57 | 7 (below) | all | yes |
| GSM4006843 (10x 3' v2) | 66.70 M | 59 | 7 (below) | all | yes |
| GSM6470208 (10x 5' v2, 411 V(D)J genes) | 121.77 M | 55 | 7 (below) | all | yes |
| pbmc1k_v3 on the merged head | 66.60 M | 57 | 7 | all | yes |
| neuron1k_v3 (mouse, GRCm39), rules 0 | - | 57 | 7 | all, mt-Atp6 16 = 16 | yes |
| neuron1k_v3, default rules (CPU-pileup fallback) | - | 57 | 6 | all, mt-Atp6 326 619 = 326 619 | yes |

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

## Intron rules of #80 (default `intron_rules` = 389) on the GPU

Branch `gpu/intron-rules` (on top of #95). `SINGLET_GPU_PILEUP=on` now runs with the default rules; `singlet.cpp` only ignores it for rules the GPU
does not implement (kContained 2, kFlank 8, kNoExonPriority 16, kUmiGene 32, kUmiGeneAll 64).

| Rule | Where on the GPU |
|---|---|
| kUnique (1), kNoSplicedIntron (4) | `k_annotate` records CIGAR N (`AA.hasn`); `k_resolve` clears the gene of an intronic primary when NH > 1 / has N (also in `aa`, which the host reads back as the classify_read gene for poly(A) and V(D)J, and in the NH 1 R9 gene of junction and SNP evidence) |
| kMtPrefer (256) | `k_resolve` replays the mm_buffer's `first_gene` / `second_gene` (a multigene primary gives its first two exon genes, `AA.pg_a/pg_b`) and the last chrM-gene alignment; a tie of one chrM and one nuclear gene counts the chrM gene. `mt_numt_rescued` is counted per read (`k_mtres`) |
| kUmiGeneCR (128) | `stage_cr` (m6.cu) after the UMI collapse. A molecule's UMI is the CPU's component root: the pairwise union is replayed per (cell, gene) in (reads desc, UMI asc) order (candidates by binary search of the 3 x UMI-length Hamming-1 neighbours), because that root is not always the best-supported UMI. Per (cell, root UMI) across genes the molecule with most reads is kept, ties and the rest dropped; dropped molecules still count in the per-cell molecule totals (as `per_cell_molecule_stats`) and in `molecules_dropped_multigene_umi` |

Gate (a), default rules, `bench/engine_gpu_gate/gate.sbatch` with `RULES=default` (modes `cpu`, `cmp`, `on`), one H100 job, `short-gpu`, binary built from
d79b0a30 (the first full round, jobs 793165, ran the same code minus the `mt_numt_rescued` counter):

| Sample | exon / intron / ambiguous / junction / SNP AD, DP / chrM / indel outputs, CPU engine vs `--engine gpu` + GPU pileup | `SINGLET_GPU_PILEUP=compare` | host core-s CPU -> GPU |
|---|---|---|---|
| pbmc1k_v3 | byte-identical | RESULT IDENTICAL | 2561 -> 838 |
| GSM6470208 | byte-identical | RESULT IDENTICAL | 3932 -> 1328 |
| GSM4006843 | byte-identical (CPU vs GPU) | not run in the gate (cpu, on only) | 2786 -> 1056 |

`out_equiv.py` differences are the ones listed above for rules 0 (read order of `nonhost_reads.fq` / `star_Unmapped`, `star_Log.std.out`, `read_stats.tsv`,
`clip5p/probe/n_reads`, byte counts, `memory_tier`), plus `mt_numt_rescued` (559 / 17 / 231 CPU vs 0 GPU) in the first round; with the counter
(rerun, job 793217, pbmc1k_v3) it matches and is no longer listed. Core-s per M reads therefore stays at the ~3.3x advantage of the
rules-0 gate.

Assumptions not separately tested: with several chrM-gene alignments in one read the CPU keeps the last one it saw (the stream's order, taken as the signature's
alignment order); a read-level drop applies to the primary alignment only. Both are exercised only if such reads exist in these samples, and the counts are identical.
Not implemented on the GPU: the other rule bits (the guard sends those runs to the CPU pileup).
