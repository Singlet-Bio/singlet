# Fable 5.1 advisor report (2026-11-06), as received

## 1. Verdict on the plan

The plan attacks the GPU first. The numbers say the host binds first. Arm A of the shared-index job ran 24.3 of 28 cores busy with the lock held 64% of the span. The per-run core budget on pbmc1k_v3 is about 630 core-s. Demux takes 142, stream 170, refs 91. Demux runs auto-K on every sample, including single-donor ones, because `src/pipeline/singlet.cpp` defaults `n_donors = -1` and `export.h:1247` only short-circuits at `n_donors == 1`. That is 22% of host work spent fitting donors that do not exist. Cut host core-s first, add the second GPU, then buy lock time.

| Rank | Item | Experiment | Expected | Go/no-go |
|---|---|---|---|---|
| 1 | Skip demux on single-donor samples | Before `run_demux`, run `select_k_heldout` for K=1 vs 2 only, on at most 2,000 called cells and the reduced SNP set, then `make_single_donor_result` when K=1. Also honor `--n-donors 1` from sample metadata. | -120 core-s/run, about +20% at the core bound | Go if core-s/M falls 15% and Kang auto-K still returns 8 |
| 2 | Both GPUs, whole processes (job 800789) | k=16, T=7, 2 slots per GPU | up to 1.7x. Host RAM risk: 306 to 340 GB at k=8 | Go if rate at k=16 is 1.5x and host peak under 700 GB |
| 3 | Merge e20w, WSHRINK, strand fix, split auto | One exclusive timing | hold -20%, core-s -12% | Already justified |
| 4 | `refs` stage, 91 core-s | `perf record` on one run. If it copies or decompresses the genome and non-host index per process, mmap the page-cached files or use STAR `SharedMemory.cpp` on the branch | -60 to -80 core-s/run | Go if over half of refs is memcpy or decode |
| 5 | Shared index fix | Section 2 | enables 3 to 4 slots without lean mode, +10 to 20% when lock-bound | Go only if the gather microbench runs at private speed |
| 6 | `stream` stage, 170 core-s | Profile; largest host stage, parts unnamed | unknown | Decide after the profile |
| 7 | Exact cross-sample cache | Section 4 | -10 to 20% hold on 3' blood and cell lines | Go if hold falls 15% on pbmc1k_v3 with core-s/M up under 2% |
| 8 | Lane A and M kernels | After WSHRINK lane B is 2.2 s of a 30 s hold; lanes A and M are 13 s under load. Re-profile `k_p3lb` and `k_p1s` | +5 to 10% hold | Only after 1 to 5 |
| 9 | Demux GPU port | Defer. After item 1 it runs only on multiplexed samples | small | Revisit later |
| 10 | Batching service | Drop for now | | |

Ceiling after items 1 to 5: cores at 56 and about 6 core-s/M give near 9 M pairs/s at full efficiency, about 6 at the measured 0.65. The device lock gives about 2.3 M pairs/s per GPU at 3 MPS slots, so 4.5 to 5 per node. The lock binds next, then host RAM caps k near 16. Lock time per run is the metric to drive after the merge.

## 2. Shared-index slowdown

Lane A 75x slower, lane M 40x, lane B 5x: the ordering of random index loads per second in each lane. Every load from the imported G, SA, SAi pays a long fixed latency: either a non-local aperture mapping or small-page mapping of the import (page-table walks per random SA access). Checks: (1) extend `bench/engine_gpu_gate/ipc_test.cu` with 10^8 random 8-byte loads over a 12 GB buffer, private vs imported, report GB/s; (2) same on memory-pool IPC (`cudaMemPoolCreate` with `cudaMemHandleTypePosixFileDescriptor`, `cudaMallocFromPoolAsync`, `cudaMemPoolExportPointer`, fd over the Unix socket with SCM_RIGHTS; client `cudaMemPoolImportFromShareableHandle`/`cudaMemPoolImportPointer`) or driver VMM (`cuMemCreate` with recommended 2 MB granularity, `cuMemExportToShareableHandle`, `cuMemMap`, `cuMemSetAccess`); replace the handle path in `shared_index.cuh` if it runs at private speed; (3) D2D-copy control; (4) `cudaPointerGetAttributes` and an ncu diff of one `k_p1s` launch.

## 3. Donor demux

The benchmark tested stale code. `bench/demux-truth` carries the May 2026 `donor_demux.h` (uint8 counts, `select_k_bic`, no doublet model). The campaign line carries the October header (uint16, `select_k_heldout`, `doublet_stage`, staged restarts). `demux_mix.cpp` only compiles against the old one. The October engine on the real Kang 8-donor pool (`bench/modalities/demux_fix/results/final_eval.txt`) gives singlet ARI 0.958, assigned accuracy 0.982, auto-K 8 of 8, doublet AUROC 0.969; Vireo on the same cells 0.923, 0.948, 0.950 and loses donor 1039. `DEMUX-GPU.md` describes the old engine. Rebase the benchmark on the campaign header before any verdict.

Model: the header ports vireoSNP's `BinomMixtureVB` (mitochondrial clone model, free Beta per SNP per donor). The donor model in vireo is the three-state genotype mixture (categorical over 0/0, 0/1, 1/1 per SNP per donor, allele probabilities shared across SNPs). The clone model has K*ns free parameters and a rich-get-richer dynamic, so K=2 drifts to one component plus debris or a non-donor split; with 2 donors PC1 is often depth or cell type and k-means++ seeds a wrong split. The October code adds genotype states only for scoring (`build_allele_tables`). Port the three-state fit (also simpler on GPU: theta is three scalars).

Diagnostics first: initialize VB from truth labels and compare ELBO with the found optimum on m2_r02 (found higher = misspecification, truth higher = seeding). Per sample, histogram pseudobulk alt fraction over covered SNPs (one donor is trimodal at 0, 0.5, 1; extra modes = pooled sample).

Provenance: S1 GSM4006843 is cultured HUVEC (GSE135356), lots often pooled from several donors. S2 GSM7474576 `D1_1_GFP+_VDJ` is a sorted, stimulated T cell VDJ library (GSE234778), SNP coverage near TCR loci. S3 GSM7901264 glioblastoma patient PBMC. S4 10x pbmc1k_v2. Every m2 mixture uses the two doubtful samples. Report per sample the median covered and informative SNPs per cell (both alleles seen in 10 or more cells). Replace S1 and S2 with two GEX PBMC samples from one study and one chemistry.

Ceiling: Vireo (Apache-2.0) reads cellSNP AD/DP matrices directly; souporcell clustering and `troublet` (MIT) accept ref/alt matrices. Run both on every mixture and on Kang. Keep the C++ reimplementation in the pipeline, no wrapping. Demuxafy (Genome Biology 2024) and a 2025 snRNA benchmark rank Vireo first among genotype-free tools at 16 or fewer donors, 85 to 90% precision on real 2-donor pools, so a K=2 failure is a bug.

Other: doublet AUROC below 0.5 from `1 - prob_max` is expected (summed depth sharpens the posterior); score doublets only with `prob_doublet`. Add a 0.6 to 0.8 read-keep fraction to planted doublets. Seed changing 47% of assignments is a convergence failure. Drop the proposed bands; set tolerances against truth after the fix (accuracy within 1 point, AUROC within 0.01, K equal). `write_donor_vcfs`, `write_donor_coverages`, `snp_ad`/`snp_dp` must stay in the private object; public outputs get labels and probabilities only.

## 4. Exact read cache

Build it after items 1 to 5. Key on the post-CR4-clip sequence (the clip reads per-base quality). Bind the table to a hash of genome directory, STAR version and parameter block. Store 128-bit hash + length + record blob; 200 M entries about 8 GB, mmap MAP_SHARED per node. Consult after device dedup on distinct sequences, drop hits from the lanes, merge records back in input order. Audit 1% of hits by re-aligning on the device. Gate with the identity harness (cache on vs off). Re-measure hit rates on 20 random GEO samples before sizing an atlas store.

## 5. Measurement

Rate has a 7.7% leg CV, so two legs cannot see effects under about 15%. Core-s/M and lock held have CVs near 1%: make them the primary decision metrics and use the demand model to translate to node rate. Put a fixed b0 arm in every job and report ratios to it (g050 and g051 differ by 35%). Exclude runs before the first completion and report a bootstrap interval over per-run rates. Interleave binaries run by run inside one leg. Fix the build commit check. Demux tolerance is against truth, not against the CPU run.
