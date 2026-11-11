# E2: truth-gated pruning of heavy reads

**Hypothesis.** This is the advisor's option 2. Reads that overflow 16 windows go to lanes M and B, where they run under STAR's full limits: 128 windows, 50 alignments per window, 20 exons. Some of these reads STAR reports as multimapped, as mapping to too many loci, or as unmapped after search. Capping their window set at the 16 best anchors should gain at least 1.3x in the alignment kernels. The truth gate should still pass, because these are the reads STAR already handles worst.

**Baseline.** Replay of the 1,991,926-sequence pbmc1k_v3 set (job 803599, frontier, 100% identical to STAR) gives lane A 174 ms, M 213 ms and B 156 ms. Lanes M and B carry 31% of the distinct reads but take 68% of the kernel time, and phase 3 is most of both. Halving M and B takes the total from 543 to 358 ms, a 1.52x gain. Removing them entirely bounds the gain at 3.1x.

**Change.** A window cap for the heavy lanes behind a named mode, never the default: first in the replay harness as `m5` flags, then in engine_gpu.cu. Reads that end unique under the cap keep their placement. The rest carry a flag, and the pileup's multimapper rules treat them as they do now. The CPU and GPU implementations must agree (oracle O5).

**Measure.** Tier 1 is replay on the standard set, giving kernel time per lane and the reads whose result changed. Truth is scored by the truth track's frozen gate (singlet-agents, `bench/ground-truth/gate`), never by a copy, in two stages:

- **Iteration loop.** PBMC-only sweep runs on the development truth-read subsets. Each takes well under an hour.
- **Final candidate.** The full gate on the four development datasets that can gate a GPU candidate.

The gate takes one BAM per dataset with FASTQ read names, plus an identity report showing that BAM mode equals the production stream on the same binary. `bench/frontier-accuracy/gpu/gpu_truth.sbatch` does that plumbing. The held-out split is never built, run or scored before the OSF registration. The pruned binary also reruns count parity, because changed alignments change the matrices.

The measures:

- **Q.** Molecule-level counting accuracy: the share of truth molecules (one cell barcode and UMI) counted to the long-read gene, relative to STAR production, with a paired cell bootstrap.
- **M1 to M7.** The gate's measures, each with its registered margin.

The reference arm is the current production (K5 at score 30, run imp_gpuK5s30), not STAR. Production already sits about one Q point above STAR, and the headroom to a perfect counter is under 0.01. Against that arm the bar is:

- Q stays inside the reference run's bootstrap interval (about ±0.0001).
- M1 wrong, M1 correct and M3 UMIs lost stay inside the reference run's intervals.
- Every gate measure passes.
- The production identity check stays at 0 differences.

A cap that leaves STAR's chosen alignments alone shows M1 and M2 identical to four decimals. M3 UMIs lost and M1's unmapped share move first when dropped windows matter.

**Falsifier.** Any one of these ends the card: the kernel gain on the standard set is under 1.3x; Q or M1 or M3 leaves the reference run's interval; any gate measure fails; the identity check shows a difference.

**Result.** Pending. The gate inputs, Q and the guardrails are settled with the truth track (their RESULTS.md, commit 8f04d99). The next step is the replay flavor with a window cap for lanes M and B.
