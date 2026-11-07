# E2: truth-gated pruning of heavy reads

**Hypothesis.** This is the advisor's option 2. Reads that overflow 16 windows go to lanes M and B, where they run under STAR's full limits: 128 windows, 50 alignments per window, 20 exons. Some of these reads STAR reports as multimapped, as mapping to too many loci, or as unmapped after search. Capping their window set at the 16 best anchors should gain at least 1.3x in the alignment kernels. The truth gate should still pass, because these are the reads STAR already handles worst.

**Baseline.** Replay of the 1,991,926-sequence pbmc1k_v3 set (job 803599, frontier, 100% identical to STAR) gives lane A 174 ms, M 213 ms and B 156 ms. Lanes M and B carry 31% of the distinct reads but take 68% of the kernel time, and phase 3 is most of both. Halving M and B takes the total from 543 to 358 ms, a 1.52x gain. Removing them entirely bounds the gain at 3.1x.

**Change.** A window cap for the heavy lanes behind a named mode, never the default: first in the replay harness as `m5` flags, then in engine_gpu.cu. Reads that end unique under the cap keep their placement. The rest carry a flag, and the pileup's multimapper rules treat them as they do now. The CPU and GPU implementations must agree (oracle O5).

**Measure.** Tier 1 is replay on the standard set, giving kernel time per lane and the reads whose result changed. The truth score comes from the truth track's gate (`gate_eval`, Q and the guardrails in singlet-agents `bench/frontier-accuracy`, branch `claude/compassionate-curran-6e8664`), run on the development sets only. The held-out panel stays untouched until the OSF registration.

**Falsifier.** The truth gate fails on any development set, or the kernel gain on the standard set is under 1.3x.

**Result.** Pending. The truth track has been asked about gate_eval's inputs, Q's threshold and the guardrails.
