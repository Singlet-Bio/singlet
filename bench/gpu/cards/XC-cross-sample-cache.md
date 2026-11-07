# XC: the cross-sample cache

**Hypothesis.** Some of a run's device and host work is identical for every sample on a genome, or repeats across samples. Keeping it resident should cut the device hold by at least 15% on pbmc1k_v3 without raising core-seconds per M pairs by more than 2%, which is the advisor's go rule for option 6.

**Baseline.** A pbmc1k_v3 run at one slot holds the device for 25 s. The non-host screen uploads its 26.5 GB index in every run: 2.15 s of copies, 4.3 s in the "nonhost: context + index" range, which is 17% of the hold. The pileup reference load takes 105 core-s (about 190 on nuclei2k_v2). The advisor's exact read cache measured cost-weighted hit rates of 23 to 37% on 3' blood, 12% on brain, 3% on 5' and 0.3% on nuclei, so its gain depends on which samples run together.

**Change.** Two halves, cheapest first:

- **(a) Resident per-genome state.** A holder process keeps the non-host index on the device, as one does for the shared genome index, and the pileup references stay in shared memory across runs.
- **(b) The exact read cache.** Alignments of distinct read sequences from earlier samples are looked up before the lanes run.

**Measure.** For (a), compare the "nonhost: context + index" range and the refs stage before and after, then run a sequential A/B on held device time with a core-s guard. For (b), first re-measure hit rates on 20 random GEO samples (the advisor's condition), then replay with the cache.

**Falsifier.** Hold falls under 15% on pbmc1k_v3, core-s per M pairs rises more than 2%, or any output changes.

**Result.** Pending.
