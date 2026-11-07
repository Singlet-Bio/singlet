# SL: more slots per GPU

**Hypothesis.** A third slot can raise throughput per GPU for deep samples too, once two things change. Admission must follow each run's estimated device need rather than a static share. And the genome index must be held once per device, which is the shared index (IDEAS item 4, identical at 2 and 3 slots).

**Baseline.** Steady rates at 1, 2 and 3 slots are 0.49, 1.46 and 1.87 M pairs/s for pbmc1k_v3, and 0.55, 0.96 and 0.67 for nuclei2k_v2. At 3 slots the E17 share is (device - 2 GB)/3, about 30 GB. nuclei2k_v2's E8 records need about 36.1 GB, so E17 stopped them in all six runs. Level 5 was off, and STAR mapped 120 chunks, holding the device for 247 s at 17.4 core-s per M pairs. pbmc1k_v3 at 3 slots waits 35 s per run for a slot and holds the device for 55 s.

**Change.** Step 1 uses existing code: the shared index (`SINGLET_GPU_SHARED_INDEX=1`) at 3 slots, which takes the index copy out of each slot's device use. Step 2 is new code. E17 would admit a run by its own estimate against the free device memory, in place of a fixed share per slot; the admission log already records "est GB" per request. A run whose need fits then waits for memory instead of dropping out of level 5.

**Measure.** Run the points k6t4s3r2 (pbmc1k_v3) and k3t9s3r2 (nuclei2k_v2) with and without the shared index on one pinned H100, with identity to the CPU reference. Then run a sequential A/B of the winner against 2 slots on throughput.

**Falsifier.** The advisor's falsifier for option 5 is that device hold fails to fall 10% at 3 slots with the shared index. Two more outcomes would also end the card: nuclei2k_v2 still drops out of level 5 at 3 slots after step 2, or the 3-slot rate is no higher than the 2-slot rate on either sample.

**Result.** Pending. Step 1 is pilot item 26-slots3-shidx.
