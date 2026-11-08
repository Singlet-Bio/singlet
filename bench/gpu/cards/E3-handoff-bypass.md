# E3: the host work left in STAR's window

**Hypothesis.** This is the advisor's option 3: build the records from the device's hand-off records and skip STAR's per-read host pass, for at least 20% fewer core-seconds per million read pairs with identical outputs. On the current frontier most of that already exists. At levels 4 and 5 (E10, E11), a read with a device record is never given to STAR's loop, and Solo counts on the device. STAR's loop maps only the reads without a record (the host lane, about 0.7 s of wall time at 27 threads). What remains open is whether the host work inside STAR's window can still yield a fifth of a run's CPU.

**Baseline.** pbmc1k_v3 at one slot and 27 threads costs 611 core-s per run, or 9.2 core-s per M pairs (item 15, run c1). The harness's "stream" stage, from "pileup streaming from STAR starts" to "STAR completed", holds about 165 core-s, 2.5 core-s per M. A 20% cut is 1.8 core-s per M, so the card holds only if about three quarters of the stream window turns out to be removable host work.

**Change.** To be named from the measurement. The candidates are the E16 formatting of the unmapped records beside STAR's loop, STAR's chunk input and output threads, and the pipeline's consumer of STAR's stream.

**Measure.** Tier 1 has to place the core-seconds of the stream window by process and thread. nsys CPU sampling cannot do it on Clipper: the compute nodes run with `perf_event_paranoid` 4, so `perf_event_open` fails and nsys reports process-tree sampling unavailable (job 803927 ran with `NSYS_SAMPLE=process-tree` and recorded no samples). The tier therefore needs a per-thread sampler of `/proc/<pid>/task/*/stat` (thread name, user and system time) for the run's process tree, at 0.5 s, aligned with the run's NVTX ranges. Tier 2 is a sequential A/B on the core-s metric (`SEQMETRIC=core`), followed by the 9-sample identity gate.

**Falsifier.** Any one of these ends the card: the removable work in the stream window comes to under 1.8 core-s per M pairs; the A/B shows under 20% fewer core-seconds; any output differs outside the gate's accepted list.

**Result.** Pending.
