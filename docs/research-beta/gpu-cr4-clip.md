# GPU prefill with `--clipAdapterType CellRanger4` (Beta aligner profile K5), 2026-10-30

Branch `gpu/cr4-clip`, stacked on PR #95. Before: `fillClip` declined CellRanger4, so STAR mapped every 10x 3' v2/v3 read on the CPU
(Beta turns K5 on for those protocols, `singlet.cpp` aligner profile, #96).

## What it does
- CR4's 5' clip is STAR's `ClipMate::clipChunk` (opal alignment of the TSO, 64 reads at a time, clip length left next to the `+` line) and its 3' clip is
  `ClipCR4::polyTail3p`, both applied by `ClipMate::clip`. `readSequences` (prefill) now runs the same code per read: the opal batch, then the two
  `ClipMate::clip` calls on a copy of the numeric read. The clip lengths go to the GPU (`Input::pre5/pre3`; `k_clip` uses them in place of the Hamming
  parameters), so the cache and the hand-off records are keyed by the clipped sequences exactly as STAR's readLoad sees them.
- The same pass hands `readLoad` the CR4 5' clip (`g_cr4Info`, indexed by read) and `mapChunk` skips `clipChunk`, so opal runs once per read, not twice.
- The CR4 probe in `singlet.cpp` forks two STAR children; they inherited `SINGLET_ENGINE_GPU` and ran the GPU prefill setup. They now unset it
  (the probe maps 200 k reads on the CPU). This was ~460 core-s and ~100 s wall per sample.

## Gate (CPU engine with K5 vs `--engine gpu` with K5; K5 is on by default for these protocols, `CR4 probe ... -> profile ON`)
pbmc1k_v3 (3' v3) and GSM4006843 (3' v2), Clipper jobs 793215 and 793238 (build 793237, head of `gpu/cr4-clip`).

| Run | Result |
|---|---|
| default intron rules (389), GPU pileup falls back to the CPU pileup: CPU engine vs `--engine gpu`, pbmc1k_v3 | 60 files identical, 6 differ (logs, read order, probe counts); every count output identical |
| `SINGLET_INTRON_RULES=0`, GPU pileup on, pbmc1k_v3 and GSM4006843 | every count output identical **except `polya_counts.1pz` and `polya_sites.tsv.gz`** (GPU: fewer sites, 54.6 kB vs 87.2 kB on pbmc1k_v3) |

**Open: poly(A) sites with the GPU pileup and clipped reads.** The GPU path rebuilds a clipped read's poly(A) tail from the hand-off entry
(`gpu_host_pass`, `clip` branch; clip bits from `ReadAlign::outputHandoffEntry`). That branch only ran on 10x 5' before (poly(A) is not counted for
that library), so CR4 on a forward 3' library exercises it for the first time. Not found by reading; the next step is a per-read dump of the
CPU and GPU site keys for a few thousand clipped reads. It does not affect the default configuration (the GPU pileup is off while the #80 intron
rules are on), but it must be fixed before the pileup is turned on again.

## Cost (host core-s per M reads, whole run, one run per cell; pbmc1k_v3, 66.6 M reads)
| Run | core-s | wall s | core-s / M |
|---|---|---|---|
| CPU engine, K5 | 2563 | 462 | 38.5 |
| GPU engine + GPU pileup, K5 (rules 0) | 958 | 386 | 14.4 |
| GPU engine + CPU pileup, K5 (default rules) | 1381 | 552 | 20.7 |
| GPU engine + GPU pileup, no K5 (earlier run, #95 gate) | 812 | 351 | 12.2 |

K5 costs the GPU path ~2 core-s per M reads (the opal clip, now in the prefill's reading threads: "reading reads" 14 s instead of 8 s).
GSM4006843 before the probe fix: 1693 core-s (25.4 / M); not rerun after it.

## Merged-head check, default rules (jobs 793385 build, 793386; head 99e0ec50)
CPU engine vs `--engine gpu`, K5 on, default intron rules (GPU pileup falls back to the CPU pileup): pbmc1k_v3 60 files identical, GSM4006843 62
identical, 6 differ on each (`nonhost_reads.fq` and `star_Unmapped.out.mate1/2` read order, `star_Log.std.out`, `read_stats.tsv`, probe counts). Every
count output is byte-identical. Host core-s: pbmc1k_v3 2572 (CPU) vs 1279 (GPU); GSM4006843 2975 vs 1679 (per M reads: 38.6 vs 19.2 on pbmc1k_v3, 44.6 vs 25.2 on GSM4006843).
