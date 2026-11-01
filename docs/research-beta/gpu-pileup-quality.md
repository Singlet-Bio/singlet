# GPU pileup with per-base quality (S8, 2026-10-31)

Before: the GPU pileup declined to run (`not run: base qualities neither constant nor tied to N`, then the CPU pileup) unless the reads' base qualities were constant or low exactly at N
bases, because it filtered SNP and chrM bases by a function of the *sequence* and shared one list per distinct sequence (signature) between all its reads. The atlas keeps ILLUMINA4-binned
qualities (SNP and mtDNA callers use them), so any sample with a low-quality base that is not an N would silently lose the GPU pileup.

## What it does now
- Each read carries a 128-bit mask: bit i = base i of the read's record (original orientation, clipped ends removed) has quality < `--min-baseq` (default 10; with ILLUMINA4 only bin 0).
  - STAR's mapping threads write it for hand-off reads (`<hand-off>.reads.q`, next to the per-read entries; `HandoffReadEntry` flag bit 6 = record over 128 bases, no mask); `SINGLET_HANDOFF_QMASK=0` turns the file off.
  - The collector builds it from the BAM record for stream reads.
- The signature lists keep the base's index in the record (SNP entries bits 25-31, chrM entries bits 17-23; `k_lists`), and the expansion to reads (`k_cnt_q`, `k_fill_hilo`, `k_fill_mt`) counts and
  writes only the entries whose base passed, as the CPU's `qual[query_pos + offset] < min_baseq` test does. Partitions gather their masks with the reads (`k_gatherq`).
- Used only when the old rule cannot reproduce the filter (`snpSkipN` would be -1), so constant or N-tied data take the unchanged path. Needs the SNP index < 2^23 and records of at most 128 bases; otherwise
  the GPU pileup declines as before and says why. `SINGLET_GPU_PILEUP_QUAL=force` ships the masks for any data (identity test).
- Cost with masks forced: +0.6 s wall, +2 core-s on pbmc1k_v3 (16 bytes per read on the device).

## Gate (binaries from the branch merged with prod/campaign d9834b8a; Clipper H100 NVL, `--engine gpu` + `SINGLET_GPU_PILEUP=on`, default rules)
All runs: `STREAM PATH OK`, `GPU PILEUP OK`; 59-74 files identical, the allowed set differs (read order, logs, provenance counters, `summary.json` `memory_tier`); every count output identical.
| Sample | Check | Result |
|---|---|---|
| pbmc1k_v3, neuron1k_v3 (job 793781) | masks forced (`PER-BASE QUALITY PATH OK`) vs CPU engine, and vs the unforced GPU run | 60 identical files each, counts identical; 302 vs 301 s wall, 978 vs 977 core-s (pbmc1k_v3) |
| SRR19421372 (5' v2), SRR19573735 (3' v2), SRR30477995 (3' v3): real full-quality runs, ILLUMINA4 (job 793810) | CPU engine vs new GPU | counts identical. All three have their low-quality bases tied to N, so they took the old path (no mask message) |
| synthq (job 793828): 8 M reads of pbmc1k_v3 with Q2 on two non-N bases of one read in three | CPU engine vs new GPU, masks chosen automatically (`low-quality masks for 7153808 reads`) | 59 identical files, counts identical |

Not covered: records over 128 bases (the GPU aligner takes up to 255 nt, STAR's limit is 650; those runs keep the CPU pileup and log it), a real sample whose low-quality bases are not N (none of the six real
runs is one), ctest.
