# E23: the GPU pileup's 28 s on neuron1k_v3 (mtpileup)

Experiment E23 ("mtpileup") of the GPU aligner program. Branch `gpu/fable/mtpileup`, from `gpu/fable-engine` at 78c46552 (the E18/E19
head). The question: the GPU pileup held the device 28 s on neuron1k_v3 (mouse, 92.9 M read pairs, 141 M chrM tuples) against 0.4–5.3 s
on the other samples (E19 gate 798027). The suspect was the mitochondrial path. The target: the pileup's device-lock hold on neuron1k_v3
at most 6 s, every output byte-identical, nothing else changed.

The fix is on by default. `SINGLET_GPU_PILEUP_E23=0` runs the kernels it replaces (the A/B switch of §3a). With the variable unset the
binary runs the new kernels; nothing outside the GPU pileup's reducers changed.

## 0. Answer in brief

- **It is not the mitochondrial path.** The chrM stages take 16 ms on neuron1k_v3 (141 M tuples) and 27 ms on pbmc1k_v3 (357 M
  tuples): linear in the tuples, and 0.06 % and 1.2 % of the GPU time.
- **It is the R9 collapse of the junction molecules: 25.75 s of the pileup's 26.8 s of GPU time** (profile job 798059, T = 8).
  - The kernel (`k_r9`) collapses the UMIs of each (cell, junction) group on one thread, testing every pair: O(n²) per group. Its time
    is the largest group's thread.
  - neuron1k_v3 has a (cell, junction) group of 14,097 UMIs (6 groups above 10,000). pbmc1k_v3's largest is 456.
  - Two more per-group serial kernels have the same shape at a smaller scale: the SNP R9 collapse (`k_r9` again: 0.35 s on neuron1k_v3,
    1.42 s on pbmc1k_v3, largest (cell, SNP) group 3,802 and 7,728 UMIs) and kUmiGeneCR's roots (`k_cr_roots`: 0.41 s and 0.50 s,
    largest (cell, gene) group 12,583 and 15,385 UMIs, n log n).
  - The groups are UMIs of one highly expressed gene in one cell: one gene per sample holds 8 (neuron1k_v3) and 7 (pbmc1k_v3) of the 8 largest (cell, gene) groups.
- **The scaling law:** the pileup's GPU time was set by the single largest (cell, feature) group, not by the number of tuples:
  `k_r9` ≈ (n_max² / 2) pair tests on one thread (48 ns per test for SNPs; 259 ns for neuron1k_v3's junction group), `k_cr_roots` ≈
  n_max · 3 · ulen binary searches (64 ns per search step).
- **The fix (`pileup.cu`, `m6.cu`): the same molecules from data-parallel kernels, whose work grows with the entries (radix sorts)
  and the pairs within small buckets, not with the square of a group.**
  - R9 (`r9_collapse`): the candidate pairs are the pairs of a bucket (group, one half of the base-5 UMI) — two UMIs one base apart
    agree on one half; each pair passing `k_r9`'s test is unioned with a lock-free union-find that always keeps the smaller index as the
    root (the root `k_r9` leaves, whatever the order); gene-assigned entries take the first entry of their gene component from a
    stable sort; the SNP class sums and the R1 vote per root as before.
  - kUmiGeneCR (`stage_cr`): `k_cr_roots`' sequential union, replayed per connected component over stage c's own edges sorted by
    (component, lower rank), one thread per component. The union steps are the same, in the same order within each component.
- **Result on neuron1k_v3 (A/B job 798076, one binary, T = 8):** GPU time **26.8 s → 0.28 s**; the three kernels 26,512 ms → 3.9 ms
  (`5.sj.r9` 25,746 → 2.0 ms, `6.snp.r9` 346 → 1.4 ms, `cr.roots` 420 → 0.5 ms); device lock held **31 s → 4.4 s** in that job, of which
  2.4 s is the A/B's own digest of every output (`SINGLET_E16_OFF=digest`, off in production); wall 147.9 → 121.1 s. The pileup digest is
  the same (`4aed2e01…`), and out_equiv gives 60 identical files.
- **pbmc1k_v3 paid for the same kernels:** 1.92 s of its 2.15 s GPU time (the SNP collapse 1.42 s, `k_cr_roots` 0.50 s). After: GPU
  time **2.15 s → 0.24 s** (`6.snp.r9` 1,422.6 → 1.8 ms, `cr.roots` 495.7 → 0.5 ms), lock held 6.6 → 4.7 s in the A/B (2.55 s of it the
  digest). Digest the same (`d2aa9309…`), 60 identical files.
- **Gate at T = 18 (job 798081, short-gpu, g052; all 10 samples, the binary before the fix (`base`, 759e2c86) and after (`gpu`,
  28356c0b) in the same job, against E19's kept CPU references):**
  - **identity: 21 of 21 verdicts IDENTICAL** (cpu:base and cpu:gpu on every sample, and the default path below). Each has E19's
    count of identical files (60 on neuron1k_v3, pbmc1k_v3, nuclei2k_v2, pbmc1k_v2, pbmc1k_cite_v3; 59, 59, 58, 57, 49 on the others);
    every other file is in E11's run-dependent set; the 10 STAR Solo files, `star_SJ.out.tab`, the mapping statistics, the read files
    (sorted) and `Gene/CellReads.stats` (sorted rows) are identical.
  - **neuron1k_v3: device lock held 28 s → 2 s, GPU 26.76 s → 0.27 s, wall 127.7 → 100.2 s (−27.5 s, −21.5 %), core-s 587.8 → 559.7.**
    The target (≤ 6 s) is met with 4 s to spare.
  - pbmc1k_v3: lock 4.1 → 2.1 s, GPU 2.15 → 0.24 s, wall 102.1 → 100.7 s.
  - Elsewhere the GPU time falls or stays (pbmc10m 288 → 41 ms, pbmc5p_v1 467 → 179 ms, hgmm1k_v3 413 → 203 ms, …), the lock hold falls
    by 0–0.3 s, and the wall time is unchanged within the run-to-run noise (§4).
  - **Default paths unchanged:** pbmc1k_v3 with `SINGLET_ENGINE_GPU_E2E=0` against E11's kept gpuD (gate 796377): IDENTICAL, 60/2/4,
    the same count as E19's gate.

## 1. Profile and scaling law

### 1a. Where the time goes

Profile job 798059 (debug-gpu, g052, T = 8): binary 759e2c86, the E19 head plus finer CUDA-event marks in the reducers (`5.sj.fill`,
`5.sj.sort`, `5.sj.runs`, `5.sj.r9`, the same for `6.snp`, `7.mt.fill`/`sort`/`rle`, `c.agg`, `cr.rank`, `cr.roots`) and
`SINGLET_GPU_PILEUP_PROF=1` (the group-size statistics of §1b, downloaded to the host: 2.3 s and 1.6 s, not counted below). The stage
line now goes to the run's log ("stages (ms): …", every partition summed; `fetch` is the host's fetch of the results).

| stage (ms) | neuron1k_v3 | pbmc1k_v3 |
|---|---|---|
| read pairs / tuples gene, sj, snp, mt | 92.9 M / 69.4 M, 11.0 M, 5.9 M, 141.2 M | 66.6 M / 52.2 M, 9.4 M, 9.1 M, 357.0 M |
| **5.sj.r9** (junction R9 collapse, `k_r9`) | **25,754.0** | 5.2 |
| **6.snp.r9** (SNP R9 collapse, `k_r9`) | **345.9** | **1,423.8** |
| **cr.roots** (kUmiGeneCR roots, `k_cr_roots`) | **409.1** | **495.7** |
| 7.mt (fill, sort, rle, rest) | 16.0 | 26.7 |
| everything else on the device | 260.9 | 207.9 |
| GPU total (without the profile's downloads) | 26,786 | 2,159 |
| host fetch | 703.5 | 832.2 |
| device lock held (with the profile's downloads) | 31 s | 5.7 s |

The three per-group serial kernels are 99.0 % of neuron1k_v3's GPU time and 89 % of pbmc1k_v3's.

### 1b. The groups

From `SINGLET_GPU_PILEUP_PROF=1` (group starts downloaded after each reducer):

| groups | neuron1k_v3 | pbmc1k_v3 |
|---|---|---|
| (cell, junction): groups / entries (UMIs) | 2,538,739 / 6,221,286 | 1,406,109 / 4,804,172 |
| largest (next four) | **14,097** (12,376, 11,779, 11,519, 10,613) | 456 (395, 383, 332, 324) |
| groups ≥ 1,000 / ≥ 10,000 | 53 / 6 | 0 / 0 |
| (cell, SNP): groups / entries | 2,400,281 / 3,381,432 | 3,343,611 / 4,860,324 |
| largest (next four) | 3,802 (2,573, 2,271, 2,145, 1,794) | **7,728** (6,256, 5,737, 4,384, 4,325) |
| (cell, gene) (`k_cr_roots`' groups): groups / entries | 8,238,191 / 24,889,351 | 5,277,223 / 16,372,731 |
| largest | 12,583 (gene 15224; the 8 largest are the same gene) | 15,385 (gene 17937; 7 of the 8 largest) |
| chrM genes: groups / entries | 143,042 / 1,542,619 | 82,558 / 1,175,918 |
| stage c's candidate buckets (segment, UMI half): largest | 24 / 37 | 26 / 53 |
| chrM: tuples / distinct (cell, position, base, strand) | 141,210,824 / 18,739,136 | 357,035,821 / 24,453,614 |

The largest groups are not chrM genes (the profile marks chrM genes; none of the 8 largest is one).

### 1c. The scaling law

All three kernels give one (cell, feature) group to one thread. With millions of groups, the kernel ends when the thread of the
largest group ends, so its time follows that group alone:

| kernel | work of a group of n entries | measured |
|---|---|---|
| `k_r9` (junctions, SNPs) | the pair loop over n(n−1)/2 pairs, and for each gene-assigned entry a scan for the first entry of its gene component (up to n) | SNP: 1,423.8 ms / (7,728² / 2) = 48 ns per pair (pbmc1k_v3); 345.9 ms / (3,802² / 2) = 48 ns (neuron1k_v3). Junction: 25,754 ms / (14,097² / 2) = 259 ns per pair (neuron1k_v3) |
| `k_cr_roots` | n · 3 · ulen neighbours, each a binary search over the group (log₂ n steps) | 495.7 ms for 15,385 · 36 · 13.9 steps (pbmc1k_v3), 409.1 ms for 12,583 · 36 · 13.6 (neuron1k_v3): 64 ns per step in both |

So the time grows with the square of the deepest (cell, junction) or (cell, SNP) group, which grows with the depth of the most expressed
gene in one cell. neuron1k_v3's deepest junction group is 31x pbmc1k_v3's (960x the pairs), hence 25.8 s against 5 ms. The chrM path is
sort + run-length encode over the tuples: 16 ms for 141 M, 27 ms for 357 M (the sort 6.1 and 15.3 ms: 2.5x the time for 2.5x the
tuples).

## 2. The fix

### 2a. R9 collapse: `r9_collapse` (`src/engine_gpu/gpu_pileup/pileup/pileup.cu`)

R9 (`order-free-rules.md`; `UmiFeatureCollapse` / `SnpConsensus::finalize` in `umi_dedup.h`) collapses the junction and SNP molecules
per (cell, feature): entries are the exact (cell, feature, UMI) runs, sorted by (cell, feature, UMI); two free entries are joined iff
their UMIs differ at exactly one base and max(w) + 1 ≥ 2 min(w) (w: the UMI's reads, or its clamped SNP class counts); a molecule is a
connected component; an entry assigned to a gene (R9 per gene) is one molecule per gene component (`rid`). `k_r9` computed, per group on
one thread: `root[i]` (the smallest index of i's component, since its union hangs the larger root under the smaller), then `isroot`
(junctions) or the summed class counts `msum` of each root, the R1 vote `val` and the vote statistics (SNPs).

`r9_collapse` computes the same arrays with kernels over the entries:

1. **Group ids:** `k_gstart` + an inclusive scan.
2. **Candidate pairs (directional rule only):** a UMI of `ulen` base-5 digits is split into a low half (`ulen / 2` digits, < 5⁶) and a high
   half (the rest, < 5⁷ < 2²⁰). Two UMIs one base apart agree on all the digits of one half. For each half, the free entries are radix
   sorted by (group, half value) (`k_r9_hkey`; gene-assigned entries get a sentinel key past every group); `k_r9_pairs` (one thread per
   entry) tests the entry against the later entries of its bucket with `k_r9`'s own test (count rule, `umi5_one_mismatch`). A pair
   one base apart is in exactly one bucket, so every pair `k_r9` would test positively is tested once.
3. **Union:** `uf_union_min` hooks the larger root under the smaller with `atomicCAS` and retries if the root changed meanwhile. Every
   component's root is its smallest entry index, which is the root `k_r9` leaves, whatever the order in which the pairs arrive.
   `k_r9_flatten` writes each entry's root.
4. **Gene-assigned entries:** a stable radix sort by (group, `rid`) (free entries: sentinel) puts each gene component's entries in
   index order; `k_r9_rroot` gives every entry of a run the run's first entry, which is `k_r9`'s "first entry of its gene component".
5. **Outputs:** `isroot[i] = root[i] == i` (junctions); or (SNPs) `msum` zeroed, each entry's clamped class counts added to its root with
   `atomicAdd` (integer sums: order-free), and `k_r9_vote`, `k_r9`'s vote code per root, with the same three statistics counters.

Scratch: five arrays of the caller that are dead at that point (the sort input, the run flags and keys, the segment ids; each at least
m entries), so the reducer's peak device memory does not grow (24 GB on neuron1k_v3 before and after). The work is three radix sorts
of m keys (one per half, one for the gene components) plus the pairs within the buckets. A bucket holds the UMIs of one group that
share half their digits (5⁶ values per half for a 12-base UMI); E23 did not profile their sizes, but the stage takes 1.4–2.0 ms on
groups of up to 14,097 UMIs (§3a). Stage c's buckets of the gene store, built the same way, have at most 53 entries (§1b).

### 2b. kUmiGeneCR roots: the replay per component (`src/engine_gpu/gpu_pileup/m6/m6.cu`)

kUmiGeneCR (intron rule 128, part of the default rules 389) needs the UMI of each molecule as the CPU's `collapse_molecules` leaves it:
the root of a sequential union over the entries in rank order (reads descending, UMI ascending), where each entry p unions every
higher-ranked neighbour q (one base apart, count rule) with `par[find(q)] = find(p)`. That root is not always the best-supported UMI,
so `k_cr_roots` replayed the union per (cell, gene) group on one thread, finding each entry's 3 · ulen neighbours by binary search.

The replay now runs per connected component, over the edges stage c already built for its own connected components:
- stage c's edges are exactly `k_cr_roots`' unions: pairs one base apart (2-bit UMI) in the same (cell, gene) segment that pass the
  (symmetric) count rule, each pair once;
- `k_cr_ekey` keys each edge by (component label, lower rank of its two entries) and keeps the higher rank; a radix sort orders them;
- `k_cr_replay` (one thread per component) applies `par[find(hi)] = find(lo)` in that order;
- `k_cr_rumi` writes each entry's root UMI.

Why the roots are the same: within one step p, every union hangs the other root under `find(p)`, and `find(p)` stays the root of p's
set through the step, so the order of the unions within a step does not matter; the steps run in rank order as on the CPU; and the
components are disjoint, so replaying each one alone gives the same parent of every root. The work of a component's thread is its
edges. Without the directional rule stage c has no edges, and neither version unions anything.

### 2c. Switch

`SINGLET_GPU_PILEUP_E23=0` runs `k_r9` and `k_cr_roots` as before (read in `run_pipeline`, per run). Unset or any other value: the new
kernels. The switch exists for the A/B of §3a and costs nothing when unused.

## 3. Identity per output

### 3a. A/B in one binary (job 798076, debug-gpu, g051, T = 8, binary 28356c0b)

Per sample, two runs of the same binary: `SINGLET_GPU_PILEUP_E23=0` (before) and `=1` (after), with `SINGLET_E16_OFF=digest`, so the
pileup logs the SHA-256 digest of everything it returns to the pipeline; then `out_equiv.py` over the two output directories (read files
set aside, as the gate does).

| | neuron1k_v3 off | neuron1k_v3 on | pbmc1k_v3 off | pbmc1k_v3 on |
|---|---|---|---|---|
| pileup digest | `4aed2e01…2c25a58` | **same** | `d2aa9309…2952052` | **same** |
| out_equiv (off vs on) | | 60 identical, 3 timing/path, 3 differ | | 60 identical, 3 timing/path, 3 differ |
| the 3 that differ | | `read_stats.tsv`, `star_Log.std.out` (timings), `pileup_stats.json` `bytes_public` (the sizes of those timing files) | | the same 3 |
| GPU time | 2.7e4 ms | **2.7e2 ms** | 2.2e3 ms | **2.4e2 ms** |
| `5.sj.r9` / `6.snp.r9` / `cr.roots` (ms) | 25,745.5 / 346.4 / 420.2 | **2.0 / 1.4 / 0.5** | 5.2 / 1,422.6 / 495.7 | **1.6 / 1.8 / 0.5** |
| sum of the stage marks, without `fetch` (ms) | 26,784 | **275** | 2,154 | **236** |
| `run_pipeline` (E16 step, s) | 27.75 | 1.30 | 3.23 | 1.32 |
| digest (A/B only, s) | 2.34 | 2.38 | 2.55 | 2.55 |
| device lock held | 31 s | **4.4 s** | 6.6 s | **4.7 s** |
| wall (T = 8) | 147.9 s | 121.1 s | 119.8 s | 120.5 s |

The digest covers every array the pileup returns (counts, molecules, the chrM tuples and events, the merges for V(D)J); it is the same
before and after on both samples. The 60 identical files include every count matrix and pileup product, among them the chrM outputs
`mt_heteroplasmy.1pz`, `private/mt_allele_counts.1pz`, `private/mt_allele_counts_rev.1pz`, `mt/mt_events.1pz`, `mt_variants.tsv` and
`mt/mt_summary.tsv`, and the junction and SNP products the fix touches (`sj_counts`, `splice_psi`, `sj_features.tsv.gz`,
`splice_events.tsv`, `private/snp_ad`, `private/snp_dp`, the donor and ancestry calls). The device lock in this job includes the A/B's
digest (2.3–2.6 s with the arena held); without it the hold is about 2 s (§4).

### 3b. Gate at T = 18 (job 798081)

`bench/fable/mtpileup/jobs/gate.sbatch` (E19's gate, §8): short-gpu, g052 (H100 NVL), T = 18, 41 min. Per sample, mode `base` (binary
`singlet-base`, 759e2c86: the E19 head with E23's stage marks only) and mode `gpu` (`singlet-gpu`, 28356c0b: the fix), both with the
defaults (`--engine gpu`, `SINGLET_GPU_PILEUP=on`, level 5, E13 ingest, E16's tail), each compared with E19's kept CPU reference
(`$W/jobs/fable/release/ref/T18`, job 797474, commit 431f4ef9; not rewritten here). All 21 runs: exit 0, the stream path, GPU pileup on, 0
fallback lines; `base` and `gpu` at level 5 of 5 with the ingest.

| sample | cpu:base | cpu:gpu | out_equiv (identical / timing-path / differ) | E19's count (798027) |
|---|---|---|---|---|
| neuron1k_v3 | IDENTICAL | IDENTICAL | 60 / 1 / 5 | 60 / 2 / 4 |
| pbmc1k_v3 | IDENTICAL | IDENTICAL | 60 / 1 / 5 | 60 / 2 / 4 |
| nuclei2k_v2 | IDENTICAL¹ | IDENTICAL¹ | 60 / 1 / 5 | 60 / 2 / 4¹ |
| hgmm1k_v3 | IDENTICAL | IDENTICAL | 49 / 1 / 5 | 49 / 2 / 4 |
| pbmc2m | IDENTICAL | IDENTICAL | 59 / 1 / 5 | 59 / 2 / 4 |
| pbmc10m | IDENTICAL | IDENTICAL | 59 / 1 / 5 | 59 / 2 / 4 |
| pbmc1k_v2 | IDENTICAL | IDENTICAL | 60 / 1 / 5 | 60 / 2 / 4 |
| pbmc1k_cite_v3 | IDENTICAL | IDENTICAL | 60 / 1 / 5 | 60 / 2 / 4 |
| pbmc1k_cite_v3_adt | IDENTICAL | IDENTICAL | 57 / 1 / 5 | 57 / 2 / 4 |
| pbmc5p_v1 | IDENTICAL | IDENTICAL | 58 / 1 / 5 | 58 / 2 / 4 |

¹ `nonhost_reads.fq`: the known non-host writer bug (E15 §5.7): other records, the same count, in every engine; as in E19.

Per output, on every sample and both modes:
- **identical (out_equiv):** every count matrix and pileup product (`exon_counts`, `intron_counts`, `ambiguous_counts`, `sj_counts`,
  `splice_psi`, `polya_counts`, `ambient_corrected_counts`, `vdj_gene_usage`; `private/snp_ad`, `snp_dp`), **the chrM outputs
  `mt_heteroplasmy.1pz`, `private/mt_allele_counts.1pz`, `private/mt_allele_counts_rev.1pz`, `mt/mt_events.1pz`, `mt_variants.tsv`,
  `mt/mt_summary.tsv`** (and `mt_reference.fa`, `private/mt_sites.tsv`, the `private/mt` consensus and variants), and the tables
  (`gene_expression.tsv`, `cell_calls.tsv`, `sj_features.tsv.gz`, `splice_events.tsv`, the donor, doublet, cell-cycle, ambient,
  saturation, sex and ancestry outputs, `cell_qc_metrics.tsv`, `read_qc_summary.json`, …), where the sample has them;
- `star_Solo.out`: 10 of 10 files identical; `star_SJ.out.tab` identical (117,254 junctions on neuron1k_v3); `star_Log.final.out`'s
  mapping statistics identical; `nonhost_reads.fq`, `star_Unmapped.out.mate1/2` the same records (sorted; neuron1k_v3: 490,605 and
  3,667,707 reads); `Gene/CellReads.stats` the same rows (sorted; 600,534 on neuron1k_v3);
- the differing files are all in E11's run-dependent set: `read_stats.tsv` and `star_Log.std.out` (stage rows and timings),
  `summary.json` (`memory_tier`, CPU vs GPU engine), `pileup_stats.json` (`bytes_public`), and `provenance.json`, whose whitelist path
  names the binary's directory (`mtpileup/bin` here, `release/bin` for the references). That last one is why the counts read 60/1/5
  instead of E19's 60/2/4: E19's binaries sat in the references' directory.

### 3c. Default paths

- **GPU path with `SINGLET_ENGINE_GPU_E2E=0`** (mode `gpuO0`, pbmc1k_v3, same job) against E11's kept gpuD outputs (gate 796377):
  IDENTICAL, out_equiv 60 / 2 / 4 (E19's gate: the same), STAR Solo 10 of 10, `star_SJ.out.tab` (126,996 junctions) and the mapping
  statistics identical, CellReads.stats the same 323,542 rows. This run uses the new kernels too (its stage line: `5.sj.r9` 1.6 ms,
  `6.snp.r9` 1.9 ms, `cr.roots` 0.5 ms).
- **CPU engine:** not run; E23 changes only kernels of the GPU pileup (`src/engine_gpu/gpu_pileup/`), and the CPU engine runs the
  CPU pileup. The references it was compared with are E19's CPU runs.
- **`SINGLET_GPU_PILEUP_E23=0`** gives the outputs of the kernels before (the A/B's "off" runs: the same digest as "on").

## 4. Timings

Gate 798081 (short-gpu, g052, H100 NVL, T = 18). Wall time and core-s are for the whole `singlet --pipeline` run (cgroup `cpu.stat`);
the lock is the pileup's own "[gpu_pileup] device lock: …, held" line; GPU is the sum of the pileup's CUDA-event stage marks without the
host fetch (the log's "GPU … ms" rounds to 2 digits). `base` ran first, then `gpu`, per sample.

| sample | lock held base → gpu (s) | GPU base → gpu (ms) | of which the 3 kernels (ms) | host fetch (ms) | wall base → gpu (s) | core-s base → gpu |
|---|---|---|---|---|---|---|
| **neuron1k_v3** | **28 → 2** | **26,757 → 274** | 26,485 → 3.9 | 711 / 728 | **127.7 → 100.2** | 587.8 → 559.7 |
| **pbmc1k_v3** | **4.1 → 2.1** | **2,148 → 237** | 1,917 → 4.0 | 841 / 856 | 102.1 → 100.7 | 551.4 → 549.3 |
| nuclei2k_v2 | 3.6 → 3.5 | 367 → 318 | 51.5 → 2.2 | 1,432 / 1,450 | 199.2 → 202.5 | 955.2 → 956.9 |
| hgmm1k_v3 | 1.2 → 1.0 | 413 → 203 | 215.7 → 3.3 | 140 / 138 | 99.3 → 80.9² | 482.6 → 416.0² |
| pbmc2m | 0.57 → 0.49 | 52 → 11 | 42.0 → 0.7 | 120 / 114 | 33.1 → 33.4 | 121.0 → 121.3 |
| pbmc10m | 1.1 → 0.85 | 288 → 41 | 248.0 → 1.3 | 348 / 347 | 45.4 → 45.4 | 192.1 → 192.2 |
| pbmc1k_v2 | 1.5 → 1.5 | 262 → 243 | 29.4 → 2.6 | 391 / 376 | 105.0 → 104.5 | 533.9 → 532.2 |
| pbmc1k_cite_v3 | 1.4 → 1.3 | 256 → 161 | 88.9 → 2.7 | 385 / 389 | 78.3 → 78.2 | 404.7 → 403.7 |
| pbmc1k_cite_v3_adt | 0.38 → 0.36 | 3.0 → 3.3 | 0.0 → 0.2 | 1 / 1 | 65.1 → 65.3 | 272.2 → 272.4 |
| pbmc5p_v1 | 4.7 → 4.4 | 467 → 179 | 293.3 → 6.3 | 3,080 / 3,060 | 122.3 → 121.6 | 505.7 → 504.1 |
| pbmc1k_v3 `gpuO0` | 7 (gpu only) | 251 | 4.0 | 1,069 | 194.1 (E19: 195.1) | 955.6 |

² hgmm1k_v3's base run was the job's first on the barnyard genome: its "probe/ref load" took 24 s against 8 s in the gpu run
(page cache). Its pileup stages differ by 1 s at most; the 18 s are not E23's. nuclei2k_v2's +3.3 s are in the non-host stage
(76 → 79 s) and the reference load (6 → 7 s); its pileup stages are the same.

**Where neuron1k_v3's 27.5 s went** (`s8_stages_e13.py`, wall s / core-s): "GPU-pileup hand-off" 31 / 65.7 → 5 / 40.5, plus
"pileup finish" 1 / 7.5; "STAR stream into pileup" 22 → 20; the rest within 1 s. GPU busy 42 → 14 s. The 28 core-s saved are the
pileup thread's wait on the device. In E19's gate 798027 the same sample's gpu run took 131.5 s with the 28 s hold.

**The rest of the hold.** After the fix the lock hold on neuron1k_v3 is GPU 0.27 s + host fetch 0.73 s + about 1 s of context, arena
and uploads. The samples with the longest holds now are pbmc5p_v1 (4.4 s, of which the fetch is 3.06 s) and nuclei2k_v2 (3.5 s, fetch
1.45 s). The device-side work is at most 0.32 s on every sample.

The A/B (§3a, T = 8, one binary) gives the same picture on the debug node: neuron1k_v3 wall 147.9 → 121.1 s, pbmc1k_v3 119.8 → 120.5
s (noise).

## 5. Surprises

- **The suspect was innocent.** The chrM path is the cheapest reducer per tuple: one radix sort and one run-length encode over 141–357 M
  tuples, 16–27 ms. What made neuron1k_v3 slow is a few deep (cell, junction) groups of one highly expressed gene (6 groups above
  10,000 UMIs). The junction reducer's `k_r9` was quadratic in them and ran them on one thread each.
- **The per-pair cost differs by 5x between junctions and SNPs** (259 vs 48 ns per pair of the largest group). `k_r9` has two quadratic
  loops, the pair test over free entries and, for each gene-assigned entry, the scan for the first entry of its gene component; the
  profile does not split the kernel's time between them. Both are gone with the fix, so the split was not pursued.
- **pbmc1k_v3 was also paying 1.9 s** (the SNP collapse 1.42 s and `k_cr_roots` 0.50 s) of its 2.2 s GPU time, for the same reason at a
  smaller scale. Its lock hold halves (4.1 → 2.1 s), but its wall time moves by only 1.3 s.
- **The wall-time saving on neuron1k_v3 (27.5 s) is the whole hold saving (26 s) plus 1–2 s:** the pileup's hand-off is on the run's
  critical path (in the stage table nothing overlaps it), so every second the device work loses comes off the wall time.
- **`k_cr_roots` was the dominant kernel on the small samples:** pbmc10m 174 ms of 288, pbmc2m 38 of 52, hgmm1k_v3 197 of 413, so the
  replay helps them most in relative terms (pbmc10m's GPU time 7x lower), although their wall times do not move.

## 6. What remains

- **The host side of the pileup is now most of the lock hold.** After the fix the GPU time is at most 0.32 s on every sample; the rest of
  the hold is the host: the fetch of the partition's results (0.7–0.9 s on neuron1k_v3 and pbmc1k_v3, 3.1 s on pbmc5p_v1),
  the uploads and the arena (E18: ~2.3 s of context, arena and pageable uploads on pbmc1k_v3). pbmc5p_v1's fetch (3.06 s, against
  0.1–1.5 s elsewhere) is the largest single item; E23 did not look into it. The lock placement itself is E20's.
- **Other per-group serial kernels.** None is left in the reducers' hot path at these depths: stage c's candidate buckets are at most 53
  entries, the chrM path has no groups. A sample with a deeper gene in one cell (a targeted panel, a very deep cell) would now scale
  with its bucket sizes in `k_r9_pairs`, not with the square of the group.
- E19's digest code updates `cell_reads` inside a comment (`pileup_gpu.cu`); harmless, not touched here.

## 7. Files touched, and why

| file | change | why |
|---|---|---|
| `src/engine_gpu/gpu_pileup/pileup/pileup.cu` | `r9_collapse` and its kernels (`k_r9_hkey`, `uf_union_min`, `k_r9_pairs`, `k_r9_flatten`, `k_r9_rkey`, `k_r9_rroot`, `k_r9_isroot`, `k_r9_msum`, `k_r9_vote`), called from `red_junction` and `red_snp` in place of `k_r9`; the switch; the finer stage marks; the "stages (ms)" log line; `SINGLET_GPU_PILEUP_PROF=1`'s group statistics (`prof_groups`, `prof_gene_groups`, chrM counts); `bits_of` moved up | the fix, and the profile that found it |
| `src/engine_gpu/gpu_pileup/m6/m6.cu` | `k_cr_ekey`, `k_cr_replay`, `k_cr_rumi`, `CrEdges`; `stage_c` keeps its edges for `stage_cr`; `stage_cr` replays them (or runs `k_cr_roots` with the switch at 0); marks `c.agg`, `cr.rank`, `cr.roots` | the fix |
| `bench/fable/mtpileup/jobs/{build,prof,ab,gate}.sbatch` | build (release build of a ref), profile, A/B and gate jobs; the gate is E19's with a `base` mode (the binary before the fix), the lock hold and GPU time per run, and `--exclude=g050` | deliverable |
| `docs/gpu-engine/fable/E23-mtpileup.md` | this report | deliverable |

Not changed: `pileup_gpu.cu` (the lock and the hand-off), `SeqCacheGpuPrefill.cpp` and `pileup_engine.h` (E20's lock placement and
prefill), `src/pipeline` (E21), anything on the CPU path.

## 8. Jobs, commits, how to run

**Branch `gpu/fable/mtpileup`:**
- 759e2c86: stage marks, the "stages (ms)" line, `SINGLET_GPU_PILEUP_PROF=1`; build and profile jobs.
- 28356c0b: the fix (`r9_collapse`, the CR replay), the switch, the A/B job.
- f2188e1d: the gate job.
- This document is the next commit.

**Jobs** (Clipper; results under `$W/jobs/fable/mtpileup/results/`, logs `$W/logs/e23-*-<job>.out`):

| job | partition, node | what | result |
|---|---|---|---|
| 798056 | short-gpu (build) | build of 759e2c86 → `bin/singlet-base` | ok |
| 798059 | debug-gpu, g052 | profile, neuron1k_v3 and pbmc1k_v3, T = 8 | `prof-798059/` (§1) |
| 798075 | short-gpu (build) | build of 28356c0b → `bin/singlet-gpu` | ok |
| 798076 | debug-gpu, g051 | A/B, neuron1k_v3 and pbmc1k_v3, T = 8 | `ab-798076/` (§3a): digests the same, 60 identical |
| 798081 | short-gpu, g052 | gate, 10 samples × (`base`, `gpu`) + pbmc1k_v3 `gpuO0`, T = 18, 41 min | `gate-798081/` (§3b–§4): 21 of 21 IDENTICAL |

Baselines used: E19's kept CPU references (`$W/jobs/fable/release/ref/T18`, gate 797474), E19's gate 798027 (per-sample hold before E23),
E11's kept gpuD (`$W/jobs/fable/solowire/results/gate-796377/gpuD`).

**Run:** build with `sbatch --export=ALL,VARIANT=gpu bench/fable/mtpileup/jobs/build.sbatch` (`VARIANT=base REV=<commit>` for the
baseline binary), then `sbatch --export=ALL,SAMPLES="neuron1k_v3 pbmc1k_v3",MODES="base gpu",PAIRS="cpu:base cpu:gpu"
bench/fable/mtpileup/jobs/gate.sbatch` (job 798081 ran `SAMPLES` = all 10, `MODES_pbmc1k_v3="base gpu gpuO0"`,
`PAIRS_pbmc1k_v3="cpu:base cpu:gpu e11gpuD:gpuO0"`). The A/B: `sbatch --export=ALL,BIN=singlet-gpu bench/fable/mtpileup/jobs/ab.sbatch`.

**Switches:** `SINGLET_GPU_PILEUP_E23=0` (the kernels before E23); `SINGLET_GPU_PILEUP_PROF=1` (group statistics in the log; downloads,
profiling only).
