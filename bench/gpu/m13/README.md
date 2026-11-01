# W11 M13: the non-host (viral / bacterial / fungal) screen on the GPU ("host step 2")

Result file: `../results/m13.json` (built by `make_m13_json.py` from `runs/<sample>/`). CPU truth: `prod/campaign` **c4a1be38**.

**Result: identical to the CPU engine on GSM4006843, GSM6470208 and GSM5411066 (mouse).** For every read of the three samples
(3.04 M, 12.27 M and 4.87 M unmapped reads) the GPU's minimizer count and species hit counts equal the CPU classifier's, and the engine
run with the GPU results injected writes byte-identical `nonhost/` files and `nonhost_reads.fq`. The 26.1 GB index (+2.8 GB directory,
+0.39 GB host Bloom filter) is resident in HBM after 8.9 s (cold read) or 3.6 s (page cache warm). The classification takes **35-79 ms
of GPU compute** per sample (87-155 M reads/s; 48-63 M reads/s with the host-to-device copy of the reads) and removes **0.98-1.21 host
core-s per M read pairs** (load of the indexes plus classify) at the campaign's 4 threads. The EM stays on the host (section 5).

## 1. What the CPU classifier computes (c4a1be38)

Code: `include/singlet/pileup/nonhost/{min_sketch,host_kmer_filter,nonhost_screener,nonhost_em,nonhost_cell_matrix}.h`, driver
`run_nonhost_em_screening` in `src/pipeline/singlet.cpp`. Called once per sample after the pileup stage; non-fatal on error.

**Input.** `${out_prefix}/star_Unmapped.out.mate1` (STAR `--outReadsUnmapped Fastx`, the cDNA mate). 4 lines per record (`getline`); a record
with an empty sequence line is skipped; the first 20,000,000 kept reads are used (`MAX_NONHOST_READS`; none of the samples reaches it). Read
order = file order (it is STAR's output order and it matters for the EM, section 1.6). Index `i` of a kept read is its rank in this list.

**1.1 Index files** (`$REF/nonhost/{viral,bacterial,fungal}.snhskidx`, format `SNHSKIDX` v2): 32-byte header (magic, version, k, w, reserved,
`n_entries`), then `n_entries` x 16-byte entries `{u64 kmer_hash, u32 species_id, u32 pad}` **sorted ascending by `kmer_hash`** (equal hashes in
file order), then a species-name manifest. k = 21, w = 11. Sizes: viral 90,310,522 entries (1.45 GB), bacterial 411,215,174 (6.58 GB),
fungal 1,128,455,398 (18.07 GB) = 26.08 GB. The engine builds two query indexes: `viral_` = the viral file; `microbial_` = `std::merge(bacterial,
fungal)` (stable: for equal hashes the bacterial entries come first). `fungal_species` = ids with a non-empty name in the fungal manifest.
Species ids are **not unique across the three files** (id 7016 is a viral name and also a fungal-set member): the hit maps are keyed by id alone, so
a bacterial and a fungal entry with the same id count into the same slot.

**1.2 Minimizers** (`extract_minimizers`). Per 2-bit canonical k-mer: `fwd` (A=0 C=1 G=2 T=3, first base in the high bits), `rc` its reverse
complement, `hash = murmurhash64A(min(fwd, rc))` where murmurhash64A is the fmix64 finalizer (`x ^= x>>33; x *= 0xff51afd7ed558ccd; x ^= x>>33;
x *= 0xc4ceb9fe1a85ec53; x ^= x>>33`). Case-insensitive ACGT; a k-mer containing any other byte has hash `INVALID = UINT64_MAX`. With `n = len - 20`
k-mers, `n < 11` gives no minimizers. For each window of 11 consecutive k-mers (`i = 0 .. n-11`) the window minimum is the smallest hash (strict `<`,
start value INVALID); an all-INVALID window is skipped; a minimizer is emitted when it differs from the previously emitted one (only consecutive
duplicates are removed, so the same hash can occur twice in a read).

**1.3 Host subtraction** (`HostKmerFilter`, `$REF/nonhost/host_21mer.bloom`, magic `HSKBLOOM`, 3,094,118,976 bits = 387 MB, 7 probes): a minimizer `m`
is "host" if for `i = 0..6` bit `(h1 + i*h2) mod n_bits` is set, with `h1 = fmix64(m ^ 0x9e3779b97f4a7c15)`, `h2 = fmix64(m ^ 0x6c62272e07bb0142) | 1`
(64-bit wrap-around before the modulo). Host minimizers are dropped. If fewer than `kMinNonhostMins = 5` minimizers remain the read gets **no hits at
all** (`np = 0`, an empty `MultiHit`, UNKNOWN). Otherwise `np` = the number of remaining minimizers: it is the denominator of every rate.

**1.4 Hit counting** (`classify_multi_batch_filtered`, Phases A-C). All kept minimizers of all reads go into one flat list of `(hash, read)` sorted by hash
only (GNU parallel sort, unstable between reads, which does not matter: reads never interact). `batch_query_presorted` merge-scans it against each index; for
every query/entry pair with equal hash it does `hits[read][species_id]++` in an `unordered_map<u32,u32>`, one map per read and per index (`vh`: viral index,
`mh`: bacterial+fungal). For one read the increments therefore happen in **ascending minimizer hash order**, duplicates of a hash back to back, and for
each minimizer the entries of the run in **file order** (bacterial run before fungal run). A species enters the map at its first increment.

**1.5 Per-read decision** (Phase D, `hits_to_multi_hit`; all float, host). `rate = float(cnt) / float(np)`; `soft = 0.05f * 0.25f`; every species with
`rate >= soft` is appended to `viral_hits` / `microbial_hits` **in `unordered_map` iteration order** (libstdc++: a function of the insertion sequence
and the bucket history, i.e. of the first-increment order of 1.4); `best_*_rate` is the largest rate (strict `>`). `top_category`: UNKNOWN if both best
rates are 0; VIRAL if `best_viral >= best_microbial` (a tie goes to viral); otherwise the microbial species with the largest rate (strict `>`, so the
**first maximum in iteration order wins ties**) decides FUNGAL (id in `fungal_species`) or BACTERIAL.

**1.6 EM and outputs** (all host, all float). `em_deconvolve(multi_hits, kingdom_map, 500, 1e-6, 1e-4, 0.05, 0.30, ..., min_abs_reads = max(500,
0.005f * n_total))`. Per read the viral and microbial rates are merged in an `unordered_map<u32,float>` (max of the two), hits below 0.05 are erased, the
survivors get species indices **in the order the merged map is iterated, as first seen over the reads in file order**; theta starts uniform; each of up to
500 iterations does the E-step (`denom += theta[k]*h`, then `theta_new[k] += theta[k]*h/denom`, reads in file order, species in map order), adds the
Dirichlet term `0.1f*prior_hr[k]`, normalises and tests `max|delta| < 1e-6f` (it hits the 500 cap on the samples here). Then per-species posterior mean
hit rate, filters (`theta >= 1e-4`, `mean_hr >= 0.30`, `round(theta*N) >= min_abs_reads`) and a `std::sort` by abundance. `build_kingdom_map` takes the
first kingdom seen. Outputs: `nonhost/nonhost_em_abundance.tsv`, `nonhost_summary.json` (kingdom tallies from `top_category`), `nonhost_per_cell.{1pz,tsv}`
(`NonHostCellMatrix::build`, uses `multi_hits` and the mate files), `nonhost_reads.fq` (only if a kingdom fraction exceeds 1%), and the optional secondary
alignment (skipped here: the per-species FASTAs are missing, as in every campaign run so far).

**Floating point lives only in 1.5 and 1.6**, both serial or per-read, on the host. What fixes their bits is the *order*: read order for the EM sums, and
first-increment order of species within a read for the maps' iteration order. 1.2-1.4 are integer.

## 2. The GPU port (`m13.cu`)

Design input (read only, ideas re-implemented, no code copied): `singlet-agents/.claude/worktrees/gpu-align-vision/bench/gpu-align/results/RESULTS.md`
(random probes into a 34 GB table 11.6 G/s, `a3_hash.cu`; cub radix sort 11 G pairs/s, `a4_sort.cu`; rolling canonical 2-bit k-mers, `b1_kmer.cu`).
Sort-merge against the 26 GB index (what the CPU does, and what exploration 4.12 planned) is not needed: the index is already sorted by hash and murmur
hashes are uniform, so a **radix directory over the top B hash bits** (about 2 entries per bucket) gives each probe one directory read plus a 1-3 entry scan.

**Resident index (host step 1).** The three files' entry arrays are copied as they are (16 B AoS, 26.08 GB) into HBM by 12 reader threads (64 MB pinned bounce
buffers, one stream each); a kernel builds `dir[b] = first entry with (hash >> (64-B)) >= b` (B = 25 / 27 / 29, 2.82 GB in total) and checks the sort order
(0 descents in all three files). The Bloom filter's words (387 MB) are copied as they are. Total 29.9 GB in HBM; the species-name manifests stay on the host.

**Classification (per batch of reads, all on the device).**
1. `k_count`, `k_fill`: one thread per read walks the bases with rolling `fwd`/`rc` and a 16-slot ring of k-mer hashes; the same window rule, the Bloom test, `np`
   (0 if < 5); `k_fill` writes the kept minimizers and insertion-sorts each read's list ascending (the order of 1.4). Exclusive scan -> offsets.
2. `k_probe`: one thread per minimizer probes viral, bacterial and fungal (`lo`, `len` of the equal-hash run in each).
3. `k_sum` + scan: occurrences per read; `k_write`: one 64-bit key per occurrence, `read<<33 | db<<32 | species_id` (db 0 = viral map, 1 = bacterial+fungal
   map), written in exactly the CPU's increment sequence (viral segment first, then per minimizer the bacterial entries and then the fungal entries).
4. `cub::DeviceRadixSort` (stable) by key, `cub::DeviceRunLengthEncode` -> one group per (read, map, species) with its count, first occurrence = minimum
   original position (`k_first`), a second radix sort by first position restores the CPU's **first-increment order** of the species in each map.
5. Output to the host: `np[reads]` and the groups `(key, count)` in that order (`NHGPURS1` file for the dev build; 12 bytes per group).

**Host side (unchanged code).** The patched `classify_multi_batch_filtered` (`m13.patch`) builds each read's two `unordered_map`s by inserting the species in the
GPU's order (same insertion sequence => same bucket history => same iteration order) and then runs the original `hits_to_multi_hit`, EM, matrix and writers.
**The EM is not ported:** its sums are tied to read order and to libstdc++'s `unordered_map` iteration order (twice: the per-read merged map and the species index), and
it costs only 0.11 core-s per M pairs; a bit-exact port is not simple. In a production GPU engine the leftover host work of this stage is Phase D + EM + writers.

Limits of this version: occurrences per batch < 2^31 (the largest sample has 38 M), species counts keyed by a 32-bit id, `k = 21, w = 11` (checked against the
index headers), no cap-subsampling logic beyond reading the first 20 M reads. Reads of any length work (ring buffer), but the in-thread insertion sort is meant for
the 20-100 minimizers of short reads.

## 3. Method and gate

One Slurm job per sample on `short-gpu` (1 H100 NVL, 28 CPUs), everything in the job's `/tmp` (`jobs/m13_run.sbatch`):
1. **CPU capture.** Dev engine (`jobs/m13_build.sbatch`: c4a1be38 + `m13.patch`, non-PGO `-march=x86-64-v3`, `$W/jobs/w11/m13/bin/singlet-m13`) runs the campaign command
   line (`--threads 24 --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off`) and stops after the non-host screen (`SINGLET_NH_STOP`). It leaves
   `star_Unmapped.out.mate1/2`, its own `nonhost/` outputs and `nonhost_reads.fq`, and the CPU truth per read (`SINGLET_NH_DUMP`): `raw.bin` (np and both hit maps in
   iteration order) and `multi.bin` (the `MultiHit`s with the float rates as bits).
2. **GPU** (`m13`, CUDA 12.4.1 devel image, `nvcc -O3 -arch=sm_90 -fmad=false`): loads the index, classifies the same unmapped reads (1 warm-up + 3 timed reps, cudaEvents,
   median; the output checksum is equal across reps) and compares its per-read result with `raw.bin` as sets (np, viral map, microbial map).
3. **Replay A** (`singlet nhscreen`, CPU path, control) and **replay B** (the GPU results injected) on the same unmapped file, each dumping `raw.bin`/`multi.bin`.
4. **Compare**: `cmp` of the three dump pairs (this includes the maps' iteration order and the float bits of every rate) and sha256 of every file under `nonhost/` and of
   `nonhost_reads.fq` for engine == replay A == replay B.
5. **Replays C (CPU) and D (GPU results) at `--threads 4`, no dumps**, for the host-time numbers, outputs checked again against the engine's.

## 4. Results

| | GSM4006843 (3' v2) | GSM6470208 (5' v2) | GSM5411066 (mouse 3' v3) |
|---|---|---|---|
| read pairs | 66,699,625 | 121,774,155 | 69,556,631 |
| unmapped reads screened | 3,035,990 | 12,265,634 | 4,872,654 |
| minimizers after Bloom (Q) / occurrences (H) / groups (G) | 24.3 M / 9.2 M / 3.3 M | 54.3 M / 38.3 M / 18.5 M | 28.2 M / 14.1 M / 6.4 M |
| reads with np > 0 / with any hit | 2,453,419 / 682,540 | 7,875,603 / 2,359,612 | 3,168,710 / 623,655 |
| per-read np, viral map, microbial map: GPU vs CPU | 0 mismatches | 0 mismatches | 0 mismatches |
| `raw.bin`, `multi.bin` engine == replay A == replay B | identical (62.8 MB, 53.7 MB) | identical (295 MB, 258 MB) | identical |
| `nonhost/` files and `nonhost_reads.fq`, engine == A == B == C == D | identical (5 files, 142 MB fq) | identical (3 files, 474 MB fq) | identical (3 files, 124 MB fq) |
| EM species reported | 3 (per-cell matrix 15 x 3) | 0 | 0 |
| GPU, all stages incl. read upload and result download (median of 3) | 62.6 ms | 195.6 ms | 85.6 ms |
| of which host-to-device copy of the reads / extract+Bloom / probe / expand+sort+group / device-to-host | 24.2 / 17.0 / 15.3 / 2.5 / 3.6 ms | 97.5 / 45.9 / 18.0 / 15.2 / 18.9 ms | 34.4 / 20.8 / 9.8 / 13.8 / 6.8 ms |
| **GPU compute only** (no copies) | **34.7 ms** | **79.2 ms** | **44.5 ms** |
| **reads/s**: compute only / with copies | **87 M** / 49 M | **155 M** / 63 M | **110 M** / 57 M |
| classify working set in HBM | 1.6 GB | 5.4 GB | 2.2 GB |

The copies dominate: the reads travel as ASCII over PCIe (1.1 GB at 11 GB/s for GSM6470208). In the GPU engine the unmapped reads are produced on the device by the aligner, so the
compute-only column is the cost of this stage there. The extract+Bloom stage is the Bloom's 7 random probes into 387 MB; probe time follows the number of minimizers.

**Index (host step 1)**, identical for every job: 26.08 GB of entries (viral 1.45, bacterial 6.58, fungal 18.07 GB), directories 2.82 GB (B = 25, 27, 29), Bloom 0.39 GB, **29.9 GB resident**.
Load to HBM with 12 reader threads: **8.86-8.98 s** with the files' client page cache dropped first (`posix_fadvise DONTNEED`; read + copy 0.9 + 2.2 + 5.5 s, Bloom 0.3 s, directory build
0.05 s in total, about 3 GB/s) and **3.6 s** when read again from the page cache. The storage servers' own cache state is unknown (the files had been read by earlier jobs), so 8.9 s is a
measurement, not a cold-start bound. The CPU engine reads and parses the same files in 38-48 s per sample (`load_indices` timer, includes the `std::merge` of bacterial and fungal).

**Host core-s removed** (this job's own process-CPU timers; replays C and D at `--threads 4`, the campaign setting; replay D still loads the indexes on the host as a dev expedient and keeps
Phase D, EM and the writers, so "removed" = C load + C classify - D classify):

| core-s | GSM4006843 | GSM6470208 | GSM5411066 |
|---|---|---|---|
| CPU index load (`load_indices`) | 38.0 | 40.6 | 47.1 |
| CPU classify (OpenMP, 4 threads) | 28.5 | 82.5 | 38.2 |
| host left with GPU classify (read results, Phase D) | 0.67 | 3.35 | 1.22 |
| **removed per M read pairs: classify only** | **0.42** | **0.65** | **0.53** |
| **removed per M read pairs: index load + classify** | **0.99** | **0.98** | **1.21** |

For comparison, section 12.9 of the exploration (m10b, campaign build, 4 threads, other runs of the same samples): classify 20.5 / 77.2 / 20.3 core-s = 0.31 / 0.63 / 0.29 per M pairs, load
33.2 / 46.2 / 28.6 = 0.50 / 0.38 / 0.41 per M, pooled over six samples 0.43 and 0.32 per M. The classify numbers here are higher on the 3' samples (different run, different build, STAR's read order; I did not
isolate why, one candidate being OpenMP threads spinning in the parallel sort). The load is a one-time cost per node with a persistent service (step 1); the classify part is
the new saving of step 2. Phase D plus the EM plus the writers stay on the host (EM 0.11 per M pairs pooled).

**Not identical / caveats.**
- Nothing differs between the GPU-fed run and the CPU run. One comparison to an *older run* does differ: the m10b run of GSM4006843 (`host/runs/GSM4006843/small/nonhost/nonhost_summary.json`, 4 threads,
  PGO build 28969b87) has the same kingdom counts and species but EM abundances that differ in the third digit (0.084205 vs 0.084967 for species 7016). The EM is capped at 500 non-converged iterations and
  its float sums depend on read order; STAR's unmapped-read order is not the same between runs (24 vs 4 threads), and the build differs. I did not separate the two causes. It does not affect the gate
  (same unmapped file, same binary in every replay), but it means **bit-exact EM output requires the GPU engine to hand the host the unmapped reads in the order the CPU engine's file would have had**; the
  exploration's rule "original index rank < 20 M" fixes the cap, not the order.
- The comparison covers the integer inputs of the EM and its outputs end to end, but only GSM4006843 has EM species (3), so the EM-dependent files (`nonhost_per_cell.*`) are verified on that sample only.
- The 20 M read cap and the 2^31 occurrence limit were not reached (the largest sample screens 12.3 M reads and has 38 M occurrences).
- The species-name manifests and the fungal set come from the host-loaded files; the GPU does not need names.
- The first (cold) rep of each timing set is separate in `gpu.json` (`warmup_total`: 93 / 321 / 137 ms; allocations of the buffers).

## 5. Jobs and budget

| job | what | elapsed |
|---|---|---|
| 789787 | build: `m13` (nvcc) + `singlet-m13` (engine c4a1be38 + patch) | 3:06 (16 CPUs) |
| 789794 | development run on pbmc2m (97,506 unmapped reads): all gates identical | 2:59 |
| 789797 | GSM4006843, failed in 3 s (its temporary `.1fq` under `$W/data` had been deleted); fixed by re-staging into job `/tmp` (`STAGE_SRR=SRR9900123`, 311 s) | 0:03 |
| 789798 / 789807 | first full pass: GSM6470208 / GSM4006843, identical (`runs/run1/`) | 19:22 / 13:16 |
| **789818 / 789819 / 789820** | **final pass with timing replays: GSM6470208 / GSM4006843 (re-staged) / GSM5411066 (re-staged, SRR14999752)** | 23:45 / 20:08 / 15:19 |

**GPU-hours used: 1.63** (limit 6; wall time of the GPU jobs, one H100 NVL each). **Core-hours used: about 45** allocated (28 CPUs x 1.58 h of run jobs + 16 CPUs x 0.05 h for the build; limit 150).
The reads, the unmapped FASTQ, the dumps and the replays only existed in the jobs' `/tmp`; `$W` holds the dev binaries, the sources, logs and small result files (`$W/jobs/w11/m13/`).
`$W/pipeline-benchmark/bin/*` and the campaign worktree were not touched.

## 6. Files and reproduction

`m13.cu` (GPU program), `m13.patch` (dev-build patch against c4a1be38: `m13_dev.h` new, `nonhost_screener.h`, `singlet.cpp`: timers, dumps, `SINGLET_NH_GPU_RESULTS` injection, `SINGLET_NH_STOP`,
`singlet nhscreen`), `jobs/m13_build.sbatch`, `jobs/m13_run.sbatch`, `make_m13_json.py`, `runs/<sample>/` (gpu.json, gpu.log, compare.txt, engine/replay logs with the `[m13t]` timers, summaries; `runs/run1/` is the first pass).
Sources are uploaded to `$W/jobs/w11/m13/`; `sbatch --partition=short-gpu jobs/m13_build.sbatch`, then
`sbatch --partition=short-gpu --time=01:30:00 --export=ALL,SET=GSM6470208 jobs/m13_run.sbatch` (add `STAGE_SRR=SRR9900123` for GSM4006843, `SRR14999752` for GSM5411066, whose temporary copies were
deleted), copy `$W/jobs/w11/m13/results/<SET>/*` to `runs/<SET>/`, `python make_m13_json.py`. The `m13` exit code is 0 only if the per-read comparison with the CPU dump has no mismatch.
