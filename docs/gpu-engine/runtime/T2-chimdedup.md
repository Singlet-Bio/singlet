# T2 chimdedup: the `--chimeric` second pass maps each distinct read sequence once

Branch `t2/chimdedup` (from `gpu/throughput-2` 58926d6d). Build `bin/t2-chimdedup` (job 819027: `ctest exit 0`, 188 of 188 tests passed, `test_chimeric_dedup` among them; binary sha256 `c7dde135e8cad2d8...`, patch sha256 `33e936f353ef78d2...`). Cluster jobs besides the build: gate 819032, runs 819034 (pbmc5p_v1), 819035 (GSM6250953), 819036 (d300_5p), 819037 (CPU engine), 819048 (GSM6250953, Solo skip): 5 jobs. Scripts: `bench/gpu-runtime/jobs/t2-chimdedup.sbatch`, `t2-chimdedup-cpu.sbatch`, `bench/gpu-runtime/t2-chimdedup-watch.sh`, `t2-chimdedup-summary.py`. Data (job results and summaries, counts, hashes and seconds only): `bench/gpu-runtime/data/t2-chimdedup/`. Remote results: `$W/jobs/gpu-runtime/results/t2-chimdedup-<job>-<tag>`.

## 1. What changed

| file | what |
|---|---|
| `include/singlet/pileup/chimeric_dedup.h` (new, 436 lines) | stage A: `build_dedup()`; `DedupParams`, `DedupResult`; `dedup_detail::{ChunkedVec, Reader, SeqSet, pack_seq, hash_bytes, barcode_codes}` |
| `include/singlet/pileup/chimeric_table.h` | `MemberTable` (line 45); `Config::members`, `Config::expanded_dump_path`; `Stats::dedup_refusal`; `build_rows` branch `if (cfg.members != nullptr)` (line 342, the per-read path below it is untouched); `write_table` returns without writing when `dedup_refusal` is set (line 619) |
| `src/pipeline/singlet.cpp` | the chimeric block (from line 10578): stage A call, refusal logic, `run_pass` lambda (the former inline STAR argument building and fork), `SINGLET_CHIM_THREADS`, `SINGLET_CHIM_DEDUP_DUMP`, fall back to the pass on all reads, cleanup of `chim_dedup.mate1/2` (std::filesystem); include of `chimeric_dedup.h` |
| `src/star/STAR.cpp` (line 406) | one condition: `getenv("SINGLET_CHIM_PASS_NOSOLO") == nullptr` before `soloMain.processAndOutput()` |
| `tests/cpp/test_chimeric_dedup.cpp`, `tests/cpp/CMakeLists.txt` | the unit test |

Switches (all read in the parent `singlet`):
- `SINGLET_CHIM_DEDUP`: on by default; `=0` runs today's path (record counting by `getline`, the pass on `unm,sc` files, no member table). Off automatically (today's path) when `SINGLET_KEEP_CHIMERIC_JUNCTION=1` (logged: `dedup off (...)`), when `--soloType` is not `CB_UMI_Simple`, and when `star_args.txt` has `--soloBarcodeMate` other than 0, `--soloCBtype` other than `Sequence`, `--soloCBposition`, `--soloUMIposition`, `--outSAMreadID` other than `Standard`, `--outSAMattrRGline`, `--twopassMode` other than `None`, or `--outFilterType` other than `Normal` (the last two make a read's junction line depend on other reads; the others change what the junction line prints). Logged as `dedup refused (<reason>); running the pass on all reads`.
- `SINGLET_CHIM_THREADS=<n>`: the pass's `--runThreadN`; unset = the item's. Works with dedup on or off (the same `run_pass`).
- `SINGLET_CHIM_DEDUP_DUMP=1` (with dedup on): writes `chim_Expanded.junction` and keeps it.
- `SINGLET_CHIM_SKIP_SOLO=1` (optional item 10, default OFF, my choice): the parent sets `SINGLET_CHIM_PASS_NOSOLO=1` in the forked child only; the child's STAR skips `Solo::processAndOutput` (its output is deleted unread).

Stderr lines (new): `[singlet] chimeric: dedup: <N> reads, <D> distinct sequences (<x> %), <t> s` (t in fixed decimals, because the stream precision of this program is 1 digit and prints 39 s as `4e+01`), and `[singlet] chimeric: dedup memory: parent VmRSS <a> MiB before, <b> MiB at the peak of stage A, <c> MiB during the pass` (all three are `VmRSS` of the parent read from `/proc/self/status`; `<c>` is sampled when stage A returns, before the fork). The `reads into the second pass` and `second pass` lines are unchanged in format and, as measured, in numbers.

### 1.1 What the printed CB and UMI are a function of (read from the code)
`ReadAlign_oneRead.cpp:164-167` calls `SoloReadBarcode::getCBandUMI(Read0, Qual0, readLengthOriginal, readNameExtra[0], readFilesIndex, readName)` once per read before `chimericDetection()`; `ChimericAlign_chimericJunctionOutput.cpp:22-23` prints `RA->soloRead->readBar->cbSeq` and `umiSeq` as the last two columns. In `SoloReadBarcode_getCBandUMI.cpp` for `CB_UMI_Simple` with sequence CBs (FASTQ input):
- `bSeq` = the whole sequence line of the barcode mate (`readSeq[barcodeRead]`, `readLen[barcodeRead]` = the unclipped original length). `bQual` is only used for the quality histogram and the 1-mismatch match string, never for the printed `cbSeq`/`umiSeq`.
- if `bL > 0` and `bSeq.size() != bL`: fatal error exit; if `bL == 0` and `bSeq.size() < cbumiL`: `N` is appended up to `cbumiL`.
- `cbSeq = bSeq.substr(cbS-1, cbL)`, `umiSeq = bSeq.substr(umiS-1, umiL)`; the strings are printed as they are (the whitelist match, `convertCheckUMI` and `addStats` change `cbMatch*` but not `cbSeq`/`umiSeq`).
So they are a function of the barcode mate's sequence line and the run's `--soloCBstart/CBlen/UMIstart/UMIlen/BarcodeReadLength` only. Not of the cDNA mate, the qualities, the read name, `readFilesIndex` or `readNameExtra` (those are used for SAM input, `CB_UMI_Complex`, SmartSeq). `Line.read` (junction field 10) is stored in `build_rows` and never read afterwards (checked: the only uses of `.read` in the header are its assignment).

### 1.2 What the member table stores
For every read of the pass input, in input order and grouped by sequence id: the cell barcode (`cb_len` bases) and the UMI (`umi_len` bases), each as one `uint64` with 3 bits per base (A 0, C 1, G 2, T 3, N 4), i.e. exactly the strings `getCBandUMI` would print, reconstructed by `barcode_codes()` from the barcode mate's sequence with the same length rule, the same N padding and the same substring bounds. It does not store the barcode mate record. `barcode_codes` returns "not reproducible" (stage A refuses, today's path runs) for: a barcode read whose length differs from a fixed `--soloBarcodeReadLength`, a read too short for the CB or UMI position, a CB or UMI with a base outside `ACGTN`, a CB or UMI longer than 21 bases. Stage A also refuses a cDNA base outside `ACGTN`, a malformed 4-line FASTQ record (no `@`, no `+`, sequence and quality of different length, sequence over 650 bases), a partial record, different record counts of cDNA and barcode files, an unwritable output. Layout: `start[n_ids+1]` (`uint64`), `cb[N]`, `umi[N]` (`uint64`), 16 bytes per read plus 8 bytes per distinct sequence.

### 1.3 How it works
- Stage A (`build_dedup`, in the parent, replacing the two counting loops): thread 1 reads (unm1, then sc1) and finds distinct sequences: each sequence is packed 3 bits per base (tight) into an arena; the table is open addressing on a 32-bit hash tag and an id; **a tag match is confirmed by comparing the length and all packed bytes (`memcmp`)**, so identity is by full sequence, never by the hash alone. The first read of a sequence is its representative; ids are assigned in order of first appearance, over both files. Thread 1 writes `chim_dedup.mate1` (`@<id>`, the representative's sequence and quality). Thread 2 reads (unm2, then sc2) in step (batches of 65,536 reads through a bounded queue), computes the CB and UMI codes of every read and writes `chim_dedup.mate2` (`@<id>`, the representative's own barcode record and quality). After both threads: the arena, hash table, per-sequence arrays are freed (end of thread 1's scope), then per-read arrays (`id`, `cb`, `umi` in chunks of 4 M) are grouped by id into the `MemberTable` (counting sort, stable) and freed. `chim_in` / `chim_sc` come from the per-file counts.
- The pass runs on the two representative files; everything else in the argument list as before.
- `build_rows` with `cfg.members`: each junction line is parsed once; the read-name field is the id; `st.lines`, `no_barcode` (short lines), `noncanonical`, `reoriented` are advanced by the member count, in the same order of checks as the per-read path (lines; fewer than 22 fields; worse alignment dropped silently; non-canonical; reorientation; barcode resolve with `ambiguous` / `unmatched`; `kept_lines`); only the barcode resolve and the line collection loop over the members (the resolve of a packed CB is memoised). **Self-check**: for every line with 22 or more fields the CB and UMI STAR printed must equal member 0's entry (and have the table's lengths); a line whose name is not a valid id is refused too. On any refusal `write_table` writes nothing and sets `Stats::dedup_refusal`; the caller logs `dedup refused (<reason>); running the pass on all reads`, runs today's pass on `unm,sc` (those four files are removed only after the table step) and builds the table without members.
- Memory before the fork: everything of stage A except the member table is freed when `build_dedup` returns.

## 2. Results

### a. Unit test (job 819027)
`test_chimeric_dedup` (`ctest`, 0.47 s) passed; `100% tests passed out of 188`, `ctest exit 0` in `rt-build-819027.out`. The test: stage A on 10 synthetic read pairs in an unmapped and a soft-clip file (4 distinct sequences, one with N, duplicates with the same and different UMIs, barcodes exact, one-mismatch, two-neighbour, N, unmatched); member table equals a naive grouping; the representative FASTQ pair equals the expected bytes; refusals (lower-case barcode, wrong barcode length with a fixed length, free length padded with N like STAR, a barcode file with more records); junction files in today's form and in dedup form with duplicates, multimapping lines (worse and best alignment), non-canonical, CT/AC, a short line without barcode columns: rows and every counter equal for canonical-only x library orientation (4 configurations), the dump equals today's lines as a multiset, a printed UMI that differs from the table and an id outside the table are refused and no table is written.

### b. Pre-send checks
No string literal broken across a line in the added lines (checked every added line of the diff against 58926d6d for an odd number of unescaped quotes: the only hits are existing lines and comment text). CRLF kept in the working tree (`git ls-files --eol`: `i/lf w/crlf`; the patch has no CR). Notes: the brief's diff command (`-c core.autocrlf=false diff b21de2cf`) on a CRLF working tree prints whole-file diffs, so I committed and sent `git diff b21de2cf HEAD`, and I added `tests/cpp` to the diff paths (the brief's paths do not include it, so ctest would not have built my test).

### d. Identity gate (job 819032, default build, `--chimeric` not used)
8 lines `VERDICT <sample> cpu:gpu IDENTICAL (except E11's run-dependent files; ...)` for pbmc2m, pbmc10m, pbmc1k_v3, pbmc1k_v2, pbmc1k_cite_v3, pbmc5p_v1, neuron1k_v3, nuclei2k_v2; `FALLBACK LINES: 1` per sample (the R11 info line). The gate runs without `--chimeric`, so the switch is not exercised by it; I did not run a second gate with `SINGLET_CHIM_DEDUP=0`.

### e. Table identity, dedup on vs `SINGLET_CHIM_DEDUP=0` (same binary)
| sample (job) | arm | md5 of decompressed `chimeric_junctions.tsv.gz` | lines | junction lines / kept / no barcode / unmatched / ambiguous / non-canonical / turned |
|---|---|---|---|---|
| GSM6250953 (819035) | off | 3573f1645c4fc6e920d37c70a0033df7 | 59,632 | 17188238 / 4498881 / 0 / 123105 / 14517 / 9749577 / 4589683 |
| | on | 3573f1645c4fc6e920d37c70a0033df7 | 59,632 | same |
| | on + dump | 3573f1645c4fc6e920d37c70a0033df7 | 59,632 | same |
| | on, pass threads 16 | 3573f1645c4fc6e920d37c70a0033df7 | 59,632 | same |
| GSM6250953 (819048) | on, Solo skip (2 runs), on | 3573f1645c4fc6e920d37c70a0033df7 (all three) | 59,632 | same |
| pbmc5p_v1 (819034) | off, off, off(KEEP) | a5d71eb6a8b45889fe667bf57f17dce3 | 24,141 | 3926694 / 27130 / 0 / 3483 / 64 / 2928198 / 21701 |
| | on, on, on+dump, on+Solo skip | a5d71eb6a8b45889fe667bf57f17dce3 | 24,141 | same |
| d300_5p (819036) | off | 1763c9dca0013d9bb3e68d7bfc419efe | 102,847 | 16886797 / 145563 / 0 / 60831 / 1369 / 12840905 / 131869 |
| | on | 1763c9dca0013d9bb3e68d7bfc419efe | 102,847 | same |

All equal the known values (GSM6250953 `3573f164...` and the counters of your brief; pbmc5p_v1 `a5d71eb6...`; d300_5p `1763c9dc...` from `results-819009.txt`). Sorted-line md5 of the tables also equal where recorded (GSM `6d771324...`, pbmc `7903ba0b...`, d300 `0b0e43a4...`). The tables' `blacklist none` as in the earlier jobs (the reference base has no `pileup_indices/fusion`).

### f. Raw-line proof (data lines with the read-name column replaced by `.`, `LC_ALL=C` sorted)
| sample | raw file of `SINGLET_CHIM_DEDUP=0` + `SINGLET_KEEP_CHIMERIC_JUNCTION=1`: lines, md5 | `chim_Expanded.junction` of dedup + `SINGLET_CHIM_DEDUP_DUMP=1`: lines, md5 | verdict |
|---|---|---|---|
| pbmc5p_v1 (819034) | 3,926,694, d9e7359a0c10b09163e2622063149394 | 3,926,694, d9e7359a0c10b09163e2622063149394 | IDENTICAL |
| GSM6250953 (819035) | 17,188,238, 5202513e0679f97a1e700c494b3bfca2 | 17,188,238, 5202513e0679f97a1e700c494b3bfca2 | IDENTICAL |
So a junction line, apart from its read name, is a function of the cDNA sequence (and the member's own CB and UMI), on these two samples, for every one of 3.9 M and 17.2 M lines. No differing line.

### g. CPU engine (job 819037, pbmc5p_v1, `--engine cpu --chimeric`, 16 threads)
dedup on: table md5 `a5d71eb6a8b45889fe667bf57f17dce3` (24,141 lines), `dedup: 4944371 reads, 4535057 distinct sequences (91.7 %), 3.2 s`, counters as in e; off: `a5d71eb6a8b45889fe667bf57f17dce3`, same counters. No `chim_*` file left in the output directory in either arm. Pass log-to-log 60 s (on) and 77 s (off) (epochs 1795143105 to 1795143165, 1795143489 to 1795143566); item wall 380 s / 400 s.

### h. Timing at 8 item threads (H100 lane settings of `t2-chimpass.sbatch`; `bin/t2-chimdedup`; the arms of one job run one after the other on the same node)
"pass" = epoch of the `reads into the second pass` log line to the epoch of the `second pass` line (it excludes stage A in the dedup arms and the old record counting in the off arms, which was not measured). "child CPU" = user+sys of the pass's STAR process (last 1 Hz sample). cgroup = job cgroup anon+shmem (GiB) at the first watch sample of the chimeric block, at the pass start (`chim_Log.out`), and the peak.

GSM6250953 (121,366,072 pairs; 50,768,292 reads into the pass; 17,636,124 distinct = 34.7 %):
| run (job) | stage A s | pass s | item wall s | child CPU s | child VmHWM GiB | cgroup first / pass start / peak GiB | parent VmRSS before / stage A peak / at fork, MiB |
|---|---|---|---|---|---|---|---|
| off, KEEP=1 (819035) | - | 865 | 1,035.4 | 6,473.0 | 24.1 | 53.0 / 53.0 / 70.5 | - |
| on (819035) | 39.0 | 594 | 794.4 | 4,481.4 | 25.1 | 52.9 / 54.4 / 71.6 | 7836 / 9737 / 8696 |
| on + dump (819035) | 39.1 | 609 | 795.0 | 4,512.5 | 25.5 | 53.3 / 55.6 / 71.9 | 7836 / 10010 / 8985 |
| on, `SINGLET_CHIM_THREADS=16` (819035) | 39.0 | 322 | 508.4 | 4,526.1 | 25.2 | 52.8 / 55.3 / 72.4 | 7396 / 9505 / 8411 |
| on (819048) | 37.9 | 604 | 789.3 | 4,530.3 | 25.1 | 53.0 / 54.0 / 71.6 | 7939 / 9783 / 8657 |
| on + Solo skip (819048) | 38.5 | 589 | 789.4 | 4,503.1 | 24.5 | 52.7 / 55.0 / 71.2 | 7252 / 9363 / 8269 |
| on + Solo skip, again (819048) | 39.2 | 595 | 782.1 | 4,513.8 | 24.8 | 52.6 / 53.8 / 71.4 | 7241 / 9403 / 8443 |

pbmc5p_v1 (60,000,000 pairs; 4,944,371 reads; 4,535,057 distinct = 91.7 %), job 819034 (order: off+KEEP, on, off, on+dump, on, off, on+Solo skip):
| run | stage A s | pass s | item wall s | child CPU s | child VmHWM GiB | cgroup first / pass start / peak GiB | parent VmRSS before / stage A peak / at fork, MiB |
|---|---|---|---|---|---|---|---|
| off, KEEP=1 | - | 130 | 196.8 | 855.7 | 24.6 | 50.9 / 50.9 / 68.4 | - |
| on | 3.2 | 112 | 166.5 | 707.6 | 24.4 | 50.7 / 52.3 / 68.2 | 7827 / 7923 / 7846 |
| off | - | 136 | 187.5 | 860.5 | 24.2 | 51.2 / 51.2 / 68.1 | - |
| on + dump | 3.3 | 114 | 169.1 | 706.1 | 24.0 | 50.4 / 51.3 / 67.9 | 7573 / 7661 / 7583 |
| on (repeat) | 3.1 | 114 | 167.1 | 708.5 | 24.4 | 50.7 / 51.2 / 68.2 | 7813 / 7943 / 7834 |
| off (repeat) | - | 137 | 187.2 | 860.5 | 24.4 | 51.4 / 51.4 / 68.2 | - |
| on + Solo skip | 3.0 | 111 | 164.7 | 707.1 | 24.6 | 50.7 / 50.7 / 68.2 | 7807 / 7930 / 7821 |
Run-to-run: the three dedup-on pairs of runs are within 112-114 s (pass) and 166.5-167.1 s (item); the off runs 130 (KEEP), 136, 137 s and 187.2-187.5 s (item).

d300_5p (299,628,383 pairs; 22,372,925 reads; 19,286,025 distinct = 86.2 %), job 819036 (on first):
| run | stage A s | pass s | item wall s | child CPU s | child VmHWM GiB | cgroup first / pass start / peak GiB | parent VmRSS before / stage A peak / at fork, MiB |
|---|---|---|---|---|---|---|---|
| on | 14.5 | 433 | 608.9 | 3,174.7 | 29.1 | 59.1 / 60.7 / 77.3 | 12096 / 13202 / 12586 |
| off | - | 560 | 709.8 | 4,096.0 | 27.7 | 58.5 / 58.5 / 75.7 | - |

Summary of the effect (computed from the tables): GSM6250953 item wall 1,035.4 s to 794.4 / 789.3 s (-23 %), child CPU 6,473 s to 4,481 / 4,530 s (-31 %), pass 865 s to 594 / 604 s including a stage A of 38-39 s on top; with the pass at 16 threads 508.4 s item wall. pbmc5p_v1 pass 130-137 s to 111-114 s; d300_5p pass 560 s to 433 s.

## 3. Memory (GSM6250953, parent process `singlet`)
- Stage A: parent VmRSS grew from 7,836 MiB to 9,737 MiB at the peak (+1,901 MiB = 1.86 GiB; +2,174 MiB in the dump run; +1.8 to +2.2 GiB across the other runs); target was about 4 GiB: met.
- Member table during the pass: VmRSS 8,696 MiB when stage A returned (+860 MiB); +718 to +1,202 MiB in the other runs; the table itself is 16 B x 50.77 M + 8 B x 17.64 M = 953 MB (computed). Target about 1.5 GiB: met. `VmRSS` here is sampled when stage A returns, not while the pass runs; the cgroup anon+shmem at the pass start is +1.5 GiB over the first sample of the on run of 819035 (+1.0 GiB in the on run of 819048) and the cgroup peak +1.1 GiB over the off run (71.6 vs 70.5), the latter includes noise between runs.
- The child's `VmHWM` is 24.1 GiB (off) and 24.5-25.5 GiB (on), one more cause being a different sequence mix.
- Peak of stage A by construction (estimate, not measured separately): arena about 1.0 GB + hash table 0.27 GB + 4 B (id) + 16 B (CB, UMI) per read before grouping, then 16 B per read after.

## 4. Stage A time
37.9-39.2 s on GSM6250953 (6 runs) with two threads (20 GB of FASTQ from the node-local disk and page cache), 14.5 s on d300_5p (21.5 M + 0.9 M reads), 3.0-3.3 s on pbmc5p_v1. Target about 40 s: met (barely; one thread for the cDNA mate does the hashing; no profile taken).

## 5. Optional item 10 (skip the Solo post-map in the pass)
Done: one condition in `STAR.cpp` plus the child's env var. Default OFF (`SINGLET_CHIM_SKIP_SOLO=1` turns it on). Measured with it on: table md5 equal (GSM6250953 two runs, pbmc5p_v1 one run). Saving: GSM6250953 pass 589 and 595 s against 594, 604, 609 s without (on arms, same job 819048: 604 s); about 10 s (estimate, run-to-run spread of the on arm is 594-609 s), not the 23 s of the earlier measurement; pbmc5p_v1 111 s against 112-114 s. The Solo records written during mapping are still produced.

## 6. Differences from the prescribed design, failures, what was not done
- Member table: packs CB and UMI (3 bits per base) instead of storing "the bytes needed"; the barcode mate record itself is not kept (only the representative's goes to `chim_dedup.mate2`).
- Stage A uses two threads (cDNA mate and barcode mate); the brief allowed the item's threads.
- Extra logic not in the brief: the argument-based refusals in 1 (twopass, BySJout, read ID, RG, barcode mate/position options), and the line `dedup off (...)` for `SINGLET_KEEP_CHIMERIC_JUNCTION=1`.
- The fall back (self-check refusal in `write_table`, then `run_pass(false)` and a second table build) and the argument-based refusals in `singlet.cpp` were **not exercised at run time** (no job triggered them; the unit test covers the refusal in `write_table` and in `build_dedup`, not the code in `singlet.cpp`). The `SINGLET_CHIM_THREADS` switch was only run with dedup on (GSM6250953, 16 threads); with dedup off it takes the same lines of code.
- `SINGLET_CHIM_DEDUP_DUMP` writes a file of 17.2 M lines on GSM6250953 (about 2.3 GB, like the raw file).
- The gate does not use `--chimeric`; the identity of the chimeric output is shown by e, f and g, not by the gate.
- The `second pass` and `Total` seconds printed by the existing log lines use a stream precision of 1 digit (`6e+02`), as before; the tables above use log-line epochs.
- The time of the old record counting (before the `reads into the second pass` line) was not measured; so "pass s" of the off arms excludes it and the on arms exclude stage A (shown in its own column).
- Not measured: where the 594 s of the dedup pass go (child CPU 4,481 s / 8 = 560 s of mapping; start, Solo post-map and `write_table` expansion not timed separately). The child CPU per input read is 127 us for all reads (off) and 254 us per distinct read (on); the 33.1 M duplicate reads cost about (6,473 - 4,481) / 33.1 M = 60 us each if the fixed costs are equal in both arms (estimate), so the duplicates are cheaper than the average read and the CPU saving (31 %) is smaller than the read saving (65 %).
- The raw-line proof covers pbmc5p_v1 (CellRanger-style clip of the 5' library) and GSM6250953; d300_5p was checked by table and counters only.
- All GPU runs were on g051 (one job at a time: the account's GRES limit serialised them); `squeue on node` at each job's start lists only that job; the node memory line of every run start is in the data files. The CPU-engine job ran on g002.
