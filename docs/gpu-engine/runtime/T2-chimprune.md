# T2 chimprune: the pair loop of `chimericDetectionMult` with the same junction lines for every read

Branch `t2/chimprune` (from `gpu/throughput-2` 9cf776f7). Code commit `85c2be4f` (the report commit on top of it carries this file, the job script and the data). Build `bin/t2-chimprune` (job 819077: `ctest exit 0`, `100% tests passed out of 188`, 746 s; binary sha256 `576fa903cadc272b...`, patch `patches/t2-chimprune.diff` sha256 `92cb119c3a185ec4...`, `git diff b21de2cf HEAD -- src include CMakeLists.txt bench/gpu/m5/stitch.cuh tests/cpp`, 161,683 bytes). Cluster jobs besides the build and the gate: 819080 (CPU engine item), 819081 (standalone pass, both samples, 5 arms each), 819082 (H100 items pbmc5p_v1 and d300_5p), 819083 (H100 item GSM6250953): 4 jobs. Gate: 819079. Scripts: `bench/gpu-runtime/jobs/t2-chimprune-run.sbatch` (new); the H100 and CPU-engine items use `t2-chimdedup.sbatch` / `t2-chimdedup-cpu.sbatch` unchanged (arms `pp1:SINGLET_CHIM_PAIRPRUNE=1`, `pp0:SINGLET_CHIM_PAIRPRUNE=0`). Data (counts, hashes, seconds): `bench/gpu-runtime/data/t2-chimprune/`. Remote results: `$W/jobs/gpu-runtime/results/t2-chimprune-819081-run`, `t2-chimdedup-819080-cpu`, `t2-chimdedup-819082-items`, `t2-chimdedup-819083-gsm`.

## 1. What changed

| file | what |
|---|---|
| `src/star/ChimericDetection_chimericDetectionMult.cpp` | the two nested loops of `chimericDetectionMult` moved unchanged into the member `pairLoopOld` (the `=0` branch; the loop text is the old text, with the three variables it updates passed by reference under their old names). New member `pairLoopNew` (the pruned loop). New static `chimPairPruneMode()` (reads `SINGLET_CHIM_PAIRPRUNE` once, in a function-local `static const int` in `chimericDetectionMult`). `chimericDetectionMult` now computes `minScoreToConsider` as before, then calls `pairLoopOld`, `pairLoopNew`, or (verify) both. Everything after the loops (the `chimScoreBest==0` return, `chimN`, the too-many-loci rule, output) is untouched. Includes `<algorithm> <cstdio> <cstdlib> <cstring>` added. |
| `src/star/ChimericDetection.h` | private declarations of `pairLoopOld`, `pairLoopNew` and three per-object vectors `flSeg`, `flScore`, `flSufMax` (buffers reused from read to read; one `ChimericDetection` per `ReadAlign`, so one per thread) |

`SINGLET_CHIM_PAIRPRUNE`: unset or any value other than `0` / `verify` = pruned loop (default); `0` = old loop; `verify` = per read the old loop into a separate vector, the new loop into `chimAligns`, compare, continue with the new result. Nothing else changes (stitching, seed search, parameters untouched).

`pairLoopNew`: flatten once per read in the old loop's order (window, then alignment in the window) into `flSeg` (a `ChimericSegment` per transcript, built once; reserved to the total count, so it never reallocates), keep only the transcripts that pass `segmentCheck()` (the old loop skips the others both as first and as second member of a pair, so the pairs left and their order are the old ones), `flScore[i] = maxScore`, `flSufMax[i]` = maximum `maxScore` over the later kept transcripts. Outer loop: skip transcript `i` if `minScoreToConsider > 0 && score1 + flSufMax[i] < minScoreToConsider`. Inner loop over later kept transcripts: skip if `minScoreToConsider > 0 && score1 + score2 < minScoreToConsider`, else strand test, `chimericAlignScore`, and the candidate block (copied from the old loop, unchanged text). The argument for both skips is written as a comment at the check (in the source, `pairLoopNew`).

Verify mode is about 40 lines (`chimericDetectionMult`, the `else` branch): it compares the count of `chimAligns`; per element the addresses of the two original transcripts (`&seg1.align`, `&seg2.align`, which is the identity of "which alignment"), both segment strands, `chimScore`, `stitchingDone`, `ex1`, `ex2`, `chimJ1`, `chimJ2`, `chimRepeat1`, `chimRepeat2`, `chimMotif`, `chimStr`, the two stitched alignments' `maxScore` and `nExons`; then `chimScoreBest`, `bestChimAlign` and the final `minScoreToConsider`. On a difference it prints one line `SINGLET_CHIM_PAIRPRUNE=verify MISMATCH: ...` with the counts to stderr and calls `std::_Exit(1)`. Ownership: the old loop's results are swapped into a local vector (so the function's start-of-call deletion of the previous read's `al1`/`al2` still works on `chimAligns` only); the old loop's allocated `al1`/`al2` (`stitchingDone`) are deleted after the comparison; the new loop's stay in `chimAligns` for the usual deletion at the next call. No leak, no double free by construction (not checked with a tool).

## 2. Findings from the code (before the change)

Line numbers are in the files of the worktree (before the edit for `chimericDetectionMult.cpp`).

(a) The constructor and `segmentCheck()` are pure functions of `P` and the transcript. `ChimericSegment.cpp` lines 3-17 (constructor): reads `align.intronMotifs[1..2]`, `align.Str`, `align.exons[][EX_R/EX_L]`, `align.nExons`, `align.Lread`, `align.readLength[0]`; writes only `str`, `roS`, `roE` of the segment itself. Lines 19-32 (`segmentCheck`): reads `align.rLength`, `pCh.segmentMin`, `align.intronMotifs[0]`, writes nothing. Neither depends on the other segment, mutates shared state or produces output. The pair loop never writes to `trAll[][]`: `ChimericAlign::chimericStitching` (`ChimericAlign_chimericStitching.cpp` lines 12-13) copies `*al1`, `*al2` into new `Transcript`s and edits the copies; `chimericAlignScore` (`ChimericDetection_chimericDetectionMult.cpp` lines 6-20) takes the segments by non-const reference but only reads them. So the values do not change between a segment's construction at the top of the loop and any later use. The old loop constructs a segment for every transcript (as `seg1`) anyway, so building them all before the loop reads nothing new. No field needed to stay per pair.

(b) `ChimericAlign` keeps copies, not references, of the segments: `ChimericAlign.h` line 17 declares `ChimericSegment seg1, seg2;` by value, copy-constructed in the constructor (`ChimericAlign.cpp` line 4). A `ChimericSegment` itself holds references (`P`, `pCh`, `Transcript &align`, `ChimericSegment.h` lines 12-15), so the stored copies refer to the `Transcript` in `trAll` and to `Parameters`. `al1 = &seg1.align` (`ChimericAlign.cpp` line 7) is the address of that `Transcript` in `trAll`, not of the segment object. Therefore a stored `ChimericAlign` depends on `trAll` and `P` staying valid until the junction output of the same call (and `chimericJunctionOutput` / BAM output run inside the same call, after the loops), and not on the loop-local segment objects. Today's lifetime: the loop-local `seg1`/`seg2` objects die at the end of their loop iteration, long before the stored copies are used. Mine: `flSeg` lives until the next call (a member), at least as long. `seg1.str`/`seg2.str` are read by `chimericStitching` (line 18) from the stored copies. No other use of `seg1`/`seg2` of a `ChimericAlign` outside the constructor and stitching (grep of `ChimericAlign_chimericJunctionOutput.cpp`, `ChimericAlign_chimericBAMoutput.cpp`, `ReadAlign_chimericDetection*.cpp`: none).

(c) `chimericAlignScore` has no side effects: lines 6-20; locals `chimScore`, `chimOverlap`, `diffMates` only; returns 0 or `seg1.align.maxScore + seg2.align.maxScore - (int)chimOverlap`.

Bound at the check. `maxScore` is `intScore` = `int` (`Transcript.h` line 46, `IncludeDefine.h` line 69). `chimOverlap` is `uint` = `unsigned long long` (`IncludeDefine.h` line 50), computed on line 9 as 0 or `end - start + 1` in a branch whose condition guarantees `start <= end` (`seg2.roS>seg1.roE ? 0 : seg1.roE-seg2.roS+1` under `seg2.roS>seg1.roS`, so `roS2 <= roE1`; `seg2.roE<seg1.roS ? 0 : seg2.roE-seg1.roS+1` under `roS2 <= roS1`, so `roE2 >= roS1`): no unsigned wrap-around, value at most `max(roE)+1`, where `roE` is a read coordinate (at most `DEF_readSeqLengthMax` = 650 per read, `IncludeDefine.h` line 140). So `(int)chimOverlap` is in `[0, 651]` and `chimScore <= score1 + score2` whenever it is not 0. An overlap above `INT_MAX` cannot occur (read coordinates are below 2^31); if it could, the cast would give a negative number and the bound would break, which is why the argument rests on the coordinate range and is written next to the check.

## 3. Results

### a. Build and checks
Build 819077: `ctest exit 0`, 188 of 188 tests passed. String-literal check of the brief on the diff: 4 hits, all closed strings on one line or a `printf` format closed before the line end (read, not broken). CRLF kept in the two edited files (`git ls-files --eol`: `i/lf w/crlf`); the patch has no CR. **No unit test**: `ChimericDetection` needs a `Parameters` object (a class with several hundred members, filled by the command-line parser), a `Genome` and a `ReadAlign`, and `tests/cpp/CMakeLists.txt` builds header-only tests that do not link the STAR objects; the scaffolding would be far above 150 lines. The proof is `verify` mode on every read of the two samples below plus the raw-line md5s.

The job `t2-chimprune-run.sbatch` and the H100 jobs were submitted with `--time=03:59:00` on the command line (the 05:00:00 / 06:00:00 of the scripts was refused: "Requested time limit is invalid").

### b. Identity gate (job 819079, default build, `--chimeric` not used)
```
VERDICT pbmc2m cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 91/7/4)
VERDICT pbmc10m cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 91/7/4)
VERDICT pbmc1k_v3 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 92/7/4)
VERDICT pbmc1k_v2 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 92/7/4)
VERDICT pbmc1k_cite_v3 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 92/7/4)
VERDICT pbmc5p_v1 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 91/5/4)
VERDICT neuron1k_v3 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 92/6/4)
VERDICT nuclei2k_v2 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 92/6/4)
```
`FALLBACK LINES: 1` per sample (the R11 info line). The gate does not use `--chimeric`, so it does not run the changed code; the chimeric identity is shown by c and d. I did not run a second gate with the switch on explicitly (the switch is on by default in this build, so the gate above is the switch-on run of the code path that every `--chimeric` item uses; without `--chimeric` that code is not reached).

### c. Standalone pass on the kept dedup inputs (job 819081, g-node CPU only, 8 threads, order per sample: `=0`, default, default, `=0`, verify)
Required md5 of the sorted raw data lines: pbmc5p_v1 `643f187626f6ffaa67d0fe9acf324d42` (2,832,142 lines), GSM6250953 `c2a0f02ef3d48de17e89fab98b17c1ac` (6,706,877 lines). Seconds are bash `time`.

| sample | arm | wall s | user s | sys s | md5 of sorted raw lines | lines |
|---|---|---|---|---|---|---|
| pbmc5p_v1 | `=0` (a) | 106.6 | 725.1 | 12.1 | 643f187626f6ffaa67d0fe9acf324d42 | 2,832,142 |
| | default (a) | 83.3 | 549.4 | 11.6 | 643f187626f6ffaa67d0fe9acf324d42 | 2,832,142 |
| | default (b) | 83.2 | 548.2 | 11.7 | 643f187626f6ffaa67d0fe9acf324d42 | 2,832,142 |
| | `=0` (b) | 105.7 | 726.2 | 11.1 | 643f187626f6ffaa67d0fe9acf324d42 | 2,832,142 |
| | verify | 109.7 | 754.0 | 11.7 | 643f187626f6ffaa67d0fe9acf324d42 | 2,832,142 |
| GSM6250953 | `=0` (a) | 611.1 | 4,710.8 | 15.8 | c2a0f02ef3d48de17e89fab98b17c1ac | 6,706,877 |
| | default (a) | 419.6 | 3,197.2 | 15.1 | c2a0f02ef3d48de17e89fab98b17c1ac | 6,706,877 |
| | default (b) | 420.2 | 3,203.0 | 15.0 | c2a0f02ef3d48de17e89fab98b17c1ac | 6,706,877 |
| | `=0` (b) | 610.9 | 4,716.8 | 15.0 | c2a0f02ef3d48de17e89fab98b17c1ac | 6,706,877 |
| | verify | 620.9 | 4,795.0 | 15.0 | c2a0f02ef3d48de17e89fab98b17c1ac | 6,706,877 |

All ten arms: rc 0, no `MISMATCH` line in any stderr tail, STAR's counters equal in all arms of a sample (pbmc5p_v1: 114,684 unique, 4,275,567 too short, 799,963 chimeric reads; GSM6250953: 4,732,909 / 12,241,451 / 3,237,917). Verify mode therefore compared old and new loop on every one of 4,535,057 and 17,636,124 reads (the distinct sequences of the pass) without a difference, and the junction lines are the same as in the old loop.

Effect (computed): default against `=0` (means of the two runs): pbmc5p_v1 wall 106.2 s to 83.3 s (-21.6 %), user 725.7 s to 548.8 s (-24.4 %); GSM6250953 wall 611.0 s to 419.9 s (-31.3 %), user 4,713.8 s to 3,200.1 s (-32.1 %). The verify arm costs 28 s (pbmc5p_v1) and 81 s (GSM6250953) of user time more than the `=0` arm (estimate of the new loop plus the comparison, from one run each; the run-to-run spread of the `=0` arms is 1 to 6 s). Against the profile's 229 and 1,904 CPU-s for the pair loop (T2-chimprof, other build, cycle counters) the saving of 177 and 1,514 user-s is about 77 % and 80 % of that (estimate; the profile's node was a different CPU type, so the comparison is only indicative), not the 90 % the profile estimated: what stays is the scan over the later transcripts (an integer add and compare per visited pair), which the prescribed design keeps because the order of pairs is fixed.

### d. Whole items on an H100 (`--chimeric`, 8 item threads, one job per sample group, arms one after the other on the same node; jobs 819082 (g-node of the job: see data), 819083)
Table md5 of the decompressed `chimeric_junctions.tsv.gz`; "pass" = epoch of the `reads into the second pass` line to the epoch of the `second pass` line; "child CPU" = user+sys of the pass's STAR process (last 1 Hz sample).

| sample (job) | arm | table md5 | rows | pass s | item wall s | child CPU s |
|---|---|---|---|---|---|---|
| pbmc5p_v1 (819082) | default | a5d71eb6a8b45889fe667bf57f17dce3 | 24,141 | 84 | 155.9 | 529.5 |
| | `=0` | a5d71eb6a8b45889fe667bf57f17dce3 | 24,141 | 111 | 164.9 | 699.1 |
| d300_5p (819082) | default | 1763c9dca0013d9bb3e68d7bfc419efe | 102,847 | 338 | 500.4 | 2,384.4 |
| | `=0` | 1763c9dca0013d9bb3e68d7bfc419efe | 102,847 | 436 | 597.1 | 3,157.4 |
| GSM6250953 (819083) | default | 3573f1645c4fc6e920d37c70a0033df7 | 59,632 | 412 | 614.0 | 3,041.5 |
| | `=0` | 3573f1645c4fc6e920d37c70a0033df7 | 59,632 | 595 | 782.4 | 4,457.5 |

All six md5 equal the required values. The `second pass` log line's counters (junction lines / kept / without barcode / unmatched / ambiguous / non-canonical / turned) are equal in both arms of every sample and equal to those of T2-chimdedup.md: pbmc5p_v1 3926694 / 27130 / 0 / 3483 / 64 / 2928198 / 21701; d300_5p 16886797 / 145563 / 0 / 60831 / 1369 / 12840905 / 131869; GSM6250953 17188238 / 4498881 / 0 / 123105 / 14517 / 9749577 / 4589683. The reads-into-the-pass lines are equal too (4944371 / 22372925 / 50768292). Effect on the item wall (one run per arm, no repeat; the A/B order was default first, then `=0`): pbmc5p_v1 -5.5 %, d300_5p -16.2 %, GSM6250953 -21.5 %; child CPU -24 %, -24 %, -32 %. The pbmc5p_v1 item wall changes less than the pass because the pass is a smaller part of it (the item wall includes the first pass). No "repeat" of these two arms, so a difference under about 4 % (the brief's run-to-run figure) would not be a finding; the differences above are larger than that.

### e. CPU-engine item (job 819080, `--engine cpu --chimeric`, pbmc5p_v1, 16 threads, g002)
default: table md5 `a5d71eb6a8b45889fe667bf57f17dce3` (24,141 rows), `=0`: `a5d71eb6a8b45889fe667bf57f17dce3` (24,141 rows); equal, counters equal to d. Pass (log epochs) 51 s against 61 s (the log prints `5e+01` / `6e+01`; epochs 1795160825 to 1795160876 and 1795161191 to 1795161252); job wall of the run 380 s against 375 s (`Elapsed` 6:20.40 / 6:15.02), i.e. the whole item is not shorter in this one pair of runs (the item is dominated by the CPU engine's first pass; the order of arms was default first).

## 4. Not done, failed, uncertain
- No unit test (reason in 3a). The identity of the loops is shown by verify mode over all 22.2 M reads of the two samples and by the raw-line md5s, not by randomised transcript sets with `minScoreToConsider` raised mid-loop.
- The gate was run once, on the default build with `--chimeric` not used; the gate does not reach the changed code (see 3b).
- The H100 items ran one job per sample group, one run per arm (no A/B/B/A for part d); only part c has the A/B/B/A order.
- The inner scan stays O(n^2) in the number of kept transcripts (per visited pair: a compare of `score1 + score2` against the threshold), by design, because the order of pairs must be kept. A further reduction (not done, not tested): skip runs of later transcripts whose scores are below `minScoreToConsider - score1` using a precomputed order that preserves the visiting order (for instance a per-block maximum), which would change no output; I did not implement it because the brief prescribed this design.
- The memory of `flSeg`, `flScore`, `flSufMax` is per thread and reused (capacity grows to the largest read: a ChimericSegment is 56 bytes; the largest read of GSM6250953 has 356 windows, a few thousand transcripts at most, estimate); the child's peak memory was not compared between arms.
- Verify mode's identity check compares the fields listed in section 1; it does not compare the full stitched `Transcript` contents beyond `maxScore` and `nExons` (the output lines, which print them, are compared by the md5s above).
- The R11 info line in the gate's fallbacks is the only fallback line (as before).
