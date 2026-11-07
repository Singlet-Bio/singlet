# CPU engine speed: results

Branch `perf/cpu-speed` from gpu/frontier 99105bd8; `perf/cpu-speed-pick` carries the nine adopted code commits and a one-line build
fix, without the bench scripts, on gpu/frontier 37166d32. Timings come from Clipper CPU nodes (short-cpu, Xeon Gold 6230R), never H100s. Binaries are
CPU-only (`-DSINGLET_ENGINE_GPU=OFF`, `-march=x86-64-v3`, `-g`), built by `build.sbatch`. The A arm is `perf/cpu-speed-base`
(frontier plus these scripts). Paired runs alternate A and B on one node (`ab.sbatch`); intervals are 95% t intervals over the
pairs.

Every adopted commit is exact. At 070d9839 the campaign's gate on CPU nodes (`gate.sbatch`, T=18, against `ref-99105bd8/T18`) gives
IDENTICAL on all nine samples (job 803920), and host ctest passes 161 of 161 (803919, one GPU test skipped). At 0bf061ec, which adds
the collapse fix, ctest again passes 161 of 161 (803971) and the gate gives IDENTICAL on all nine samples (803968). The paired runs also hash every output against the first A run: only the known run-dependent files differ.

## Verdicts

| # | candidate | verdict | numbers |
|---|---|---|---|
| 1 | barcode-sharded streaming pileup | sharding kept off; two exact loop speedups adopted | the loop binds at T >= 16; with the speedups pbmc1k_v3 at T=16 runs 22% faster |
| 2 | CPU tail after mapping | adopted, seven commits | gsm7064234s export 231 s to 108 s; GSM7656149 see below |
| 3 | distinct-read mapping | kept off | an unbounded cache lifts hits from 39% to 48% on GSM7656149 but saves 3.7% core-s and 1.4% wall |
| 4 | node packing | adopted as an operating recipe (no code) | 8 x T4 per 32-core node with a shared genome: 3.0x the samples per node-hour of one T32 job |
| 5 | TSO/adapter clip before STAR | rejected | already in place on both engines for 10x 3' (CR4 clip) |
| 6 | batched interleaved SA search | kept off | research only; the GPU covers mapping |
| 7 | K5 profile with outFilterScoreMin 30 as the CPU default | already in place | frontier turns it on for 10x 3' v2/v3 whenever the CR4 probe passes |
| 8 | compiled FRN-10 resolver before winAnchorMultimapNmax 10 | not started | truth-gated and owned by the truth track; outside the D-103 window |
| 9 | early end of STAR's BAM stream (found by profiling) | kept off, unmeasured | STAR runs 26 s (pbmc1k_v3) to 185 s (GSM7656149) after its last record while the pileup waits |

**Where a CPU-only run spends its time.** Two single runs of the pick branch at T=18 (803974, 803975; adb942b6 on frontier 654109ad,
one run each, so treat them as shares, not timings):

| sample | reads | wall | core-s per M reads | before STAR | stream (STAR and pileup) | STAR tail after last record | finalize | non-host | export |
|---|---|---|---|---|---|---|---|---|---|
| pbmc1k_v3 | 66.6 M | 266 s | 32.7 | 8 s | 156 s | 26 s | 26 s | 19 s | 31 s |
| GSM7656149 | 263.4 M | 1727 s | 48.3 | 26 s | 935 s | 185 s | 238 s | 60 s | 283 s |

Mapping, with the pileup streaming beside it, is 59% and 54% of wall; with STAR's tail it is 68% and 65%, inside the review's 61-87%.
The ps sampler in ab.sbatch matched only threads named "singlet*", so frontier's role names (654109ad) hid the per-thread split; the
campaign's threadsampler.py reads them and is the tool for the next profile.

**1. Streaming pileup.** The serial stream loop is the ceiling of a CPU-only run at 16 or more threads. On GSM7656149 at T=16 its
thread was 84% busy while STAR's workers were about 70% idle on the pipe; on pbmc1k_v3 T=32 (434 s) was no faster than T=16 (448
s). Sharding the 900-line loop by barcode would lift the ceiling furthest, but it rewrites every accumulator path three weeks before
the freeze and does nothing for GPU jobs, whose records bypass the loop (E8). Two exact changes take work off the loop instead: a
reader thread decodes the records and resolves their CBs ahead of it (b8ee0c75), and each aligned block finds its first panel SNP in
a 1 kb bucket instead of binary-searching the chromosome (070d9839). On pbmc1k_v3 at T=16, over three pairs, the stream phase falls
from 334 s to 255 s and the run from 465 s to 362 s (-102 s, CI -126 to -79). Core-s rise 1.9%.

**2. CPU tail.** Profiles of GPU job 803428 (GSM7064234) and of CPU runs showed serial or idle-thread stages after mapping. Fixed:
the intron-rich ambient fit ran all 32 consensus members although the estimate is discarded (4a0eb780); the barcode universe ran on
one thread (3aa9ccd5); demux restarts ran on one thread each with spare threads idle (3affd0f4); the non-host step read both
unmapped mate files when no species was confirmed (3b837285); the 32 consensus members ran one after another (5229f56e); and the
molecule collapse at the end of the stream ran on one thread after STAR had finished (676d7d0a, CPU path only). The first version of that last change freed each group on the worker
that collapsed it; that memory belongs to glibc's main arena, so sixteen workers queued on its lock and the collapse took 101-137 s on
GSM7656149 against 79-81 s serial. 0bf061ec frees the groups together afterwards, and the collapse now takes 52 s (finalize 175 s
against 200-207 s). On gsm7064234s
(T=16, first four commits, two pairs) the export falls from 231 s to 108 s (CI -144 to -102 s) and the time to STAR's fork from
32.7 s to 18.0 s. On GSM7656149 (T=16, base against 070d9839, two pairs) the run falls from 1694 and 1584 s to 1332 and 1382 s;
the time to STAR's fork from 37 s to 18 s, the ambient step from 82 s to 47 s and demux from 156 s to 104 s. In one more pair on that
sample (803969) the collapse fix runs in 1297 s against 1359 s for 070d9839, its collapse taking 52 s against 129 s; the outputs
match except the run-dependent logs (the CR4 probe's STAR logs among them).

**3. Distinct reads.** SeqCache at its 4 GB default hits 39.3% of GSM7656149's 262 M lookups; at 64 GB, where it never evicts, it
hits 48.4%, which is the exact-duplicate share. A whole-sample distinct pass can save no more than that unbounded cache, which
cut core-s by 3.7% and wall by 1.4% (one run each, T=16), and the cache variant costs 60 GB per job.

**4. Packing.** `singlet genome load` once, then jobs with `--genome-shared on`: each pbmc1k_v3 job holds about 10 GB private
(STAR 3.1 GB, parent 5.5 GB, probes) beside one 25 GB genome. Eight T=4 jobs on 32 cores finished in 1163 s, 24.8 samples per
node-hour against 8.3 for one T=32 job at a time. Contention raises consumed core-s per M reads from 45.5 (T=4 alone) to 56.6, but
allocated core-s per M reads fall from 108 (T=16) and 208 (T=32) to 70. CPU-only samples should run this way.

**5. Clipping.** The CR4 clip already runs on CPU STAR and in the GPU's k_cr4 for 10x 3' v2/v3, with identical results. Further
clipping changes alignments on both engines and would need the truth gate (M1-M7); it is out of scope before the freeze.

**6. SA search.** Seed search is 48% of STAR's CPU and memory-latency bound; per-read prefetch did not help before, and the GPU
maps the campaign's large samples. Left as research.

## Adopted commits (cherry-pick `perf/cpu-speed-pick`, oldest first)

| pick | perf/cpu-speed | change |
|---|---|---|
| 2a559fc4 | 4a0eb780 | ambient: an intron-rich library fits only consensus member 0 |
| e0a9a26c | 3aa9ccd5 | barcode universe on star_threads threads |
| 330263d7 | 3affd0f4 | demux: the K = 1 folds side by side (the restart half landed on frontier as 8812402c) |
| ada212a6 | 3b837285 | non-host: no unmapped-mate pass when the EM confirmed no species |
| e78cee94 | 5229f56e | ambient: consensus members side by side, summed in member order |
| d9ddefae | 676d7d0a | molecule labelling on the finalize threads, same molecules in the same order |
| 57d98a62 | b8ee0c75 | stream pileup: records decoded and CBs resolved on a reader thread |
| b629310b | 070d9839 | SNP pileup: first panel SNP found within its 1 kb bucket |
| f309a41c | 0bf061ec | molecule labelling: the groups freed after the parallel collapse, on one thread |
| 5a20f3ac | | build: singlet_star finds singlet/thread_name.h in a CPU-only build (frontier 9f7ca5d7 broke `-DSINGLET_ENGINE_GPU=OFF`) |

The commits were gated on the 99105bd8 base. The rebased branch on 37166d32 applied cleanly apart from the demux overlap, but it was
not built or gated there: its build and gate (804107-804109) were cancelled when the session wound down for the freeze.

## Left for later

- On GPU ingest the barcode universe asks for R11's histogram before STAR's prefill has written it, so the .1fq is scanned a second
  time (135 s on GSM7064234, beside the 108 s decode). Computing the universe after the prefill would remove the scan.
- The post-stream SNP-consensus and junction-UMI finalizers are serial (33 s and 46 s on GSM7656149); their R9 gene-root cache is
  not thread-safe, so they need more than a parallel loop.
- The early end of STAR's BAM stream (perf/cpu-speed fcfd3965) closes the pipe's write end once the BAM is closed, as E16 "eof"
  already does at GPU level 5, so the pileup finalizes while STAR frees memory, writes SJ.out.tab and counts Solo. Expected saving is
  STAR's tail, 26 s on pbmc1k_v3 and 185 s on GSM7656149; its A/B and gate (804099, 804100, 804098) were cancelled before they ran.
