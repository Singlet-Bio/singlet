# CPU engine speed: results

Branch `perf/cpu-speed` from gpu/frontier 99105bd8; `perf/cpu-speed-pick` carries the same eight code commits, without the bench
scripts, on gpu/frontier 654109ad. Timings come from Clipper CPU nodes (short-cpu, Xeon Gold 6230R), never H100s. Binaries are
CPU-only (`-DSINGLET_ENGINE_GPU=OFF`, `-march=x86-64-v3`, `-g`), built by `build.sbatch`. The A arm is `perf/cpu-speed-base`
(frontier plus these scripts). Paired runs alternate A and B on one node (`ab.sbatch`); intervals are 95% t intervals over the
pairs.

Every adopted commit is exact. At the tip 070d9839 the campaign's gate on CPU nodes (`gate.sbatch`, T=18, against
`ref-99105bd8/T18`) gives IDENTICAL on all nine samples (job 803920), and host ctest passes 161 of 161 (803919, one GPU test
skipped). The paired runs also hash every output against the first A run: only the known run-dependent files differ.

## Verdicts

| # | candidate | verdict | numbers |
|---|---|---|---|
| 1 | barcode-sharded streaming pileup | sharding kept off; two exact loop speedups adopted | the loop binds at T >= 16; with the speedups pbmc1k_v3 at T=16 runs 22% faster |
| 2 | CPU tail after mapping | adopted, six commits | gsm7064234s export 231 s to 108 s; GSM7656149 see below |
| 3 | distinct-read mapping | kept off | an unbounded cache lifts hits from 39% to 48% on GSM7656149 but saves 3.7% core-s and 1.4% wall |
| 4 | node packing | adopted as an operating recipe (no code) | 8 x T4 per 32-core node with a shared genome: 3.0x the samples per node-hour of one T32 job |
| 5 | TSO/adapter clip before STAR | rejected | already in place on both engines for 10x 3' (CR4 clip) |
| 6 | batched interleaved SA search | kept off | research only; the GPU covers mapping |

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
molecule collapse at the end of the stream ran on one thread after STAR had finished (676d7d0a, CPU path only). On gsm7064234s
(T=16, first four commits, two pairs) the export falls from 231 s to 108 s (CI -144 to -102 s) and the time to STAR's fork from
32.7 s to 18.0 s.

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
| ce5e40fc | 4a0eb780 | ambient: an intron-rich library fits only consensus member 0 |
| b90444c5 | 3aa9ccd5 | barcode universe on star_threads threads |
| 55a69df5 | 3affd0f4 | demux: restarts use the spare threads; K = 1 folds side by side |
| 481ffb2f | 3b837285 | non-host: no unmapped-mate pass when the EM confirmed no species |
| 5fe3f9d9 | 5229f56e | ambient: consensus members side by side, summed in member order |
| b58ade02 | 676d7d0a | molecule labelling on the finalize threads, same molecules in the same order |
| b1afd64c | b8ee0c75 | stream pileup: records decoded and CBs resolved on a reader thread |
| c04ff079 | 070d9839 | SNP pileup: first panel SNP found within its 1 kb bucket |

## Left for later

- On GPU ingest the barcode universe asks for R11's histogram before STAR's prefill has written it, so the .1fq is scanned a second
  time (135 s on GSM7064234, beside the 108 s decode). Computing the universe after the prefill would remove the scan.
- The post-stream SNP-consensus and junction-UMI finalizers are serial (33 s and 46 s on GSM7656149); their R9 gene-root cache is
  not thread-safe, so they need more than a parallel loop.
- gpu/frontier now runs a second non-host EM for the candidate table; the two EMs are independent and could run side by side.
