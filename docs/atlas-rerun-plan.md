# Atlas re-run plan: fixed pipeline, every run, before the v2 migration

Status: plan, 2026-10-24, branch `plan/atlas-rerun`. No production jobs were submitted for it. Only measurement probes ran:
- ENA run reports;
- download benchmarks;
- a depth probe on three deep samples;
- a performance matrix with output-equivalence checks.

**Route chosen (Zach, 2026-10-24): Clipper only, with the pipeline optimized first.**

Code and inputs are in `bench/atlas_rerun/`.

Decision (Zach, 2026-10-24): re-run the atlas with the fixed pipeline (#22 + #25). Use every SRR and all reads, or a documented
cap sampled across runs, before the v2 migration. Plan it first: cost on Clipper vs Anvil, schedule and order, with empty
bundles and strand-affected samples first.

## Summary

- **What is re-run: 70,558 GEX samples.** Median 250M read pairs per sample, and 42% have more than one run. All reads of all
  runs is 21.7 trillion pairs, 13× what the 2026 campaign read.
- **Optimized pipeline, identical outputs.** Four changes:
  - the download moves out of the compute job;
  - 4-thread jobs;
  - a determinism fix (now in #22);
  - one shared genome per node.

  Together they cut core-hours **1.5–1.7×** (1.8–2.3× if the non-host screen is deferred). Every count matrix is byte-identical
  to the present setting.
- **On Clipper the network sets the calendar, not the cores.** Downloads into Clipper top out at ~150 MB/s (~13 TB/day) from
  NCBI and from AWS alike.

| depth | present core-h | optimized core-h (screen / no screen) | Clipper calendar, optimized, 300 cores |
|---|---|---|---|
| full | 1,091k | 624k / 475k | ~145 days |
| 200M cap | 561k | 357k / 282k | ~70 days |
| 100M cap | 328k | 217k / 175k | ~39 days |
| 50M cap | 189k | 130k / 108k | ~21 days |

  Add ~1 week of preparation (merges, I/O stage, EmptyDrops, validation run).
- **A 200M cap keeps 95–98% of cells and 63–77% of UMIs per cell** of full depth on the two deep samples probed. A
  head-of-run cap is nearly unbiased (−2% UMIs).
- **Modalities:**
  - gene expression and ADT are equivalent to STARsolo / Cell Ranger, and gene expression to scBaseCount at equal depth;
  - cell calling and strand are worse today; `called_v2` and #25 fix them in this re-run;
  - the library-type screen is better than scBaseCount's;
  - splice junctions, mtDNA, donor demultiplexing, doublets, ambient RNA and non-host are not benchmarked yet (§8).


## 1. What is re-run

**Universe.** Two sources are combined:
- the 74,269 published atlas samples (`/api/gsm` status DONE) in the 8,306 v2 plan studies (`plan-09fd92b149`);
- the 5,031 GEO samples the catalog skipped as `skip_plate_based`, in which scBaseCount finds valid 10x barcodes
  (`docs/scbasecount-comparison.md`).

That is 79,300 samples. The library-type screen (`docs/library-type-screen.md`, high and medium confidence) then routes them:

| route | samples | what happens |
|---|---|---|
| gene expression (GEX) | **70,558** | full pipeline (this plan) |
| ADT / HTO / CMO libraries | 4,010 | ADT modality (PR #29), not the GEX pipeline |
| V(D)J, guide, ATAC, other DNA | 4,732 | excluded from GEX; listed in the study manifest with the class |
| mRNA siblings not yet processed | 710 | added to the GEX queue, in the wave of their study (depth not yet looked up) |

**Waves** (GEX samples):

| wave | samples | why first |
|---|---|---|
| a. studies with empty matrices in the published bundle (557 studies, `bench/scbasecount/empty_bundle_suspects.tsv`) | 5,276 | they serve nothing today |
| b. strand re-run list from #25 (`docs/strand-detection/rerun_samples.csv`, minus wave a) | 6,272 | today's counts are antisense noise |
| c. skipped 10x samples (scBaseCount comparison) | 5,030 | new coverage; median 6.4k cells each in scBaseCount |
| d. everything else, by study size (total cells), largest first | 53,980 | most cells corrected per hour |

Wave a overlaps the strand list; a sample is counted in its first wave. Wave d could instead be ordered by user traffic once
the API logs are joined. Cells per study is a stand-in for value until then.

## 2. Inputs per sample: runs and depth

ENA's run report was fetched for every SRA project behind the plan studies, plus every run of every SRX/SRR the atlas lists
(`fetch_runinfo.py`, `fetch_missing.py`). It covers 79,093 of the 79,300 samples (246,287 + 41,934 runs). The 207 not found
are counted at the median depth.

GEX samples (70,558):

| | p10 | p25 | median | p75 | p90 | p99 |
|---|---|---|---|---|---|---|
| runs per sample | 1 | 1 | 1 | 2 | 4 | 16 |
| read pairs per sample (all runs) | 29M | 109M | **250M** | 425M | 625M | 1.29B |

- **42% of GEX samples have more than one run.** The old campaign read only the first. 3' 10x libraries are often sequenced
  over several lanes or flow cells, so the first run can be anything from a quarter to all of a sample.
- **The old campaign used a median 15% of each sample's reads** (`docs/scbasecount-comparison.md`), and 82% of samples used
  less than half.
- **Full depth is 21.7 trillion read pairs** for the GEX universe, against ~1.7 trillion read in the 2026 campaign.
- **Per-run library check.** 497 GEX samples have runs whose read lengths differ more than 1.5× between runs. These are
  candidates for a GSM that holds several libraries (for example GEX and ADT in one GEO sample). Their runs must not be
  pooled blindly; see the gates in §6.

### What a cap costs in counts

`probe_depth.sbatch` ran STARsolo `GeneFull` with EmptyDrops_CR at 1/16 to all of the reads of two deep human 3' v3 samples.
Reads were taken as every k-th pair across all runs.

| sample | read pairs used | cells | median UMIs/cell | saturation |
|---|---|---|---|---|
| GSM5815984 (adrenal, 2 runs) | 31M | 9,527 | 1,814 | 3% |
| | 61M | 9,675 | 3,471 | 7% |
| | 122M | 9,772 | 6,461 | 13% |
| | 245M | 9,926 | 11,327 | 23% |
| | **489M (all)** | **10,126** | **18,089** | 37% |
| GSM7666314 (lymphoma, 2 runs) | 22M | 5,160 | 1,909 | 8% |
| | 45M | 5,481 | 3,406 | 14% |
| | 89M | 5,798 | 5,681 | 26% |
| | 179M | 6,326 | 8,360 | 42% |
| | **358M (all)** | **6,663** | **10,912** | 60% |

Deep GEO libraries are far from saturation, so UMIs per cell keep rising with depth:

| cap | cells, share of full depth | UMIs per cell, share of full depth |
|---|---|---|
| ~200M | 95–98% | 63–77% |
| ~100M | 87–96% | 36–52% |

**A cap mostly costs counts per cell, not cells.**

Taking the *first* quarter of each run instead of every 4th pair gave the same cells (5,778 vs 5,798) and 2% fewer median
UMIs. So a head-of-run cap, the only kind that saves download, is nearly unbiased. The third probe sample (GSM5068934) was
still downloading when this was written.

## 3. Where the time goes, and what was optimized

All measurements are on PBMC 3' v3 (66.6M read pairs) with Singlet's campaign flags, 16 reserved cores, each config run twice.
Two CPU types took part: Xeon Gold 6230R and Xeon 6730P. Code: `bench/atlas_rerun/perf_run.sbatch`, `perf_summary.py`,
`perf_equiv.py`, `dl_bench.sbatch`, `dl_equiv.sbatch`.

**One sample at 4 threads (1,120 s):**

| stage | seconds |
|---|---|
| STAR mapping (274M reads/h) | 718 |
| BAM sort | ~100 |
| STARsolo counting | ~45 |
| genome load (from page cache) | 14 |
| reference load | 30 |
| BAM index | 18 |
| barcode refresh | 25 |
| pileup | 42 |
| export | 14 |

STAR is 87% of the time. Singlet's own stages (pileup, export) are 5%, so there is little to gain by optimizing Singlet's code.

**Outside that run, the campaign had two more costs:**
- **downloading:** a remote `singlet download` at a median 0.065M reads/s (campaign logs), about 15 s per million reads,
  inside the same 4-core job;
- **the non-host screen** (viral / bacterial / fungal), on by default.

"Identical" below means the same matrices, byte for byte, as the present setting (`perf_equiv.py`: every count matrix, the
barcode list and the stats JSON).

| change | effect, per sample | outputs vs present | used in quote |
|---|---|---|---|
| **download in a separate 1-core stage**: `prefetch` then `singlet download <local .sra>` (1M reads/s) instead of remote VDB inside the 4-core job | removes ~15 s/M of idle 4-core time: **−45% of core-hours at full depth**. Download of one run: 163 s vs 1,054 s | decoded reads **identical** (`dl_equiv`) | yes |
| **4 threads per sample** (vs 8 or 16) | core-h per sample 1.24 (4 threads) vs 1.59 (8) vs 1.85–2.41 (16) | identical after the determinism fix below | yes |
| **determinism fix** (`umi_dedup.h`, pushed to #22) | none | the first fix gave slightly different feature-level matrices at 8/16 threads (0.13% of exon-feature entries, same totals); now **byte-identical** at 4/8/16 | yes |
| native (`-march=native`) build | −2% (within noise) | identical | optional |
| genome in shared memory (`--genome-shared`) | −1% time; one genome per node instead of one per job | identical | yes, for packing |
| 4 samples on 16 cores (realistic packing) | +15% per sample vs one alone (memory bandwidth) | identical | yes (×1.15) |
| **non-host screen off** | −27% time (1,120 vs 1,420 s), −35 GB RAM | non-host members absent | decision (§9) |
| non-host indices cached in `/dev/shm` | no gain: the cost is screening 2.6M unmapped reads, not loading | identical | no |
| disk-backed BAM sort (`--bam-sort-mode disk`) | none: 1,113 s vs 1,120 s; peak RSS 18.6 GB instead of a sort buffer sized to the job's memory | identical | yes |
| STARsolo counting off + streaming pileup (`CB_samTagOut`, no sorted BAM) | would save ~15–18% of time | **not equivalent**: exact-only barcode matching, and a different stored barcode list | no (future work) |
| cascade aligner (transcriptome first) | — | not equivalent by design (skips introns, junctions and mtDNA for routed reads) | no |

**Memory is the hidden throughput limit.**
- A present-style job holds the 30 GB genome, STAR's in-RAM sort buffer (sized from the job's memory; the 122M-pair probe was
  killed at 48 GB) and, with the screen on, 35 GB of non-host indices.
- The campaign requested 80 GB per 4-core job. On Clipper's nodes (40–64 cores, 0.5–0.77 TB) that leaves 25–60% of cores idle.
- **Lean packing** (one shared genome per node + disk sort): 4 concurrent samples each peaked at 20 GB RSS, an upper bound since
  it counts the shared genome pages. They ran at 1,338 s per sample (+19% vs one alone, in line with the ×1.15 packing factor),
  with all 4 outputs identical. A node then needs ~30 GB once plus ≤20 GB per 4-core job, so every core of every Clipper node
  can be used (a 64-core / 512 GB node: 16 jobs in ~350 GB).
- With the non-host screen on, each job adds ~35 GB of indices (read into private memory today). That fits on the 768 GB nodes
  but leaves ~40% of a 512 GB Granite Rapids node idle. Memory-mapping the indices (`MinSketchIndex::load`, one shared copy per
  node) would remove that; it is a small code change, not yet made.

**Network is the hard limit on Clipper.** Aggregate download into Clipper measured 116–160 MB/s, whatever the source:

| test | aggregate |
|---|---|
| NCBI `prefetch`, 8 streams on one node | 125 MB/s |
| NCBI `prefetch`, 16 streams on one node + 3 nodes × 4 streams | ~160 MB/s |
| AWS S3 (`sra-pub-run-odp`), 16 streams | 116 MB/s |
| ENA HTTPS | 2.5 MB/s |

That is ~1.2 Gbit/s, or **~13 TB/day**, and it caps every Clipper-only schedule below. Two things could lift it:
- a 24-hour test at 32 streams;
- asking GVSU IT whether Clipper has (or can get) a Science-DMZ path. A 10 Gbit/s path would make the network irrelevant.

SRA Lite (no base qualities, about half the bytes) was not available through the paths tested. It would also change the
quality-filtered mtDNA / SNP calls.

## 4. Quote: Clipper only, present vs optimized

Model (`bench/atlas_rerun/cost_model_v2.py`), from the measurements above. Per sample, with *U* = read pairs used
(millions):
- compute wall = 300 s + 12.3 s × *U*, or 16.8 s × *U* with the non-host screen;
- × 1.15 for packing, at 4 threads;
- present: + 15 s × *U* of in-job remote download;
- optimized: a 1-core I/O stage;
- + 20% retries;
- download volume = *U* × bases per pair × 0.44 B/base, at ~150 MB/s aggregate.

Calendar days are max(compute at the given cores, download at 150 MB/s).

| depth | pipeline | core-hours | download | download days | calendar, 200 cores | calendar, 300 cores | calendar, 400 cores |
|---|---|---|---|---|---|---|---|
| full | present | 1,091k | 1.9 PB | 145 | 227 | 152 | 145 |
| full | **optimized** | 624k (screen) / 475k (no screen) | 1.9 PB | 145 | 145 | 145 | 145 |
| 300M | present | 734k | 1.2 PB | 95 | 153 | 102 | 95 |
| 300M | **optimized** | 456k / 357k | 1.2 PB | 95 | 95 | 95 | 95 |
| 200M | present | 561k | 0.9 PB | 70 | 117 | 78 | 70 |
| 200M | **optimized** | 357k / 282k | 0.9 PB | 70 | 74 / 70 | 70 | 70 |
| 100M | present | 328k | 0.5 PB | 39 | 68 | 46 | 39 |
| 100M | **optimized** | 217k / 175k | 0.5 PB | 39 | 45 / 39 | 39 | 39 |
| 50M | present | 189k | 0.3 PB | 21 | 39 | 26 | 21 |
| 50M | **optimized** | 130k / 108k | 0.3 PB | 21 | 27 / 23 | 21 | 21 |

The "present" calendar assumes every reserved core is used. With present memory requests (80 GB per 4-core job) that is not
possible on Clipper's nodes, so the true present calendar is 1.3–2.5× longer than shown.

**How to read it:**
1. **Optimizing cuts core-hours 1.5–1.7× at equal outputs, or 1.8–2.3× with the screen off.** The largest single gain is
   moving the download out of the compute job.
2. **On Clipper the calendar is set by the network, not the cores.** With the optimized pipeline, 300 cores keep up with a
   150 MB/s link at every depth, with or without the screen. Full depth means ~5 months of downloading, a 200M cap ~10 weeks,
   100M ~6 weeks, 50M ~3 weeks.
3. **A faster link changes the answer.** At ~1 GB/s (a Science-DMZ path), compute sets the pace again: a 200M cap in ~7 weeks
   at 300 cores (357k core-h with the screen), full depth in ~3 months.
4. Capped samples stream only their share of each run (remote reads, ~5 MB/s per 1-core stream, ~30 streams to fill the
   link). Uncapped samples are fetched whole and converted locally.

Clipper capacity, for reference:
- 576 cores on 12 `cpu` nodes (6 × 40 + 4 × 52 + 2 × 64; the two 64-core Granite Rapids nodes are ~22% faster);
- 192 on 4 `bigmem` nodes (1.5 TB each);
- preemptible `scavenger` GPU nodes.

200–300 cores sustained is realistic without displacing the lab.

## 5. Order of work and schedule (Clipper)

| step | what | duration | gate |
|---|---|---|---|
| 0 | merge #22 (now deterministic) and #25; add the I/O stage, provenance and EmptyDrops calls (§6); pin the binary | ~1 week | benchmark numbers reproduced; 4/8/16-thread outputs identical |
| 1 | 24-hour bandwidth test (32 streams) and a question to GVSU IT about the uplink | 1–2 days, in parallel with step 0 | sets the real calendar |
| 2 | 50-sample validation run | 1 day | matches the benchmark; strand right on wave-b samples; EmptyDrops recall vs authors ≥ 0.95 |
| 3 | wave a (5,276) → b (6,272) → c (5,030) | ~2.3 weeks at 200M, ~1.3 at 100M (download-bound) | per-study QC and republish as studies finish |
| 4 | wave d (53,980), largest studies first | the rest of the calendar in §4 | rolling per-study QC, v2 publish |

Waves a–c are 23% of the GEX samples and of the bytes, so they finish in about a quarter of the total calendar.

**Within each wave:**
- samples go in study order, so a study is republished as one unit;
- two job types per sample: a 1-core I/O job (download, per-run screen, encode to node-local `.1fq`) and a 4-core compute job
  (align + pileup → export → EmptyDrops → results and a ledger row), chained with `--dependency`;
- the I/O pool is sized to fill the link (~30 streams), the compute pool to the cores free;
- failures are retried once, then parked in a ledger for triage.

## 6. Prerequisites and gates

1. **#22 and #25 merged with green CI.**
   - #22 is the molecule-label fix.
   - #25 is the strand fix. Its CI is its first real Python run.
   - Re-run `bench/pipeline` on the merged `main` and require the §4 numbers of `pipeline-validation.md` (exon vs STARsolo
     `Gene` ratio 0.97–1.02; worker invariance) plus the 5' sample correct without `--reverse-strand`.
2. **Cell calling: EmptyDrops**, not STARsolo's CellRanger 2.2 knee. Either option works:
   - STARsolo `--soloCellFilter EmptyDrops_CR` in the alignment step;
   - `called_v2` (`docs/called-v2.md`) on the stored counts, which needs no extra pass and keeps every barcode.

   Prefer `called_v2` with the plateau `lower`. It is validated on 42 samples and handles the low-RNA blind spot. Store the
   per-barcode FDR.
3. **Per-run screen before pooling runs** (borrowing scRecounter's tests):
   - on 1M reads of each run, test barcode whitelists (3' v2/v3/v4, 5', Multiome GEX) and both strands;
   - keep runs whose best whitelist matches ≥30% of reads, with the same chemistry as the sample's majority;
   - drop runs whose R2 looks like a feature barcode or V(D)J (`docs/library-type-screen.md` signatures);
   - record kept and dropped runs.

   Pooling all runs blindly would mix GEX with ADT/HTO libraries in GSMs that hold both (497 samples with mismatched run read
   lengths). This screen also corrects the 1,961 5'-labelled-3' samples at the source.
4. **Cap implementation.** The cap is "the same share of every run":
   - each run contributes reads ∝ its size, up to *cap* in total;
   - taken from the head of each run (§2 checks head vs uniform);
   - record `reads_available`, `reads_used`, `runs_used` per sample.
5. **Provenance in every bundle** (the drift finding of `pipeline-validation.md` §1): engine git commit, binary sha256,
   reference build, cap, runs used, worker count, cell-call method and FDR. The 2026 bundles do not say which engine wrote them.
6. **Publish gates** (from `docs/scbasecount-comparison.md`):
   - a sample whose matrices have nnz = 0 while `n_cells > 0` does not publish;
   - strand QC (wrong-strand share < 0.5) is required;
   - library-type flags route to the ADT modality or to exclusion.
7. **Ledger.** One row per sample: status, reads, runs, chemistry, strand, cells, timing, cluster, retries. It lets the
   campaign be resumed, priced and audited. The 2026 campaign's `processing_status` and API status disagree in places, and
   the 557 empty studies slipped through.

8. **Pinned, deterministic binary.** One binary (git commit + sha256) for the whole campaign. Before wave a, check that 4-, 8-
   and 16-thread runs of the validation samples give byte-identical matrices (`perf_equiv.py`). They do for PBMC 3' v3 since
   the determinism fix.
9. **I/O stage.** A separate 1-core download job per sample: `prefetch` + local decode for uncapped samples, a streamed
   head-of-run read for capped ones. Check that its decoded reads equal the present path (`dl_equiv.sbatch`).
10. **Memory sizing.** Request memory from the measured peak of the chosen settings (§3), not the campaign's 80 GB, so whole
    nodes can be used.

## 7. Hand-off to the v2 migration

**Write v2 sample objects directly from the re-run, per study, and skip the legacy-bundle → migrate path for re-run samples.**

**Why:**
- the v2 migration was built to repackage legacy bundles losslessly (verify-or-store recipes, D3/D15 in
  `docs/v2-migration.md`). Its main job, bit-equivalence with the legacy members, does not apply to new counts;
- writing legacy bundles first would double the I/O for 70k samples;
- the publisher already emits v2 only (D11): the pipeline writes TP1Z members, `pack_gse` / `scripts/v2_publish.py` packs
  them;
- the re-run's per-sample result directories are exactly that input.

**So:**
1. pipeline → per-sample `.1pz` members, including the new `ambiguous_counts`, provenance and EmptyDrops calls;
2. the v2 packer builds the sample objects and the study manifest as soon as a study's samples are all done or parked;
3. upload and swap the manifest, which is atomic per study;
4. the legacy bundle of that study is archived under D12.

**Before the first wave publishes:**
- **Schema changes for the v2 packer and readers:**
  - `ambiguous_counts` as a member;
  - the provenance block;
  - `called_v2` with FDR;
  - `reads_used` / `reads_available`.
- **What happens to legacy-only studies.** Studies outside the re-run, for example a study whose every sample is excluded,
  keep the lossless migration path.
- **Which run is authoritative.** A study is never half old, half new in one manifest. Studies wait in the queue until all
  their samples are done or parked.

**Order relative to the v2 campaign:** run the re-run first, then v2 publication per study as it completes, as Zach decided.
The v2 lossless migration of the untouched legacy bundles is only needed for studies the re-run cannot reach (missing SRA
data, failed screens). That is a much smaller campaign.

## 8. Is every modality at state-of-the-art?

No, not yet. Gene expression is benchmarked and, after the fixes, matches the reference tools. Most other per-cell modalities
have never been compared with an established tool. Status per modality, with the evidence:

| modality / step | reference tool(s) | status | evidence |
|---|---|---|---|
| gene counts, exonic | STARsolo `Gene`, Cell Ranger | **equivalent** after #22: per-cell ratio 0.97–1.02, per-cell gene r 0.96–0.99 | `pipeline-validation.md` §4.3 |
| gene counts, exon + intron | STARsolo `GeneFull`, alevin-fry USA | **equivalent within tool spread**: +2% to +10% vs GeneFull, alevin-fry −3% to +18% | same |
| vs scBaseCount | scBaseCount (STARsolo, all reads) | **equivalent at equal depth**: within its own thinning noise (0.01–0.04). The published atlas is not equivalent (truncated depth); the re-run fixes that | `scbasecount-comparison.md` §3 |
| spliced / unspliced / ambiguous | STARsolo Velocyto, alevin-fry | unspliced **agrees** (ratio 0.96–1.08). Spliced vs ambiguous **differs by definition**: collapsed gene model vs per-transcript | `pipeline-validation.md` §4.4 |
| strand / chemistry detection | scRecounter per-sample tests | **worse today** (17% wrong strand). #25 fixes it; it is validated in #25 but not on the whole atlas yet | `strand-detection.md` (#25) |
| cell calling | EmptyDrops_CR, Cell Ranger | **worse today** (knee; 73% of the authors' cells). `called_v2` (EmptyDrops on stored counts) is validated on 42 samples and must be in the re-run | `called-v2.md`, `empty-droplets.md` |
| library type (GEX vs ADT / HTO / V(D)J …) | none in scBaseCount | **better than scBaseCount** once applied: the screen exists, scBaseCount ships ~1.6k non-GEX SRX as GEX | `library-type-screen.md` |
| ADT / HTO counts | Cell Ranger feature barcoding | **equivalent**: UMI ratio 1.0002, r 0.999995 on pbmc_1k_protein_v3 | PR #29 |
| splice junctions, PSI | STARsolo `SJ`, leafcutter / BRIE-style PSI | **not benchmarked**. The legacy PSI member had a 0/1 bug; true PSI is computed client-side | `v2-migration.md` D8 |
| mtDNA heteroplasmy / variants | mgatk, MQuad | **not benchmarked** | — |
| donor demultiplexing (genotype-free) | vireo, souporcell | **not benchmarked** | — |
| doublets | scDblFinder, barnyard truth | **not benchmarked**. The barnyard mixed-species rate (0.1%) reflects conservative calls, not doublet detection | `pipeline-validation.md` §4.7 |
| ambient RNA | SoupX, CellBender | **not benchmarked** (only the empty-droplet study's CellBender comparison of calls) | `empty-droplets.md` §9 |
| V(D)J usage from GEX reads | none (V(D)J libraries use Cell Ranger vdj) | **no reference**; a by-product, not a V(D)J assay | — |
| non-host (viral / microbial) | Kraken2 / sylph | **not benchmarked** | — |
| sex, ancestry, cell cycle | XIST/RPS4Y1 rules, Seurat CellCycleScoring | **not benchmarked** | — |

So today Singlet can claim, with evidence:
- gene-expression counts equivalent to STARsolo / Cell Ranger and, at equal depth, to scBaseCount;
- ADT equivalent to Cell Ranger;
- a library-type screen scBaseCount lacks.

After the re-run, with #25 and `called_v2`, it can also claim strand handling and cell calls on a par with scBaseCount.

**Not yet claimable:** splice junctions/PSI, mtDNA, donor demux, doublets, ambient and non-host. Each needs a benchmark like
`pipeline-validation.md`'s on data with a known answer before it is advertised. Mixed-genotype pools, mito-variant lineage
datasets and spike-in viral samples all exist publicly. These benchmarks do not block the GEX re-run, since the re-run
regenerates these modalities anyway, but they should finish before the modalities are marketed.

## 9. Decisions needed

1. **Depth cap.**
   - **200M (recommended): ~10 weeks on Clipper.** It keeps nearly all cells and two-thirds or more of UMIs.
   - 100M: ~6 weeks, keeps 36–52% of UMIs.
   - Full depth: ~5 months at today's bandwidth.
   - A faster uplink (decision 3) would make full depth ~3 months.
2. **Non-host screen during the re-run.**
   - It adds ~25% core-hours and 35 GB per job.
   - At ≥300 cores it does not lengthen the calendar, which is set by the download.
   - Running it later would mean downloading everything again.
   - **Recommended: keep it on if ~300 cores are available, off if the campaign gets fewer.**
3. **Ask GVSU IT about Clipper's uplink and run a 24-hour 32-stream bandwidth test.** This is the largest lever on the
   calendar. Nothing else in this plan moves it.
4. **Memory-lean packing** (shared genome, disk sort; optionally memory-mapped non-host indices, a small code change): adopt it
   if §3 shows equal outputs at a tolerable time cost.
5. **Anvil** stays out, as decided. One note: its faster network is the other way past the Clipper bandwidth ceiling.

## Reproducing

| step | script | where |
|---|---|---|
| catalog and plan inspection | `inspect_catalog.py` / `.sbatch` | Clipper |
| ENA run report per project | `fetch_runinfo.py` / `.sbatch` | Clipper |
| runs for samples the project reports missed | `fetch_missing.py` / `.sbatch` | Clipper |
| universe, waves, routes | `build_universe.py` | anywhere, on the pulled tables |
| cost per option (first draft, Anvil-oriented) | `cost_model.py` | anywhere |
| Clipper quote, present vs optimized | `cost_model_v2.py <universe.parquet>` | anywhere |
| performance matrix | `perf_encode.sbatch`, `perf_run.sbatch <config> <dataset> <rep>`, `perf_summary.py` | Clipper |
| output equivalence between configs | `perf_equiv.py` / `.sbatch` | Clipper |
| download paths and aggregate bandwidth | `dl_bench.sbatch <mode> <SRR,...>`, `dl_equiv.sbatch` | Clipper |
| download timing | `probe_download.sbatch <GSM> <SRR,...>` | Clipper |
| depth saturation | `probe_depth.sbatch <GSM>` (array K = 1, 2, 4, 8, 16) | Clipper |

Tables live under `/mnt/projects/debruinz_project/debruinz/singlet/data/atlas-rerun/`.
