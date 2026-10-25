# Atlas re-run plan: fixed pipeline, every run, before the v2 migration

Status: plan, 2026-10-24, branch `plan/atlas-rerun`. No production jobs were submitted for it. Only measurement probes ran:
- ENA run reports;
- download benchmarks;
- a depth probe on three deep samples;
- a performance matrix with output-equivalence checks.

**Decisions (Zach, 2026-10-24/25):**
- **All reads of every run, unconditionally; no cap.**
- Clipper is the default. The pipeline is optimized first.
- Anvil is quoted (§4); no SUs spent.

Code and inputs are in `bench/atlas_rerun/`.

Decision (Zach, 2026-10-24): re-run the atlas with the fixed pipeline (#22 + #25). Use every SRR and all reads, or a documented
cap sampled across runs, before the v2 migration. Plan it first: cost on Clipper vs Anvil, schedule and order, with empty
bundles and strand-affected samples first.

## Summary

- **What is re-run:** 70,558 gene-expression samples, **all reads of all runs**. That is 21.7 trillion read pairs (median 250M per sample), 13× what the 2026 campaign read.
- **Pipeline speed.** Every change below gives outputs byte-identical to stock STAR + the fixed engine:
  - separate download stage;
  - 4-thread jobs;
  - shared genome;
  - disk sort;
  - determinism fix;
  - **STAR acceleration (PR #34):** sequence cache, lazy window table, huge pages, PGO.

  Mapping is 1.54× faster and a whole sample 1.35× faster than #22's binary. Together with the job-shape changes, **core-hours fall from 1,091k (present) to 316k (no non-host screen) / 466k (with it)**. The sequence cache gains more at full depth: 66% hits on a 489M-read sample, against 48% on PBMC.
- **Calendar and cost, all reads:**

| | compute | download | binding limit |
|---|---|---|---|
| Clipper, 300 cores | 44–65 days | **~145 days** at ~150 MB/s | network |
| Anvil, whole nodes | ~365k–535k SU (3.3–4.8× the 112k balance) | not measured; Anvil's network is faster | SU balance |

- **Production binary (PR #31).** Cell calling is `called_v2` (EmptyDrops on exon + intron, plateau `lower`). It was ported to C++ and matches Python exactly on 10,349 barcodes. Strand is correct for 5' libraries without flags, and a per-run library screen and provenance are in. A fix round (`prod/modality-fixes`) is in progress for the problems the modality benchmarks found (§8).
- **Modalities, against established tools on data with a known answer (§8):**
  - **Equivalent:** gene counts, ADT, junctions/PSI, non-host.
  - **Better:** donor demux with a given donor count.
  - **Worse or missing, being fixed:** doublets, ambient, automatic donor count, the mtDNA variant list.
  - **Biggest single fix:** a false 5' clip that trimmed 11 bp from every read of some 3' samples.
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

### What depth buys (why all reads)

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

**Depth mostly adds counts per cell, not cells.** A 200M cap would have kept 63–77% of the UMIs per cell. The re-run uses all reads.

Taking the *first* quarter of each run instead of every 4th pair gave the same cells (5,778 vs 5,798) and 2% fewer median
UMIs. So if bandwidth ever forces a cap on a few extreme samples, a head-of-run cap is nearly unbiased.

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
| **STAR acceleration (PR #34)**: sequence→alignment cache (2 GB), lazy window table, huge pages, faster read parsing, PGO | mapping 793 → 515 s, whole run 1,138 → 845 s (Xeon 6226); 4-way packed batch 1.24× faster. Cache hits 44–51% on PBMC (1–4 GB), 66% simulated at 489M reads | all 60.7M BAM records and every matrix **identical to stock STAR**. The adjacent-read dedup in #22 changed ~300 reads (wrong `trBest` on a hit); removed | yes |
| transcriptome-first step (projection of single-gene, annotated-junction reads) | 38.5% of mapped reads qualify on PBMC (`bench/aligner/txome_eligible.py`); est. ≤1.3× more mapping speed, overlapping the cache | identical by construction if built as specified | not built (§9) |
| STARsolo counting off + streaming pileup (`CB_samTagOut`, no sorted BAM) | would save ~15–18% of time | **not equivalent**: exact-only barcode matching, and a different stored barcode list | no (future work) |
| existing cascade aligner (`txome_aligner.h`) | — | a scaffold: never wired into the read path, stats hard-coded, seeding matches ~1 in 22 reads | no |

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

## 4. Quote: all reads, Clipper and Anvil

Model: `bench/atlas_rerun/cost_model_v2.py`, from the measurements in §3. Per sample:
- compute wall = 300 s + 12.3 s × *U* without the non-host screen, or 16.8 s × *U* with it;
- STAR mapping (10.8 s/M of that slope) divided by the measured speedup;
- × 1.15 for packing;
- + 20% retries.

*U* is all read pairs.

| pipeline | core-hours, no screen / screen | Clipper calendar, 300 cores | Anvil SU (whole nodes) |
|---|---|---|---|
| present (in-job download, 80 GB jobs) | — / 1,091k | ~152 days compute (more with memory limits) | — |
| optimized job shape | 442k / 592k | **145 days** (download-bound; compute 61 / 82) | 505k / 676k |
| + STAR acceleration (PR #34, ×1.54 mapping) | **316k / 466k** | **145 days** (compute 44 / 65) | **~365k / ~535k** |
| + a 2× mapping speedup overall | 262k / 412k | 145 days (compute 36 / 57) | 300k / 471k |

Add ~30k core-h for the 1-core download stages.

**Clipper.** Download into Clipper tops out at ~150 MB/s (~13 TB/day) from NCBI and AWS alike. 1.9 PB of SRA data takes ~145 days at that rate, 2–3× longer than the compute. Faster alignment no longer moves the Clipper calendar; only the network does. A 10 Gbit/s path (Science DMZ) would bring it to ~45–65 days, set by compute.

**Anvil.**
- **Shape:** whole nodes of 128 cores and 257 GB, charged all 128 cores. One genome sits in `/dev/shm`, with 28 four-thread samples per node.
- **Memory:** fits with disk sort (≤20 GB per job measured). With the non-host screen, it fits only if its indices are memory-mapped once per node; today each job reads a private 35 GB copy.
- **Cost:** ~365k SU without the screen, ~535k with it. That is 3.3–4.8× today's 112,226 SU balance. About 4–6 nodes for 30 days.
- **Allocation:** needs an ACCESS upgrade or credit exchange (allocations.access-ci.org, by the PI). Per-core speed on Anvil's EPYC 7763 and its download rate need a one-node pilot (~200 SU) before committing.

**Split option.** Clipper downloads and computes at its link rate while Anvil spends the existing balance in parallel. 112k SU covers ~21–31% of the samples, so Clipper's share takes ~100–115 days.

## 5. Order of work and schedule

| step | what | duration | gate |
|---|---|---|---|
| 0 | land #22, #25, #31 (production binary), the modality fixes and #34 (STAR acceleration) into one pinned, PGO-built binary | ~1–2 weeks | benchmark numbers reproduced; 4/8/16-thread outputs identical; modality benchmarks re-run |
| 1 | 24-hour bandwidth test (32 streams); ask GVSU IT about the uplink; Anvil one-node pilot if Anvil is used | 1–2 days, during step 0 | sets the real calendar |
| 2 | 50-sample validation run | 1 day | matches the benchmarks; strand right on wave-b samples; EmptyDrops recall vs authors ≥ 0.95 |
| 3 | wave a (5,276) → b (6,272) → c (5,030) | ~23% of the total calendar (~5 weeks on Clipper) | per-study QC and republish as studies finish |
| 4 | wave d (53,980), largest studies first | the rest | rolling per-study QC, v2 publish |

**Within each wave:**
- samples go in study order, so a study is republished as one unit;
- two job types per sample, chained with `--dependency`:
  - a 1-core I/O job: download, per-run screen (`singlet screen-runs`), encode to node-local `.1fq`;
  - a 4-core compute job: align + pileup → export → `called_v2` → results and a ledger row;
- the I/O pool is sized to fill the link, the compute pool to the cores free;
- failures are retried once, then parked in a ledger.

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
4. **All reads.** Every kept run (item 3) is used in full; record `reads_available`, `reads_used` and `runs_used` per sample.
   1% of samples have >1.3B pairs; they need 16-thread jobs or sharding by run on a 5-day partition, and disk sort (the
   in-RAM sort OOM'd at 80 GB on 182M reads).
5. **Provenance in every bundle** (the drift finding of `pipeline-validation.md` §1): engine git commit, binary sha256,
   reference build, runs used, worker count, cell-call method and FDR. The 2026 bundles do not say which engine wrote them.
6. **Publish gates** (from `docs/scbasecount-comparison.md`):
   - a sample whose matrices have nnz = 0 while `n_cells > 0` does not publish;
   - strand QC (wrong-strand share < 0.5) is required;
   - library-type flags route to the ADT modality or to exclusion.
7. **Ledger.** One row per sample: status, reads, runs, chemistry, strand, cells, timing, cluster, retries. It lets the
   campaign be resumed, priced and audited. The 2026 campaign's `processing_status` and API status disagree in places, and
   the 557 empty studies slipped through.

8. **Pinned, deterministic binary.** One PGO-built binary (git commit + sha256) for the whole campaign, including #34. Before wave a, check that 4-, 8-
   and 16-thread runs of the validation samples give byte-identical matrices (`perf_equiv.py`). They do for PBMC 3' v3 since
   the determinism fix.
9. **I/O stage.** A separate 1-core download job per sample: `prefetch` of each kept run, then a local decode
   (6.5× faster than remote reads). Check that its decoded reads equal the present path (`dl_equiv.sbatch`).
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

Every modality was benchmarked against an established tool on data with a known answer:
- `docs/pipeline-validation.md` (#22) for gene counts;
- `docs/modality-benchmarks/` for the rest.

The fixes listed are on `prod/modality-fixes`, re-checked with the same harnesses.

| modality | reference tool, truth | today | fix |
|---|---|---|---|
| gene counts, exonic / exon+intron | STARsolo `Gene`/`GeneFull`, Cell Ranger, alevin-fry | **equivalent** after #22 (ratio 0.97–1.02) | — |
| vs scBaseCount | same reads | **equivalent at equal depth**; the re-run uses all reads | — |
| cell calling | EmptyDrops_CR, authors' calls | **fixed in #31**: `called_v2` in C++, identical to the validated Python | — |
| strand / chemistry | — | **fixed in #31** (5' counted reverse without flags) | per-run screen added |
| library type | none in scBaseCount | **better than scBaseCount** | — |
| ADT / HTO | Cell Ranger | **equivalent** (r 0.999995) | PR #29 |
| splice junctions | STARsolo SJ, PBMC | **equivalent**: per-cell r 0.9999, r 0.994 at ≥10 molecules | STARsolo-like filter flags (177k vs 90k junctions, 134k singletons) |
| PSI | PSI from STARsolo SJ | **equivalent**: r 0.996, MAE 0.008; the 0/1 bug is gone | — |
| non-host | Kraken2 + minimap2 oracle, SARS-CoV-2 infected vs mock (GSE166766) | **equivalent**: 554k vs 553k viral reads, per-cell Spearman 0.985, AUROC 0.727 vs 0.732 | per-cell table lost on a fresh dir; unconfirmed taxa shown per cell |
| donor demux (K given) | vireo, demuxlet truth (Kang 2018, 8 donors) | **better**: ARI 0.986 vs 0.92 | — |
| donor demux (auto K) | same | **broken**: picks 2 of 8 (ARI 0.18) | model selection |
| doublets | Scrublet, scDblFinder; genotype and barnyard truth | **worse**: AUROC 0.83 vs 0.88; calls 20–25% of cells | calibrated threshold; genotype doublets (hard-coded 0 today; vireo AUROC 0.95) |
| ambient RNA | SoupX, CellBender, DecontX; human–mouse truth | **worse**: Spearman 0.41 vs 0.49–0.69; correction never produced | per-cell mixture estimate; corrected matrix |
| mtDNA | pysam pileup, donor truth | per-cell VAF **sound** (r 0.85). Variant list **unreliable** (12% supported). Homoplasmic variants dropped | rCRS reference, raw allele counts, stricter caller |
| SNP allele counts | — | **never produced**: no `--snps` in the campaign | `--snps` by default, private storage |
| 5' adapter clip | — | **bug**: clipped 11 bp from every PBMC 3' v3 read (−14% splices) | probe on ≥10k reads |

**Signals added so they are not lost with the BAM:**
- per-cell unmapped reasons, mismatch rate, antisense, intergenic and no-barcode counts;
- poly(A)-site × cell counts with an internal-priming flag;
- raw chrM allele counts;
- SNP allele counts (private).

## 9. Decisions needed

1. **Where to compute all reads.**
   - **(a) Clipper only:** ~145 days, set by the download link.
   - **(b) Anvil:** ~365k–535k SU, which needs an ACCESS upgrade (3.3–4.8× the balance).
   - **(c) Split:** Clipper plus Anvil's current balance, ~100–115 days.
2. **Ask GVSU IT about Clipper's uplink and run the 24-hour bandwidth test.** On Clipper this is now the only lever on the calendar.
3. **Non-host screen.**
   - It adds ~30% compute, which does not lengthen the Clipper calendar.
   - Unmapped reads are lost once the BAM is gone.
   - On Anvil it adds ~170k SU and needs memory-mapped indices.
   - **Recommended: keep it on.**
4. **SNP allele counts.** Store per-cell SNP allele counts privately (not public), so demux, ancestry and ASE can be redone without realigning. Only donor assignments would be public.
5. **Transcriptome-first step.** Build it only if Anvil is used, where SUs matter. It would save ≤1.3× more on mapping, and only once #34's cache is in. On Clipper it would not change the calendar.
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
| transcriptome-first eligible reads | `bench/aligner/txome_eligible.py` / `.sbatch` | Clipper |
| modality benchmarks | `bench/modalities/{genotype_pool,ambient,sj_nonhost}/` | Clipper |
| download paths and aggregate bandwidth | `dl_bench.sbatch <mode> <SRR,...>`, `dl_equiv.sbatch` | Clipper |
| download timing | `probe_download.sbatch <GSM> <SRR,...>` | Clipper |
| depth saturation | `probe_depth.sbatch <GSM>` (array K = 1, 2, 4, 8, 16) | Clipper |

Tables live under `/mnt/projects/debruinz_project/debruinz/singlet/data/atlas-rerun/`.
