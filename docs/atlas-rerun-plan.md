# Atlas re-run plan: fixed pipeline, every run, before the v2 migration

Status: plan, 2026-10-24, branch `plan/atlas-rerun`. No production jobs were submitted for it. Only measurement probes ran:
- ENA run reports;
- three download timings;
- a depth probe on three deep samples.

Code and inputs are in `bench/atlas_rerun/`.

Decision (Zach, 2026-10-24): re-run the atlas with the fixed pipeline (#22 + #25). Use every SRR and all reads, or a documented
cap sampled across runs, before the v2 migration. Plan it first: cost on Clipper vs Anvil, schedule and order, with empty
bundles and strand-affected samples first.

<!-- SUMMARY -->

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

<!-- DEPTH -->

## 3. Cost model

All rates are measured, not assumed.

**Compute** (`docs/pipeline-validation.md` §4.8, 7 datasets, fixed engine, campaign flags):
- Singlet uses **~50 CPU-s per million read pairs** for alignment + pileup + export, plus ~5 CPU-s per million for `.1fq`
  encoding.
- At 4 threads it runs at 81% CPU efficiency: **15–17 s of wall time per million read pairs**.
- At 16 threads efficiency drops to ~42%: STAR scales, the export and cell-calling tail does not. So **4-thread jobs are the
  unit** for cost.
- Model per sample, with *U* = read pairs used (millions):

  wall<sub>4 threads</sub> = 300 s + 17.5 s × *U*; core-hours = 4 × wall.
- Cross-check against the 2026 campaign (52,756 samples, ~32M reads each): that campaign spent 1.56 reserved core-h per sample
  including genome loads from NFS and optional screens. The model gives 0.94 for 32M, close to the hpc skill's rule of thumb
  (≈1 SU for a 30M-read sample on 4 cores, mouse campaign).
- The model is being checked at full depth on GSM5815984 (489M pairs, `probe_singlet.sbatch`); see §2.

**Download** (`probe_download.sbatch`, three samples, 9 runs, Clipper compute nodes):
- **~30 MB/s per SRA run** with `prefetch`, three in parallel without slowing;
- SRA files are **0.44 bytes per base**.
- With a cap, each run is read only up to its share of the cap (*U* × reads<sub>run</sub> / reads<sub>sample</sub>), so
  download scales with *U*, not with the full depth. The first N spots of an SRA run are not ordered by cell, so a head-of-run
  sample is close to a random one; §2 checks this against an every-k-th-read sample.

**Options** (70,558 GEX samples):

| option | read pairs used | samples capped | core-h | core-h + 20% retries | SRA download | p99 sample, 4 threads |
|---|---|---|---|---|---|---|
| full depth | 21.7 T | 0% | 446k | **535k** | 1.57 PB | 6.3 h (max 52 h) |
| cap 300M pairs | 14.4 T | 43% | 303k | **364k** | 1.02 PB | 1.5 h |
| cap 200M pairs | 10.8 T | 59% | 234k | **281k** | 0.76 PB | 1.1 h |
| cap 150M pairs | 8.6 T | 67% | 191k | **229k** | 0.60 PB | 0.8 h |
| cap 100M pairs | 6.1 T | 77% | 141k | **170k** | 0.42 PB | 0.6 h |
| cap 50M pairs | 3.2 T | 85% | 86k | **103k** | 0.22 PB | 0.3 h |
| *2026 campaign, for scale* | *1.7 T* | | *82k* | | | |

Per wave, at a 200M cap / at full depth: a 18.7k / 35.0k core-h, b 17.7k / 34.1k, c 17.7k / 31.5k, d 180k / 345k.

**Storage.**
- Scratch per job: the sample's SRA files + `.1fq` + the sorted BAM in `/dev/shm`. That is ≤ 2.5 × 0.44 B/base, about
  25–35 GB per job at a 200M cap, against 3.3 TB of node `/tmp`. Nothing is kept: SRA files and `.1fq` are deleted after
  the sample is done, and no BAMs are kept.
- Outputs: legacy-format results are ~30 MB per sample at 32M reads today. Assume ~60–100 MB per sample at 100–200M, which
  is **4–7 TB** under `$WORK` (47 TB free) before v2 packing.
- Full depth has 1% of samples over 1.3B pairs. Those need sharding by run or 16-thread jobs on the 5-day partitions; the
  others fit `short-cpu`'s 4 h limit.

## 4. Clipper vs Anvil

### Clipper (free)

**Capacity.** Clipper has no binding per-user limit: QOS allows 75 nodes per user. The limit is the shared hardware:
- `cpu` / `short-cpu`: 12 nodes, 40–64 cores and 0.5–0.77 TB each, ~600 cores in all, shared with the lab;
- preemptible `scavenger` adds idle GPU nodes, up to 22 nodes.

**Observed today:**
- one lab user held most of `short-cpu`;
- our arrays ran 4–12 tasks at a time (`QOSMaxNodePerUserLimit` / `Priority` waits);
- `scavenger` started jobs immediately on idle GPU nodes.

Realistic sustained throughput is **100–200 cores** (25–50 four-thread jobs), with bursts to ~400 when the cluster is quiet.

**Queues.**
- `short-cpu` (4 h) starts faster than `cpu` (5 days) on the same nodes.
- `scavenger` jobs can be preempted, so they need `--requeue` and idempotent per-sample outputs. The benchmark's jobs ran this
  way.
- Each job loads the STAR index from NFS (2–5 min). Node-local shared-memory genome loading (`singlet genome load`,
  `--genome-shared`) amortises this when one job processes several samples in sequence.

**Downloads.** ~30 MB/s per SRA run. Aggregate from Clipper is untested above 3 parallel streams. Plan on 0.5 GB/s, meaning
~15–20 concurrent downloads, until a 20-stream probe says otherwise.

### Anvil (ACCESS, `bio260157`)

**Balance:** **112,226 SU** on 2026-10-24 (`mybalance`; 400,001 limit, 287,775 used).

**Charging.** 1 SU = 1 core-hour on the reserved cores. On `shared`, the charge is max(cores, memory / 1.9 GB). A Singlet job
holds the 30 GB STAR index privately: 17.7 GB RSS at 4 threads, so it is charged as ~9 cores, **2.3× the core-hours**.

**The efficient shape is whole nodes:**
- 128 cores / 257 GB per node;
- the genome loaded once into `/dev/shm`;
- 32 four-thread samples at a time, each ~4–6 GB without the genome;
- SU ≈ core-hours.

This needs a one-node pilot first: a few SU, to be approved.

| option | SU (whole-node, + retries) | share of today's balance | Anvil nodes for a 14-day run |
|---|---|---|---|
| full depth | 535k | 476% | 12.4 |
| cap 300M | 364k | 324% | 8.5 |
| cap 200M | 281k | 250% | 6.5 |
| cap 150M | 229k | 204% | 5.3 |
| cap 100M | 170k | 151% | 3.9 |
| cap 50M | 103k | 92% | 2.4 |

**Only a 50M cap fits the current allocation, and it would use all of it.** Anything deeper needs one of:
- a supplement or exchange of ACCESS credits (the PI can request on allocations.access-ci.org);
- running most of the work on Clipper.

Anvil's advantages are concurrency (hundreds of nodes) and faster networks to NCBI. Its maintenance outage
**2026-11-07 00:00 – 11-08 18:00 EST** falls inside any schedule below.

### Recommended split

**Waves a–c run on Clipper.** These 16,578 samples are the urgent, visible part: empty studies, antisense samples, new 10x
coverage.

**Wave d runs on Clipper and on Anvil whole nodes in parallel.** Anvil is held to ~80k SU, keeping ~30k SU in reserve.

Wall-clock days of computing after the gates, all core-hours including 20% retries:

| cap | total core-h | waves a–c core-h | waves a–c on Clipper, 150 cores | all on Clipper, 150 / 200 cores | Clipper + 80k SU on Anvil, 150 / 200 cores |
|---|---|---|---|---|---|
| 300M | 364k | 84k | 23 d | 101 / 76 d | 79 / 59 d |
| **200M** | 281k | 65k | 18 d | 78 / 59 d | **56 / 42 d** |
| 150M | 229k | 53k | 15 d | 64 / 48 d | 41 / 31 d |
| **100M** | 170k | 39k | 11 d | 47 / 35 d | **25 / 19 d** |
| 50M | 103k | 24k | 7 d | 29 / 21 d | 7 / 5 d (fits Anvil alone) |
| full | 535k | 121k | 34 d | 149 / 111 d | 126 / 95 d |
- **Full depth is not recommended:**
  - it is 2× the cost of a 200M cap;
  - it takes 4–8 months on Clipper or 4.8× the Anvil balance;
  - 1% of samples need >6 h each;
  - the gain over a cap above saturation is small (§2).

## 5. Order of work and schedule

Calendar at a 200M cap. Clipper runs at ~150 cores; Anvil uses ≤80k SU on whole nodes after its pilot. Times are wall-clock
time after each step's gate.

| step | what | where | duration | gate to pass before the next step |
|---|---|---|---|---|
| 0 | merge #22 and #25 (CI green); add EmptyDrops calling and the provenance stamp (§6); pin the binary | — | ~1 week of PRs | build reproduces the benchmark numbers |
| 1 | **50-sample validation run**: 7 benchmark datasets + 43 atlas samples across waves, protocols and species | Clipper | 1 day | matches the benchmark; strand right on all wave-b picks; EmptyDrops recall vs the authors ≥ 0.95 where they deposited calls |
| 2 | one-node Anvil pilot: 32 samples, shared-memory genome | Anvil (~200 SU, ask first) | 1 day | SU per sample within 10% of the model |
| 3 | **wave a** (empty studies, 5,276) | Clipper | ~6 days | republish per study as each finishes |
| 4 | **wave b** (strand, 6,272) | Clipper | ~6 days | strand QC on every sample |
| 5 | **wave c** (skipped 10x, 5,030) | Clipper | ~6 days | library-type and chemistry screen per run |
| 6 | **wave d** (53,980), by study cells | Anvil from step 2 on; Clipper after wave c | ~5 weeks after wave c | rolling: per-study QC, then v2 publish |

**Total at 200M:** ~1 week of preparation plus ~8 weeks of computing (56 days) with both clusters. On Clipper alone it is
~11 weeks.

**Total at 100M:** ~1 week plus ~3.5 weeks (25 days) with both clusters, or ~7 weeks on Clipper alone.

**Within each wave:**
- samples go in study order, so a study is republished as one unit;
- each array task is one sample: download → per-run screen → encode → align + pileup → export → EmptyDrops → write results and
  a ledger row;
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

## 8. Decisions needed

<!-- DECISIONS -->

## Reproducing

| step | script | where |
|---|---|---|
| catalog and plan inspection | `inspect_catalog.py` / `.sbatch` | Clipper |
| ENA run report per project | `fetch_runinfo.py` / `.sbatch` | Clipper |
| runs for samples the project reports missed | `fetch_missing.py` / `.sbatch` | Clipper |
| universe, waves, routes | `build_universe.py` | anywhere, on the pulled tables |
| cost per option | `cost_model.py` | anywhere |
| download timing | `probe_download.sbatch <GSM> <SRR,...>` | Clipper |
| depth saturation | `probe_depth.sbatch <GSM>` (array K = 1, 2, 4, 8, 16) | Clipper |

Tables live under `/mnt/projects/debruinz_project/debruinz/singlet/data/atlas-rerun/`.
