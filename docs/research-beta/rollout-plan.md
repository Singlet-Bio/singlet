# Research Beta: rollout plan and workstreams

Written 2026-10-27 by the orchestrating session after a five-part audit. The evidence, with file and line references, is in
[audit-2026-10-27.md](audit-2026-10-27.md). This plan supersedes §5 ("Path to Research Beta") of
[../research-beta.md](../research-beta.md). Nothing in it has been launched. §8 lists the decisions taken on 2026-10-27 and those still open.

## 0. Decisions on record (Zach)

**Platform**
- **One codebase.** Alpha and Beta use the exact same code and package. Beta only gets its own Cloudflare R2 bucket, data
  Worker, D1 database and Pages branch at **beta.singlet.bio** (2026-10-27).
- **Rolling publication.** beta.singlet.bio is updated continuously as results arrive from Clipper, Anvil and any other
  cluster.
- **Launch gate.** The software scope and state must be excellent and proven **before** the reprocessing campaign
  starts. Re-running ~70k samples twice is not affordable.

**Campaign**
- **No SUs on downloads.** Downloads must not spend Anvil SUs; run them on the Anvil login nodes. **Approved by Amiya
  (Purdue RCAC)** for the fetcher on login nodes (Zach, 2026-10-27). Keep it network-only and within the 2-core /
  12 GiB cgroup.
- **Acceleration.** Make the remaining pipeline as fast as possible before launch. FASTASSIGN (#41) goes in if its
  outputs are equivalent, plus a rebuilt PGO binary.
- **Scope and order.** Transcriptomics runs first, then the other modalities. **Every modality is developed and debugged
  before anything runs at production scale** (2026-10-27): the BSF covers ADT/HTO, V(D)J/CRISPR, ATAC/Multiome/spatial,
  mtDNA lineage and cell type as well as GEX.
- **Alpha has priority on Clipper.** Development and benchmark jobs may run during the Alpha freeze only if they cannot
  delay Alpha's jobs (§9).
- **Models.** Sonnet 5.5 for the workstreams (the gruntwork); Opus 5.5 for orchestration only; Haiku 5.5 for sweeps.
- **Depth.** All reads of every run; no cap.
- **Clusters.** Clipper first, Anvil only after Zach confirms. State the SU estimate before anything over ~1,000 SU.

**Data and release rules**
- **Format.** `.s1` with the pz2 codec is the only data format. Nothing is named "store" or "v2".
- **Public vs private.** Genotypes are never public. Inferred sex, ancestry and cell type are public cell metadata.
- **Licences.** Code MIT, data CC0.
- **Benchmarks.** Real data with real ground truth only. 10x's published outputs are the comparator: don't run 10x
  software. No MiXCR.

## 1. Where things stand

| Area | State on 2026-10-27 |
|---|---|
| Research Alpha | **.s1 transcode:** done (7,461 studies, 236.5 M cells). **Cutover (website #74):** merged. **Still running:** the cell index, data QC and the catalog load. PR #54 is open. The Alpha freeze holds until Alpha ships. |
| Rename | Done through Alpha: `s1/` prefix, D1 038/039, Worker. [rename.md](rename.md) is history now. |
| Beta engine | **Production stack (#31, #36, `prod/campaign`, #34):** unmerged and pre-rename. **Draft fixes (#41, #45, #46, #47):** unmerged. **Pinned candidate:** `singlet-campaign` 02aa9b9. **Open bugs:** 10. **Benchmarks:** 0 of 22 capabilities pass. |
| Orchestrator (#42) | **Built** and smoke-tested on 8 samples, but **without whitelists**, so its timings are not representative. **Anvil:** disabled. **I/O and compute:** already separate stages, but on Anvil the I/O is billed 1-core `shared` jobs. |
| .s1 producer | **Missing.** The migrator reads only legacy bundles. Beta has no release policy and no channel concept. |
| Network | Clipper's download ceiling is ~150 MB/s (~13 TB/day). Raw SRA is ~1.58 PB (1.9 PB with the 20% retry margin). |
| Anvil | **Balance:** 111,416 SU. **Outage:** Nov 6 17:00 to Nov 8 18:00 EST. **Login nodes:** 2 cores and 12 GiB per user per node. **Scratch:** 100 TB and 1 M files, purged at 30 days by access time. |
| CI | **Stacked PRs:** no tests or lint. **Lint gaps:** `scripts/` and `bench/` are not linted. **Release:** the PyPI 1.0.0 publish failed, and the docs deploy has been red since 10-25. |

## 2. The launch gate: Beta Software Freeze (BSF)

The campaign starts when one **pinned binary** and one **pinned Python/s1 commit** pass the checklist in §7. In outline:
1. Every open correctness bug is fixed, or its layer is explicitly omitted by the Beta release policy.
2. Outputs are deterministic across thread count, run order, barcode order and reruns.
3. Every speed-up is proven output-identical on real data, or its difference is benchmarked and accepted.
4. Every public layer passes its real-data benchmark at the pinned commit. The Beta release policy is generated from
   those results.
5. A 50-sample validation through the orchestrator, on both clusters' paths, round-trips into `.s1`. It is published
   to the Beta stack, re-run, and found identical.
6. Fetch and staging work within policy on both clusters, and the calendar and SU projection is approved.

## 3. The two big levers

### 3a. Separate download from compute

The I/O stage today mixes two kinds of work:
- **a network fetch:** `prefetch` needs only sra-tools;
- **a CPU and memory transform:** decode, screen and pool. Pooling has hit 16 GB, more than a login node allows.

So the design splits on that line.

**Fetch tier (network only).** It writes `stage/<GSM>/runs/<SRR>.sra`, a `fetch.json` with sizes and checksums, and an
inbox event.
- **On Anvil:** one fetcher process per login host.
  - **Placement:** pinned in tmux, kept within 2 cores and 12 GiB, with 4–8 concurrent `prefetch` or S3 streams.
  - **Work queue:** it claims `fetchq/` tickets that the controller writes.
  - **Liveness:** it sends heartbeat events. If the fetcher dies, its samples go back to `released`, never `lost`.
- **On Clipper:** the same fetch code, run as cheap 1–2 core I/O jobs as today. Free, but the jobs are network-only.
- **Policy:** approved by Amiya at Purdue RCAC (2026-10-27). Record the approval in the runbook. The fallback, a 1–2 core
  `shared` fetcher job at ~24 SU per day, stays in the code for outages and login-node maintenance.

**Compute tier.** The node job decodes each run on the node's spare cores. Each Anvil node has 16 cores outside the
112 four-thread slots, so this costs about 0 extra SU. It then runs `screen-runs` and #47's concatenation
(`--runs-dir --run-screen`), then the engine.
- **Pooling goes:** decode → encode → decode is gone (670 s and 16 GB at 122 M pairs).
- **Mismatched runs:** samples whose runs differ in read length fall back to per-run decode-and-merge.

**Drop the remote head screen.**
- Pre-drop runs from ENA metadata instead.
- Screen the full run on the compute node.
- The 1 M-pair remote stream saves no bandwidth, and its OOM is what made the smoke test fail.

**Staging rules.**
- **Age:** kept under 14 days, because of the 30-day atime purge.
- **Caps:** bytes and files, with files counted from events and never with `find`.
- **Deletion:** only after the compute stage verifies its outputs.
- **Shipping:** out/ goes to private R2 or Clipper storage within the purge window.

**What it saves.**
- **Anvil SUs:** ~30k core-h of I/O arrays atlas-wide, pro rata to Anvil's share.
- **Idle nodes:** compute nodes never wait on the network.
- **Calendar:** on Clipper it saves nothing. Clipper is network-bound, so the only Clipper levers are fewer bytes and a
  faster link:
  - SRA Lite;
  - Anvil taking a larger share;
  - a GVSU Science-DMZ path.

**SRA Lite decision.**
- About −60% bytes on runs that have it.
- Our `.1fq` already drops qualities, so counts should not change.
- The mtDNA and SNP callers do use quality. W7 measures Lite against full quality on real genotype and MAESTER truth.
  Use Lite only if those layers are unaffected or are not public.

### 3b. Accelerate the engine

The cost model is 316k core-h without the non-host screen and 466k with it. It undercounts at depth: measured walls are
1.3–2.5× the model above ~110 M pairs, mostly because of the disk BAM sort.

Changes in order of value:

| # | change | expected saving | must prove |
|---|---|---|---|
| 1 | **Retire the coordinate-sorted BAM.** Stream alignments into the pileup with STAR's CB/UB correction replicated from pass-1 counts. Take strand from the header and run screen. | 17–27% of wall on idle nodes, 23–62% at depth: ~75–230k core-h | identical matrices and calls on real 3′, 5′ and multi-run data |
| 2 | **FASTASSIGN (#41)** plus a PGO rebuild trained on 3′ and 5′ | 8–16%: 35–95k | identity logs of jobs 787387–787393, then more datasets |
| 3 | **Non-host screen:** a pre-merged, memory-mapped index shared per node; drop the discarded dump | the model overcharges ~150k; the real saving is a few % plus ~30 GB less RAM per job | identical non-host tables |
| 4 | **Serial stages:** pass 1 from the header, overlap the reference load, parallel CB refresh, 4 GB sequence cache | 4–5%: ~20–30k | identical |
| 5 | **Packing:** shared genome plus mapped screen index; 1–2 threads or 1.3× oversubscription where memory allows | 20–30% of core-seconds where it applies | memory.peak at 28-way |

**What retiring the sorted BAM changes** (Zach asked, 2026-10-27):
- **Nothing published changes if the route holds.** The BAM is already deleted after pileup and never published.
- **STAR's barcode correction must be rebuilt.** STAR writes the corrected CB/UB tags only while sorting. The pileup would
  apply the same 1-mismatch correction with N-base pseudocounts from pass-1 exact counts. This is the main risk, and the
  identity test must cover it.
- **The pileup becomes one streaming pass in read order.** Counts and junctions don't need position order. SNP, mtDNA and
  intron calls need per-position tallies kept in memory instead of walking a sorted file. That is cheap for chrM and a
  fixed SNP panel; memory must be measured for whole-genome intron and SNP tallies.
- **No random access.** The strand re-run and the strand probe can't reopen an index; strand comes from the header and
  the run screen.
- **Determinism.** Unsorted STAR output arrives in thread order, so every caller must be order-independent. That is the
  same work as bug 5 (barcode-order dependence).
- **Debugging.** A `--keep-sorted-bam` flag keeps today's path for the benchmark cohort and for IGV inspection.

**Targets.**
- **With #1:** ~250–330k core-h, with depth no longer penalised.
- **Without #1:** ~330–420k.
- **Timebox for #1:** two weeks. If identity is not reached, sort in RAM where memory allows, and keep the disk sort
  elsewhere.

**GPU alignment** (Zach asked, 2026-10-27): not orders of magnitude, and not for Beta.
- **The bottleneck is memory latency.** STAR's seed search walks a ~30 GB suffix array with random, data-dependent access.
  An H100's memory bandwidth (~3–4 TB/s) is ~10× a CPU node's, which caps a well-built GPU aligner at about 10× a node.
- **Reported results match that cap:**
  - NVIDIA's GPU STAR (Parabricks `rna_fq2bam`): users report ~8–9× on a 4-GPU node against CPU STAR, and none with some
    parameter sets.
  - Earlier GPU seed-and-extend aligners report ~10×.
- **Clipper's scale:** Clipper has two H100 NVL GPUs. Two GPUs would equal a few CPU nodes, and Clipper is limited by its
  network link, not by compute.
- **Cost of a custom build:** months of work, and proving it identical to STAR for the freeze would be very hard.
- **Revisit:** after Beta, for a GPU-rich allocation.

### 3c. Calendar and SU model

**Assumptions:**
- Anvil spends ~105k of the 111k SU at ~1.15 SU per core-hour, which buys ~91k core-h.
- Clipper downloads ~13 TB/day.
- Bytes include retries.

| software state | atlas core-h | Anvil share of samples | Clipper bytes | Clipper calendar |
|---|---|---|---|---|
| today's pin, unscaled model | 466k (up to ~2× at depth) | 10–20% | 1.5–1.7 PB | 117–131 days |
| + items 2–5 | ~330–420k | 22–28% | 1.37–1.48 PB | 105–114 days |
| + item 1 (BAM retired) | ~250–330k | 28–36% | 1.22–1.37 PB | 94–105 days |
| + SRA Lite where it applies (if allowed; assumes half the bytes are in runs that have Lite) | same | same | ~0.85–0.96 PB | ~66–74 days |
| + an ACCESS supplement of 100k SU (instead of Lite) | same | +26–35 points | −0.5 to −0.66 PB | ~43–67 days |

**Throughout:** Clipper stays network-bound. Its compute share needs only ~22–33 days of 300 cores.

**Calendar to Q1 2027:** BSF around the first week of December plus ~95–105 days lands in mid-March 2027. SRA Lite, a
supplement, or the GVSU link gives margin.

**Rolling publication:** beta.singlet.bio fills study by study from the first week of the campaign.

### 3d. Long-term storage: no reads or BAMs kept (Zach, 2026-10-27)

**Decision:** intermediate files are not kept. Even `.1fq` (~0.35–0.63 PB for the atlas) does not pay off. A coordinate-
sorted BAM would be ~3–6 PB.

**Budget:** about **10 TB** of long-term storage, on Clipper only. Anvil keeps nothing: `$PROJECT` is deleted 90 days
after the allocation ends, and scratch is purged after 30 days.

**What the 10 TB is for:**
- references and indexes;
- private outputs: genotype and depth tables, and the restricted sidecars;
- campaign ledgers and logs;
- the **non-host read archive** (§3e). It is small: only reads assigned to a detected species, with their CB and UMI.
  Estimated well under 1 TB for the atlas; W3 measures it on the validation samples.

**Consequences:**
- A re-run means downloading again. That is why the software freeze has to be right the first time.
- Re-download is free in money, but costs calendar time on Clipper's ~13 TB/day link.
- Re-compressing reads strongly enough to change this (e.g. reference-based `.1fq` tier 2, ~0.4 GB per 100 M reads) is not
  in Beta scope.

### 3e. Non-host screen: what it does and what it costs (Zach asked, 2026-10-27)

**Not an aligner like STAR.** It is a k-mer sketch classifier in the style of sylph and Kraken2, run on the cDNA reads that
STAR leaves unmapped (`include/singlet/pileup/nonhost/`). Steps:
1. **Input:** STAR's unmapped mate 1 (cDNA) and mate 2 (barcode + UMI), capped at the first 20 M reads.
2. **Sketch:** canonical 21-mer minimizers, window 11, for each read.
3. **Host subtraction:** a ~313 MB Bloom filter of host-genome minimizers removes host k-mers. Reads with fewer than 5
   non-host minimizers are `UNKNOWN`. These are mostly unmapped host repeats.
4. **Lookup:** each read's minimizers are matched against the viral and microbial (bacterial + fungal) indexes. These are
   flat sorted arrays, ~25 GB on disk; the fungal index alone has ~1.06 B entries. The match is a sort-merge scan over
   the whole index, so its cost grows with the index size, not with the number of reads.
5. **Abundance:** an EM, as in Bracken and sylph, turns per-read multi-hits into species abundances with HIGH, MEDIUM and
   LOW confidence.
6. **Confirmation:** a seed-chain check against each detected species' genome FASTA, if the FASTA is installed.
7. **Per cell:** a cell × species matrix of reads and UMIs (`nonhost_per_cell.1pz`). Reads and UMIs are never published.

**What it reports:** species (and kingdom) per cell. It gives no genes or transcripts (`n_genes` is always 0, because no
annotation is loaded).

**Transcript level would be feasible as an add-on:**
- **How:** after the EM, align only the reads assigned to detected species against those species' annotated
  transcriptomes. That gives a cell × microbial-gene matrix.
- **Cost:** small, because it touches only detected-species reads (usually well under 1% of reads). A bacterial genome
  indexes in about a second. It becomes expensive only for heavily contaminated samples with dozens of species, so cap
  the number of species.
- **Biology limits what it shows:**
  - 10x 3′ and 5′ chemistries capture polyadenylated RNA. Polyadenylated viruses, such as SARS-CoV-2, HIV, EBV and
    influenza mRNAs, are captured well, and their gene-level counts are informative.
  - Bacterial mRNA is not polyadenylated, so bacterial gene profiles would be sparse and biased.
- **Recommendation:** viral gene level for Beta if W7 can benchmark it on real infected-sample truth. Bacteria and fungi
  stay at species level.

**Genome positions per read: a two-stage design** (Zach asked, 2026-10-27)

Goal: map each detected species' reads to its genome, so we can tell apart vaccine exposure, active infection, latent
infection and contamination.

*Stage 1, per sample, inside the campaign:* the sketch screen as above. It writes every read assigned to a detected
species (cDNA, CB, UMI, species, score) to the private non-host read archive on Clipper. A cap per species and sample
keeps the archive bounded.

*Stage 2, batch per species, across samples:* after each campaign wave, for every species detected in at least one
sample:
- **References:** build a STAR index from a curated reference set: the reference genome plus major strains or
  genotypes, with annotation.
- **Alignment:** align all archived reads for that species from all samples in one job. STAR handles spliced viral
  transcripts (HIV, EBV, HPV, adenovirus, herpesviruses).
- **Outputs, per sample:**
  - per-read genome positions;
  - per-cell gene counts;
  - coverage profile and strand;
  - splice junctions;
  - consensus variants (strain or lineage);
  - virus–host chimeric reads, which mark integration sites.
- **Cost:** small. A viral index builds in seconds, a bacterial one in about a minute, and the read sets are tiny.
- **Complexity:** moderate, about one extra stage for the orchestrator. Batching is simpler than per-sample alignment,
  because each species is indexed once.

What it lets us distinguish (each needs real-truth validation by W7 before any label is published):

| Situation | Signature |
|---|---|
| mRNA vaccine (e.g. BNT162b2, mRNA-1273) | reads match the codon-optimised vaccine spike sequence (added as its own reference), not the wild-type genome; spike only, no other ORFs |
| Active SARS-CoV-2 infection | reads across many ORFs; leader–TRS subgenomic junctions show replication |
| Latent herpesvirus (EBV, CMV) | latency gene programs (e.g. EBV LMP1/EBNA) without lytic genes |
| Integrated retrovirus (HIV) | virus–host chimeric reads at integration sites, plus the expression pattern |
| Viral vectors (lentivirus, AAV, gene therapy, lab constructs) | vector elements (WPRE, CMV promoter, transgene) and no wild-type genes |
| Contamination | the same species and strain across unrelated studies or batches; reads spread evenly across barcodes, not in cells |

The archive is private. Reads never reach public outputs. Only per-cell gene counts and per-sample summaries that pass W7
become public layers.

**Making it cheap** (W3):
- **Load cost:** today each job zero-fills and copies 25–35 GB of index into its own heap. Change this to one pre-merged
  index file per node, memory-mapped read-only and shared through the page cache: loading becomes ~free, and per-job
  memory drops by ~30 GB.
- **Scan cost:** the whole-index scan is about 25 GB of sequential reads per job. Use binary-search lookup instead when a
  sample has few unmapped reads (most 3′ samples), and keep the merge scan for large batches.
- **Read count:** deduplicate identical unmapped reads (counting multiplicity) and drop low-complexity reads before the
  sketch.
- **Discarded dump:** stop writing `nonhost_reads.fq`.
- **Cost model:** refit it from measured timers at 0.5, 2.6 and 20 M unmapped reads. Today's model charges 4.5 s per M
  reads (~150k core-h); the audit estimates 10–47k.
- **Target:** under 3% of the wall time and under 2 GB of extra RSS per job.

## 4. Alignment with Alpha

**Code**
- **One code path.** Every Beta change lands on `main` behind CI. Alpha keeps running from the same code, so an `.s1`
  regression job runs on every s1-touching PR:
  - `verify --policy` on the Alpha pilot studies;
  - a read-back of the Alpha read-path tests.
- **Sequencing with the Alpha session.**
  - W5 starts only after #54 merges.
  - Website changes merge to `main` only after Alpha ships, or with the Alpha session's agreement.
  - Beta's Cloudflare resources are new and separate, but each creation or deploy is still announced and approved by
    Zach.
- **Freeze.** No data-generating Beta cluster jobs until Alpha ships, unless Zach lifts the freeze for
  development and benchmark jobs (decision 1).

**Data and policy**
- **Same format, separate stack.** Keys, manifest, cells table, packs and the D1 `s1_*` schema are identical. Channel
  differences are config only:
  - the bucket;
  - the Worker route `data.beta.singlet.bio`;
  - the D1 database;
  - the Pages branch;
  - `SINGLET_DATA_BASE` and `SINGLET_S1_PACKS_REMOTE` for the clients.
- **Release policy.** A Beta policy object generalises the Alpha one: per layer `public | preview | omit | private`,
  generated from the benchmark results at the pinned commit. `private/*` is always private (add it to `s1/policy.py`
  and the website's `restricted.ts`).
- **Labels.** The same validated labeller with a new `annot_version`, revalidated on Beta counts. Ancestry depth tables
  move from `private/` into the restricted sidecar the labeller already reads.

**Every Alpha known issue gets a Beta fix and an owner**

| Alpha issue | Beta fix | Owner |
|---|---|---|
| D1 wrong strand (8,270 samples; mouse never classified) | per-sample strand detection and per-run screen (#31) | W2 verifies on 5′ and mouse |
| D2 non-expression libraries processed as RNA | runs dropped with a reason; modality campaigns later (decision 4) | W4 / W2 |
| D3 empty matrices | publish gate in the adapter: nnz = 0 with cells is refused | W5 |
| D4 molecule double count; counts depend on `--threads` | molecule labels (#22 in #31) plus the determinism fixes | W2 |
| D5 first run, 30 M reads | all reads, every run | W4 |
| D6 conservative knee | `called_ed` (EmptyDrops-class) as the published call, after the truth decision | W2 / W7 |
| D7 false 5′ clip | the #36 probe fix | W2 verifies |
| D8 Visium as droplets | the #48 spot path, or park those samples (decision 4) | W4 |
| §2c ambient, doublets | #45 / #43, or omitted by policy | W2 / W7 |
| §2c spliced/unspliced inflated | #22 plus the #46 intron rules | W2 |
| §2c demux auto-K | genotype-concordance merge | W2 |
| §2c legacy PSI 0/1 | true PSI recipe and junction table | W5 |
| §2c per-cell non-host | gate on the non-host benchmark; the sample level stays public | W7 |
| §2c mtDNA alleles | private, until the lineage-safe design (#53) | W5 policy |

## 5. Workstreams

**Format.** Each workstream is proposed as its own session in its own worktree, started with a fresh context that points
at this file. Each one:
- owns a disjoint set of files;
- writes a handoff note `docs/research-beta/ws-<id>.md` at every stop;
- ends its turns with `wrap_up`.

**Models** (Zach, 2026-10-27):
- Sonnet 5.5 for every workstream session.
- Opus 5.5 only for the orchestrating session, which reviews designs and resolves cross-workstream questions.
- Haiku 5.5 for sweeps and checklists.

### W1. Integration, naming and CI (Sonnet 5.5). Starts first.

- **Goal:** one Beta engine on `main`, so every later PR gets full CI.
- **Owns:**
  - branch topology;
  - mechanical renames across the repo, during phase A only;
  - `.github/workflows/`;
  - lint config (`pyproject.toml` ruff and pyright sections, a new `.clang-format`);
  - `r/_pkgdown.yml`, `CHANGELOG.md`;
  - the release workflow.
- **Phase A, 3–5 days.** Rebase the production stack (#31 → #36 → `prod/campaign` with #34) onto `main`:
  - drop the `python/singlet/v2` tree, the `.s2` objects and the `v2/` prefixes in favour of main's `s1`;
  - rename `called_v2` to `called_ed` in the C++ (header and namespace, `cell_calls.tsv` columns `called_ed` /
    `called_ed_fdr`, CLI value, provenance keys, golden vectors, 73 files);
  - allowlist provenance fields so cluster paths never reach a manifest.

  **Phase A gate:** the rebuilt binary's outputs are byte-identical to 02aa9b9 apart from the renamed columns and keys,
  on the PBMC and hgmm references and one 5′ dataset. Then merge to `main` as one "Beta engine" PR. Rebase the open
  drafts (#41, #45, #46, #47) onto it and hand #41 and #47 to W3 and #45 and #46 to W2. Fix #41's CRLF commit.
- **Phase B, 2–3 days, then ongoing.**
  - **CI:** run on every PR base, or at least `main` and `beta/**`.
  - **Python lint:** ruff over `scripts/` and `bench/`.
  - **C++ lint:** clang-format checks on changed lines only, plus a warnings budget.
  - **Release:** fix the PyPI trusted publisher and the pkgdown index; move or delete the pilot `v1.0.0` tag.
  - **Website and singlet-cloud:** make `eslint` and `tsc` blocking once they are clean.
  - **Dependabot:** triage (majors one at a time).
- **Does not touch:** engine logic, `python/singlet/s1/` (other than the rename), or the website beyond CI.
- **Clusters:** Clipper `debug-cpu` / `short-cpu` for the identity check only (decision 1).

### W2. Engine correctness (Sonnet 5.5). After W1 phase A.

- **Goal:** every open correctness bug fixed and proven on real ground truth.
- **Owns:**
  - `include/singlet/pileup/**` except `nonhost/`;
  - `include/singlet/fq/**` for the 101-nt barcode read;
  - their tests;
  - `bench/engine-fixes/`.
- **Work**, in order:
  1. **NUMT / mt-gene multimapper drop:** a pseudogene-aware tie-break (#46).
  2. **Intron rules (#46):** pick the default rule from real truth (cross-species, STARsolo agreement, CITE).
  3. **Determinism:**
     - sort junction names before numbering them;
     - remove barcode-order dependence in the ambient, doublet and mtDNA callers.
  4. **Encoder:** stop dropping every read when R1 is 101 nt (GSE230239).
  5. **Ambient (#45):** finish the calibration, or recommend omitting it.
  6. **Demux auto-K:** merge clusters with near-identical genotypes (~0.80 concordance).
  7. **donor-snps genotype cut-offs** (`donor_demux.h:2511`).
  8. **`called_ed`:** default selection and the label input, from the independent-truth results (Kang, hgmm).
  9. **Re-verify on 5′ and on mouse:** strand and the false 5′ clip.
- **Coordination:** W2 owns `pileup_engine.h` until its fixes land. W3's streaming-pileup change to that file goes in
  afterwards, reviewed by W2.
- **Done when:** each fix has a real-data result in its PR, and 4/8/16 threads plus shuffled run and barcode order give
  identical outputs.

### W3. Engine performance (Sonnet 5.5). After W1 phase A.

- **Goal:** the lowest cost per sample with identical outputs (§3b).
- **Owns:**
  - `src/star/**`, `include/singlet/star/**`;
  - `include/singlet/pileup/nonhost/**` and the sketch index;
  - stage plumbing in `src/pipeline/singlet.cpp`;
  - the PGO scripts;
  - `src/star/bench/`.
- **Work:**
  1. Measure first (audit §3 "Measure first").
  2. Read FASTASSIGN's identity logs. Merge #41 only if its outputs are equivalent on real 3′, 5′ and deep data (Zach);
     then merge #47.
  3. Non-host, per §3e: a shared memory-mapped index, adaptive lookup, read dedup, no dump; measured against the
     <3% wall and <2 GB RSS targets. Prototype viral gene-level counts, cost-capped.
  4. Serial-stage trims and the 4 GB sequence cache.
  5. BAM retirement: a separate session (W3b) follows [bam-stream-protocol.md](bam-stream-protocol.md), with a
     stop rule at phase 2 and a two-week timebox for the prototype.
  6. Thread count and packing study (memory.peak at 28-way on Clipper; the Anvil numbers come from the pilot).
  7. PGO rebuild on 3′ + 5′ data and the final pin.
- **Done when:**
  - the pinned binary has a measured stage profile at 66 M, ~110 M and ~260 M pairs on both 3′ and 5′;
  - an identity report is in place;
  - the cost model (`cost_model_v2.py`) is refitted from those measurements together with W4.

### W4. Download tier and orchestrator (Sonnet 5.5). Starts now.

- **Goal:** §3a built and tested; the orchestrator ready for both clusters.
- **Owns:**
  - `scripts/atlas_rerun/**`;
  - `bench/atlas_rerun/cost_model_v2.py`;
  - `docs/atlas-rerun-*.md`;
  - Anvil `$PROJECT` setup (sra-tools, references, whitelists), which costs no SUs.
- **Work:**
  1. `fetch_stage` with prefetch or S3 and checksums.
  2. The `cmd_fetcher` daemon with tickets, heartbeat and `.part` files; `compute_stage.prepare_reads` (decode, screen,
     concat, and the fallback); remove `pool_runs`.
  3. Plan from ENA `sra_bytes`, with an ENA metadata pre-drop in place of the head stream.
  4. Staging age, byte and file caps.
  5. Ship out/ before the purge.
  6. An SRA Lite option, used only on W7's verdict.
  7. Record Amiya's login-node approval in the Anvil runbook, with the fetcher's limits (2 cores, 12 GiB, 4–8 streams).
  8. A login-node fetch bandwidth test of ~200 GB to `$SCRATCH` (no SUs; needs Zach's OK).
  9. Re-run the Clipper smoke test **with whitelists**, on terms that cannot delay Alpha (§9).
  10. Routing for non-GEX runs and Visium, so each modality's runs are queued for its own campaign.
  11. Storage (§3d): keep only references, private outputs, ledgers and the non-host read archive, within ~10 TB on
      Clipper, and delete staged reads after verification.
  12. The non-host stage-2 batch job (§3e): one job per detected species after each wave, reading the archive.
- **Done when:**
  - a dry run on both clusters shows the projection;
  - the fetcher survives a kill and restart without parking samples;
  - the smoke test is green with whitelists.

### W5. Beta `.s1` producer and release policy (Sonnet 5.5). After #54 merges.

- **Goal:** orchestrator handoffs become verified `.s1` objects, with no legacy-bundle detour.
- **Owns:**
  - `python/singlet/s1/**`;
  - `scripts/s1_*.py`;
  - `python/singlet/_restricted.py`;
  - the labeller hooks in `python/singlet/celllabels/**`;
  - the modality lists in the Python and R clients and MCP;
  - `docs/format-s1.md`.
- **Work:**
  1. `PipelineDirBundle`, which maps `out/<GSE>/<GSM>/*` and hides `private/`.
  2. A study-file writer.
  3. Facts from `handoff/<GSE>.json`.
  4. `called_ed` read from `cell_calls.tsv`.
  5. The `ambiguous` modality.
  6. The publish gate: nnz = 0 with cells is refused.
  7. A Beta policy object: a generator from `bench/public/results`, `private/*` always private, the policy name not
     "alpha".
  8. Packing `private/<GSE>/<GSM>` into the restricted sidecar, with depth tables where the labeller reads them.
  9. A new `annot_version` and revalidation of the labels on Beta counts.
  10. Junction table v2.
  11. Whitelists for 3′ v4 and 5′.
  12. Documenting the Alpha and Beta additions in `format-s1.md` §7 and §9.
  13. An `.s1` regression job for Alpha.
- **Done when:** the 8 smoke samples and the 50 validation samples transcode, verify (`verify --policy beta`) and read
  back through the Python and R clients.

### W6. Beta stack and site (Sonnet 5.5). Design now, deploy after Zach approves.

- **Goal:** beta.singlet.bio serves `.s1` studies from its own resources and updates per study.
- **Owns** (singlet-website):
  - Worker and Pages config for the Beta channel;
  - a D1 clone with schemas 032–039;
  - the proxy;
  - the `restricted.ts` `private/*` rule;
  - per-channel handling of `alpha_exclusions`;
  - the rolling-ingest workflow (per study: objects to R2, then manifest last, then D1 SQL);
  - the Beta benchmark hub and roadmap pages from #66, re-labelled "measured on Beta";
  - fixing the dead `v2-performance.md` link.
- **Gates:** each Cloudflare resource creation and deploy is listed and approved by Zach. No `main` merges until Alpha
  ships.
- **Done when:** the validation studies are live on beta.singlet.bio, and the Python, R and MCP clients read them with
  only `SINGLET_DATA_BASE` changed.

### W7. Benchmarks and the release gate (Sonnet 5.5). After W1 phase A; continuous.

- **Goal:** every public layer is justified on real data at the pinned commit.
- **Owns:**
  - `bench/public/**`;
  - folding the result branches (`bench/public-*`, `bench/exact-vs-pseudo`, `bench/cellcall-independent-truth`) into
    #40;
  - `bench/public/results/*.json`;
  - the gate file that W5's policy generator reads.
- **Work:**
  1. Re-run every capability writer on each candidate pin.
  2. Reframe the 10x-truth cell-calling metrics as "agreement with Cell Ranger", and keep the independent-truth F1 as
     the headline.
  3. Build the missing truth sets:
     - Kang doublet artefacts;
     - the genotype cohort rows;
     - the extra hgmm libraries.
  4. **Doublet and empty-droplet benchmarks** (Zach requires both before the doublet combiner #43 is decided). Use real
     truth only:
     - **Doublets:** genotype cross-donor doublets (Kang and the genotype pools), HTO multiplets from hashed libraries, and
       human-mouse barnyard mixtures. Compare against scDblFinder, DoubletFinder and Scrublet, plus 10x's published
       outputs where they exist. Report heterotypic vs homotypic separately.
     - **Empty droplets:** the independent-truth set (Kang genotype F1 0.991) plus barnyard and hashed libraries,
       against CellBender, EmptyDrops_CR and Cell Ranger's published calls.
  5. Measure SRA Lite on SNP and mtDNA truth.
  6. Non-host: species-level truth (spike-ins or infected samples with orthogonal detection), and viral gene-level
     truth if W3's prototype is kept.
  7. Report pass or fail per layer.
- **Done when:** every capability file is `final` at the pinned commit, and the gate file exists.

### W8. Independent QC and audit (Sonnet 5.5, with Haiku 5.5 sweeps). Starts now; continuous.

- **Goal:** nobody grades their own homework.
- **Owns:**
  - the BSF checklist (§7) and its status;
  - `docs/research-beta/qc/`;
  - QC scripts, which are read-only against outputs.
- **Work:**
  - review every engine, s1 and orchestrator PR at medium effort;
  - re-derive the identity and determinism claims from logs;
  - check that every benchmark number uses real data and the pinned commit;
  - scan validation outputs for reads, BAMs, genotype or path leaks;
  - run Alpha's data-QC sweep (`aqc`) on the Beta validation samples;
  - keep the open-decision list current.
- **Writes:** no product code, ever.

### W9. Campaign operations (Sonnet 5.5). Starts at BSF.

- **Work:**
  - the 50-sample validation (both clusters' paths);
  - the Anvil one-node pilot (~200 SU, with approval);
  - the cost-model refit and the final projection for approval;
  - waves a–d with rolling publication through W6's ingest;
  - weekly status into `docs/research-beta/campaign-status.md`.
- **Single cluster owner:** W9 is the only session that submits campaign jobs.

### W10. Other modalities to release quality (Sonnet 5.5, one session per modality). After W1 phase A.

Zach (2026-10-27): every modality is developed and debugged before anything runs at production scale. Transcriptomics
still runs first; the other modalities then run as campaigns on the same orchestrator and the same pinned software.

| Session | PR and note | Done when |
|---|---|---|
| W10a ADT/HTO | #29, [adt-hto.md](adt-hto.md) | ADT counting matches published outputs; HTO demultiplexing passes on real hashed truth |
| W10b V(D)J and CRISPR | #51, [vdj-crispr.md](vdj-crispr.md) | clonotypes and guide calls pass against 10x's published outputs; no MiXCR |
| W10c ATAC, Multiome, spatial | #48, [atac-spatial.md](atac-spatial.md) | fragments, peaks and Visium spots pass; the correct scATAC whitelist ships |
| W10d mtDNA lineage | #53, [mtdna.md](mtdna.md) | lineage calls pass on MAESTER or other real truth; deterministic |
| W10e cell type | #32, #44, [cell-type.md](cell-type.md) | the model is re-validated on Beta counts at the pinned commit |

**Rules for W10:**
- **Rebase:** each session rebases its PR onto the Beta engine and owns only its modality's files.
- **Shared engine files:** changes to them go through W2 (pileup) or W3 (STAR and plumbing) as reviewed PRs.
- **Benchmarks:** each modality's capability page is written by W7 from real truth.
- **Pacing:** at most two W10 sessions run at once, to keep token use down; the order is the table order.
- **Doublet/ambient study (#43):** goes to W7 (doublet and empty-droplet benchmarks), then a decision.

**The orchestrator** records every dropped run with its library type, so the modality campaigns pick their runs up without
re-screening.

## 6. Sequencing

```
now ─────────► Alpha ships ─────────► +~3 wks ─────────► BSF (~early Dec) ─────────► campaign (~95-105 days)
W1 phase A ──► W1 phase B (CI) ───────────────────────────────────────────────────────────────────►
W4 fetch tier + orchestrator ─────────► smoke w/ whitelists ──► 50-sample validation (W9)
W8 checklist + reviews ───────────────────────────────────────────────────────────────────────────►
W6 design ───────────────► W6 deploy (approved) ──► validation studies live ──► rolling ingest ───►
           W2 correctness (after W1-A) ──────────► fixes merged
           W3 performance (after W1-A) ──────────► final pin (PGO)
           W7 benchmarks (after W1-A) ───────────► gate file at the pin
                          W5 (after #54) ────────► producer + policy
           W10a-e modalities (after W1-A, two at a time) ──► every modality at release quality
```

The BSF now includes every modality (decision on record, §0), so it moves later than "early Dec". W8 re-estimates the date
once W10's sessions report their remaining work.

**File ownership is disjoint.**
- **W1:** CI, config and phase-A renames.
- **W2:** pileup callers and fq.
- **W3:** STAR, non-host and stage plumbing.
- **W4:** `scripts/atlas_rerun`.
- **W5:** `s1` and the clients.
- **W6:** website.
- **W7:** `bench/public`.
- **W8:** QC docs only.
- **W10:** each modality's own files.

**One shared file:** `pileup_engine.h`, where W3's change waits for W2's.

**One Clipper bridge:** sessions use `clipper.py` but never start or restart `bridge.py`. If it drops, they pause and
report.

## 7. Beta Software Freeze checklist

**Engine**
- [ ] Bugs 1–9 of research-beta.md §4 are fixed with real-data evidence, or their layer is `omit` in the Beta policy.
- [ ] Outputs are identical across 4/8/16 threads, shuffled run order, shuffled barcode order and two reruns. This
      covers count matrices, calls, junction tables, demux, doublets, ambient and mtDNA.
- [ ] Every speed-up has an identity report on real 3′, 5′ and multi-run data, or a benchmarked difference that has been
      accepted.
- [ ] The download quality decision (full vs Lite) is made on real SNP and mtDNA truth.
- [ ] The edge cases all work: 101-nt R1, 5′ v1/v2, 3′ v2/v3/v4, multi-run, mismatched read lengths, and the non-10x
      and split-seq gates.
- [ ] One PGO binary is pinned (sha256, commit, build flags), and every output's provenance records the engine commit.

**Benchmarks**
- [ ] Every capability JSON is `final` at the pinned commit, on real data with real truth, including the Pareto and reach
      panels.
- [ ] Doublet and empty-droplet benchmarks are published, and the #43 combiner decision is taken from them.
- [ ] Every modality (W10a–e) passes its benchmark at the pinned commit.
- [ ] The non-host screen is under 3% of wall time and 2 GB of extra RSS, with identical tables.
- [ ] The gate file is generated from those results, and the Beta policy is generated from the gate file, with its
      sha256 pinned.

**Campaign**
- [ ] The fetch tier runs on both clusters within the login-node limits RCAC approved.
- [ ] Long-term storage stays within ~10 TB on Clipper (§3d), as measured on the validation samples.
- [ ] The unsorted-BAM protocol ([bam-stream-protocol.md](bam-stream-protocol.md)) has reached its phase 6 decision.
- [ ] The staging age and byte/file caps work, and so does shipping before the purge.
- [ ] The smoke test is green with whitelists, and the 50-sample validation is green.
- [ ] A validation rerun is identical.
- [ ] No reads, BAMs, genotypes or cluster paths appear in any public output (W8 scan).
- [ ] The cost model is refitted, and the calendar and SU projection is approved by Zach. Extra ACCESS credits have
      been requested if needed (§8).

**Publication**
- [ ] The `.s1` producer verifies all validation samples (`verify --policy beta`), and the clients read them back.
- [ ] The restricted sidecar holds the private outputs, and the labeller finds the depth tables there.
- [ ] Labels are revalidated on Beta counts, under a new `annot_version`.
- [ ] The Beta stack is live, with rolling ingest tested on the validation studies.
- [ ] The Alpha `.s1` regression is green.

**Repository**
- [ ] CI runs on every PR base. Lint covers `scripts/` and `bench/`. C++ formatting is checked on changed lines.
- [ ] The PyPI trusted publisher and pkgdown are fixed; tags are tidy; CHANGELOG is updated.
- [ ] `format-s1.md`, research-beta.md and the runbooks are current.

**Operations**
- [ ] Alpha has shipped and the freeze is lifted.
- [ ] The GVSU uplink inquiry has an answer.
- [ ] The Anvil maintenance window has passed or is planned around.

## 8. Decisions

**Taken (Zach, 2026-10-27):**
1. **Freeze scope.** Alpha finishes everything it needs on Clipper first. Development and benchmark jobs may run only on
   terms that cannot delay Alpha (§9).
2. **Login-node downloads.** Approved by Amiya (Purdue RCAC). No help-desk question is needed.
3. **Acceleration.** FASTASSIGN goes in if equivalent, plus a rebuilt PGO binary. BAM retirement stays a two-week
   timeboxed attempt, now that its implications are written down (§3b).
4. **Scope.** Transcriptomics first, then the other modalities, but every modality is developed and debugged before any
   production-scale run (W10).
5. **Non-host screen.** Make it cheap and keep it on (§3e).
6. **Doublets and empty droplets.** Both benchmarks come before the #43 decision (W7).
7. **Models.** Sonnet 5.5 for the workstreams, Opus 5.5 for orchestration, Haiku 5.5 for sweeps.
8. **Storage.** No BAMs or reads are kept. About 10 TB of long-term storage, on Clipper only (§3d).
9. **No Anvil work** of any kind until the campaign is ready to deploy.
10. **GPU alignment** is not pursued for Beta (§3b).

**Still open:**
1. **Non-host stage 2 (§3e).** Which viral, bacterial and vaccine references go into the curated set, and whether
   stage-2 gene counts and the vaccine, infection and latency signatures are public in Beta or only after W7 validates
   them.
2. **Viral gene-level non-host counts:** in Beta or later (§3e).
3. **Anvil pilot.** The ~200 SU one-node pilot after the Nov 6–8 outage.
4. **More Anvil credits.** ACCESS allows a PI several Explore, Discover or Accelerate projects ("multiple requests
   allowed"). Limits per project: Explore 400k, Discover 1.5M, Accelerate 3M credits. The credit limit of a project
   type cannot be exceeded by supplements; the route to more is a higher tier.
   - **A second project** must have a distinct research objective. For example, a benchmarking and methods project
     separate from the atlas re-run. A second small project for the same campaign could be read as splitting to dodge the
     limit.
   - **The clean route:** upgrade the existing project to the next tier with a short proposal.
   - Ask ACCESS (or Amiya) which route fits before applying. Each 100k SU moves ~0.5 PB off Clipper's link.

## 9. Rules for every workstream

**Data and release**
- **Real data.** Real ground truth only; simulation only as a corroborated supplement.
- **No unapproved changes.** Nothing changes R2, D1, Workers, DNS or Pages without Zach's explicit approval. No Anvil
  SU without approval.
- **Private stays private.** Genotypes and `private/` never reach a public object.

**Clusters**
- **Login nodes:**
  - **Clipper:** only `sbatch`, `squeue`, `sacct` and log reads.
  - **Anvil:** nothing until the campaign is ready to deploy; then the approved fetcher only.
- **One bridge.** Never run `hpc/clipper/sync.sh` while Alpha or campaign jobs run.
- **Clipper jobs** use `short-cpu` or `debug-cpu` where they fit, and say how much compute they need before
  submitting.
- **Alpha first.** Until Alpha ships, development and benchmark jobs:
  - run with `--nice=10000`, so Alpha's jobs, also under account `debruinz`, start first;
  - run with array throttles of `%4` or less;
  - stay off any partition where Alpha has jobs pending.
  Check `squeue -u $USER` before each submission, and stop submitting if an Alpha job is waiting on resources.

**Process**
- **Every change goes through a PR on `main`** with CI. Stacked branches are temporary.
- **Handoff notes.** Committed under `docs/research-beta/` at every stop, so a fresh session can resume.
- **Token economy.** Use the cheapest model that can do the job well; read with `git show` instead of checking out;
  don't poll.
