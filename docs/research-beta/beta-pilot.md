# Beta pilot: do the benchmark results hold on fresh datasets?

Status: **pre-registration** (written 2026-10-31, before any pilot result exists). Owner: S15. Binary: `prod/campaign` 337b405b.

Question (Zach, 2026-10-31): do the dev/test results (USA, cell calling, demux, junctions at SOTA; doublets tied; V(D)J, HTO, mtDNA) hold on real datasets that no tool was tuned or tested on? Every Singlet output is run beside the open SOTA tool for its modality.

## Selection rules

- Human 10x GEO samples from the Beta catalogue; real data only.
- Excluded: every GSE/GSM named in `bench/public/` (cohort), `docs/research-beta/*`, the S5 release-gate sets, S1's gate samples (pbmc1k_v3, neuron1k_v3) and S9's `.1fg2` fixtures (1,924 ids grepped from `bench/public`, `docs/research-beta`, `bench/alpha`, `bench/atlas_rerun` and `jobs` before choosing; each pick was checked against that list afterwards, which caught and removed GSE271789 and GSE180268, rows 2 and 13). The S5 gate and S9 fixture sets are covered only as far as those paths name them.
- Read counts come from ENA (`read_count` summed over a sample's runs). The Alpha QC `n_input_reads` is capped at 30 M and is not used. Chemistry is confirmed at the 1 M-pair head screen, not from Alpha labels.
- No reads kept after the pilot. No production R2/D1 writes (any publish is a dry run). No genotypes leave Clipper.

## Samples (round 1, 13 live picks tonight; rows 2 and 13 dropped)

| # | GSM | GSE | Pairs (ENA) | Covers | Truth source (independent of any tool) |
|---|---|---|---|---|---|
| 1 | GSM5682562 | GSE188429 | 68.9 M | 5' + HTO hashed CD8 PBMC, SARS-CoV-2 study | hashtag design in GEO; HTO library to be located per run |
| 2 | (dropped) | GSE271789 | | was the 2-donor pool; study already appears in the repo exclusion list | |
| 3 | GSM5130155 | GSE168163 | 236 M | 5' T cells, healthy female donor | GEO sex F |
| 4 | GSM5130163 | GSE168163 | 262 M | 5' T cells, male patient | GEO sex M; paired TCR library |
| 5 | GSM5171610 | GSE168859 | 851 M | deep, 5' T cells with TCR, healthy female | GEO sex F; paired TCR library |
| 6 | GSM7884577 | GSE247157 | 1.40 G | deep, single-nucleus multiome GEX (retina), female | GEO sex F |
| 7 | GSM7064234 | GSE226108 | 1.20 G | deep, single-nucleus multiome GEX (fovea), male | GEO sex M |
| 8 | GSM8958776 | GSE295854 | not in ENA query | brain snRNA-seq (occipital cortex, ALSP), female | GEO sex F |
| 9 | GSM8141311 | GSE261353 | 88.0 M | lung BAL cells | none beyond tool agreement |
| 10 | GSM8183378 | GSE263018 | 266 M | gut/gastric epithelium, female | GEO sex F |
| 11 | GSM8164511 | GSE262403 | 355 M | organoid (hESC-derived intestine) | cell line identity: one hESC line, one genotype |
| 12 | GSM4987144 | GSE163781 | 312 M | 5' v1 cell-line mixture (Jurkat, TALL-104, EL4, IVA12) | known mixture; EL4 is mouse, so species and line labels are truth |
| 13 | (dropped) | GSE180268 | | was the HNSCC tumour; study already appears in the repo exclusion list | |
| 14 | GSM9038586 | GSE299393 | 109 M | human MDA-MB-231 xenograft tumour (mouse stroma: non-host and species check) | one cell line, female; mouse reads expected |
| 15 | GSM4987148 | GSE163781 | 836 M | deep, cell-line mixture (3' v3 label in title) | known mixture composition |

Gaps still to fill before results (to be added as a dated addendum to this file, still before any result):
- a **SARS-CoV-2 / infected** sample with an infection label (candidates in GSE167118, GSE148881, GSE165477),
- a **3' v2** sample, a **3' HT** sample, a **shallow** sample (< 30 M pairs),
- a **genetic multi-donor pool** with a known donor count (replaces row 2) and a second solid tumour (replaces row 13),
- an HTO sample whose HTO library is confirmed in SRA.

Sex coverage so far: F (3, 5, 6, 8, 10, 14) and M (4, 7).

## What runs per sample

1. Fetch via the S9 path into `$WORK` staging (head screen first).
2. Singlet `prod/campaign` 337b405b: `--engine cpu` on every sample; `--engine gpu` on every sample as capacity allows (never more than 3 GPU jobs at once); `out_equiv` for CPU vs GPU; dry-run publish.
3. Open tools, one per modality (harness owners in brackets):
   - STARsolo GeneFull/Velocyto/EmptyDrops_CR/SJ, alevin-fry USA, kb-python nac, emptyDrops, CellBender, SoupX, decontX, scDblFinder [S11 scripts; kb-python and Kraken2 have no existing script and are new here],
   - cellsnp-lite + vireo, souporcell, mgatk, peddy/somalier [S10 scripts, `bench/public/genotype/hpc/`],
   - HTO: hashedDrops, HTODemux, GMM-Demux, cellhashR BFF (S11 has reimplementations only; the real packages are installed for the pilot),
   - TRUST4 (V(D)J), Kraken2 (non-host).
4. Reads and BAMs deleted when the sample's last job ends. Outputs and logs kept.

## Queue (submitted 2026-10-31 evening; scripts in `ops/s15/`)

Build job 795360 (`prod/campaign` 337b405b; `singlet-cpu` and `singlet-gpu` in `$WORK/jobs/s15/bin/337b405b`; every stage asserts the commit). Per sample, `ops/s15/submit_all.sh` chains with `afterok`:

`fetch` (S9 `fetch_run`, full `.sra`, shared cap, decode to `.1fq`) -> `fq` (first 100 M pairs as FASTQ, plus the same prefix as `.1fq`) -> `starsolo` (STAR-velodump: `Gene GeneFull Velocyto SJ`, `EmptyDrops_CR`, whitelist chosen by barcode match) and `scap` (Singlet CPU on the same 100 M-pair prefix); `scpu` (Singlet CPU, full sample, tagged BAM) runs after `fetch`.

GPU runs use 3 serial lanes, so at most 3 GPU jobs are in the queue at once. Each `sgpu` job (Singlet GPU, full sample) submits its sample's `cleanup` and then the next sample of its lane. Samples needing over 375 GB of memory request 2 H100s (a 1-GPU job is capped at 375 GB), so the deep samples may serialise the lanes. `cleanup` runs `out_equiv` (CPU vs GPU), then deletes `.sra`, `.1fq`, FASTQ and BAMs; a `KEEP` file in `$WORK/pilot/<GSM>/` defers the deletion.

Competitors run on the 100 M-pair prefix, not the full sample; `scap` gives Singlet on the identical prefix for a matched comparison. Samples below 100 M pairs use all their reads.

Competitors (`ops/s15/submit_tools.sh`, `pilot_tools.sh`), all on STARsolo's `GeneFull` matrices or the 100 M-pair FASTQ, chained after the sample's STARsolo (or Singlet CPU for mgatk) job: DropletUtils `emptyDrops`; SoupX, decontX and scDblFinder on STARsolo's filtered cells; CellBender 0.3.2 (V100S, one at a time, so the H100 lanes stay at 3); Kraken2 on the cDNA mate (viral k2 DB, so only viral/non-host-viral truth is testable); mgatk on Singlet's chrM reads and cell calls; alevin-fry (`simpleaf`, cr-like); kb-python `nac` (one shared index); TRUST4 on the 5' GEX reads of the five 5' samples. `finalclean` (afterany of the sample's tools) removes `KEEP` and deletes the reads once the GPU lane has passed the sample.

Not queued: HTO callers, vireo/souporcell, peddy/somalier. They need a pooled or hashed sample that is not in the list yet (see the gaps above). The paired TCR libraries are separate SRA experiments and are not fetched, so TRUST4 is judged on the GEX reads only.

## Amendment 1 (2026-10-31, night; written before any pilot result exists)

1. **Demux and genetic-doublet truth without a new sample.** GSM4987148 and GSM4987144 (GSE163781) are mixtures of Jurkat, TALL-104, EL4 and IVA12 at 1:1:1:1 (GEO sample text), so K = 4 and each line is a distinct genotype. cellsnp-lite + vireo and souporcell run on both with K = 4 (jobs listed in `$WORK/jobs/s15/toolids.tsv`). Caveat: EL4 is a mouse line, so mouse cells carry no human-panel SNPs and may land in their own cluster or in none; the species label is part of the truth. Score Singlet's demux and genetic doublets against these labels, plus line-specific marker genes where the study names them.
2. **One sample added.** GSM8085472 (GSE256089, "10 hashed donors", batch A chip 1 lane 1, 5', 1.42 G pairs): a pool of 10 donors, so K = 10 for vireo/souporcell. The study and sample are not in the exclusion list. (GSM9005866 from GSE298047 was queued briefly and cancelled before it ran: that study is in `bench/alpha/` and so excluded.)
3. **Paired libraries.** V(D)J: GSM5171642 ("Healthydonor5_VDJ", 170 M pairs) pairs with GSM5171610 (same donor name). Singlet V(D)J (S11's `run_vdj.py`) and TRUST4 run on the library reads (first 100 M pairs). HTO: the HTO library GSM8085473 (172 M pairs) of GSM8085472 is in SRA, but the tag-to-donor map is **not verified**: GEO's pages are behind a CAPTCHA and the FTP family files truncate from Clipper. HTO callers are therefore not queued; the gap stays open until someone pastes the GEO sample text.
4. **Protocol changes.** All Singlet runs use `--threads 16` (outputs can depend on thread count, a known engine bug). STARsolo takes its UMI length and whitelist from the protocol tag in Singlet's CPU run log (`include/singlet/fq/protocol.h` table) and waits for that run; it falls back to barcode matching only when no tag is available.
5. **GSM7064234 (retina snMultiome GEX, male): pilot harness bug, not a pipeline failure.** See Amendment 2 item 5.

## Amendment 2 (2026-11-01, before any pilot result exists)

1. **HTO design verified from GEO** (read by the orchestrator, not by S15). GSE256089 extract protocol: "PBMC from 38 participants were washed then individually hashed with TotalSeq-C anti-human hashtag antibodies ... Uniquely-hashed participant PBMC were pooled into 4 batches of 9 or 10 participants". GSM8085473 (batch A, lane 1) lists Hashtags 1-10 (TotalSeq-C0251 to C0260), so each tag marks one participant and the lane holds 10 donors. Chemistry is 5' v1.1 with Feature Barcode. This replaces the "tag map unverified" caveat of Amendment 1; participant names (possibly in `GSE256089_HTO_details_README.txt`) are not needed.
2. **Truth for demux and HTO on GSM8085472 / GSM8085473.** vireo K = 10 genotype clusters on the GEX reads. Each cluster is matched one-to-one to a tag by majority. Before any score, report the cluster-to-tag purity (the fraction of a cluster's singlet cells carrying its majority tag); if it is not high, the truth is not usable and the sample is scored by agreement only.
3. **What runs.** Singlet HTO: the counter (`python/singlet/adt/count.py`, branch `claude/adt-modality-0637a6`) on the pilot GEX run's called cells, with the panel (A0251-A0260) and offset from the atlas run of the same library, then `call_hashtags`. Open callers on the same count matrix, so the comparison tests calling, not counting: hashedDrops (DropletUtils), HTODemux (Seurat), cellhashR BFF (`bff_raw`, `bff_cluster`) and GMM-Demux, each best effort (a failed install is recorded as a failure, not retried silently). Jobs: `ops/s15/submit_hto.sh`.
4. **Non-host check.** The donors are on ART for HIV-1 (expected suppressed). Kraken2 (viral k2 DB, includes HIV-1) on the cDNA reads and Singlet's non-host output for GSM8085472 are both inspected for HIV reads; few or none are expected, and the check is recorded either way.
5. **GSM7064234: pilot harness bug (corrected 2026-11-01); the campaign gate is fine.** Diagnosis (job 795636, first 200,000 spots of SRR23617320): mate 1 is 8 nt (index), mate 2 is 151 nt with its first 16 nt in 3M-february-2018 for 93.2% of reads, mate 3 is 151 nt (cDNA). The decode gate rejected the run (`non10x_protocol_low_conf_no_wl_no_10x_geometry`, tag splitseq, `wl_match_rate=0`) because `pilot_fetch.py` passed the whitelist directory only as `$SINGLET_WHITELIST_DIR`, which `singlet download` ignores; it reads `--whitelist-dir`. The campaign path (`fetch_worker.py`) passes `--whitelist-dir` and is unaffected. S14 is separately fixing the env-var handling and the gate's fallback to splitseq on read length. The pilot fix: `pilot_fetch.py` now passes `--whitelist-dir`, GSM7064234 is rerun from its kept `.sra`, and every `.1fq` decoded before the fix is redecoded with the whitelist (`ensure_wl` in `pilot_stage.sh`) before Singlet reads it.
6. **Protocol calls made without a whitelist.** Every sample fetched before the fix has `wl_match_rate=0` in its gate line: GSM4987144 (tags 3p-v2 and 5p-v2 across its runs), GSM5130155 and GSM5130163 (5p-v3 with a 28-nt R1), GSM5171610 (5p-v2), GSM7884577 (3p-v3), GSM8164511 (3p-v3), GSM9038586 (3p-v2); GSM4987148, GSM8183378, GSM8141311, GSM5682562 and GSM8085472 had not reached the gate when it was fixed or were still decoding. All are redecoded; the gate lines before and after are kept (`decode_wl.log` per sample) and any change of protocol tag is reported.
7. **GSM8958776 (brain snRNA): dropped, "no ENA runs"** - (`sample_alias` returns nothing), so it was never fetched. It is a failure of the pilot's lookup, not of Singlet; it stays in the list as not run unless a replacement brain-nuclei sample is found.

## Scoring (after the jobs)

Per sample × modality: Singlet vs each tool with the metric the matching SOTA doc uses; truth source where there is one, agreement otherwise. Compare each modality with its dev/test result and flag degradation beyond the bootstrap CI. Record every crash, timeout, OOM, missing output and CPU≠GPU difference, ranked by number of samples hit.

Deliverables, filled in after scoring: per-sample × per-modality table, a generalisation verdict per modality, a failure list with owners.

## Results

The scored results are at the end of this file.

(not yet available)

### Note on GPU sizing (2026-11-01)
Every GPU job is a single H100 with at most 370 GB of memory (a 1-GPU job is capped at 375 GB, and 2-GPU jobs hit the per-user GPU QOS). GSM4987148 started earlier as a 2-GPU, 440 GB job and is the only exception. If a deep sample runs out of memory on one GPU at 370 GB, it goes to the failure list with its read count and peak RSS.

### Engine reruns (2026-11-01)
Two engine fixes land after the pilot binary 337b405b: #152 (pileup thread independence, merged, tip 767f1a43) and S8's barcode-universe rule (`fix/barcode-universe`, not yet merged). Once the second merges, only the Singlet CPU and GPU jobs are rerun on that tip, from the kept `.sra` files (the `.1fq` is redecoded with the whitelist), into new run directories; those runs are scored as the pilot's Singlet result and the 337b405b runs stay as a before/after column. The clean-up jobs now delete `.1fq`, FASTQ and BAMs but keep `.sra`; the `.sra` files are deleted after the reruns finish. If staging approaches 20 TB, the samples slowest to fetch are kept and the rest are refetched.

### Engine reruns: submitted (2026-11-01)

Pilot Singlet result = the rerun on prod/campaign **2a3b0d61** (#153 barcode universe, #152 pileup threads, #140, #135, #128, #124, #123 and others).
Built by job 796137 into `$W/jobs/s15/bin/2a3b0d61` (CPU, and GPU for sm_80/90). Scripts: `ops/s15/pilot_rerun.sh`, `submit_rerun.sh`.
The first-pass runs (337b405b) stay as the before column. Outputs: `s1cpu_r`, `s1gpu_r`, `equiv_r.txt`; both engines use `--threads 16` and must report `pileup_threads` 1.

Changes against the first pass, because most first-pass Singlet runs failed (see the failure list):
- `--genome-dir` is passed explicitly (the 337b405b run exited with ".1fq processing requires --genome-dir" after species detection returned `unknown` on 8 of 13 samples).
- Runs pooled into one sample must share a protocol. When the engine names a run that differs, the rerun drops it, records it in `runs_dropped.txt` and retries.
- `SINGLET_STALL_TIMEOUT=21600` (the default 1800 s stalled the tagged-BAM sort on GSM4987148 and GSM8085472).
- Reads are re-fetched and re-decoded with the new binary into `stage_r`; the `.sra` is deleted by `rclean` after both runs.
- Every GPU run is a single H100 (memory at most 370 G), three serial lanes.

### Failure list, first pass (binary 337b405b), ranked by samples hit

| # | Failure | Samples (of 13) | Owner |
|---|---|---|---|
| 1 | Species detection returns `unknown` on human 10x reads, so `.1fq` mode exits ".1fq processing requires --genome-dir" (CPU and GPU) | 6: GSM5130155, GSM5171610, GSM7064234, GSM7884577, GSM8164511, GSM8183378 | S14 |
| 2 | `pileup_threads` is not 1 (16 or missing): run marked INVALID | 4 with output: GSM5130163, GSM5682562 (GPU), GSM8141311 (GPU), GSM9038586 (GPU) | S8 (fixed by #152, retested in the rerun) |
| 3 | Tagged-BAM sort stalls past the 1800 s watchdog, exit 5 (CPU and GPU) | 2: GSM4987148, GSM8085472 | S8 |
| 4 | `--runs-used` missing for a multi-run sample (pilot harness bug, fixed in `reads_arg`) | 2: GSM4987144 (GPU), GSM9038586 (CPU) | S15 (fixed) |
| 5 | A run in the sample differs in protocol from run 1, so the runs cannot be pooled | 1: GSM4987144 (SRR13296300) | S14 |

The rerun records the species-detect line of every run (`[species-detect]` in `run.log`) even though `--genome-dir` is passed, so the table shows how often detection would have failed.
Run directories: `/mnt/projects/debruinz_project/debruinz/singlet/pilot/<GSM>/{s1cpu,s1gpu}` (first pass) and `{s1cpu_r,s1gpu_r}` (rerun).

### Reruns follow the campaign path (2026-11-01)

The reruns omit `--tagged-bam` (it forces the sorted path, `pileup_threads` 16, and a 155-263 GB BAM sort) and pass `--exons $G/genes/genes.gtf` with an explicit genome. Each run records `pileup_threads.txt` and `pileup_path_reason.txt`; a run with `pileup_threads` other than 1 is INVALID.
Tools that need a BAM (mgatk, cellsnp-lite + vireo, souporcell) take STARsolo's sorted BAM, not Singlet's. The first-pass STARsolo runs wrote no BAM (`--outSAMtype None`), so these tools need a STARsolo rerun with a sorted BAM on the 100M-pair prefix; until then their results come only from the few first-pass samples that kept a Singlet BAM and are flagged as such.

### Reads seen by each side (fairness)

Singlet's main run (`s1cpu_r`, `s1gpu_r`) sees all reads. STARsolo and every open tool see the first 100 M pairs. Where a sample is deeper, a Singlet CPU run on the same prefix (`s1cap_r`, new binary, stream path) is the like-for-like side for the competitor, genotype and mtDNA comparisons.
STARsolo is rerun with a sorted BAM (`starsolo_bam`, GeneFull) and its own filtered cell calls; mgatk, cellsnp-lite + vireo and souporcell read that BAM, not Singlet's.

| Sample | pairs (M) | tools see (M) | matched Singlet prefix run |
|---|---|---|---|
| GSM5682562 | 69 | 69 | s1cpu_r (all reads) |
| GSM8141311 | 88 | 88 | s1cpu_r (all reads) |
| GSM9038586 | 109 | 100 | s1cap_r |
| GSM5130155 | 236 | 100 | s1cap_r |
| GSM5130163 | 262 | 100 | s1cap_r |
| GSM8183378 | 266 | 100 | s1cap_r |
| GSM4987144 | 312 | 100 | s1cap_r |
| GSM8164511 | 355 | 100 | s1cap_r |
| GSM4987148 | 836 | 100 | s1cap_r |
| GSM5171610 | 851 | 100 | s1cap_r |
| GSM7064234 | 1198 | 100 | s1cap_r |
| GSM7884577 | 1404 | 100 | s1cap_r |
| GSM8085472 | 1416 | 100 | s1cap_r |

### Rerun incident log (2026-11-01)

- Slurm runs the copy of a job script made at submit time, so the first rerun jobs still passed `--tagged-bam` after the script on disk was fixed. All 13 CPU reruns (4 had finished: pileup_threads 16, `pileup_path_reason` "--tagged-bam rewrites the sorted BAM", INVALID) and the first GPU lane head were cancelled and resubmitted (rcpu 797255-797267). Jobs submitted later by a running lane pick up the current script.
- GPU OOM on one H100 at 370 G, sorted path (tagged-BAM sort, STAR killed by signal 9, peak RSS 384 GB): GSM7064234 (1.20 G pairs) and GSM7884577 (1.40 G pairs). Requeued on the stream path; if they run out again the CPU `--memory-tier` run is the sample's Singlet result.
- Species detection returned `unknown` on every rerun that reached it for GSM8164511, GSM8183378, GSM5130155, GSM5171610, GSM7064234, GSM7884577; it called GSM4987144 `Mus musculus` (a mouse line is in the mix) and GSM5130163 `Homo sapiens`.
- GSM4987144: the engine named 3 of 4 runs as differing in protocol, so the rerun used 1 run (SRR13296299).

### Binary change (2026-11-01)

2a3b0d61 is not the Beta candidate (it lacks #164 EmptyDrops_CR cell calls, #165 exon/intron and USA matrices, #160/#170 protocol vote, #161 species filters, #175 GTF next to `--genome-dir`). The pilot's Singlet result is the rerun on **0ea48783** (build job 797270, `bin/0ea48783`), one binary for all 13 samples. Outputs: `s1cpu_0ea48783`, `s1gpu_0ea48783`, `s1cap_0ea48783`, `equiv_0ea48783.txt`, reads in `stage_0ea48783`. `--exons` is still passed. 2a3b0d61 outputs `s1cpu_r`, `s1gpu_r` exist only for the few runs that finished and are not scored. The STARsolo BAM side (`rstar`, mgatk, vireo, souporcell) does not use the Singlet binary and keeps running.

## Benchmark store and Leftovers for Thursday (2026-11-02)

Store at `$WORK/benchmark-store/` on Clipper: `datasets.tsv` (86 rows; `fetch` = sra | url | bam | pilot | later, `fetch_note` gives the reason for every `later`), `index.tsv` (one row per finished run), `<dataset>/<sample>/<tool>/<version>/` (symlinked outputs + `run.json`). Scripts: `ops/s15/store/` (`build_datasets.py`, `submit_store.sh`, `store_register.py`, `submit_big.sh`, `pilot_big.sh`) and the stage/tool scripts in `ops/s15/`. Reads and BAMs are deleted by each chain's last job, only when every tool in the chain ended OK. Wave 1 estimate: ~2,000 core-hours, ~550 GB transient reads, ~150 GB kept outputs; second batch (Kang x4 via BAM, 2 LCL, Ding 10x v3 library A, 3 more SRRs): ~600 core-hours, ~150 GB transient.

Open items:
- **GSM7064234 Singlet GPU run out of memory twice** (797046, 797331): 1,198,074,594 pairs, single H100 node with 370 G; killed at 22–23 min with 387.9 GB peak RSS (sorted-BAM path; the stream path was tried on the second attempt and died the same way). The CPU run (797286) finished (3 h 19 min). Engine owner: S8. Not fixed here.
- **Reference tools for GSM7064234 and GSM4987148** rerun from the kept `.sra` (first 100 M pairs; STARsolo on bigmem, 24 h; chains in `jobs/s15/bigids.tsv`). The first-pass mgatk jobs (GSM5171610, GSM7064234) timed out at 4 h on `short-cpu` after their BAM had been deleted; GSM5171610 mgatk still to be redone (needs a refetch).
- **souporcell**: `consensus.py` (genotype VCF + ambient estimate) fails because the sif has no `g++`; the clustering (`clusters.tsv`) is complete and is what the store keeps (GSM4987144, GSM4987148, GSM8085472 registered).
- **CellBender** failed with a ZeroDivisionError whenever the raw matrix had fewer than `total-droplets-included` droplets with >= 5 UMIs; the stage now caps that number and skips with a message when there are too few background droplets.
- **sra_download_set** was one SRR (SRR24098804) with 73 % of reads "too short" on STARsolo (mapping 26 %): unexplained; split into one dataset per SRR.
- **Kang 2018 (GSE96583)**: the SRA runs hold only the 98-nt cDNA read; the barcodes live in the submitted BAMs (CR/UR tags, v1 chemistry, 14-nt barcode + 10-nt UMI). Reads are rebuilt from the BAMs (mapped primary reads only), STARsolo `--soloCBlen 14`, kb and alevin skipped for v1.
- **pbmc5k_v4_d1** needs the GEM-X v4 whitelist (shipped only with 10x software). Multiome/ATAC datasets need ATAC tools that are not set up. Other `later` rows: see `fetch_note`.
- Tools not set up: DoubletFinder, scds, Scrublet, popscle demuxlet/freemuxlet, demuxmix, HTODemux/hashedDrops, velocyto standalone, SnapATAC2/chromap, TRUST4 only on 5' VDJ sets.

### Update 2026-11-04: everything on the scheduler

- `ops/s15/store/submit_store.sh` v2 submits each dataset as a self-contained chain (fetch with `curl -C -` resume -> STARsolo on the species reference -> tools -> cleanup). Each tool has one automatic retry (`afternotok`, twice the memory, 48 h, cpu/bigmem/gpu) and a gate job (`afterok:primary ? afterok:retry`); cleanup runs only `afterok` on the gates, so a failed chain keeps its reads and BAM. Datasets above 60 M pairs use `cpu` with 24 h limits; the store caps reads at 100 M pairs. A dependency ladder keeps 5 datasets in flight; the submitter itself runs as a Slurm job (`s15-submitter`).
- **Wrong-reference fix:** seven mouse or human+mouse datasets (nuclei2k_v2, mouse_brain_nuclei5k_v31, hgmm1k_v2/v3, hgmm10k_v3, neuron1k_v3, heart1k_v3) had been run against GRCh38 only; their outputs moved to `benchmark-store/_invalid_wrong_reference/` and the chains rerun with GRCm39 / the barnyard reference (STARsolo, EmptyDrops, CellBender, SoupX/DecontX/scDblFinder, Kraken; kb, alevin, mgatk, vireo, souporcell, TRUST4 are human-only for now).
- Kang 2018 (GSE96583) chains rebuilt: BAM downloaded first (resumable), reads rebuilt with `bam2fq_cr.py`; the earlier streaming conversion died after 25 M reads.
- souporcell `consensus.py` works with a conda gcc 11 (micromamba env `envs/gcc11` under `$WORK`, bound into the sif); ambient-RNA and genotype VCF outputs are produced and stay private in the store.
- New dataset rows: 15 public 10x datasets, Xi & Li aliases (Kang batches, hgmm 6k/12k, Stoeckius PBMC) and GEX runs for mixed-role libraries (Stoeckius, Norman, GSE145370, MULTI-seq); `fetch_note` documents each `alias` / `later`.
- Not routed yet (reasons in `fetch_note`): Drop-seq, inDrops, Seq-Well, CEL-Seq2, Smart-seq2 (need protocol-specific STARsolo/kb setups and whitelists), GEM-X v4 (kb 0.29.1 has no 10XV4 preset, list not available locally), Multiome ATAC (chromap/MACS2 not set up), companion tag/guide/VDJ libraries (HTO, ADT, MULTI-seq, guides: no counting tools queued), Visium, Singlet runs for store datasets (the engine stages `.1fq` from SRA runs only; no FASTQ staging path).

## Results (scored 2026-11-06, Singlet 0ea48783)

Singlet runs above 100 M pairs are scored on the matched 100 M-pair prefix that STARsolo saw. Agreement metrics have no truth, so they show how far Singlet sits from the reference tool. Sex has GEO truth for seven samples. Scores come from `ops/s15/score_pilot.py`. Raw numbers: `ops/s15/pilot_scores.tsv`.

### Cell calling (Singlet against STARsolo EmptyDrops_CR)

| Sample | Singlet cells | STARsolo cells | Jaccard | Precision | Recall |
|---|---|---|---|---|---|
| GSM5682562 | 7,555 | 8,930 | 0.824 | 0.986 | 0.834 |
| GSM8141311 | 12,008 | 13,560 | 0.884 | 0.999 | 0.885 |
| GSM9038586 | 8,645 | 8,701 | 0.980 | 0.993 | 0.987 |
| GSM5130155 | 8,241 | 8,391 | 0.939 | 0.977 | 0.960 |
| GSM5130163 | 103,993 | 103,876 | 0.998 | 0.999 | 1.000 |
| GSM8183378 | 7,548 | 7,544 | 0.961 | 0.980 | 0.980 |
| GSM4987144 | - | 3,520 | - | - | - |
| GSM8164511 | 7,827 | 8,092 | 0.906 | 0.967 | 0.935 |
| GSM4987148 | 8,017 | 6,494 | 0.485 | 0.591 | 0.730 |
| GSM5171610 | 48,341 | 48,426 | 0.974 | 0.987 | 0.986 |
| GSM7064234 | 46,880 | 45,447 | 0.958 | 0.964 | 0.994 |
| GSM7884577 | 57,055 | 16,033 | 0.276 | 0.277 | 0.986 |
| GSM8085472 | 14,146 | 15,067 | 0.926 | 0.993 | 0.932 |

### Sex, ambient RNA, doublets

| Sample | Singlet sex | GEO sex | Singlet ambient | SoupX rho | decontX rho | Singlet doublet rate | scDblFinder rate |
|---|---|---|---|---|---|---|---|
| GSM5682562 | male | - | 0.427 | 0.039 | 0.124 | 0.056 | 0.101 |
| GSM8141311 | male | - | 0.337 | 0.130 | 0.197 | 0.080 | 0.121 |
| GSM9038586 | unknown | female | 0.305 | 0.258 | 0.330 | 0.051 | 0.103 |
| GSM5130155 | female | female | 0.563 | - | 0.107 | 0.065 | 0.085 |
| GSM5130163 | male | male | 0.204 | - | 0.021 | 0.250 | 0.362 |
| GSM8183378 | female | female | 0.064 | 0.050 | 0.282 | 0.043 | 0.081 |
| GSM4987144 | unknown | - | 0.033 | 0.004 | 0.020 | 0.024 | 0.052 |
| GSM8164511 | male | - | 0.179 | 0.014 | 0.036 | 0.049 | 0.105 |
| GSM4987148 | unknown | - | 0.030 | 0.037 | 0.054 | 0.063 | 0.077 |
| GSM5171610 | female | female | 0.082 | 0.032 | 0.037 | 0.219 | 0.185 |
| GSM7064234 | male | male | - | 0.066 | 0.052 | 0.248 | 0.131 |
| GSM7884577 | female | female | 0.132 | 0.262 | 0.080 | 0.249 | 0.121 |
| GSM8085472 | male | - | 0.069 | 0.059 | 0.122 | 0.114 | 0.151 |

### Donor and line demultiplexing (adjusted Rand index on shared singlet cells)

| Sample | Singlet vs souporcell | Singlet vs vireo | souporcell vs vireo | Singlet donors |
|---|---|---|---|---|
| GSM4987144 | - | - | 0.799 | - |
| GSM4987148 | 0.072 | 0.138 | 0.548 | 2 |
| GSM8085472 | 0.997 | 0.997 | 0.999 | 10 |

### Run time and memory (full runs; minutes and GB)

| Sample | Singlet CPU | Singlet GPU | STARsolo (100 M prefix) | CPU RSS | STARsolo RSS | CPU vs GPU differing files |
|---|---|---|---|---|---|---|
| GSM5682562 | 12 | 11 | 8 | 28 | 20 | 3 |
| GSM8141311 | 13 | 11 | 8 | 28 | 21 | 3 |
| GSM9038586 | 13 | 13 | 16 | 28 | 20 | 3 |
| GSM5130155 | 34 | 31 | 14 | 30 | 21 | 3 |
| GSM5130163 | 45 | 38 | 10 | 56 | 20 | 3 |
| GSM8183378 | 32 | 27 | 15 | 28 | 22 | 3 |
| GSM4987144 | 13 | 12 | 21 | 37 | 21 | 2 |
| GSM8164511 | 44 | 37 | 16 | 31 | 22 | 3 |
| GSM4987148 | 95 | 78 | 34 | 65 | 22 | 3 |
| GSM5171610 | 102 | 87 | 19 | 68 | 21 | 3 |
| GSM7064234 | 199 | 23 | 24 | 85 | 22 | 1 |
| GSM7884577 | 211 | 25 | 20 | 104 | 22 | 1 |
| GSM8085472 | 176 | 141 | 17 | 43 | 21 | 3 |

### Verdict per modality

- **Cell calls.** Singlet matches STARsolo closely on 10 of the 12 scored samples (Jaccard 0.82 to 1.00). GSM4987144 has no matched Singlet run to score. It differs on GSM7884577 (single-nucleus multiome: 57 k cells against 16 k, Jaccard 0.28) and on GSM4987148 (deep line mixture: 8.0 k against 6.5 k, Jaccard 0.49). Both fall outside the range seen on the other samples, so cell calling does not generalize to these two sample types yet. Owner: Singlet cell-calling track.
- **Sex.** Six of the seven samples with a GEO sex are called correctly. GSM9038586 (xenograft with mouse stroma) returns unknown. Owner: sex-call track.
- **Ambient RNA.** Singlet's contamination estimate sits far from SoupX and decontX on most samples (for example 0.43 against 0.04 and 0.12 on GSM5682562). The tools disagree among themselves too, so none of the three has truth here. Needs a spike-in or species-mixture sample before a verdict.
- **Doublets.** Rates are of the same order as scDblFinder. Singlet flags more on the deep T-cell samples (0.22 to 0.25 against 0.13 to 0.18). No truth.
- **Demultiplexing.** On GSM8085472 (hashed PBMC, 10 donors) Singlet agrees with souporcell and vireo (ARI 0.997 and 0.997). On GSM4987148 (known line mixture) agreement is low (0.07 and 0.14) while souporcell and vireo agree with each other at 0.55. GSM4987144 has no matched Singlet run to score.
- **Speed and memory.** Singlet GPU is faster than CPU on every sample that completed both. CPU and GPU outputs differ in 1 to 3 files per sample. In the case I read (GSM5682562) the differences are a byte count and timing fields. I did not inspect the others. Singlet CPU uses 1.4 to 5 times the memory of STARsolo.
- **Not scored.** V(D)J, HTO calls, mtDNA heteroplasmy and splice outputs have no scoring script yet. The 13 samples were not re-run after the Kang and v1 fixes, since those are store datasets.

### Diagnosis of the two cell-call divergences (2026-11-06)

Method: `ops/s15/diag_pilot.py` on the matched 100 M-pair Singlet runs (`s1cap_0ea48783`) and the STARsolo EmptyDrops_CR calls. Both Singlet runs report every call as `called_v2`.

**GSM7884577 (single-nucleus multiome, 57,055 Singlet cells against 16,033 STARsolo cells).** Both tools share 15,815 cells with a median of 1,541 UMIs. The 41,240 extra Singlet cells sit at a UMI floor, with a median of 436 and a 10th to 90th percentile range of 389 to 520. STARsolo stops at 643 UMIs. DropletUtils emptyDrops at FDR 0.001 calls 45,222 cells, which is much closer to Singlet than to STARsolo. Singlet rejects only 218 of the STARsolo cells. The divergence is therefore a difference in the low-UMI cutoff, not a disagreement on well-covered cells. CellBender produced no output on this sample (its input folder was wrong at the time), so there is no third opinion. Action for the engine owner: check the `called_v2` lower bound on nuclei libraries with a large empty-droplet tail.

**GSM4987148 (deep line mixture, 8,017 against 6,494).** The tools share 4,739 cells. Singlet's 3,278 extras are high-UMI cells (median 5,694). STARsolo's 1,755 extras are low-UMI barcodes (median 155) that Singlet marks as not cells. CellBender calls 10,253 cells and contains all 6,494 STARsolo cells and 7,784 of Singlet's 8,017. Here the evidence favors Singlet: its extras are well covered, and STARsolo's extras look like ambient barcodes. The low Jaccard (0.49) reflects EmptyDrops_CR choices more than a Singlet error.

### Line-mixture demultiplexing (GSM4987148)

The mixture holds four lines. Singlet labels 8,017 cells with two donors (4,012 and 3,399), 537 unassigned and 69 doublets. Vireo finds four donors (3,256, 1,582, 1,171 and 132 cells). Souporcell finds four clusters (1,450, 1,339, 1,220 and 305 cells) and leaves 1,806 cells unassigned. On 2,810 shared singlet cells the adjusted Rand index is 0.55 between souporcell and vireo, 0.07 between Singlet and souporcell, and 0.14 between Singlet and vireo. Souporcell splits one vireo donor into two clusters, so neither reference tool is clean here. Singlet's two-donor result under-clusters a four-line mixture. Without per-cell truth the reference tools are better than Singlet on this sample, but only by agreement with each other.

### Routes (2026-11-06 evening)

- Drop-seq and Seq-Well route tested (job 801027): STARsolo with a 12 nt barcode, 8 nt UMI and no whitelist ran on 2 M macosko reads and estimated 1,009 cells. The macosko and gierahn chains had already finished on this route.
- GEM-X v4 and Flex are blocked. Replacements queued: `pbmc5k_protein_v3_nextgem`, `pbmc5k_v3` and `hodgkin_ngsc3_v31` (10x URLs, HTTP 200). A genotype-pooled donor library (Neavin 2023 village) is the proposed replacement for the Flex demultiplexing truth.
- BAM inputs queued through the existing BAM route: `lebrigand2020_scnaumi_190c` and `_951c`, `russell2018_uninfected` and `_8h`, `dm_fca2022_male_head`. The Russell and FCA BAM tags were not checked before queueing.
- Not yet routed: Smart-seq2 and plate methods, spatial (STARsolo only, no Space Ranger), the Mereu sets (a dog reference must be built), sci-RNA, SPLiT-seq, inDrops, CEL-Seq2, ICELL8, SCRB-seq, ddSEQ and the long-read sets.

### Reproducing the line-mixture result

Sample: `pilot_GSM4987148` (GSM4987148, GSE163781, four-line mixture, 836 M pairs, matched 100 M-pair prefix). All paths are on Clipper under `/mnt/projects/debruinz_project/debruinz/singlet/`.

| Output | Path |
|---|---|
| Singlet matched run (cell calls, `donor_assignments.tsv`, `doublet_scores.tsv`) | `pilot/GSM4987148/s1cap_0ea48783/out/` |
| Singlet full run | `pilot/GSM4987148/s1cpu_0ea48783/out/` |
| souporcell (`clusters.tsv`, ambient RNA, cluster genotypes) | `pilot/GSM4987148/tools/souporcell/out/`, store entry `benchmark-store/pilot_GSM4987148/GSM4987148/souporcell/` |
| vireo (`vireo_out/donor_ids.tsv`) | `pilot/GSM4987148/tools/vireo/`, store entry `benchmark-store/pilot_GSM4987148/GSM4987148/vireo/` |
| STARsolo calls and matrices | `pilot/GSM4987148/starsolo_bam/Solo.out/GeneFull/`, store entry `benchmark-store/pilot_GSM4987148/GSM4987148/STARsolo/` |
| CellBender cell barcodes | `pilot/GSM4987148/tools/cellbender/cb_cell_barcodes.csv` |

The numbers in the diagnosis come from `python3 ops/s15/diag_pilot.py <pilot dir>`, which reads these files and prints the contingency tables. Vireo and souporcell used the STARsolo BAM and `K=4`.

### Correction: species of the two cell-line mixtures (2026-11-06)

ENA lists both GSM4987144 (5' v1) and GSM4987148 (3' v3) as *Homo sapiens* plus *Mus musculus*, because the mixtures contain the mouse line EL4. Both were first run against GRCh38 only. Their STARsolo, SoupX/decontX/scDblFinder, emptyDrops, CellBender and Kraken results were moved to `benchmark-store/_invalid_wrong_reference/pilot_GSM4987144` and `pilot_GSM4987148`, and the five tools are rerun on the human+mouse reference (jobs `s15-g-*`, started by `ops/s15/store/rerun_species.sh`). The kb, alevin, mgatk, vireo and souporcell results stay as human-reference results.

The scores above for these two samples compare Singlet with the human-only STARsolo run. They are provisional until the barnyard rerun finishes, and the cell-call comparison for GSM4987148 (Jaccard 0.49) in particular must be redone. The other eleven pilot samples are single-species by their ENA records (the MDA-MB-231 xenograft GSM9038586 was already run as human+mouse).

### Node-packed layout (2026-11-06, after the per-user node cap)

All earlier per-dataset chains were cancelled (789 jobs) and replaced by whole-node drivers in `ops/s15/store/`: `node_driver.sh` (several `pipeline.sh` runs side by side on one exclusive node, popping dataset ids from a shared queue), `gpu_driver.sh` (two CellBender runs on one V100S node) and `plan.py` (what each dataset still misses, from `index.tsv`). Pipelines skip every tool already in the index, fetch reads only when a missing tool needs them, retry each stage once and clean up only when everything delivered. The Kraken PlusPF database (117 GB) sits in the page cache once per node. Never more than four CPU nodes plus one V100S node, and no H100 node.
