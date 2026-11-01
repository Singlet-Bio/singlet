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
