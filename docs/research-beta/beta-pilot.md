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

Not yet queued (no usable template, or needs per-sample edits; to be added before `cleanup` runs, using `KEEP`): alevin-fry, kb-python `nac`, CellBender, DropletUtils/SoupX/decontX/scDblFinder, HTO callers, TRUST4, Kraken2, vireo/souporcell/mgatk, peddy/somalier.

## Scoring (after the jobs)

Per sample × modality: Singlet vs each tool with the metric the matching SOTA doc uses; truth source where there is one, agreement otherwise. Compare each modality with its dev/test result and flag degradation beyond the bootstrap CI. Record every crash, timeout, OOM, missing output and CPU≠GPU difference, ranked by number of samples hit.

Deliverables, filled in after scoring: per-sample × per-modality table, a generalisation verdict per modality, a failure list with owners.

## Results

(not yet available)
