# Singlet Research Beta: scope, status and handoff

Status 2026-10-26. **Research Alpha** is the current atlas, stabilized and served now at singlet.bio with an "alpha" tag (plan:
singlet-agents `docs/RESEARCH_ALPHA_PLAN.md`). **Research Beta** is the full re-run of every sample with the fixed, benchmarked
pipeline, released at beta.singlet.bio (estimated Q1 2027). This document is the entry point for Beta: what was built and
measured between 2026-10-22 and 2026-10-26, the decisions taken, the open bugs, and the ordered path to release. All work below is
on branches and draft PRs unless marked merged; nothing in this list changed R2, D1 or a Worker except where stated.

A freeze is in force until Alpha ships: no new cluster jobs on Clipper or Anvil, no release of held jobs, no website merges and no
D1/R2/Worker changes except from the Alpha session.

## How to resume

1. Read this file, then the note for the area you pick (section 1). Each note states the goal, what was done, results, decisions,
   open problems, the exact next steps and the Clipper job ids and paths of results left on the cluster.
2. Check out the area's branch or PR (section 1); every branch is pushed and every PR is a draft that says its status.
3. Before any cluster work, confirm the Research Alpha freeze has been lifted (singlet-agents `docs/RESEARCH_ALPHA_PLAN.md`).
4. Clipper: one bridge only (`clipper.py status`), never `hpc/clipper/sync.sh`; results of jobs that finished after this handoff are
   at the paths listed in each note.
5. Work in priority order: open bugs (section 4), then the path to Beta (section 5).

**Since 2026-10-27, start from [research-beta/rollout-plan.md](research-beta/rollout-plan.md).** It defines workstreams W1-W9, the
launch gate (Beta Software Freeze) and its checklist, and it supersedes section 5. Beta uses the same codebase as Alpha with its own
R2 bucket, data Worker, D1 database and Pages branch at beta.singlet.bio.

All notes in `docs/research-beta/` are copies made at the stop on 2026-10-26 from the feature branches and PR descriptions named in each file.

## 1. Where everything is

| Area | Handoff note | PRs (Singlet-Bio/singlet unless noted) |
|---|---|---|
| Pipeline benchmark, production binary, re-run plan | [pipeline-benchmark.md](research-beta/pipeline-benchmark.md) | #30 plan; #31 production engine; #36 modality fixes; #34 STAR speedups; #41 fast window assignment; #42 re-run orchestrator; #45 ambient calibration; #46 intronic + NUMT; #47 deep samples; #22 / #25 superseded by #31 |
| Public benchmarks: contract, cohort, harness, notebook, sex, ancestry | [benchmarks.md](research-beta/benchmarks.md) | #40 |
| Cell type | [cell-type.md](research-beta/cell-type.md) | #32 model and harness; #44 capability results |
| ADT / HTO | [adt-hto.md](research-beta/adt-hto.md) | #29 |
| V(D)J and CRISPR guides | [vdj-crispr.md](research-beta/vdj-crispr.md) | #51 |
| ATAC / Multiome and spatial | [atac-spatial.md](research-beta/atac-spatial.md) | #48 |
| mtDNA clonality / lineage | [mtdna.md](research-beta/mtdna.md) | #53 (full report docs/mtdna-lineage.md on bench/mtdna-lineage) |
| Doublet / ambient models | | #43 |
| **Rollout plan, workstreams and launch checklist (2026-10-27)** | [rollout-plan.md](research-beta/rollout-plan.md); evidence in [audit-2026-10-27.md](research-beta/audit-2026-10-27.md) | supersedes §5 below |
| Rename "v2" to ".s1" (Singlet 1.0) | [rename.md](research-beta/rename.md) (history: done through Research Alpha, `s1/` prefix) | #39, #56 merged; zdebruine/singlet-website#73, #74 merged |
| Benchmarking and roadmap pages | [website-benchmarking.md](research-beta/website-benchmarking.md) | zdebruine/singlet-website#66 (Beta preview) |
| Ontology filters | [ontology.md](research-beta/ontology.md) | zdebruine/singlet-website#54 |

Merged into main in this period: the per-sample storage format and clients (#13 with #12, #14, #15, #16, #18, #19, #23, #26, #28),
the empty-droplet study (#17, with #20), the library-type screen (#21), the scBaseCount comparison (#27), CI that reports the
required check on every PR (#24), test-flake fixes (#33, #35), and called_v2 2.1 (#37, #38). Junction table v1 is published in R2
(`v2/ref/`, 145 packs; index 16 to 161 entries). Website: MCP modalities (#53), retirement readiness (#52), cell-index endpoints
(#55, Worker not deployed).

## 2. What the benchmarks show today

Every published number uses real data with real ground truth; simulation is only a supplement corroborated by real data
(contract 1.1.0 enforces this). Status today: none of 22 capability pages passes yet; the website shows failing metrics as measured.

- **Equivalent or better:** gene counts and splice junctions/PSI against STARsolo (junction r 0.9999, PSI r 0.996); ADT counting
  against Cell Ranger (r 0.999995); non-host detection against Kraken2; chromosomal sex (97.0% of cells correct vs cellXY 68.0%);
  donor demultiplexing with an unknown donor count (exact on 6 of 6 held-out pools, vireo 0 of 6); cell type (leads CellTypist,
  scANVI, Seurat mapping and SingleR at every Cell Ontology depth); cell calling on independent truth (called_v2 F1 0.991 on Kang
  genotype truth vs CellBender 0.985, EmptyDrops_CR 0.954, Cell Ranger 0.942).
- **Behind, with a fix in progress:** spliced/unspliced and intronic assignment (#46); mouse mitochondrial genes lost to the chr1
  NUMT (#46); ambient bias (#45); donor SNP concordance (0.961 vs vireo 0.978) and false alternate alleles; HTO demultiplexing
  (2 of 8); exact vs pseudo-alignment combined accuracy (STARsolo 1.00, Singlet 0.70, alevin-fry 0.62).
- **Speed:** runtime is shown on a quality-vs-runtime Pareto chart, not as a pass/fail gate; Singlet is an exact genome-wide aligner
  and pays for it in runtime compared with alevin-fry, in exchange for reach (introns, junctions, SNPs, mtDNA, non-host).

## 3. Decisions taken (Zach)

- Re-run the whole atlas before any storage migration, with all reads of every run (no depth cap). Clipper by default; Anvil only
  quoted (505k-676k SU against a 112k balance).
- Real data with real ground truth for every published metric.
- Compare against 10x's published outputs; do not run 10x software. No MiXCR (licence key).
- Bundle the Visium v1 spot list and the Multiome ATAC-GEX pairs rebuilt from 10x public CC BY 4.0 data, with attribution.
- Rename "v2" to ".s1" (no "store" naming; keep magic bytes, `called_v2` to `called_ed`): code merged (#39, #56); the R2 `s1/`
  prefix, D1 and Worker cutover were done through Research Alpha (zdebruine/singlet-website#74). The C++ `called_v2` rename is
  still open on the production branches (rollout-plan W1).
- Alpha and Beta share one codebase; Beta gets its own R2 bucket, data Worker, D1 database and Pages branch at beta.singlet.bio,
  updated continuously as clusters deliver; the campaign starts only after the Beta Software Freeze (2026-10-27).
- Downloads must not spend Anvil SUs: fetch on Anvil login nodes, compute on the nodes (2026-10-27).
- Login-node fetching approved by Amiya (Purdue RCAC); every modality is developed and debugged before any
  production-scale run, with transcriptomics run first; keep reads as `.1fq`, not BAMs; Sonnet for workstreams, Opus for
  orchestration (2026-10-27; rollout-plan §8).
- Speed is a Pareto trade-off against pseudo-aligners, not a gate.

Open decisions: Anvil spend; non-host screen on or off in the re-run; GVSU uplink bandwidth test; storing SNP allele counts privately;
doublet combiner in 1.0.0; ancestry label for admixed Americans (AMR vs "admixed"); HTO caller.

## 4. Open bugs, in priority order

1. Mouse mitochondrial genes dropped as NUMT multimappers (#46).
2. Intronic multimapper rescue: rules ready, not default (#46).
3. Encoder drops every read when the 10x barcode read is 101 nt (GSE230239; see benchmarks note).
4. Default download is lossy (drops qualities the SNP and mtDNA callers use).
5. Barcode-order dependence in ambient, doublet and mtDNA callers.
6. Ambient calibration (#45, partial).
7. Automatic donor count can split a donor (merge fix specified).
8. Deep-sample BAM sort, multi-run input, download memory (#47).
9. The bundled scATAC whitelist `737K-cratac-v1.txt` is a copy of the 3' v2 list (#48 replaces it).
10. Unmapped / non-host reads must never reach public outputs (the orchestrator excludes them; keep it that way).

## 5. Path to Research Beta

Superseded on 2026-10-27 by [research-beta/rollout-plan.md](research-beta/rollout-plan.md); kept for history.

1. Fix bugs 1-9 on the production branches; every page of the benchmark passes on the 52-dataset cohort (#40), using the
   contract and notebook.
2. Merge the production stack (#31, #36, #34, #41, #46, #47) and pin the binary; every output records the engine commit.
3. Exploratory screen on Clipper with the orchestrator (#42) on a diverse cohort; validate correctness and completeness.
4. Full re-run of all samples (every run, all reads) on Clipper, with Anvil if approved; empty bundles (557 studies) and
   strand-affected samples (8,270) first.
5. Run the rename ops (R2 copy, D1, Worker, website #67; code is merged) before writing new objects, then the storage migration, cell index build and Worker deploy.
6. Publish beta.singlet.bio with the benchmark pages (#66) and roadmap; retire legacy data afterwards.

## 6. Lessons learned

- Benchmarks must be judged on independent truth: Cell Ranger's own calls as "truth" favour EmptyDrops_CR by construction.
- Re-check every headline against real data; one simulated performance figure (66% cache hits) had to be replaced.
- Read the whole input: the old campaign kept the first SRR and its first 30M reads (median 15% of reads used).
- Shared-state bugs hide in test infrastructure (LocalStore temp names, per-URL transport caches): fix them where found.
- Clipper: one bridge only (bridges share one token); `hpc/clipper/sync.sh` deletes non-branch worktrees; sbatch copies the script at
  submission; the GVSU VPN drops often, so jobs must run unattended.
- Many parallel agent sessions cost a lot of tokens; keep orchestration narrow, prefer small models for mechanical work, and hand
  off through notes in the repo.
