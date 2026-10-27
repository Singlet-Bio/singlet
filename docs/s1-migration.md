# Singlet .s1 migration: decisions, workstreams, rules

Started 2026-10-22. Branch `feat/singlet-v2` (singlet-bio, on top of PR #12) and `feat/singlet-v2` in a separate worktree of
`singlet-website`. Format: `docs/format-s1.md`. Evidence for every decision: `bench/compression/PROPOSAL.md`.

## Decisions (orchestrator; revisit only with new measurements)

| # | decision | reason |
|---|---|---|
| D1 | Matrix payload = class-order 32-lane rANS (codec 1), generic gaps+zstd fallback (codec 2) | -15% payload, no per-column sort, 3-20x faster than today; fallback covers m >= 2^21, odd values, tiny files |
| D2 | Per-sample immutable content-addressed objects `s1/s/<GSM>/<d>.s1`; mutable study manifest `s1/m/<GSE>.json` | every sample under the 512 MB CDN limit (max 132 MB); stale offsets become impossible |
| D3 | Verify-or-store for every component and member | bit-equivalence by construction; recipes only used where they reproduced the bytes of that sample |
| D4 | Keep legacy column order; chunk index at a fixed offset; default `chunk_cols` chosen by experiment (candidates 256/1024) | no permutation cost; called cells already sit near the front; random cell access = one range per chunk run |
| D5 | Exon and intron share the sample `cells` table (barcodes once per sample + per-cell stats) | barcodes are 5% of v1 bytes and identical across the two matrices |
| D6 | Packs for row vocabularies (all six builds, 5.4 MB total) and 10x whitelists; Python wheel ships human+mouse+whitelists | names identical per build; R/others fetch once with sha256 |
| D7 | Raw STARsolo `matrix.mtx` stored as a standalone pz2 matrix plus a text-writer recipe now; conditional coding on the exon matrix later | lossless now, about half its bytes saved; conditional coding is a v2.1 codec feature that changes no serving code |
| D8 | PSI: never reproduce the 0/1 bug for users; `singlet.psi()` computes true PSI from `sj_counts`; the legacy member is a verified derived member (rint/trunc/stub per sample) | fixes the defect while keeping exact legacy regeneration |
| D9 | Serving: a Worker on `data.singlet.bio/s1/*` (R2 binding, Cache API, CORS `*`, Range, ETag); no dashboard Cache Rule needed | our API token cannot edit zone cache rules; Workers routes it can |
| D10 | Search: sample filters in D1 (`s1_sample`, `s1_study`); cell filters on per-sample `cells` tables; atlas cell index later | D1 cannot hold 297 M cells within its size and read budget; per-sample tables scale with the selection |
| D11 | Pipeline keeps writing TP1Z internally for now; the publisher (`pack_gse`) emits store only | the C++ pipeline change is separable; distribution is what migrates |
| D12 | Retirement: legacy bundles copied to a private archive bucket, removed from the public bucket after every study verifies; archive deleted after 30 days | reversible for 30 days at about $27/month (Infrequent Access) |
| D14 | New studies are published by the Anvil scripts in `singlet-website` branch `docs/anvil-new-studies` (`scripts/anvil/publish_study_v3.py`, `cf_bundle_v3.py`); they get a store publisher (`scripts/s1_publish.py`) and a patch | otherwise new studies would keep arriving in the legacy format |
| D15 | Bit-equivalence criterion for every `.1pz` member (stored or derived) is equality of the decoded canonical matrix; container bytes are additionally reproduced where the writer's zstd build is available (psi: 53/53 with the right writer) | TP1Z container bytes depend on the zstd build (1.5.5 vs 1.5.7) that wrote the file, not on the data |
| D13 | Immediate production fix: rewrite stale `bundle_index` rows from live central directories (laptop, wrangler d1) and add the size/ETag guard | half the atlas serves wrong offsets today; store takes days |

## Workstreams (agents; disjoint paths)

| id | owner | paths | delivers |
|---|---|---|---|
| W1 | codec (Sonnet) | `include/singlet/pz2/`, `tests/cpp/test_pz2_*.cpp`, `bench/pz2/` | C++ codec 1+2, matrix API, dispatch, digest, tests, timings, spec sections |
| W2 | container (Sonnet) | `python/singlet/s1/` except `recipes*.py`, `tests/python/test_s1_*.py` (container) | `.s1`/cells/packs/manifest reader+writer, transcoder, verifier |
| W3 | recipes (Sonnet) | `python/singlet/s1/recipes*.py`, `tests/python/test_s1_recipes*.py` | verify-or-store recipes for every legacy member type, golden tests |
| W4 | bindings + clients (Sonnet) | `src/bindings/python/_pz2.cpp`, `python/singlet/` loaders, `r/` | Python/R read path on the .s1 objects, cells slicing, local MCP |
| W5 | website (Sonnet) | `singlet-website` worktree `feat/singlet-v2`, mirrored to `singlet-cloud` | data Worker, D1 schema, MCP store tools, stale-index guard |
| W6 | campaign (Sonnet) | `singlet-agents` `hpc/clipper/v2/` | Slurm arrays: transcode + verify + upload, ledger |

## Rules for agents working in the repo

1. Edit only your paths. Read anything. Never run `git commit`, `push`, `checkout`, `reset`, `stash` or `rebase`; the orchestrator commits.
2. Build and test locally with the Rtools GCC toolchain (`bench/compression/build.sh` shows the flags and PATH; static zstd is in
   `C:\RBuildTools\4.4\x86_64-w64-mingw32.static.posix`). Python: `python -P` with numpy; the compiled legacy reader is
   `sys.path.insert(0,'python'); from singlet._pz import read_1pz`.
3. Test data (read-only, untrusted downloads, never execute): `bench/compression/corpus/` (14 matrices + `bundles/GSE145370.singlet`),
   `bench/compression/corpus_w4_corpus/` (39 samples, all builds and protocols), `corpus_w4_mtx/` (gen-B STARsolo members),
   `corpus_w4_bugs/` (GSE145370 members). Do not call Cloudflare, D1, R2 or MCP tools; public GETs to data.singlet.bio are fine.
4. Every claim of equality is checked byte for byte; write tests that keep checking it.
5. Write `bench/compression/notes/m_<id>.md` early and keep it current; finish with what is done, what is not, how to resume.
6. Timing runs take `bench/compression/results/.timing_lock` (mkdir; remove only your own) and report clock and load.
7. No polling loops, no timers left running.

## Status log

(orchestrator appends here)

- 2026-10-22 09:20 EDT: W1 published the stable API `include/singlet/pz2/pz2.h`. Wave 1 running: W1 codec, W2 container, W3 recipes,
  W4 clients, W5 site, W5a index repair prep, W6 campaign. Context map: no in-repo uploader; the publisher is the Anvil branch (D14);
  catalog sizes (`gse.r2_bundle_bytes`) are written outside the four repos; download URLs are hard-coded in `study-core.ts:21-24`,
  `_loader.py:85-87`, `load.R:22-34`.
- 2026-10-22 ~10:00 EDT: stale-index repair prepared by W5a (3,861 bundle_index rows, 3,860 bundle_manifest rows; validated 20/20,
  rehearsed with backup/undo). D1 time-travel bookmark before applying: `000016f2-00000000-0000510c-309865b013ffad741bb4a4aedb7f44cf`.
  Decision: apply bundle_index + bundle_manifest now (~7.9k row writes); defer `gse.r2_bundle_bytes` (66-84k row writes through FTS
  triggers; 325 rows were also rewritten wrongly by an external job at 11:04 UTC today) until catalog tables replace it.
- 2026-10-22 10:34 EDT: stale-index repair APPLIED and verified: 27 bundle_index batches + 10 bundle_manifest batches, all success
  (about 7,900 rows written). Post-check: bundle_index 8,202/8,202 and bundle_manifest 8,200/8,200 equal live size / member count;
  sampled member offsets land on their local headers. Remaining: gse.r2_bundle_bytes 4,186 rows (deferred); edge-cached
  /api/bundle responses expire within 24 h (cache-key bump ships with the website change set CS1).
- W6 campaign tooling ready (72 tests, local stand-ins only); catalog rows will go in via wrangler from the laptop (no ingest token
  needed); archive bucket created via the Cloudflare tools at retirement.
- 2026-10-22: W3 recipes DONE: 14 recipe ids exact on 275 member rows / 58 samples (psi 58/58 matrix-exact, 53/53 byte-exact with
  the matching writer; cell_calls 64/65; STARsolo raw/filtered matrix, barcodes, features, ambient, gene_expression, 10 per-cell
  tables exact). End to end through migrate: GSM4317429 13.08 -> 2.54 MB (5.1x), gen-B GSM5220137 74.04 -> 2.58 MB (28.7x).
  New facts: FPKM normaliser = pileup_stats.mapped_reads; mouse STARsolo gene order needs a 172 kB per-build permutation (package
  resource, now in package-data).
- 2026-10-22 11:15 EDT: production D1 schemas 031 (bundle_index etag/last_modified/verified_at) and 032 (s1_study, s1_sample,
  s1_migration) applied. Data Worker `singlet-data` deployed (version 732fe7f6) on route `data.singlet.bio/s1/*`: route wins over
  the R2 custom domain (X-Served-By: singlet-data), CORS * and OPTIONS 204 verified, legacy `data/` paths untouched. Website change sets
  CS1-CS5 pass `npm run check` (624 tests); independent review running before the PR is merged.
- 2026-10-22 ~12:30 EDT: website PR zdebruine/singlet-website#51 (3 commits: data Worker, serving layer, UI) CI green, preview smoke
  52/52; schema 033 applied; data Worker redeployed (15ee1554). Packs: all 16 + index.json uploaded to R2 s1/ref/ via wrangler and
  verified byte-identical through https://data.singlet.bio/s1/ref/. singlet-bio branch pushed (88adbf5). Clipper unreachable from the
  laptop (clipper.gvsu.edu resolves only inside GVSU: needs the GVSU VPN); campaign waits for the bridge.
- 2026-10-22 ~13:00 EDT: full-atlas /entry sweep on the preview: 8,201/8,202 studies 200 (GSE246714 502 is pre-existing on production:
  its bundle has no manifest.json). PR #51 merged (ed2f776), deployed in ~195 s; production smoke 52/52.
- 2026-10-22 ~13:05 EDT: singlet-cloud mirror PR Singlet-Bio/singlet-cloud#14 merged (f7ff553). singlet-bio PR Singlet-Bio/singlet#13 open
  (wheel CI: first MSVC/ARM builds of _pz2). Client default: legacy fallback ON during the transition (_LEGACY_FALLBACK_DEFAULT).
- 2026-10-22 ~13:10 EDT: PRODUCTION PILOT from the laptop: GSE145370 migrated by the real campaign code (187.1 MB -> 31.9 MB, verify ok),
  14 .s1 objects + ledger uploaded with wrangler, each verified byte-identical via https://data.singlet.bio/s1/, manifest uploaded last,
  catalog SQL applied (47 row writes). s1_migration verified; /api/gse shows series.store; /api/bundle index answers from the .s1 objects; MCP gate
  open (17 tools). Python client over the network: identical AnnData to the legacy reader (X, spliced, unspliced, obs/var names, obs
  columns), 4.3 s cold vs 20.1 s legacy local decode (+187 MB download).

## 2026-10-26: "store" renamed to ".s1"

Owner decision: no "store" naming anywhere. The format is named by its extension, `.s1`.

**What changed:**

| before (#39) | now |
|---|---|
| R2 and URL prefix `store/` (`store/s/`, `store/m/`, `store/ref/`, `store/ledger/`, `store/cidx/`, `store/a/`, `store/cells/`) | `s1/` with the same sub-prefixes |
| package `singlet.store` | `singlet.s1` (`singlet.store` and `singlet.v2` stay as deprecated aliases) |
| `scripts/store_*` | `scripts/s1_*` |
| `docs/format-store*.md`, `store-migration.md`, `store-performance.md` | `docs/format-s1*.md`, `s1-migration.md`, `s1-performance.md` |
| tests `test_store_*` | `test_s1_*` |
| R `R/store.R`, `.store_*` functions | `R/s1.R`, `.s1_*` |
| manifest `"singlet-store"`, pack index `"singlet-store-packs"` | `"singlet-s1"`, `"singlet-s1-packs"` |
| `StoreBundle`, `StoreSample`, `StoreContext`, `StoreError`, `read_store`, `is_store`, `store_gsms` | `S1Bundle`, `S1Sample`, `S1Context`, `S1Error`, `read_s1`, `is_s1`, `s1_gsms` (the old names are aliases) |
| `SINGLET_STORE_*` | `SINGLET_S1_*` (`SINGLET_STORE_*`, then `SINGLET_V2_*`, are still read as fallbacks) |
| verify `--store DIR` | `--s1 DIR` (`--store` still accepted) |
| D1 tables `store_study`, `store_sample`, `store_migration`, column `store_bytes`, indexes `idx_store_*`; ingest endpoint `/api/ingest/store-study` | `s1_study`, `s1_sample`, `s1_migration`, `s1_bytes`, `idx_s1_*`; `/api/ingest/s1-study` |
| campaign state `$WORK/state/store/` and caches `~/.cache/singlet/store` | `$WORK/state/s1/` and `~/.cache/singlet/s1` |

**What did not change:** the magic bytes (`SINGLET2`, `PZ2M`, `SPK2`, `CEL2`, `SCX1`) and the digest domains. Sample objects and packs are therefore byte-identical, and only the keys and the manifest strings differ.

**Readers** accept the `s1/`, `store/` and `v2/` prefixes, and the `singlet-s1`, `singlet-store` and `singlet-v2` format strings. **Writers** emit only `s1`.

**The website must match** (#67):
- the data Worker key patterns on `/s1/*`;
- D1 schema 036 with the `s1_*` tables;
- the ingest endpoint `/api/ingest/s1-study`;
- `series.s1` in the API.

## Research Alpha: the release policy in the transcode (2026-10-26)

The Alpha data release transcodes every included sample of the published atlas into .s1 objects; no read is reprocessed. The
campaign takes the policy JSON (`version`, `withheld_studies`, `excluded_samples`) with `s1_campaign.py run --alpha-policy FILE`
(`s1_migrate.sbatch`: `ALPHA_POLICY=...`, optionally `ALPHA_POLICY_SHA256=...`). `singlet.s1.policy` applies it:

- **Withheld study:** skipped before anything is read. Nothing is written: no object, manifest, ledger or catalog row.
- **Excluded sample:** dropped from the legacy view before the transcoder reads it, so it gets no .s1 object, manifest entry or
  ledger line. `manifest.json` and `study_meta.json` are rewritten without it. The study fails if any kept study file or small text
  member still names an excluded accession. A study with no sample left counts as withheld (migrate exits 3).
- **Withheld layers**, never written to a public object (`policy.omitted_layer`, a tested superset of `_restricted.is_restricted_member`):

  | layer | members |
  |---|---|
  | `genotype` | `donor/*` except `donor_assignments.tsv`, `snp_*`, `*.vcf`/`.bcf`, `ase_counts.tsv`, aligned reads (`*.bam`, 4 samples) |
  | `mtdna_alleles` | `mt/*` except `mt_summary.tsv`, `mt_heteroplasmy.1pz`, `mt_variants.tsv`, `*_mt_consensus*`, `*_mt_variants*` |
  | `ancestry_call` | `ancestry_call.json` |
  | `ambient` | `ambient_contamination.*`, `ambient_profile.tsv` |
  | `doublet_scores` | `doublet_scores.tsv` |
  | `nonhost_per_cell` | `nonhost/*per_cell*`; the sample-level `nonhost_summary.json` and `nonhost_em_abundance.tsv` stay |
  | `splice_psi_legacy` | `splice_psi.1pz`; clients compute true PSI from `sj_counts` (D8) |
  | `sex_call_legacy` | `sex_call.json`, the legacy pipeline call, for every species; replaced by the validated per-cell `sex_chrom` labels |

- **Manifest:** `release_policy` holds `{name, version, omitted_layers}`, the generic layer names only. It never holds an excluded
  accession or a per-study count of them. The per-study record of what was dropped is `policy-audit.json`, next to the .s1 tree,
  copied to `$OUT/policy-audit/` on the cluster and never uploaded.
- **Verify** (`singlet.s1.verify --policy`) compares the store with the policy's view of the legacy bundle, using decode equality
  for every kept member as before. It also checks:
  - the policy version;
  - that no excluded sample or withheld member is present;
  - that no excluded accession appears in the manifest or in any study file.

**Extras computed in the same pass, from members already in memory:**
- `samples[i].provenance` holds:
  - `reads_used`, from STAR's "Number of input reads", else `pileup_stats.total_reads`;
  - `reads_available`, which is null unless `--sample-facts` supplies it, because neither the bundle nor the catalog records it;
  - the strand wrong-fraction `wrong_strand / (wrong_strand + exon_hits)`, with the class from the PR #25 thresholds;
  - the library-type verdict;
  - the policy version.
- `samples[i].junction_support` holds the junction count and how many junctions reach 2 molecules in the sample, with a threshold note.
- The `cells` table gets `called_ed` and `called_ed_fdr` as additional columns (EmptyDrops 2.1, `docs/called-ed.md`), with
  `validated: false` for inDrop and Drop-seq. `called` stays the knee call and remains the default.
- Per-cell labels (`--labels`; `LABELS=1 AF=... RESTRICTED_BUCKET=singlet-restricted` in the sbatch) use `singlet.s1.labels_step`,
  which runs the validated labeller (`docs/cell_annotation.md`, `method="guide"`) on the counts the transcoder has already decoded:
  - cell type: the full CL path with per-level confidences (label schema 2: `cell_type_path`, `cell_type_p1..p6`). The default
    display levels are 1–2 (compartment, lineage); deeper levels are marked as previews;
  - chromosomal sex, for human and mouse only;
  - ancestry, for human donors whose per-donor depth tables exist in the bundle or in the restricted sidecar. Only those members are
    fetched from the sidecar, by ranged reads, and they are read in memory.

  The output is a canonical sidecar `s1/a/<GSM>/<d16>.parquet` referenced from `samples[i].annotations[a2026-10]`, which the atlas
  cell index ingests. Species without a model get no sidecar. A sample whose labelling exceeds its budget (`--labels-timeout`,
  `--labels-max-gb`) or fails is written without labels, and the ledger records why.
- Bundles whose central directory disagrees with a member's local header CRC are read on the local header when the data match it
  (`LegacyBundle.crc_repaired`, logged in the ledger). 14 published bundles have such a `manifest.json`.
