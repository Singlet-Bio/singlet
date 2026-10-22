# Singlet v2 migration: decisions, workstreams, rules

Started 2026-10-22. Branch `feat/singlet-v2` (singlet-bio, on top of PR #12) and `feat/singlet-v2` in a separate worktree of
`singlet-website`. Format: `docs/format-v2.md`. Evidence for every decision: `bench/compression/PROPOSAL.md`.

## Decisions (orchestrator; revisit only with new measurements)

| # | decision | reason |
|---|---|---|
| D1 | Matrix payload = class-order 32-lane rANS (codec 1), generic gaps+zstd fallback (codec 2) | -15% payload, no per-column sort, 3-20x faster than today; fallback covers m >= 2^21, odd values, tiny files |
| D2 | Per-sample immutable content-addressed objects `v2/s/<GSM>/<d>.s2`; mutable study manifest `v2/m/<GSE>.json` | every sample under the 512 MB CDN limit (max 132 MB); stale offsets become impossible |
| D3 | Verify-or-store for every component and member | bit-equivalence by construction; recipes only used where they reproduced the bytes of that sample |
| D4 | Keep legacy column order; chunk index at a fixed offset; default `chunk_cols` chosen by experiment (candidates 256/1024) | no permutation cost; called cells already sit near the front; random cell access = one range per chunk run |
| D5 | Exon and intron share the sample `cells` table (barcodes once per sample + per-cell stats) | barcodes are 5% of v1 bytes and identical across the two matrices |
| D6 | Packs for row vocabularies (all six builds, 5.4 MB total) and 10x whitelists; Python wheel ships human+mouse+whitelists | names identical per build; R/others fetch once with sha256 |
| D7 | Raw STARsolo `matrix.mtx` stored as a standalone pz2 matrix plus a text-writer recipe now; conditional coding on the exon matrix later | lossless now, about half its bytes saved; conditional coding is a v2.1 codec feature that changes no serving code |
| D8 | PSI: never reproduce the 0/1 bug for users; `singlet.psi()` computes true PSI from `sj_counts`; the legacy member is a verified derived member (rint/trunc/stub per sample) | fixes the defect while keeping exact legacy regeneration |
| D9 | Serving: a Worker on `data.singlet.bio/v2/*` (R2 binding, Cache API, CORS `*`, Range, ETag); no dashboard Cache Rule needed | our API token cannot edit zone cache rules; Workers routes it can |
| D10 | Search: sample filters in D1 (`v2_sample`, `v2_study`); cell filters on per-sample `cells` tables; atlas cell index later | D1 cannot hold 297 M cells within its size and read budget; per-sample tables scale with the selection |
| D11 | Pipeline keeps writing TP1Z internally for now; the publisher (`pack_gse`) emits v2 only | the C++ pipeline change is separable; distribution is what migrates |
| D12 | Retirement: legacy bundles copied to a private archive bucket, removed from the public bucket after every study verifies; archive deleted after 30 days | reversible for 30 days at about $27/month (Infrequent Access) |
| D14 | New studies are published by the Anvil scripts in `singlet-website` branch `docs/anvil-new-studies` (`scripts/anvil/publish_study_v3.py`, `cf_bundle_v3.py`); they get a v2 publisher (`scripts/v2_publish.py`) and a patch | otherwise new studies would keep arriving in the legacy format |
| D15 | Bit-equivalence criterion for every `.1pz` member (stored or derived) is equality of the decoded canonical matrix; container bytes are additionally reproduced where the writer's zstd build is available (psi: 53/53 with the right writer) | TP1Z container bytes depend on the zstd build (1.5.5 vs 1.5.7) that wrote the file, not on the data |
| D13 | Immediate production fix: rewrite stale `bundle_index` rows from live central directories (laptop, wrangler d1) and add the size/ETag guard | half the atlas serves wrong offsets today; v2 takes days |

## Workstreams (agents; disjoint paths)

| id | owner | paths | delivers |
|---|---|---|---|
| W1 | codec (Sonnet) | `include/singlet/pz2/`, `tests/cpp/test_pz2_*.cpp`, `bench/pz2/` | C++ codec 1+2, matrix API, dispatch, digest, tests, timings, spec sections |
| W2 | container (Sonnet) | `python/singlet/v2/` except `recipes*.py`, `tests/python/test_v2_*.py` (container) | `.s2`/cells/packs/manifest reader+writer, transcoder, verifier |
| W3 | recipes (Sonnet) | `python/singlet/v2/recipes*.py`, `tests/python/test_v2_recipes*.py` | verify-or-store recipes for every legacy member type, golden tests |
| W4 | bindings + clients (Sonnet) | `src/bindings/python/_pz2.cpp`, `python/singlet/` loaders, `r/` | Python/R read path on v2, cells slicing, local MCP |
| W5 | website (Sonnet) | `singlet-website` worktree `feat/singlet-v2`, mirrored to `singlet-cloud` | data Worker, D1 schema, MCP v2 tools, stale-index guard |
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
  triggers; 325 rows were also rewritten wrongly by an external job at 11:04 UTC today) until v2 catalog tables replace it.
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
- 2026-10-22 11:15 EDT: production D1 schemas 031 (bundle_index etag/last_modified/verified_at) and 032 (v2_study, v2_sample,
  v2_migration) applied. Data Worker `singlet-data-v2` deployed (version 732fe7f6) on route `data.singlet.bio/v2/*`: route wins over
  the R2 custom domain (X-Served-By: data-v2), CORS * and OPTIONS 204 verified, legacy `data/` paths untouched. Website change sets
  CS1-CS5 pass `npm run check` (624 tests); independent review running before the PR is merged.
- 2026-10-22 ~12:30 EDT: website PR zdebruine/singlet-website#51 (3 commits: data Worker, serving layer, UI) CI green, preview smoke
  52/52; schema 033 applied; data Worker redeployed (15ee1554). Packs: all 16 + index.json uploaded to R2 v2/ref/ via wrangler and
  verified byte-identical through https://data.singlet.bio/v2/ref/. singlet-bio branch pushed (88adbf5). Clipper unreachable from the
  laptop (clipper.gvsu.edu resolves only inside GVSU: needs the GVSU VPN); campaign waits for the bridge.
- 2026-10-22 ~13:00 EDT: full-atlas /entry sweep on the preview: 8,201/8,202 studies 200 (GSE246714 502 is pre-existing on production:
  its bundle has no manifest.json). PR #51 merged (ed2f776), deployed in ~195 s; production smoke 52/52.
- 2026-10-22 ~13:05 EDT: singlet-cloud mirror PR Singlet-Bio/singlet-cloud#14 merged (f7ff553). singlet-bio PR Singlet-Bio/singlet#13 open
  (wheel CI: first MSVC/ARM builds of _pz2). Client default: legacy fallback ON during the transition (_LEGACY_FALLBACK_DEFAULT).
- 2026-10-22 ~13:10 EDT: PRODUCTION PILOT from the laptop: GSE145370 migrated by the real campaign code (187.1 MB -> 31.9 MB, verify ok),
  14 .s2 objects + ledger uploaded with wrangler, each verified byte-identical via https://data.singlet.bio/v2/, manifest uploaded last,
  catalog SQL applied (47 row writes). v2_migration verified; /api/gse shows series.v2; /api/bundle index answers from v2; MCP gate
  open (17 tools). Python client over the network: identical AnnData to the legacy reader (X, spliced, unspliced, obs/var names, obs
  columns), 4.3 s cold vs 20.1 s legacy local decode (+187 MB download).
