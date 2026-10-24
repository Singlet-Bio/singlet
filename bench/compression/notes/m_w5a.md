# m_w5a: repair of stale bundle_index / bundle_manifest / gse rows (PREPARED, NOT APPLIED)

Status: COMPLETE (2026-10-22 09:50 EDT). Owner `m_w5a`. Nothing was written to D1 or anywhere remote: only read-only wrangler queries
(10 statements, list below) and public GET/HEAD to data.singlet.bio (about 15,000 requests, 8 in parallel, all 200/206/404 as expected).
No git command was run. Scripts `tools/m_w5a_*.py`, outputs `results/m_w5a/` (384 MB; do not commit the heavy intermediates, see "Files").

## 1. Result in numbers

| item | count |
|---|---|
| bundles checked (union of bundle_index = gse(r2_bundle_bytes>0) = 8,202; bundle_manifest 8,200) | 8,202, all HTTP 200, all Accept-Ranges |
| stale bundle_index rows (live size != D1 bytes; the same 3,861 have Last-Modified 2026-10-20/21 and `indexed_at` 2026-09-06 or 09-30) | **3,861** (47.1%); 0 of them have an unchanged size |
| bundles untouched since June/July/Sept (size and Last-Modified agree with D1) | 4,341 |
| stale bundle_manifest rows (bytes and n_files) | 3,860 (the 3,861st study, plus one more, has no bundle_manifest row) |
| stale gse.r2_bundle_bytes | **4,186** = 3,861 on rewritten bundles + 325 on bundles never rewritten (see anomalies) |
| central directories that failed to parse / failed a consistency check | 0 / 0 |
| validation of the port (step 1) | **20/20 byte-identical** |

Catalog effect: for the 3,861 studies D1 says 2,469 GB, the objects hold 1,469 GB (-1.0 TB, -40%); sum of gse.r2_bundle_bytes over all
8,202 studies is 3.89 TB now, the objects total 2.62 TB. Members: 1,116,168 indexed now, 931,118 after the repair (184,050 removed
members). Every new member set is a strict subset of the old one (no member added), retained members keep method/compressed/
uncompressed size except `manifest.json` (3,861 studies) and `samples/*/ancestry_call.json` (4,264 files, 1,381 studies), the two
members the migration rewrites. Removed kinds: ase_counts.tsv, donor/{snp_ad,snp_dp}.1pz, donor/donor{0,1}.vcf, donor/donor{0,1}_coverage.tsv,
mt/{mt_events.1pz, donor*_mt_variants.vcf, donor*_mt_consensus.fa}.

## 2. Step 1: the port and its validation
`tools/m_w5a_zip.py` ports `parseZipSource` (bundle-reader.ts:111-190) line by line: tail = last min(256 KiB, size); EOCD scan backwards;
u16/u32 fields; zip64 locator at eocd-20, zip64 record from the tail or a 56-byte range; central directory from the tail or one range;
per record method=u16(+10), csz=u32(+20), usz=u32(+24), off=u32(+42), name = UTF-8 (BOM stripped, U+FFFD on bad bytes); zip64 extra field
0x0001 read in the order uncompressed, compressed, offset, only for the 0xFFFFFFFF fields; names ending "/" are dropped; entry =
{p,n,c,u,o} in that key order, serialised like JSON.stringify (no spaces, non-ASCII raw, `\u00xx` lowercase). `o` is the LOCAL HEADER offset.
There is no filtering of restricted members at index-write time in this code (none in the audited tree; the live central directory is
simply what is stored, GSE111113's 36 missing members were removed from the object itself). `n_files` = number of entries
(catalog-refresh.ts:313), `n_gsms_in_bundle` = distinct `samples/(GSM[0-9]+)/` ids (all agree already, not touched).
Validation: 20 bundles with live size == D1 bytes and Last-Modified < indexed_at (3 most members, 3 fewest, 4 largest objects, 10 random,
seed 20261022). D1 `entries` string vs my JSON: 20/20 identical, and `bytes` == Content-Length. 4 of them are zip64 with the directory
outside the tail (883 KB of JSON for 9,721 members), 16 are not zip64 (`results/m_w5a/validation.tsv`). n_files equals the entry count for all 20.
Extra checks on the 3,861 recomputed rows: records parsed == EOCD count, bytes consumed == directory size, no entry reaches past the
directory, 939 (1 in 4 by id hash) also had the local header of their highest-offset member read back (signature `PK\3\4` and the same name): 939/939 ok.
GSE111113 now parses to 179 entries and its `exon_counts.1pz` header sits at 5,703,179 (w4_cf section 0).

## 3. What the repair does (all UPDATEs are compare-and-set, one per row, values inlined)
- `bundle_index`: `SET bytes = <live>, entries = '<json>', indexed_at = <UTC second the directory was fetched> WHERE gse_id = X AND bytes IS <old> AND indexed_at IS <old>`.
  214 of the 3,861 `entries` strings exceed 80 KB, D1's statement limit is 100 KB, so those are written as chunk 1 plus
  `SET entries = entries || '<chunk>' ... AND length(entries) = <chars so far>` statements (the length guard makes a second run a no-op;
  my first version appended again on re-run and the local rehearsal caught it). `wrangler d1 execute --file --remote` runs the file
  through D1's import API, atomic per file, so nobody reads a half-built string.
- `bundle_manifest`: `SET bytes = <live>, n_files = <entries> WHERE gse_id = X AND bytes IS <old> AND n_files IS <old>`. `n_gsms_in_bundle`, `gsm_ids`, `audited_at` untouched (all consistent).
- `gse`: `SET r2_bundle_bytes = <live> WHERE id = X AND r2_bundle_bytes IS <old>`. `last_updated` untouched.
Local rehearsal (`tools/m_w5a_rehearse.py`, in-memory SQLite loaded with the real old rows, no network): all repair files applied -> every row equals
the computed value (0 mismatches, 3,861 + 3,860 + 8,202 checked); rows outside the stale set unchanged; a second application changes 0 rows;
applying the backup files restores the original snapshot exactly; a row someone else changed first is left alone. Longest statement 78,534 bytes.

## 4. Row writes (estimate)
bundle_index 3,861 + 214 continuation UPDATEs = 4,075; bundle_manifest 3,860 (its secondary index is not on the changed columns: no extra writes).
**gse is the expensive one**: production has FTS5 external-content `fts_gse` and trigger `gse_au AFTER UPDATE ON gse` (D1 query 6), which does an FTS
`delete` plus insert of id/title/abstract/organism on EVERY update, whatever column changed. A local SQLite 3.45 copy (8,200 rows, abstracts of
80-260 words) measured 13.8 row changes per UPDATE inside one transaction and 18.1 with one transaction per statement, so 4,186 UPDATEs write about
58,000 to 76,000 rows (3,861 of them in the "stale" files, 325 in the "other" files). Total about **66,000 to 84,000 rows written** (7,935 without gse).
D1 free tier allows 100,000 rows written per day; paid includes 50 M per month. Measure first: apply a 1-statement gse file, read `rows_written` from the output, scale by 4,186.
Rows read: about one per UPDATE (primary-key lookup).

## 5. Anomalies (results/m_w5a/anomalies.tsv, 327 rows)
- **325 studies whose bundle was NOT rewritten (Last-Modified June-July 2026, size equals bundle_index.bytes and bundle_manifest.bytes) but whose gse.r2_bundle_bytes differs from the object**,
  by 1.01x to 19,507x (e.g. GSE149224: 1,424,603,593 vs 37,902,159; GSE132105: 43,701,703 vs 35,755,311). 323 of them have gse.last_updated 2026-10-22T11:0x:xxZ, i.e. a job
  wrote today (at about the time 6,187 gse rows were touched) a value that is not the object size. The same job left every rewritten study at its OLD size (3,797 of 3,861 equal bundle_index.bytes,
  64 hold some third value). I put these 325 in separate files `repair_gse_other_*.sql` so you can decide; semantics of r2_bundle_bytes in the code is the object size (search-core size filter, get_study, get_download_url).
- 2 studies have a bundle_index row but no bundle_manifest row (GSE246714, GSE272895): manifest not repaired for them (one of them is stale).
- Bundles in D1 but 404 live: **0**. Live bundles missing from D1: **0 found**: the 1,932 studies of the public search index (9,942 studies, built 2026-10-22T04:56Z) that are not in D1's bundle set were all HEADed and are all 404.
  This cannot see a bundle for a study that is in neither list (R2 cannot be listed anonymously). Conversely 192 studies in D1's bundle set are absent from the search index (e.g. GSE104156, GSE117788): informational.
- New directories failing to parse or failing a consistency check: 0.
- Downstream derived data that will change once bundle_index is repaired (not changed by this repair):
  * `LAYER_FILES` stats (`/api/stats/layers`, 12 h meta_cache): file `snp_ad.1pz` is held by 3,786 of the stale studies now and by 0 after, so the "SNP genotypes" layer count drops by up to 3,786 studies; the other nine layer files are unchanged for all 3,861.
  * the public search index (`index.json` layer flag `snps`, built from bundle_index by scripts/search-index/export-samples.sql) needs a rebuild to drop it.
  * edge-cached recipes: `/api/bundle/<gse>/entry` answers are cached 24 h with the old keys (w4_cf 0.6), so recipes for the 3,861 studies can stay wrong at the edge for up to a day after the D1 fix unless the cache key is bumped.
  * `sample_qc.usable`/matrix bytes derive from matrix members that were not rewritten: no change expected.
- The migration may not be finished: 4,341 bundles are still the June originals and the newest Last-Modified is 2026-10-21 11:56 GMT (nothing newer as of 09:49 EDT 22 Oct; a random re-HEAD of 400 bundles at that time found 0 changes).
  Any further rewrite makes that row stale again: re-run head -> compute -> sql (below), or ship the w4_cf step 0.1/0.2 guard first.

## 6. Commands for the orchestrator (nothing below has been run)
Paths: `R=C:/Users/zacha/Projects/singlet-bio/bench/compression/results/m_w5a`, run from `C:/Users/zacha/Projects/singlet-website`.
0. Safety net (read-only): `npx --no-install wrangler d1 time-travel info singlet-catalog` and write down the bookmark. (Restoring it rolls back the WHOLE database, other writers included; the backup files below are the surgical undo.)
1. Just before applying: `python -P tools/m_w5a_precheck.py` (8,202 HEADs; or pass `200` for a sample). It must print `changed ... 0`; otherwise re-run head -> compute -> sql for the changed studies.
2. Repair, bundle_index first (the user-visible bug). Each `--file` runs through the import API (atomic per file, the DB answers no queries while a file is processed; files are 3 MB, so seconds):
   `for f in $R/repair_bundle_index_*.sql; do npx --no-install wrangler d1 execute singlet-catalog --remote -y --json --file "$f" > "${f%.sql}.applied.json" || break; done`
   Do the first file alone, read its `rows_written` (expect 150 plus the continuation statements of those studies, about 155) and run the smoke script (`scripts/smoke.sh`, plus `/api/bundle/GSE111113/index` now says bytes 170777036 and 179 entries) before the loop.
3. bundle_manifest: same loop over `repair_bundle_manifest_*.sql` (10 files of up to 400 statements, 3,860 rows).
4. gse, measured first: `head -n 2 $R/repair_gse_stale_000.sql > $R/probe.sql` (one comment line plus one UPDATE), apply it and read `rows_written` (expect 14 to 18; if it is above 40, stop and rethink).
   Then the loop over `repair_gse_stale_*.sql` (26 files, 150 statements). Decide separately about `repair_gse_other_*.sql` (3 files, 325 statements; section 5).
   Note the probe row is repeated in file 000; re-applying it is a no-op.
5. Verify (read-only): dump `SELECT gse_id, bytes, length(entries) AS entries_chars, indexed_at FROM bundle_index` to `$R/after_bundle_index.json`,
   `SELECT id, r2_bundle_bytes FROM gse WHERE r2_bundle_bytes > 0` to `$R/after_gse.json`, `SELECT gse_id, bytes, n_files FROM bundle_manifest` to `$R/after_manifest.json`
   (same `--remote --json > file` form as the earlier dumps), then `python -P tools/m_w5a_verify.py`. Expect 0 mismatches (gse: 325 if you skip the "other" files).
6. Undo, if needed (surgical, compare-free): `backup.sql` (gse and bundle_manifest, 620 KB, one `--file`) and `backup_bundle_index_000..026.sql` (the old index rows with their old `indexed_at`).
   Rehearsed locally: restores the original state exactly.
Re-running the pipeline for a later wave: `python -P tools/m_w5a_head.py` (after moving live_head.tsv aside) -> `m_w5a_compute.py` -> D1 dump of the old rows (tools/m_w5a_validate.py backupq writes the query files) -> `m_w5a_sql.py` -> `m_w5a_rehearse.py`.

## 7. D1 queries used (read-only, wrangler --remote --json > file; 10 statements)
1 schema of the three tables; 2 `SELECT gse_id, bytes, indexed_at FROM bundle_index` (8,202 rows); 3 gse id/r2_bundle_bytes/key/n_gsms/last_updated (8,202);
4 bundle_manifest (failed with "fetch failed" during the DNS outage, no result); 5 bundle_manifest retry (8,200 rows); 6 triggers and indexes of the three tables;
7 `entries` of the 20 validation studies; 8-10 `gse_id, bytes, entries, indexed_at` of the 3,861 stale studies in three groups (37.7 MB each, by
`gse_id IN (...)` through `node node_modules/wrangler/bin/wrangler.js` because cmd.exe limits a command line to 8,191 characters). Rows read in total about 36,000.

## 8. Files (results/m_w5a/ unless noted)
- Repair: `repair_bundle_index_000..026.sql` (27 files, 80 MB), `repair_bundle_manifest_000..009.sql`, `repair_gse_stale_000..025.sql`, `repair_gse_other_000..002.sql`.
- Undo: `backup.sql` (gse + bundle_manifest), `backup_bundle_index_000..026.sql` (27 files, 96 MB).
- Tables: `summary.tsv` (4,186 rows: gse_id, old/live bytes, old/new entries, Last-Modified, old indexed_at, source of old entry count, gse/manifest old values, new GSM count, tables changed),
  `expected.tsv` (what each new bundle_index row must contain: bytes, n_files, entries length and sha256, indexed_at), `anomalies.tsv`, `validation.tsv`.
- Evidence: `live_head.tsv` (8,202 HEADs: status, Content-Length, ETag, Last-Modified), `live_head_extra.tsv` (1,932 search-index studies not in D1: all 404), `d1_*.json` (the read-only dumps),
  `computed.jsonl` (new rows plus request counts, 85 MB), `search_index.json` (study list), `*_run.log`.
- Heavy and re-derivable (delete or gzip before any commit): `computed.jsonl`, `d1_backup_*.json` (113 MB), `d1_validate.json`, the SQL files after they have been applied and archived.
- Tools: `tools/m_w5a_zip.py` (port), `_net.py`, `_head.py`, `_validate.py`, `_compute.py`, `_sql.py`, `_rehearse.py`, `_precheck.py`, `_verify.py`.
