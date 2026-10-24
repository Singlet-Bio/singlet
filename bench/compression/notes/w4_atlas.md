# w4_atlas: what the public data layer contains today (180-bundle sample)

Status: written 2026-10-22 under a hard stop (07:35 EDT). Owner: the `w4_atlas` agent. Everything below is MEASURED on
180 bundles unless a row says "study" (number taken from the earlier compression study, not re-measured here) or
"candidate" (not verified here). Tables with all digits: `results/w4_atlas_tables.md` (generated), `results/w4_atlas_levers.md`.

## 0. Five numbers

1. **Restricted-type members in the live public bundles: 0 of 34,252** (0 of 180 bundles; 95% upper bound 2.0% of bundles).
   The restricted-sidecar migration has been applied to every bundle in the sample that needed it (section 4).
2. **Two generations**: gen A (86 bundles, 47.8% of bundles, 39.0% of bytes, built 2026-06-18, `.1pz` members only) and gen B
   (94 bundles, 52.2% of bundles, 61.0% of bytes, later builds, `.1pz` plus STARsolo/text members, **rewritten 2026-10-20/21**).
3. **61.0% of bytes are STORED `.1pz`, 39.0% are DEFLATE text**; no other zip method exists. In gen B, 63% of bytes are deflated text.
4. **16.1% of bundles (29/180, CI 11.1-21.7%) are over 512 MB and hold 52.3% of the bytes (CI 38.9-63.2%)**; every individual
   member is under 44.0 MB and every sample under 131.6 MB, so a per-sample or per-member layout is 100% cache-eligible.
5. **Every response is `cf-cache-status: DYNAMIC`, with no Cache-Control, no CORS headers, OPTIONS preflight = 403**; HTTP/2 and
   TLS 1.3 are negotiated (ALPN `h2`), though curl 8.9 on this laptop and Python http.client speak HTTP/1.1.

## 1. Method and request budget

- `tools/w4_atlas_scan.py`: for each of the 180 ids in `w4_atlas_ids.txt` one `Range: bytes=-65536` GET (the 206 carries
  Content-Range total, ETag, Last-Modified and the cache headers, so a separate HEAD per bundle was folded in; explicit HEADs
  were done in the header audit and gave the same header set and identical ETags to the tail GETs for all 20). A second ranged GET only when the
  central directory did not fit. Own ZIP parser (EOCD, ZIP64 locator/EOCD, central directory incl. ZIP64 extras); **186 requests**.
- Members table: `results/w4_atlas_members.tsv` (34,252 rows: path, category, group, method, compressed and uncompressed size,
  CRC-32, local-header offset, restricted flag, generation); bundles table `results/w4_atlas_bundles.tsv`.
- Other requests: 16 manifest.json reads (`results/w4_atlas_manifests.tsv`), header audit on 20 bundles (about 68 requests,
  `results/w4_atlas_audit.tsv`), 41 member downloads (108 MB) for the recoding sample. Total about 310 requests. No Cloudflare/D1/R2/KV/Singlet-MCP tool was used.
- Estimators: the ids are a uniform random sample of bundles, not of bytes. Byte shares are ratio estimators
  `sum(q_i)/sum(b_i)`; atlas figures multiply the share by a total. Two totals are shown because the catalog total looks stale (section 2):
  3.89 TB (catalog as given) and 2.48 TB (Horvitz-Thompson `8,202 x sample mean`). CIs: 10,000 bootstrap resamples of whole bundles.

## 2. Sampling caveats that change the sizing

- Sample mean bundle 302.6 MB (CI 247.7-368.9) vs catalog mean 474.3 MB; sample max 3.84 GB vs catalog max 29 GB. The bootstrap
  cannot generate bundles bigger than the biggest observed, so **shares of bytes in the top tail (bundles over 512 MB, large studies) are lower bounds**.
- **Hypothesis to test (not verified by me, I may not query the catalog): the catalog byte counts predate the migration.**
  The 94 gen-B bundles lost their restricted members; the Sept 1/40 index implies 31% of bytes were `snp_*` etc. Solving
  `pre_B - 33.2 GB = 0.31 (21.2 GB + pre_B)` on this sample gives a pre-migration sample total of about 79 GB, mean about 439 MB,
  close to the catalog's 474 MB, and live x 3.89/ (79/54.5) = about 2.7 TB. The HT estimate (2.48 TB, CI 2.03-3.03) agrees. The parent can
  test it exactly: join the catalog `size_bytes` of the 180 ids in `results/w4_atlas_bundles.tsv` with my live `total`. Prediction: gen A equal, gen B larger in the catalog (GSE111113: index 324.7 MB vs live 170.8 MB).
- Sample reference-build shares by bytes (from `feature_vocab.json` clusters): GRCm39 57.1%, GRCh38 38.3%, barnyard 4.6%, GRCz11 0.1%
  (catalog: 58 / 32 / 2.3). Species and generation are confounded: all 84 GRCm39 bundles are gen B; 76 of 84 GRCh38 bundles are gen A.

## 3. Composition (compressed bytes inside the zip, 54.47 GB sampled)

| group | share of bytes (95% CI) | atlas at 3.89 TB (95% CI) | atlas at 2.48 TB |
|---|---|---|---|
| exon + intron `.1pz` | 42.89% (36.87-49.84) | 1.67 TB (1.43-1.94) | 1.06 TB |
| STARsolo `matrix.mtx` (raw 13.59%, filtered 8.57%) | 22.16% (16.93-27.30) | 862 GB (659-1060) | 550 GB |
| sj_counts + splice_psi `.1pz` | 16.68% (14.39-19.17) | 649 GB (560-746) | 414 GB |
| STARsolo `barcodes.tsv` (raw 7.10%) | 7.12% (5.35-8.91) | 277 GB (208-347) | 177 GB |
| `ambient_profile.tsv` | 3.75% (2.92-4.56) | 146 GB (113-177) | 93 GB |
| vdj_gene_usage + mt_heteroplasmy `.1pz` | 1.45% (1.21-1.78) | 56 GB | 36 GB |
| other per-cell tsv (cell_cycle, auto_barcodes, splice_events, cell_calls, doublet, ambient_contamination) | 1.18% (0.96-1.43) | 46 GB | 29 GB |
| donor/mt public tables (mt_summary 0.58%, donor_assignments 0.50%) | 1.08% (0.82-1.38) | 42 GB | 27 GB |
| STARsolo `features.tsv` (raw+filtered, identical) | 1.00% (0.77-1.22) | 39 GB | 25 GB |
| gene_expression.tsv / cell_qc_metrics.tsv / star_SJ.out.tab | 0.90 / 0.82 / 0.81% | 35 / 32 / 31 GB | 22 / 20 / 20 GB |
| feature_vocab + manifest + study_meta | 0.15% (0.12-0.19) | 5.9 GB | 3.8 GB |
| logs, small json, zip headers + central directories | 0.04% | 1.4 GB | 0.9 GB |

By member: exon_counts 24.96%, intron_counts 17.93%, raw matrix.mtx 13.59%, sj_counts 10.45%, filtered matrix.mtx 8.57%, raw barcodes 7.10%,
splice_psi 6.23%, ambient_profile 3.75%, vdj 1.07% (full 50-row table in `results/w4_atlas_tables.md`).

## 4. Generations and the restricted-member finding

Rule (by member presence): **gen B** = every sample directory carries text/STARsolo members (any of `ambient_profile.tsv`,
`gene_expression.tsv`, `cell_qc_metrics.tsv`, `star_SJ.out.tab`, `star_Solo.out/...`); **gen A** = none. No bundle is mixed.

| | gen A | gen B |
|---|---|---|
| bundles | 86 (47.8%, CI 40.6-55.0) | 94 (52.2%, CI 45.0-59.4) |
| bytes | 21.2 GB = 39.0% (26.9-51.9) | 33.2 GB = 61.0% (48.1-73.1) |
| median / mean / max bundle | 97 / 247 / 3,843 MB | 266 / 353 / 2,080 MB |
| members per sample | 10.2 | 43.9 |
| manifest `created_at`, `singlet_version` | 2026-06-18, 2.0.0 | 2026-06-27 to 2026-09-18, 1.0.0 or unknown |
| manifest `profile` | absent | `public-1` |
| Last-Modified | 2026-06-19 (86/86) | 2026-10-20 (52) and 2026-10-21 (42) |
| ETag | multipart with ~8 MiB parts (75) or single PUT (11) | multipart with **64 MiB parts** (83/83), single PUT (11) |
| Content-Type | `application/octet-stream` (86/86) | none (83) or `application/zip` (11) |
| within-gen bytes | `.1pz` 99.6%, text and json 0.4% | `.1pz` 36.4%, deflated text 63.5% (matrix.mtx 36.3, barcodes 11.7, ambient 6.1) |

Restricted-type members (repo rule `singlet._restricted.is_restricted_path`, loaded from the repo): **0 of 34,252**. No `snp_*`, `*.vcf`,
`donor*_coverage`, `ase_counts`, `donor/` or `mt/` member other than the allowlist (`donor/donor_assignments.tsv` 485 samples,
`mt/mt_summary.tsv` 474 samples, root `donor_assignments.tsv` 500). Reading: gen A (built 2026-06-18) never had those outputs (10.2 members per sample, no
`profile` key, untouched since 06-19); all 94 gen-B bundles show the migration fingerprint (rewritten Oct 20/21 with the script's 64 MiB transfer size, `public-1`, no Content-Type).
The Sept 1/40 index (31% of bytes in `snp_dp`/`snp_ad`) is stale: **restricted removal is already realized in live bytes**; it is not a lever left to pull.
Looser name check: `mt_variants.tsv` and `mt_reference.fa` (500 samples each, 5.5 MB, 0.01%) sit at the sample root and remain public under the allowlist;
that is a policy question, not a size one.

## 5. Compression methods

STORED 33.2 GB = 61.0% of bundle bytes (CI 53.1-69.5), all `.1pz` (zstd frames inside, entropy-coded already). DEFLATE 21.2 GB = 39.0% (CI 30.5-46.9),
all text/json, none of it `.1pz`. No other method (no zstd, xz, bzip2 members). Deflate ratios: matrix.mtx 4.18x, barcodes 5.93x, ambient_profile 5.30x, features 5.99x,
gene_expression 4.24x, cell_qc 5.40x, star_SJ 3.27x, feature_vocab 11.4x. Zip overhead (local headers plus central directory) is 0.01% of bytes, 165 B per member.

## 6. Size distributions and cacheability

- Bundle size (MB) p10/p25/p50/p75/p90/p99/max: 32 / 69 / 181 / 375 / 709 / 1,565 / 3,843. Samples per bundle median 4 (max 224); members per bundle median 106 (max 2,243).
- **Over 512 MB (5.12e8): 29 of 180 = 16.1% of bundles (CI 11.1-21.7), 52.3% of bytes (CI 38.9-63.2).** Atlas projection 1,321 bundles (911-1,777), 2.03 TB
  at 3.89 TB scale (1.51-2.46). Over 512 MiB: 27 bundles, 50.4% of bytes. Over 5 GB: none in the sample (the catalog has some, up to 29 GB, so these shares are lower bounds).
  Gen A: 10 of 86 over 512 MB (52.5% of its bytes); gen B: 19 of 94 (52.2%).
- Largest member 44.0 MB (raw matrix.mtx); no member over 100 MB. Per-sample bytes (MB), p50/p90/max: all members gen A 18.6/33.7/54.8, gen B 70.5/103/132;
  exon+intron `.1pz` 15.1/22.8/34.5 (exon alone 8.7/13.6/22.9). A client that wants one sample's counts needs about 9-23 MB, and the zip it sits in is
  a median 181 MB object. Some samples are empty stubs (p10 of gen A is about 0 MB).
- Central directory: 84 B per member; median 8.6 kB, max 197 kB (2,243 members). The 64 KiB tail holds the whole directory for 174 of 180 bundles; the 6 exceptions have 795-2,243 members
  (tail holds about 776 members). A 256 KiB tail would have covered all 180; the catalog's 29 GB bundle may need more, so read the EOCD first and then the exact directory range.

## 7. Duplicated content (CRC-32 + size from the central directory)

- `feature_vocab.json`: 80.9 MB in-zip over 180 bundles (0.15% of bytes, CI 0.12-0.18; atlas 5.8 GB at catalog scale), 287-849 kB in-zip (3.1-9.7 MB uncompressed) each.
  **Only 4 distinct contents**, one per reference build: GRCm39 6,570,144 B (84 bundles), GRCh38 3,094,668 B (84), barnyard 9,664,718 B (11), GRCz11 2,675,886 B (1).
  Same build, same compressed size, same CRC: identical. Bytes saved by one copy per build: 97.5% of the file's bytes, but only 0.15% of the atlas; the real gain is one fewer fetch per study.
- `manifest.json` (median 4.5 kB, max 80 kB) and `study_meta.json` (median 1.1 kB) are unique per bundle and negligible.
- Raw STARsolo `barcodes.tsv` (the chemistry whitelist, 7.10% of bytes): **3 distinct contents in 467 members; 99.6% of its bytes are repeats**.
  `raw/features.tsv` and `filtered/features.tsv` are byte-identical in 467/467 samples, with 2 distinct contents. `mt_reference.fa`: 3 distinct contents.
  Count matrices, matrix.mtx and per-cell tables are all distinct (as expected).

## 8. Header audit (20 bundles from 0.4 MB to 3.84 GB, both generations)

HEAD (fresh TLS connection), then the 64 KiB tail GET twice on the same connection, Origin: https://example.org sent each time. Also OPTIONS preflight on 6 and Accept-Encoding on 4.

| item | result (20/20 unless stated) |
|---|---|
| `cf-cache-status` | DYNAMIC on every HEAD and every GET (also 180/180 tail GETs in the scan); the repeated tail read is still DYNAMIC, no `Age` |
| `Cache-Control`, `Expires`, `Vary` | absent |
| CORS | no `Access-Control-Allow-Origin`/`Expose-Headers` even with an Origin; **OPTIONS preflight returns 403** (6/6): a browser page on another origin cannot range-read these objects |
| Accept-Ranges | `bytes`; suffix ranges (`bytes=-65536`) work and return 206 with `Content-Range` |
| ETag | quoted; `md5-N` multipart on 17, plain md5 on 3 (single PUT); identical on HEAD and both GETs |
| Content-Type | gen A `application/octet-stream`; gen B none (9) or `application/zip` (1): inconsistent, a side effect of the migration upload |
| Content-Encoding with `Accept-Encoding: gzip, br, zstd` | none added (4/4) |
| other headers | `Server: cloudflare`, `X-Content-Type-Options: nosniff`, NEL/Report-To, no `Content-Disposition`, no `Alt-Svc` |
| protocol | ALPN selects `h2` over TLS 1.3 (tested with Python ssl); my HTTP clients used HTTP/1.1 |
| timing (this laptop, single thread) | TCP+TLS connect median 194 ms; HEAD TTFB median 91 ms (64-156); tail-GET TTFB median 108 ms first, 92 ms repeat (max 426); 64 KiB tail total 145 ms. Under 3-way concurrency in the scan the median TTFB was 197 ms |

"MCP-advertised download URL form": I did not call the Singlet MCP (not allowed). The repo's loader and docs define only
`https://data.singlet.bio/data/<GSE>/<GSE>.singlet`, the form audited here. If `get_download_url`/`get_partial_download` return a different host or a signed URL, the parent
should repeat `tools/w4_atlas_probe.py audit` with that URL; I cannot say whether it is different.

## 9. Stale member index for rewritten gen-B bundles

Verified by the parent (D1 `bundle_index` and `/api/bundle/GSE111113/index` say 324,676,242 bytes / 215 entries; the live object is 170,777,036 bytes / 179 members;
the entry endpoint returns 502; range recipes point at non-TP1Z bytes). In my sample, **all 94 gen-B bundles (52.2% of bundles, CI 45.0-59.4) were rewritten on 2026-10-20/21**
(Last-Modified, 64 MiB-part ETags, `public-1` manifests), so every one of them has a member index that predates its current bytes unless the index was rebuilt after the rewrite.
Atlas projection: about 4,283 bundles (3,691-4,875) and **61.0% of live bytes (CI 48.1-73.1)**, i.e. 2.37 TB at catalog scale (1.87-2.85) or 1.51 TB at HT scale.
Gen-A bundles (86, untouched since 06-19) have consistent members. 11 of the 94 are single-PUT objects below 64.8 MB; the rest are multipart. The rewritten zips have fewer members (179 vs 215 entries for GSE111113) and different offsets, so any offset recorded before 10-20 is wrong for them. The fix is a reindex of those 94/180 of the atlas from the live central directory (one tail read per bundle, as done here).

## 10. Recoding potential (reduced scope after the coordinator's instruction)

Scope: zstd-19 and xz-9e versus the bundle's deflate on a 6 MB text slice of one gen-B member per category (no CSC parse, no b_rans16, no brotli; derivability is w4_mtx's).
Members were fetched by range (`tools/w4_atlas_recode.py fetch`, CRC-32 verified when complete; a deflate prefix decodes from its start, text cut at a line boundary).
Baseline = deflate level 6 on the same slice (`harness/w4_atlas_bench.cpp`, `tools/w4_atlas_quick.py`).

| member (gen B) | source | slice (of text) | in-bundle deflate ratio, whole member | deflate-6 on slice | zstd-19 | xz-9e |
|---|---|---|---|---|---|---|
| raw matrix.mtx | GSE249349/GSM7937397 (first 10.6 MB of deflate stream) | 6.0 of 45.4 MB | 0.233 | 0.246 (1.47 MB) | 1.12 MB (-24.1%) | 0.98 MB (-33.5%) |
| filtered matrix.mtx | GSE296281/GSM8968011 | 6.0 of 40.4 MB | 0.258 | 0.273 (1.64 MB) | 1.22 MB (-25.9%) | 0.95 MB (-42.3%) |
| raw barcodes.tsv (whitelist) | GSE311511/GSM9326203 | 6.0 of 12.5 MB | 0.173 | 0.172 (1.03 MB) | 0.33 MB (-67.9%) | 0.07 MB (-93.0%) |
| ambient_profile.tsv | GSE130184/GSM3733526 | 6.0 of 22.4 MB | 0.186 | 0.171 (1.02 MB) | 0.63 MB (-38.3%) | 0.59 MB (-42.5%) |

Caveat on the baseline: deflate-6 on a 6 MB slice is 5-6% larger than the in-bundle deflate ratio for the two matrix.mtx members (0.246 vs 0.233, 0.273 vs 0.258) and 8% smaller for ambient_profile (0.171 vs 0.186), so the percentage savings against in-bundle bytes may differ by a few points. Barcodes: the xz result (-93%) shows how regular a sorted whitelist is, but the whole file is a 3-content duplicate (section 7), so dedup, not a codec, is the answer.

Not done: the 41-member, 3-per-category sample with brotli-11 and columnar transposition is downloaded (`corpus_w4_atlas/recode/*.txt`, selection in `results/w4_atlas_recode_sel.tsv`) but not
analysed; the CSC parse path (`harness/w4_atlas_bench.cpp mtx`) was smoke-tested on 400k raw-matrix entries and gave b_rans16 350,840 B vs 1,424,638 B deflate vs 735,833 B production `.1pz` (4.1x smaller than deflate;
roundtrip OK, one slice, not a result). The machine was loaded (zstd-19 ran about 6x slower than normal), which is why the sample was cut.

## 11. Atlas-wide savings by lever (share of live bytes, bundle-bootstrap 95% CI)

Generated by `tools/w4_atlas_levers.py` -> `results/w4_atlas_levers.md`. CIs cover bundle sampling only; fractions applied are point values.

| lever | basis | saved share of live bytes (95% CI) | TB at catalog scale 3.89 TB | TB at HT scale 2.48 TB |
|---|---|---|---|---|
| restricted-member removal | REALIZED: 0 of 34,252 members restricted in the live sample | 0.00% (0.00% to 0.00%) | 0.00 | 0.00 |
| names/vocab A: whitelist raw/barcodes.tsv -> chemistry id | measured: 3 distinct contents in 467 members (CRC), 99.6% of bytes are repeats | 7.07% (5.30% to 8.86%) | 0.27 | 0.18 |
| names/vocab B: raw+filtered features.tsv -> vocab id | measured: byte-identical raw==filtered in 467/467, 2 distinct contents | 1.00% (0.77% to 1.22%) | 0.04 | 0.02 |
| names/vocab C: feature_vocab.json once per build | measured: 4 distinct contents in 180 bundles | 0.14% (0.12% to 0.18%) | 0.01 | 0.00 |
| names/vocab D: names, permutation, ptr, colsums, barcodes inside exon/intron .1pz | study (Track A, 14 corpus matrices 114.4->53.5 MB = -53.2%), NOT re-measured on atlas members | 22.82% (19.62% to 26.41%) | 0.89 | 0.57 |
| derived members A: splice_psi.1pz regenerated from sj_counts.1pz | study (E4): 14/14 files byte-identical, gen-A bundle only | 6.23% (4.93% to 7.59%) | 0.24 | 0.15 |
| payload recoding: b_rans16 on exon/intron payload | study: -15.1% byte-weighted on the payload that remains after lever D (0.468 x 0.151 of today's .1pz bytes) | 3.03% (2.61% to 3.51%) | 0.12 | 0.08 |
| legacy-text recoding: raw matrix.mtx with xz-9e | measured here, n=1 member, 6 MB slice: -33.5% vs deflate-6 (zstd-19: -24.1%) | 4.55% (3.52% to 5.55%) | 0.18 | 0.11 |
| legacy-text recoding: filtered matrix.mtx with xz-9e | measured here, n=1 member, 6 MB slice: -42.3% vs deflate-6 (zstd-19: -25.9%) | 3.62% (2.73% to 4.53%) | 0.14 | 0.09 |
| legacy-text recoding: ambient_profile.tsv with xz-9e | measured here, n=1 member, 6 MB slice: -42.5% vs deflate-6 (zstd-19: -38.3%) | 1.59% (1.25% to 1.94%) | 0.06 | 0.04 |
| **sum of the measured/study levers above (overlaps removed: none overlap)** | | **50.04% (49.15% to 50.96%)** | 1.95 | 1.24 |
| CANDIDATE (w4_mtx is testing): filtered/matrix.mtx derivable from raw | upper bound if fully derivable | 8.57% (6.46% to 10.71%) | 0.33 | 0.21 |
| CANDIDATE (untested): ambient_profile.tsv derivable from empties counts | upper bound if fully derivable | 3.75% (2.94% to 4.56%) | 0.15 | 0.09 |
| CANDIDATE (untested): gene_expression.tsv derivable from counts + lengths | upper bound if fully derivable | 0.90% (0.71% to 1.08%) | 0.03 | 0.02 |

Reading: the levers that need no codec (whitelist, features, vocab, derived splice_psi, container redundancy) are about 37% of live bytes; payload recoding adds 3%;
the legacy-text rows above are what general-purpose codecs add if the text members stay. If w4_mtx confirms that filtered matrix.mtx, ambient_profile and gene_expression are derivable, up to
13.2% more disappears and the recoding value of those members disappears with it (do not also count legacy-text recoding on those members). Restricted removal is 0 because it is done. The bold sum is a sum of point fractions (about 37 points are codec-free and mostly counted from CRCs; 22.8 points rest on the earlier study's -53.2%, 3 points on its -15.1%, 9.8 points on one-member xz slices), so its narrow CI reflects bundle sampling only and understates the real uncertainty; read the per-row basis column.

## 12. What is verified, what is projected, how to resume

- Verified on the sample: composition, generations, zero restricted members, method shares, size and cacheability shares, header audit, CRC-based duplication, manifest fields on 16 bundles.
- Study-derived (not re-measured on atlas members): names/perm/colsums -53.2%, payload -15.1%, splice_psi regenerable (gen A only, 14 files). Gen-B `.1pz` files were not parsed.
- Open: the stale-catalog test (join catalog sizes), the MCP URL form, brotli and CSC numbers for the recoding sample, `.1pz` container shares on atlas members (`singlet._pz.info_1pz` on 10-20 downloaded `.1pz`).
- Resume: `python -P tools/w4_atlas_scan.py w4_atlas_ids.txt && python -P tools/w4_atlas_members.py && python -P tools/w4_atlas_stats.py && python -P tools/w4_atlas_levers.py`;
  recoding: `tools/w4_atlas_recode.py run --tag=oth --skip=matrix_mtx --skip=barcodes_tsv__3` (hours at current load; consider `cap` 3 MB in `harness/w4_atlas_bench.cpp text`).
