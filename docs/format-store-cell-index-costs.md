# Cost model: atlas cell index and gather on Cloudflare

Companion to `docs/format-store-cell-index.md`. Prices are Cloudflare's published list prices read on 2026-10-23 (sources at the end);
volumes come from the pilot measurements (`bench/cidx/results_local_pilot.json`, `bench/cidx/results_chunking_pilot.json`) and the
format benchmark (`bench/cidx/formats.py`, `scan.mjs`). Where a number is an assumption it says so. All costs are marginal, beyond the
Workers Paid base ($5/month, which includes 10 M requests and 30 M CPU-ms a month).

## 1. Prices

| item | price | included per month |
|---|---|---|
| Worker request | $0.30 / million | 10 M |
| Worker CPU time | $0.02 / million CPU-ms | 30 M CPU-ms |
| Worker subrequests (R2 binding calls, Cache API, service binding to itself) | not billed | (10,000 per invocation by default) |
| R2 Class B (GetObject, HeadObject) | $0.36 / million | 10 M |
| R2 Class A (PutObject, ListObjects) | $4.50 / million | 1 M |
| R2 storage, Standard | $0.015 / GB-month | 10 GB |
| R2 egress (to the Internet, to Workers) | free | |
| Cache API (`caches.default`) | not billed separately (a hit costs the Worker request, no R2 operation) | |
| Durable Objects request | $0.15 / million | 1 M |
| Durable Objects duration | $12.50 / million GB-s | 400,000 GB-s |
| Durable Objects SQLite rows written | $1.00 / million | 50 M |
| Workers Rate Limiting binding | no price published (treated as free) | |
| D1 rows read (API key check, memoised 60 s per isolate) | negligible | |

Limits that shape the design: 6 connections waiting for headers per invocation (R2 gets and cache lookups count; the seventh queues),
1,000 Cache API calls per invocation, 128 MB memory per isolate, CPU 30 s by default (up to 5 min).

## 2. Unit costs

Notation: `B` = Class B price $0.36e-6, `W` = request price $0.30e-6, `C` = $0.02e-6 per CPU-ms, `D` = DO request $0.15e-6.

### 2.1 Storage and builds

| | size | per month |
|---|---|---|
| one index version (shards ~3.1-4.3 GB at 10.4-14.4 B/cell, plan blocks ~90 MB, summary ~20 MB) | ~4.4 GB | $0.066 |
| three retained versions | ~13 GB | $0.20 |
| optional Parquet export per version | ~3.5 GB | $0.05 |
| label sidecars (one annotation version, ~3-5 B/cell) | ~1-1.5 GB | $0.02 |

Building: ~650k ranged reads of data.singlet.bio (`cells` tables, pz2 headers, chunk indexes): 650k x (W + B) = **$0.43** through the
Worker (or $0.23 R2-only with a cluster-side mirror). Uploading a version: ~150 shards + ~280 plan blocks + 3 small objects = ~435 Class
A = **$0.002**. Uploading one annotation version of sidecars: ~71k Class A = **$0.32**.

### 2.2 Queries

| query | Worker request | CPU | R2 Class B | Durable Object | total, warm | total, cold data centre |
|---|---|---|---|---|---|---|
| count / grouped count (summary) | 1 W | ~5 ms (pilot); 50-150 ms for an unfiltered atlas-wide grouped count (3 M summary rows) | 0 warm; ~5-10 cold (pointer, directory, summary header and groups, plus fills) | none | **$0.4e-6** | ~$4e-6 |
| row scan, typical ("T cells in lung, n_umi >= 1000": ~3 M candidate rows after zone maps, 4 columns) | 1 W | ~300 ms (V8 decode ~10 M rows x 4 columns per s, measured) | 0 warm; ~50 group reads + ~2 shard fills cold | 1 check (no write) + 1 add (1 row write) | **~$8e-6** | ~$3e-5 |
| row scan at the anonymous cap (8 M rows) | 1 W | ~1.3 s | ~120 cold | as above | ~$3e-5 | ~$7e-5 |
| cells page (20k-100k cells) | 1 W | scan + ~10 ms/10k cells to serialise | as above | as above | ~$1e-5 | ~$4e-5 |

So: **a million summary counts cost about $0.40; a million typical filtered cell searches about $8-30**, CPU-dominated. The per-subject
daily row budget (200 M rows anonymous, 5 G with a key) caps a single subject's scan CPU at ~$0.0006 / ~$0.016 a day.

### 2.3 Gather, per 1 M cells

Two regimes bracket every gather: **dense** (most cells of the samples touched, e.g. "all T cells of these 300 samples") and **sparse**
(a few cells from each of many samples, e.g. "1,000 random T cells from every tissue").

Bytes per cell (exon + intron, prefixes excluded because clients cache them):
* dense: the matrix bytes of the called cells, ~2-3 KB per cell for 10x data at ~3,000 UMIs (pilot: 199 B/cell at ~100 UMIs);
* sparse: one chunk per member per cell. Pilot, current writer: **323 KB per random cell**; with chunks cut at 48,000 non-zeros
  (recommended, spec 10.2): **119 KB**. Assume 120-320 KB.

| | dense, via gather | sparse, via gather (48k-nnz chunks / today's chunks) | sparse, client's own range requests |
|---|---|---|---|
| bytes moved | ~2.5 GB | ~120 GB / ~320 GB | same |
| R2 reads (Class B, cache misses) | ~1,000 (2-4 per sample, <= 8 MiB each) = $0.0004 | ~2 M (one per member per cell) = $0.72 | ~2 M = $0.72 |
| Worker requests | 100 (10,000 cells per request) = $3e-5 | 100-300 (10,000 cells, <= 4,000 reads and 1 GiB per request) = $9e-5 | ~2 M = $0.60 |
| Worker CPU (~0.5 ms per MB streamed, assumed) | ~1.3 s = $3e-5 | 60-160 s = $0.001-0.003 | per request ~1 ms: 2,000 s = $0.04 |
| cache fills (whole sample object, once per data centre) | <= 300 x B | <= 50k x B = $0.018 (second miss only) | <= 50k x B = $0.018 (first miss) |
| quota object | 100 charges = $1.2e-4 | 100-300 charges = $3.5e-4 | none |
| **total** | **~$0.0005** | **~$0.74** (bytes do not change the price; R2 reads do) | **~$1.38** |
| wall time for the client (assumed: R2 first byte 40 ms in Cloudflare, client at 32 MB/s per stream as measured from this laptop) | dominated by transfer: ~80 s for 2.5 GB | Worker side 2 M reads / (16 parts x 6) x 40 ms = ~14 min; transfer 120 GB = ~1 h at 32 MB/s | 2 M round trips of 150-275 ms (measured from this laptop) at 16 in flight = ~6-10 h, plus the same transfer |

Edge-cache hits remove the R2 line: with a hit fraction `h`, sparse gathers cost about `0.72 (1 - h) + 0.02` dollars per million cells.
Hot samples (repeatedly gathered) are cached after their second ranged miss in a data centre.

Takeaways:
1. **Dense gathers are effectively free** (well under a cent per million cells); egress is free and the bytes are the user's wait, not
   our cost.
2. **Sparse gathers cost ~$0.74 per million cells**, almost all R2 Class B, i.e. one read per touched chunk. The gather endpoint halves
   the cost of clients doing the same range requests themselves (it removes the per-request Worker charge) and turns hours of client
   round trips into minutes inside Cloudflare.
3. **The 48k-nnz chunk cut** (spec 10.2) does not change the read count but cuts sparse bytes, Worker CPU and the user's transfer time
   ~2.7x. It is the single most valuable change for this workload and must precede the migration campaign.

## 3. Bounds: one request never costs more than $0.01

### 3.1 Per request

Every request is planned before it runs, and its **worst case** is computed with the prices of section 1: every read misses the cache
and starts a whole-object cache fill (a second Class B), CPU is 5 ms fixed + 0.25 ms per 1,000 index rows decoded + 1 ms per MB copied
(both several times what the V8 measurements in `bench/cidx` show), plus the quota object's request and row write. A request whose
worst case exceeds **$0.01**, or any cap below, is not run: the answer is 413 with a plan the client can run in smaller pieces or
itself (spec 8.2). Answers report `estimate` (the worst case) and `cost` (from the reads that actually missed).

| request (default caps) | the largest such request | worst case | measured on the pilot (local) |
|---|---|---|---|
| summary count / grouped count | ~10 reads, no rows | $0.000008 | $0.0000004 warm, $0.000004 cold |
| row-scan count | 32 M rows (key) / 8 M (anon), ~620 reads | **$0.0006** | $0.000003 cold (40k rows) |
| cell sample (`return: "cells"`) | 10,000 cells from 10 studies; all their matching rows decoded (<= 32 M) | **$0.0006** | $0.0000016 cold, estimate $0.000016 |
| gather | 10,000 cells, 10 studies, 4,000 reads, 1 GiB | **$0.0030** | $0.000014 cold (100 cells, 14 samples), estimate $0.00005 |

The caps, not the estimate alone, keep the bound: the largest request any cap admits costs at most ~$0.003, a third of the budget (tested
in `src/test/data-store-cidx.test.ts`, "cost bound"). The estimate is a second line of defence (it refuses any plan whose own worst case is
over the budget, e.g. when caps are raised by configuration).

What grows with the caps is not money but work per request: a uniform sample of 10,000 cells from 10 studies touches nearly every chunk
of those studies' samples, i.e. moves roughly their whole called region (up to ~1 GB at today's chunking, ~0.4 GB with 48k-nnz chunks),
which is what the client waits for. `sampling: "chunks"` takes whole chunks and moves far less (pilot: 0.57 MB instead of 9.2 MB for 1,000
cells). So **10,000 cells and 10 studies are good defaults**: raising cells to 50,000 would still cost under $0.01 (reads are bounded by
chunks, not cells) but would make single requests slower; lowering them only adds requests. Fewer than 10 studies per request makes the
ranking (most eligible cells, or the caller's own order) decide more of the answer.

### 3.2 Per subject and day

The `CidxQuota` Durable Object keeps per-subject daily totals (UTC): cells, bytes, rows scanned, and **worst-case dollars**: every gather
is charged its estimate before it streams, every row scan the worst case of what it read. Summary counts are not metered (they cost
$0.0000004 each and are rate-limited).

| subject | daily allowance | worst case per day |
|---|---|---|
| API key | $1.00 worst-case spend, 2 M cells, 200 GiB, 5 G rows scanned; 300 requests/min | $1.00 + summary counts at the rate limit (432k x $0.0000004 = $0.17) = **$1.17** |
| anonymous address | $0.05, 200,000 cells, 4 GiB, 200 M rows; 30 requests/min | $0.05 + $0.02 = **$0.07** |

Actual cost is usually far below the charged worst case (cache hits, fewer reads than the cap), so these are ceilings. Distributed
anonymous abuse is bounded per address only; a global anonymous budget (one `CidxQuota` object named `anon:global`) would cap it absolutely
and is a one-line change if needed. Note also that plain ranged GETs of the public objects (what `via="direct"` clients and every store
download use) are outside these endpoints: each costs one Worker request plus one Class B on a miss ($0.66 per million) and is not
rate-limited today.

## 4. A plausible month

1 M summary counts, 200k filtered cell searches, 500 M cells gathered dense, 20 M cells gathered sparse (40% cache hits), one index
rebuild and one annotation version:

| | cost |
|---|---|
| counts | $0.40 |
| cell searches | $1.6-6 |
| dense gathers | $0.25 |
| sparse gathers | $9 |
| storage (3 index versions + sidecars) | $0.22 |
| rebuild + sidecar upload | $0.75 |
| **total, before the $5 plan's inclusions** | **~$12-17** |

With the 10,000-cell cap, the 20 M sparse cells are 2,000 requests of at most $0.003 each (worst case $6, ~$4.50 expected).

Much of this falls inside the Workers Paid inclusions (10 M requests, 30 M CPU-ms, 10 M Class B). The dominant and only
volume-sensitive term is sparse gathering, at ~$0.45-0.74 per million cells.

## Sources

* R2 pricing: https://developers.cloudflare.com/r2/pricing/ (Standard: $0.015/GB-month, Class A $4.50/M, Class B $0.36/M, free tier
  10 GB / 1 M / 10 M, egress free; GetObject/HeadObject are Class B, PutObject/ListObjects Class A).
* Workers pricing: https://developers.cloudflare.com/workers/platform/pricing/ (Paid: $5 base, 10 M requests + 30 M CPU-ms included,
  $0.30/M requests, $0.02/M CPU-ms; subrequests not billed; Durable Objects $0.15/M requests, $12.50/M GB-s, SQLite rows written $1.00/M).
* Workers limits: https://developers.cloudflare.com/workers/platform/limits/ (6 simultaneous connections waiting for headers, which
  includes R2 and Cache API calls; 10,000 subrequests per invocation by default; Cache API 1,000 calls per request and 512 MB objects;
  128 MB memory; CPU 30 s default, 5 min max).
* Rate Limiting binding: https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/ (periods of 10 or 60 s, per
  location, eventually consistent; no price stated).
