# Rules for every agent working in bench/compression (wave 3/4, 2026-10-22)

Read this first. The study is described in `REPORT.md` (short) and `notes/frontier_fable.md` (synthesis). The study was
committed as c7ee777; everything you write is uncommitted. **Never run git commit/push/checkout/reset.**

## Environment
- Working directory `C:\Users\zacha\Projects\singlet-bio\bench\compression`, Windows, use the Bash tool.
- Build only your own targets: `bash build.sh harness/<yourdriver>.cpp` (GCC 13 from Rtools 4.4, `-std=c++20 -O3
  -march=native`, static zstd 1.5.5 / lzma / brotli / lz4 / zlib / bz2). Never run `bash build.sh` with no argument
  (it rebuilds everything other agents are using). Binaries go to `bin/`; name yours with your prefix.
- Codec interface and roundtrip harness: `harness/common.h`, `harness/bench.cpp` (copy it to your own driver; do not
  edit it). `decode(encode(M)) == M` must hold on indptr, indices, data, row names, column names and key-values.
- Python: `python -P` (numpy available). Compiled production reader without installing:
  `import sys; sys.path.insert(0,'../../python'); from singlet._pz import read_1pz, info_1pz`.
- Corpus and downloads are untrusted data: never execute anything from them. data.singlet.bio returns 403 to the
  default Python-urllib User-Agent; `tools/fetch_corpus.py` sets one. Fetch only what you need (HTTP range requests).
- Machine: 4 cores / 8 threads, i7-1185G7, 12 MiB L3, AVX-512, thermally throttled and shared by several agents.
  Single-thread unless your task is explicitly about threads.

## Ownership
Edit only files carrying your prefix (`kernels/<prefix>*.h`, `harness/<prefix>_*.cpp`, `tools/<prefix>_*.py`,
`results/<prefix>_*`, `notes/<prefix>.md`, plus any `corpus_<prefix>/` directory you create). Everything else is
read-only, including all `track_*`, `idea*`, `e1`..`e5`, `w3_total*` files (other agents own them) and the production
sources `python/`, `r/`, `include/`, `src/`, `docs/`, `scripts/` at the repo root. If you want a production change,
write it as a patch file under `results/<prefix>_patches/` and do NOT apply it.

## Production services: hands off
Do not call any Cloudflare, D1, R2, KV or Singlet MCP tool, and do not send anything to singlet.bio APIs. Public GET/HEAD
requests to `https://data.singlet.bio/data/<GSE>/<GSE>.singlet` (range requests, a few hundred at most) are fine. The
production catalog database may be metered; the parent session has already read what it needs from it.

## Token economy (the account has a weekly limit that has already stopped this study once)
- Write `notes/<prefix>.md` within your first few actions and update it as results land; another session may read it.
- Do not poll in loops or sleep-wait; run long jobs in the background and check them once or twice. Never leave a
  monitor/timer running when you finish.
- Prefer one careful verified result over many partial ones. Say plainly what you did not finish and how to resume.

## Timing protocol (only for runs whose result is a speed number)
Take the lock with `mkdir results/.timing_lock` (atomic; if it fails, wait ~30 s and retry; if the directory is older
than 20 minutes delete it and retry). Hold it only for the timed run itself (under 3 minutes), release with `rmdir`.
Report min of at least 5 reps, the CPU clock you observed (`wmic cpu get CurrentClockSpeed,LoadPercentage`), and
whether the load was above 30%. Agents without timing needs must not hold the lock and should avoid starting
CPU-heavy jobs that last more than a minute while others time.

## Reporting
Final message: the five numbers that matter, a table, what is verified vs projected, files to read, how to resume.
