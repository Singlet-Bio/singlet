# singlet-gpu — Blockers (user-gated)

Only externally-gated items. Every entry has a one-line "what user action unblocks this." Anything the orchestrator can resolve internally goes to `state/dag.md` or `state/followups.md`, not here.

## 🔴 Active

### INFRA-GIT-INIT
- **Blocked**: auto-commit + auto-push at end of each cycle (Rule 35 placeholder), git-tag-based releases.
- **What unblocks**: `cd ~/Singlet-AI && git init && git remote add origin <url> && git push -u origin main` — or a decision to keep the repo file-based.
- **Workaround in place**: `state/cycle-log.md` is the append-only durable record. `state/release-policy.md` describes versioning without git tags.
- **Severity**: low — does not block development. Blocks public release artifacts.

### ✅ RESOLVED-2026-04-28: INFRA-WEBSITE-FRONTIER-SYNC
- **Was blocking**: `singlet.bio/benchmarks` showing live frontier data; Phase G.1 of every cycle.
- **Resolution**: user supplied service-role key. Stored at `~/.config/singlet/supabase.env` (chmod 600, outside repo). `scripts/load_secrets.sh` sources it. `pip install --user supabase==2.29.0`. `frontier_sync.py` ran successfully — 24 rows uploaded to `gpu_frontier` table, all 11 frontier features visible. (Move this entry to `state/followups.md` archive after one cycle.)
- **Note**: keys posted in chat must be considered leaked. Recommend rotating in the Supabase dashboard at next convenience.

### INFRA-CUVS-CUGRAPH-INSTALL
- **Blocked**: feature 8 (kNN CAGRA tier 2), feature 9 (Leiden), feature 10 (UMAP). Affects 3 P0 features.
- **What unblocks**: `pip install cuvs-cu12` + RAPIDS cuGraph on g001/g008/g051. User confirmation required before installing system-wide.
- **Workaround**: feature 8 Exact backend is on frontier without CAGRA. Features 9 and 10 in `todo`.
- **Severity**: high — gates 3 roadmap features.

### INFRA-R-SCRAN-INSTALL
- **Blocked**: feature 2 scran deconvolution Test 5 (`RealData_GSM4037629_vs_R_scran`). Wall-SOTA gate cannot complete.
- **What unblocks**: install R + Bioconductor `scran` on g001/g008/g051. Or install `scranPY` Python port as fallback.
- **Workaround**: 4/5 tests pass; correctness signed without R reference comparison.
- **Severity**: low — feature is on frontier, only the wall-SOTA comparison row is missing.

### INFRA-WEBSITE-BLOG-PUBLISH
- **Blocked**: Phase G.4 blog publishing on major frontier promotions.
- **What unblocks**: clone `singletai-website` locally OR provide path. Verify `scripts/etl/publish_blog.py` exists and runs.
- **Workaround**: changelog entries written to `CHANGELOG.md` locally.
- **Severity**: low — blogs are optional.

## 🟡 Watch

### INFRA-MEM-TRACKING-H100
- `cudaMemGetInfo` delta returns 0 on H100 due to driver pooling. Peak memory unreliable on g051/g052.
- **What unblocks**: integrate `cudaMallocAsync` pool stats, or use NVML `nvmlDeviceGetMemoryInfo` snapshots before/after.
- **Severity**: medium — frontier promotions still happen, but memory column shows 0.0 on H100 runs.

### INFRA-G008-LIBSTDCPP
- libstdc++ ABI variance across nodes (g001 needs explicit `gcc-toolset-13 libstdc++.a` link; g050 does not).
- **What unblocks**: documented in `state/style-rules.md`; new tests should auto-detect via CMake.
- **Severity**: low — not blocking, just a per-node footgun.

## ✅ Resolved (move to followups.md after one cycle)

(none currently)

## Conventions

- 🔴 = blocks a P0/P1 cycle
- 🟡 = annoyance, not blocking
- ✅ = resolved (move out within one cycle)
- One sentence each in **Blocked** / **What unblocks** / **Workaround** / **Severity**.
- If a 🔴 item is here for >14 days, escalate to user explicitly in the next cycle's wakeup message.
