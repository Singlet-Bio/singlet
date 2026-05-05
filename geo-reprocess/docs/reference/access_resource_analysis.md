# ACCESS Resource Analysis — Actual Memory Needs

## Executive Summary

**simpleaf/piscem is the memory bottleneck**, not kraken2. From 516 completed v5b
jobs on clipper's 1.5 TB bigmem nodes (22 cores, no memory limit):

| Metric | MaxRSS (GB) | Elapsed (hrs) |
|--------|-------------|---------------|
| P10    | 100.5       | 1.90          |
| P25    | 130.0       | 4.90          |
| Median | 183.4       | 8.60          |
| P75    | 234.6       | 14.03         |
| P90    | 317.9       | 19.88         |
| P95    | 373.3       | 23.97         |
| Max    | 622.8       | 43.41         |
| Mean   | 193.2       | 10.22         |

**Recommended resource: TAMU Launch** — 32 cores, 371 GB/node, 0.125 credits/SU.
200K credits can process **~42,000–60,000 samples** (of 213K remaining) using a
smallest-first strategy. Full catalog requires a Research allocation (~5–7M credits).

---

## 1. Pipeline Memory Decomposition

The pipeline runs **sequentially**: download → detect → simpleaf quant → QC → kraken2 → cleanup.
MaxRSS = max(peak_simpleaf, peak_kraken2).

### Component Memory Footprints

| Component | Memory | Notes |
|-----------|--------|-------|
| Piscem index (human) | 3.3 GB | Loaded during simpleaf quant |
| Piscem index (mouse) | 2.0 GB | Loaded during simpleaf quant |
| simpleaf working memory | 10–600+ GB | **Scales with read count** — dominant consumer |
| Kraken2 PlusPF DB | 83 GB (hash.k2d) | Fixed cost, loaded into RAM |
| Kraken2 working set | ~2–5 GB | classified_reads dict + streaming |
| Python/OS overhead | ~2–3 GB | Process infrastructure |

### Memory vs Runtime (proxy for input size)

| Elapsed Time Bin | Jobs | Mean MaxRSS (GB) | Interpretation |
|------------------|------|-------------------|----------------|
| < 5 min          | 16   | 2.0  | Quick failures before quant |
| 5–30 min         | 24   | 3.3  | Small samples or early failures |
| 30 min – 2 hrs   | 12   | 73.3 | Small successful runs |
| 2–8 hrs          | 182  | 136.1 | Standard workload, simpleaf dominates |
| > 8 hrs          | 282  | 234.7 | Large FASTQs, simpleaf dominates |

**Key insight**: For jobs > 2 hrs (89% of successful completions), simpleaf peak > 85 GB,
so simpleaf dominates memory — not kraken2.

---

## 2. Input Size Distribution (212,961 unprocessed samples)

| Read Count Bin | Samples | % | Cumulative % |
|----------------|---------|---|-------------|
| < 10M          | 42,679  | 20.0% | 20.0% |
| 10–50M         | 64,309  | 30.2% | 50.2% |
| 50–100M        | 37,160  | 17.4% | 67.7% |
| 100–300M       | 48,755  | 22.9% | 90.6% |
| 300–500M       | 13,904  | 6.5%  | 97.1% |
| > 500M         | 6,154   | 2.9%  | 100%  |

Median read count: **49.5M** reads. Mean: **106M** reads. Max: **6.0 billion** reads.

---

## 3. Clipper Baseline (What Actually Ran)

| Parameter | Value |
|-----------|-------|
| Partition | bigmem (b003/b004) |
| Node RAM | **1,545 GB** (1.5 TB) |
| Cores allocated | 22 |
| Memory reserved | **None** (no `--mem` flag) |
| Memory enforcement | cgroups enabled but no per-job limit |
| Max concurrent | 2 jobs per node (22+22 = 44 of 46 effective cores) |

The 183 GB median MaxRSS was real — the jobs truly consumed that much RAM. On ACCESS
with strict per-job memory enforcement, we must request at least that much.

---

## 4. Optimization: kraken2 `--memory-mapping`

kraken2 supports `--memory-mapping` which uses mmap instead of loading the full
83 GB hash.k2d into process-local RAM. The OS kernel handles page caching.

**Impact**:
- Kraken2 process RSS: **~85 GB → ~5 GB**
- For small samples (kraken2 was peak): memory drops **~80 GB**
- For large samples (simpleaf was peak): **no change**
- Classification accuracy: **unchanged** (same DB, same algorithm)
- Speed: **~unchanged** (OS caches hot pages; may be marginally slower on cold start)

**Code change required** in `scgeo/pipeline/kraken2.py`, line ~247:
```python
kraken2_cmd = [
    "kraken2",
    "--db", str(db_path),
    "--memory-mapping",  # ← ADD THIS
    "--threads", str(config.kraken2.threads),
    ...
]
```

This helps the ~24 jobs in the 30min–2hr bin and small jobs, but does NOT reduce
memory for the median job where simpleaf already exceeds 85 GB.

---

## 5. Cost Model — TAMU Launch

### Launch Node Specs
- 32 cores/node, 371 GB RAM, AMD EPYC 7763
- Billing: SU/hr = max(cores_requested, ⌈mem_GB / 11.6⌉)
- Rate: 0.125 credits/SU

### Optimal Core Count Strategy

For memory-bound jobs, request cores = ⌈mem_GB / 11.6⌉ so memory-SU equals
core count, avoiding waste on either side.

| Estimated Memory | Optimal Cores | SU/hr | Credits/hr |
|-----------------|---------------|-------|------------|
| 30 GB           | 3             | 3     | 0.38       |
| 70 GB           | 7             | 7     | 0.88       |
| 130 GB          | 12            | 12    | 1.50       |
| 200 GB          | 18            | 18    | 2.25       |
| 320 GB          | 28            | 28    | 3.50       |
| 371 GB (full)   | 32            | 32    | 4.00       |

### Estimated Per-Job Costs

Using clipper's 22-core baseline (median 8.6 hrs) with sublinear thread scaling:

| Read Bin | Est. Memory | Cores | Runtime | Credits/Job |
|----------|------------|-------|---------|-------------|
| < 10M    | 30 GB      | 3     | ~4 hrs  | 1.5         |
| 10–50M   | 70 GB      | 7     | ~12 hrs | 10.5        |
| 50–100M  | 130 GB     | 12    | ~16 hrs | 24.0        |
| 100–300M | 200 GB     | 18    | ~20 hrs | 45.0        |
| 300–500M | 320 GB     | 28    | ~24 hrs | 84.0        |
| > 500M   | 400+ GB    | ACES  | ~30 hrs | ~250        |

### Do We Need Bigmem?

**For Launch (371 GB/node)**: covers samples up to ~500M reads (P95 of catalog).
If MaxRSS exceeds 371 GB on Launch, the job needs ACES or larger.

- ~95% of v5b jobs (≤ 373 GB) → **Launch** at 0.125 credits/SU
- ~4% of v5b jobs (373–488 GB) → **ACES** at 0.125 credits/SU (488 GB/node)
- ~1% of v5b jobs (> 488 GB) → **Bridges-2 EM** at 1.0 credits/core-hr (4 TB nodes)

Answer: **No, you don't need bigmem for 95% of samples.** Launch handles them.

---

## 6. Budget Analysis — 200K Credits

### Weighted Average Cost (all samples)
200K credits / 31.9 credits mean = **~6,300 average-complexity jobs**

### Smallest-First Strategy (recommended)

Process by ascending read count to maximize sample throughput:

| Phase | Bin | Samples | Credits/Sample¹ | Phase Credits | Cumulative |
|-------|-----|---------|-----------------|---------------|------------|
| 1     | <10M | 42,679 | 1.14           | 48,700        | 48,700     |
| 2     | 10–50M | 17,390² | 8.70       | 151,300       | 200,000    |

¹ Includes 30% failure rate (failed jobs cost ~0.3–1.0 credits each)
² Budget exhausted partway through this bin

**Result: ~60,000 samples attempted → ~42,000 successful quantifications**
(28% of remaining catalog, 100% of samples < 10M reads)

### Full Catalog Estimate

To process all 212,961 samples: **~5–7M ACCESS credits** required.
This implies a **Research allocation** after the Startup proves the pipeline.

---

## 7. Concurrency & Time-to-Completion

Total SU budget: 200,000 / 0.125 = **1,600,000 SU**

| Concurrent Nodes | SU/hr | Wall-Clock to Exhaust Budget |
|-----------------|-------|------------------------------|
| 5               | 160   | 10,000 hrs (417 days)        |
| 10              | 320   | 5,000 hrs (208 days)         |
| 20              | 640   | 2,500 hrs (104 days)         |
| 50              | 1,600 | 1,000 hrs (42 days)          |
| 100             | 3,200 | 500 hrs (21 days)            |

For the smallest-first strategy with small jobs (3 cores each):
- 10 Launch nodes × 10 jobs/node = **100 concurrent jobs**
- <10M phase: 42,679 / 100 × 4 hrs = **1,707 hrs ≈ 71 days**
- Budget consumed: 48.7K credits (24%)
- Remaining 10-50M phase would take additional ~100 days at same concurrency

**Recommendation**: Request at least **20 concurrent nodes** for viability.
At 20 nodes: full 200K budget consumed in ~104 days (3.5 months).

---

## 8. Pipeline Changes Required for ACCESS

### Mandatory
1. **Explicit memory requests**: clipper used no `--mem` flag; ACCESS requires it
2. **Tiered SLURM templates**: memory + cores based on ENA read_count from catalog
3. **Thread count adjustment**: match cores to memory tier (3–32 cores)

### Recommended Optimizations
1. **Add `--memory-mapping` to kraken2**: saves 80 GB RSS for small samples
2. **Decouple download from compute**: download with 2 cores (cheap), then
   process with full cores (avoid paying HPC rates for I/O wait)
3. **Memory prediction from read_count**: use ENA metadata to estimate memory
   before submission, avoiding OOM kills and over-provisioning

### Optional (Higher Risk, Higher Reward)
4. **Smaller kraken2 database**: Standard DB (~8 GB) instead of PlusPF (83 GB) —
   sufficient for contamination detection (bacterial, viral, host cross-contamination)
5. **Skip kraken2 in first pass**: run simpleaf-only, add contamination annotations
   as a separate lightweight job later (requires keeping FASTQs or re-downloading)

---

## 9. 10K Credit Pilot Strategy (277 Remaining Clipper Batches)

### Refined Memory Distribution (541 completed v5b batch substeps)

| Memory Tier | Jobs | % | Avg MaxRSS | Avg Wall (hr) | Fits On |
|---|---|---|---|---|---|
| < 256 GB | 447 | 84.0% | 139 GB | 8.4 | Launch 256G |
| 256–371 GB | 60 | 11.3% | 304 GB | 16.5 | Launch full node |
| 371–488 GB | 19 | 3.6% | 410 GB | 22.1 | ACES full node |
| ≥ 488 GB | 6 | 1.1% | 634 GB | 23.5 | Skip / defer |

**Critical finding**: Big-memory jobs are also long-running (16–24 hr avg vs
8.4 hr for standard jobs). This compounds their cost.

### Recommended: 3-Tier OOM Escalation (Strategy A)

All jobs start on Launch `--mem=256G`. OOM failures escalate to larger tiers.

| Tier | Resource | `--mem` | Cores | SU/hr | Projected Jobs | Credits |
|---|---|---|---|---|---|---|
| 1 | Launch | 256G | 23¹ | 23 | 233 | 5,621 |
| 1→2 OOM waste | Launch | — | — | — | 44 fail | 253 |
| 2 | Launch | 371G | 32² | 32 | 31 | 1,637 |
| 2→3 OOM waste | Launch | — | — | — | 13 fail | 104 |
| 3 | ACES | 488G | 96² | 96 | 10 | 1,063 |
| 4 | — | — | — | — | 3 skip | 0 |
| **Total** | | | | | **274/277** | **8,678** |

¹ ⌈256/11.6⌉ = 23 effective SU — request 23 cores to match (free core)
² Full-node memory → all cores are free; use them all for faster execution

**Budget: 8,678 / 10,000 credits → 1,322 contingency (13.2%)**

### Wall-Clock Timeline

| Phase | Jobs | Concurrent | Runtime | Wall Clock |
|---|---|---|---|---|
| 1: Launch 256G | 277 start | 20 | 8.4 hr avg | ~5 days |
| 2: Launch 371G retries | ~44 | 20 | 13.2 hr avg | ~1.4 days |
| 3: ACES retries | ~13 | 5 | 8.9 hr avg³ | ~1 day |
| **Total** | | | | **~7 days** |

³ ACES 96 cores estimated ~2.5× faster than clipper's 22 cores (cores are free
due to memory-proportional charging, so use all 96).

### Key Optimizations Built Into Strategy A

1. **Memory-proportional core matching**: On Launch 256G, ⌈256/11.6⌉=23,
   so request 23 cores (1 free core vs the 22 used on clipper)
2. **Full-node core exploitation**: On Tier 2 (Launch full) and Tier 3 (ACES),
   memory cost already equals full-node cost, so ALL cores are free
3. **OOM waste minimization**: Jobs OOM during simpleaf mapping (early in quant),
   wasting only ~2 hr × tier rate per failure

### Alternative Strategies Considered

| Strategy | Total Credits | Remaining | Drawback |
|---|---|---|---|
| **A: 3-tier OOM (recommended)** | 8,678 | 1,322 | Requires OOM monitoring |
| B: All Launch full 371G + ACES | 11,094 | -1,094 | **Over budget** |
| C: Only <256GB jobs | 5,621 | 4,379 | 44 jobs deferred (16%) |

Strategy B fails because 84% of jobs are charged at 32 SU/hr (full-node
memory rate) when they only need 23 SU/hr — a 39% overhead.

### Pipeline Config Changes for ACCESS

```python
# scgeo/slurm/submit.py — new ACCESS tier defaults
TIER_CONFIG = {
    "launch_256": {"partition": "standard", "cpus": 23, "memory": "256G", "time": "12:00:00"},
    "launch_371": {"partition": "standard", "cpus": 32, "memory": "371G", "time": "24:00:00"},
    "aces_488":   {"partition": "standard", "cpus": 96, "memory": "488G", "time": "36:00:00"},
}
```

## 10. Production Exchange Strategy (200K Credits)

| Phase | Exchange | Resource | Purpose |
|-------|----------|----------|---------|
| 1     | 10K credits | Launch SUs | Pilot batch (277 remaining clipper jobs) |
| 2     | 190K credits | Launch SUs + 5K to ACES | Production run, smallest-first |
| 3     | Research allocation | Launch (1M+ credits) | Full catalog processing |

**Do NOT exchange to**: Bridges-2 RM, Delta, Expanse, Stampede2,
REPACSS, Derecho — all have ≤6 GB/core, making memory-heavy jobs 4–50× more
expensive than Launch.

---

## Appendix: Key Data Sources

- **MaxRSS data**: `sacct` for v5b arrays 229190, 229256, 229630, 230076, 246982, 252356
  (516 completed batch substeps, saved to `/tmp/v5b_rss_vs_time.txt`)
- **Catalog**: `archive/data/phase3_expansion_unprocessed.csv` (212,961 samples)
- **Index sizes**: `du -sh` on `index/human_splici/index/` (3.3G), `mouse_splici/index/` (2.0G)
- **Kraken2 DB**: `du -sh` on `kraken2_db/pluspf/` (84G total, hash.k2d = 83G)
- **Clipper nodes**: `scontrol show node b003` → 48 cores, 1,545 GB RAM
- **Pipeline code**: `scgeo/pipeline/api.py` (sequential stages), `quantify.py`, `kraken2.py`
