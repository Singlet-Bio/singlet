# ACCESS Execution Plan — Human Transcriptomics in 2 Weeks

## Award: BIO260157 — 200,000 ACCESS Credits

---

## 1. Scope of Work

### Target: All human single-cell transcriptomics with SRA FASTQ data

From `stage7_multimodal_catalog.parquet`:

| Category | SRX Samples | GSEs |
|----------|-------------|------|
| **Already in quant/** | 63,671 | 2,062 |
| **New (not yet started)** | 142,659 | 2,535 |
| **Total human RNA** | **206,330** | **4,597** |

Assay breakdown (new samples):

| Assay Class | Samples |
|-------------|---------|
| rna_only | 103,911 |
| citeseq | 15,200 |
| multiome | 10,953 |
| visium | 8,481 |
| perturbseq | 4,114 |

### Pipeline per sample (sequential)
`download → simpleaf quant → QC → kraken2 → cleanup`

---

## 2. Memory Analysis (from 549 v5b jobs on clipper)

### MaxRSS Distribution (GB)

| P10 | P25 | Median | P75 | P85 | P90 | P95 | P99 | Max |
|-----|-----|--------|-----|-----|-----|-----|-----|-----|
| 1.1 | 109.3 | 163.3 | 222.8 | 256.7 | 305.7 | 360.7 | 476.7 | 998.7 |

### Memory Tiers with Wall Time

| Tier | MaxRSS Range | Jobs | % | Avg GB | Avg Hrs | Resource |
|------|-------------|------|---|--------|---------|----------|
| A | 0–96 GB | 114 | 21% | 17.3 | 1.6 | Launch 128G |
| B | 96–192 GB | 215 | 39% | 142.5 | 8.2 | Launch 192G |
| C | 192–256 GB | 135 | 25% | 218.9 | 11.4 | Launch 256G |
| D | 256–371 GB | 60 | 11% | 303.6 | 12.7 | Launch 371G (full) |
| E | 371–488 GB | 19 | 3.5% | 409.8 | 12.9 | ACES 488G |
| F | 488+ GB | 6 | 1.1% | 634.0 | 8.3 | Bridges-2 EM / skip |

**Key insight**: simpleaf has NO memory-limit flag. Memory is determined entirely
by input read count and index size. Cannot be capped.

### CPU Efficiency: 13.2% average

Pipeline is largely serial. simpleaf saturates at ~8 threads during mapping; other
steps use 1–8 threads. On Launch with memory-proportional billing, extra cores
beyond the memory-equivalent are free — no waste.

---

## 3. Code Optimizations

### 3a. Enable kraken2 `--memory-mapping`

**Status**: Implemented in `scgeo/pipeline/kraken2.py`

Kraken2 loads the PlusPF database (83 GB `hash.k2d`) into RAM by default.
With `--memory-mapping`, it uses mmap instead, reducing RSS from ~85 GB to ~5 GB.

- **Accuracy**: Unchanged (same DB, same algorithm)
- **Speed**: Negligible impact on SSD-backed filesystems (OS page cache handles hot pages)
- **Impact**: Eliminates 83 GB memory floor for small samples (Tier A)
- **For large samples** (simpleaf peak > 85 GB): No change — simpleaf dominates

### 3b. Optimal CPU Request

On Launch, billing = max(cores_in_SU, ⌈mem_GB / 11.6⌉). For any memory request
above 12 × 11.6 = 139 GB, the memory-proportional charge dominates. Requesting
more physical cores costs nothing extra, so use all available cores within SU.

For 256G request: 23 SU → 23 × 6 = 138 physical cores available. Give simpleaf
all of them — even if saturation is at ~8, the mapping step benefits and costs
nothing extra.

---

## 4. TAMU Launch Specs (CORRECTED)

| Parameter | CPU Nodes | GPU Nodes |
|-----------|-----------|-----------|
| Physical cores | 192 (2×96 AMD Genoa 9654) | 192 |
| RAM | 384 GB DDR5 (371 GB usable) | 768 GB |
| Count | 35 | 10 |
| Billing | 32 SU/node-hour | — |
| Exchange | 0.125 credits/SU | — |

**Key**: 1 SU = 6 physical core-hours. Full node = 32 SU/hr = 4.00 credits/hr.

### Memory-Proportional Billing

| `--mem` Request | SU/hr | Credits/hr |
|-----------------|-------|------------|
| 128G | 12 | 1.50 |
| 192G | 17 | 2.12 |
| 256G | 23 | 2.88 |
| 371G (full node) | 32 | 4.00 |

---

## 5. Credit Budget — Three Strategies

### Batch Structure

- **50 samples per batch** (current default)
- New processing: **2,854 tasks** × 8.3 hrs average
- Recheck (already-processed GSEs): **1,274 tasks** × 0.5 hrs average (mostly skips)
- **Total task-hours: 24,325**

### Option A: 371G Fixed on Launch (Simplest)

No OOM for 95.3% of tasks. Remaining 4.7% overflow to ACES.

| Item | Credits |
|------|---------|
| New tasks (2,854 × 8.3 hrs × 32 SU × 0.125) | 94,753 |
| Recheck tasks (1,274 × 0.5 hrs × 32 × 0.125) | 2,548 |
| OOM waste at 371G (~5%, 3 hrs each) | 1,608 |
| ACES retry (134 tasks × 8.3 hrs × 96 SU × 0.125) | 13,346 |
| **Total** | **112,255** |
| **Remaining** | **87,745** |

### Option B: 256G + OOM Escalation (RECOMMENDED)

Start at 256G (catches 84.5%), retry OOMs at 371G, then ACES.

| Item | Credits |
|------|---------|
| Pass 1 at 256G (84.5% succeed: 2,411 × 8.3 × 23 × 0.125) | 57,532 |
| OOM waste at 256G (443 tasks × 3 hrs × 23 × 0.125) | 3,821 |
| Retry at 371G (305 succeed: 305 × 8.3 × 32 × 0.125) | 10,126 |
| OOM waste at 371G (138 × 3 × 32 × 0.125) | 1,656 |
| Retry at ACES (138 × 8.3 × 96 × 0.125) | 13,745 |
| Recheck tasks (1,274 × 0.5 × 23 × 0.125) | 1,831 |
| **Total** | **88,712** |
| **Remaining** | **111,288** |
| **Savings vs A** | **23,543 (21%)** |

### Option C: 192G + Multi-Level Escalation (Cheapest)

More complex management but maximally efficient.

| Item | Estimated Credits |
|------|-------------------|
| Total through all tiers | ~81,000 |
| **Remaining** | ~119,000 |

---

## 6. Concurrency & Timeline

### Required Concurrent Tasks for 14-Day Completion

```
Total task-hours: 24,325
Available: 14 days × 24 hrs = 336 hrs
Required concurrent: ⌈24,325 / 336⌉ = 73 tasks
```

At 73 concurrent tasks with 256G each → occupies ~73/3 = ~24 Launch CPU nodes
(each node fits ~3 jobs at 256G with 384 GB total). This is 69% of the 35 CPU
nodes — high but feasible if queue is not congested.

### Timeline at Different Concurrency Levels

| Concurrent Tasks | Days | Nodes Used (at 256G) |
|-----------------|------|---------------------|
| 50 | 20.3 | 17 |
| 75 | 13.5 | 25 |
| 100 | 10.1 | 34 |

**Recommendation**: Target 75 concurrent (`%75` in SLURM array).

---

## 7. Experimentation Phase (Days 1–3)

### Day 1: Environment Setup + Billing Verification

1. **Get Launch access**: SSH to `launch.hprc.tamu.edu`, verify ACCESS account
2. **Transfer data**: Globus or rsync the piscem index, kraken2 DB, and catalog
3. **Install environment**: conda/pip install scgeo dependencies
4. **Submit 1 test job** (2 samples, `--mem=256G`, `--cpus-per-task=24`, `--time=2:00:00`)
5. **CRITICAL**: After completion, run `seff <job_id>` to verify:
   - Actual SU charged (confirms billing formula)
   - Memory utilized vs requested
   - CPU efficiency
6. **Verify kraken2 `--memory-mapping`**: Check that MaxRSS does NOT include 83 GB spike

### Day 2: Calibration Runs

1. Submit 10 batches (500 samples) across range of expected sizes
   - 5 batches at `--mem=192G`
   - 5 batches at `--mem=256G`
2. Monitor for OOM kills at 192G → determines optimal starting allocation
3. Profile wall-time distribution → calibrate time limits
4. Test download bandwidth from ENA → may bottleneck throughput

### Day 3: Pilot + Parameter Lock

1. Submit 50 batches (2,500 samples) at chosen allocation
2. Verify pipeline outputs are correct (manifests, .1pz files, QC metrics)
3. Calculate **actual credits/sample** from `sacct` data
4. Lock parameters for production:
   - `--mem`: validated allocation
   - `--time`: P95 of observed wall time + 30% margin
   - `--cpus-per-task`: can be high (free with memory billing)
   - `samples_per_batch`: keep 50 unless data suggests smaller batches
   - `max_concurrent`: set based on queue responsiveness

**Experimentation budget**: ~3,000–5,000 credits (513 samples at ~6 credits/sample avg)

---

## 8. Production Phase (Days 4–17)

### Wave 1: 256G on Launch (Days 4–14)

```bash
# Generate full catalog of remaining human RNA samples
sc-geo batch submit \
    human_rna_remaining.parquet \
    --job-name scgeo-human-256g \
    --partition normal \
    --cpus 96 \
    --memory 256G \
    --time 24:00:00 \
    --samples-per-batch 50 \
    --max-concurrent 75
```

- Tasks: ~4,127 (including rechecks)
- Expected success: ~84.5% of new processing tasks
- Credits consumed: ~60,000 (Days 4–14)

### Wave 2: 371G Retry on Launch (Days 12–15)

Collect OOM failures from Wave 1, resubmit at full-node allocation:

```bash
sc-geo batch submit \
    oom_retry_256g.csv \
    --job-name scgeo-human-371g \
    --memory 371G \
    --cpus 192 \
    --time 36:00:00 \
    --max-concurrent 20
```

- Tasks: ~443
- Credits: ~14,000

### Wave 3: ACES 488G (Days 15–17, if needed)

For samples that exceed Launch's 371G:

```bash
# On ACES
sc-geo batch submit \
    oom_retry_371g.csv \
    --job-name scgeo-human-488g \
    --memory 488G \
    --cpus 96 \
    --time 36:00:00 \
    --max-concurrent 10
```

- Tasks: ~138
- Credits: ~14,000

### Monitoring

- Daily: `sacct -j <job_id> --format=JobID,State,MaxRSS,Elapsed,ExitCode`
- Track OOM kills: `sacct -s OOM` — extract failed sample IDs for retry
- Credit burn rate: `myproject` or ACCESS portal

---

## 9. Data Movement Strategy

### To Transfer to Launch

| Data | Size | Method |
|------|------|--------|
| Piscem human index | 3.3 GB | Globus / rsync |
| Kraken2 PlusPF DB | ~100 GB | Globus |
| Stage 7 catalog | ~200 MB | rsync |
| scgeo codebase | ~5 MB | git clone |

### Storage on Launch

- `/home`: 10 GB quota (code only)
- `/scratch`: 1 TB default (FASTQs, temporary)
- **Request storage increase** via TAMU HPRC ticket if needed

### Output

Each sample produces:
- `sample_manifest.json` (~2 KB)
- `quant/` directory with count matrix in `.1pz` format (~1–50 MB)
- Cleanup removes FASTQs and intermediates automatically

Estimated total output: ~5–10 TB for all 206K samples. May need `/scratch`
quota increase or project storage allocation.

---

## 10. Summary

| Metric | Value |
|--------|-------|
| **Samples to process** | 206,330 (142,659 new + 63,671 recheck) |
| **SLURM tasks** | ~4,127 (50 samples/batch) |
| **Primary resource** | TAMU Launch at 256G (`--mem=256G`) |
| **Concurrency** | 75 tasks (`%75` SLURM array) |
| **Total credits (Option B)** | ~89,000 (44% of budget) |
| **Remaining credits** | ~111,000 (for other species, retries, etc.) |
| **Timeline** | 3 days experiment + 14 days production |
| **Code changes** | kraken2 `--memory-mapping` (saves 80 GB for small samples) |
| **Critical Day 1 task** | Verify SU billing with `seff` on test job |

### Risk Register

| Risk | Mitigation |
|------|-----------|
| SU billing higher than estimated | Day 1 verification; fall back to Option C (192G start) |
| Queue congestion limits concurrency | Submit off-peak; split into multiple smaller array jobs |
| ENA download bandwidth bottleneck | Use parallel curl (8 segments); retry with SRA fallback |
| Storage quota exceeded | Request increase; process in waves with cleanup between |
| >5% jobs exceed 371G | Route to ACES; for >488G, investigate individually |
| Hardware mismatch (clipper → Launch) | Calibration runs on Day 2 validate memory profiles transfer |
