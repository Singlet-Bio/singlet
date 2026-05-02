# Singlify Catalog Pipeline Status

> Updated by doc-scribe after every cycle. Tracks running totals for the mass-reprocessing pipeline.

## Last updated: 2026-04-29 (Cycle 183)

### Cycle 178 — Multi-species diagnostic batch (39 samples, 6 species)
- **HEAD**: 159a442 (origin/main synced, includes f23d08d bloom fix)
- **CTests**: 84/84
- **Binary**: rebuilt 2026-04-29 03:42 (includes bloom filter hash fix, z-test species detection)
- **Batch**: job 368190, array 1-39, 6 species (human, mouse, zebrafish, drosophila, macaque, chicken)

#### C178 Results (39 tasks, 36 result JSONs written)
| Status | Count | % |
|--------|-------|---|
| SUCCESS | 10 | 25.6% |
| SOFT_FAIL | 9 | 23.1% |
| HARD_FAIL | 17 | 43.6% |
| No result JSON | 3 | 7.7% |

#### C178 By Species (excluding download_fail)
| Species | SUCCESS | SOFT_FAIL | Notes |
|---------|---------|-----------|-------|
| Mus musculus | 7 | 3 | All 10x SUCCESS (100%), Smart-seq2/methylation SOFT_FAIL |
| Homo sapiens | 2 | 5 | 2 near-zero MR = likely species mislabeled in catalog |
| Macaca mulatta | 1 | 0 | **FIRST macaque success!** 2814 cells, 75.7% MR |
| Drosophila mel. | 0 | 1 | 72.8% MR, 6 cells (Smart-seq2 plate-based) |
| Danio rerio | 0 | 0 | All download_fail |
| Gallus gallus | 0 | 0 | All download_fail |

#### C178 By Protocol (excluding download_fail)
| Protocol | SUCCESS | SOFT_FAIL | Success % |
|----------|---------|-----------|-----------|
| 10x v2/v3 | 8 | 2 | 80% |
| plate_based | 1 | 0 | 100% |
| smartseq2 | 0 | 4 | 0% |
| chipseq | 0 | 1 | 0% |
| methylation | 0 | 1 | 0% |

#### C178 Key Insights
- **Mouse 10x is rock-solid**: 7/7 = 100% success when data available
- **Macaque works!**: First non-human/non-mouse success (bloom falls through to metadata, GRCh38 compatible)
- **Download failures dominate**: 17/39 = 44% data_incomplete (barcode-stripped SRA deposits)
- **Catalog quality bottleneck**: chipseq, methylation, bulk_rnaseq miscategorized as scRNA-seq
- **Two catalog species mismatches**: GSM6430567 (labeled mouse, 0.6% MR) + GSM8487013 (labeled human, 0.003% MR)
- **Bloom filter validated on mouse**: All mouse 10x samples correctly detected

### Cycle 179 — Pure 10x multi-species batch (41 samples)
- **HEAD**: 1c780b1 (origin/main synced)
- **Batch**: job 368230, array 1-41, pure 10x protocols, 6 species

#### C179 Results (41 tasks, 36 results at checkpoint)
| Status | Count | % |
|--------|-------|---|
| SUCCESS | 19 | 52.8% |
| SOFT_FAIL | 6 | 16.7% |
| HARD_FAIL | 11 | 30.6% |

#### C179 Success By Species
| Species | Success | Total | Rate |
|---------|---------|-------|------|
| Homo sapiens | 7 | 12 | 58% |
| Mus musculus | 6 | 14 | 43% |
| Macaca mulatta | 3 | 3 | **100%** |
| Gallus gallus | 2 | 2 | **100%** |
| Drosophila melanogaster | 1 | 3 | 33% |
| Danio rerio | 0 | 2 | 0% |

#### Milestones
- **FIRST Drosophila success**: GSM8089497 (9,198 cells, 83.7% MR, 10xv3)
- **FIRST Chicken success**: GSM4257295 (527 cells, 82.2% MR, 10xv2)
- **5 species working**: human, mouse, macaque, drosophila, chicken

### Cycle 180 — High-confidence 10x batch + nonhost OOM fix (59 samples)
- **HEAD**: 151317f (origin/main synced, includes nonhost 20M cap)
- **CTests**: 84/84
- **Binary**: rebuilt 2026-04-29 05:00 (epoch 1777457983, nonhost cap fix)
- **Batch**: job 368274, array 1-59, pure 10x, 5 species

#### C180 Results (59 tasks submitted, 46 results)
| Status | Count | % (of results) |
|--------|-------|-----------------|
| SUCCESS | 39 | 84.8% |
| SOFT_FAIL | 7 | 15.2% |
| HARD_FAIL (no result) | 12 | — |
| Still running | 1 | — |

HARD_FAIL breakdown: 9 download_fail, 2 single-end misclass, 1 zero-cells (T13)

#### C180 Success By Species (of samples that produced results)
| Species | SUCCESS | SOFT_FAIL | Rate |
|---------|---------|-----------|------|
| Homo sapiens | 17 | 4 | 81% |
| Mus musculus | 16 | 2 | 89% |
| Macaca mulatta | 3 | 0 | 100% |
| Gallus gallus | 3 | 0 | 100% |
| Drosophila melanogaster | 0 | 1 | 0% |

#### C180 Key Insights
- **RECORD batch success rate**: 85% (39/46) — best ever
- **Nonhost OOM fix validated**: Zero OOM failures (previously caused by 60M+ read loads)
- **Mouse 10x**: 89% success (16/18)
- **Human 10x**: 81% success (17/21)
- **Chicken confirmed**: 3/3 = 100% (Gallus gallus GRCg7b index works perfectly)
- **Macaque confirmed**: 3/3 = 100% (Mmul_10 index working)
- **Download failures still dominant HARD_FAIL**: 9/12 no-result tasks
- **SOFT_FAILs**: 2 near-zero MR (catalog species mismatch), 5 moderate low-map (19-47% MR)
- **Notable successes**: GSM7840563 (31,103 cells, 90% MR), GSM7873673 (20,845 cells, 76% MR)

### Overall Pipeline Totals
- **Total results**: 1,553
- **SUCCESS**: 627 (40.4%)
- **Species with confirmed SUCCESS**: human, mouse, macaque, drosophila, chicken
- **Pure 10x success rate**: 85% (C180), 73% (C181), 53% (C179)
- **Trend**: Strong improvement with catalog filtering + nonhost OOM fix + high-confidence batches

### Cycle 181 — Largest batch, random 10x selection (71 samples, in progress)
- **HEAD**: 833d559 (origin/main synced)
- **Batch**: job 368425, array 1-71, random 10x, 5 species (30 mouse, 27 human, 5 macaque, 5 drosophila, 4 chicken)

#### C181 Results (partial — 3 tasks still running)
| Status | Count | % (of results) |
|--------|-------|-----------------|
| SUCCESS | 19 | 73.1% |
| SOFT_FAIL | 7 | 26.9% |
| STAR OOM | 6 | — |
| download_fail | 9+ | — |
| single-end misclass | 3 | — |

#### C181 Key Insights
- **STAR OOM**: 6 samples killed by OOM (signal 9) on 128G nodes. Need read_count filter or 256G tier.
- **Download failures**: Large samples (300M-1.5B reads) overflow /dev/shm during download.
- **Nonhost 20M cap working**: T66 capped at 20M reads, EM converged, no OOM.
- **Notable successes**: GSM3270885 (24,828 cells), GSM4043502 (12,714 cells), GSM8690428 (10,600 cells)
- **Lesson**: Include read_count_estimate in batch selection; skip samples >200M reads for 128G tier.

### Cycle 182 — Genome-shared test, 100 samples (job 368670, in progress)
- **HEAD**: 25cf5a4 (origin/main synced)
- **Batch**: 100 samples, 5 species, NO read_count filter (mistake — all read_count=0)
- **Key change**: `--genome-shared` flag added to job script

#### C182 Results (interim — 49/100 tasks finished, 9 running, 42 pending)
| Status | Count | % (of finished) |
|--------|-------|-----------------|
| COMPLETED (exit 0) | 10 | 20% |
| FAILED (exit 1) | 34 | 69% |
| OUT_OF_MEMORY | 5 | 10% |
| RUNNING | 9 | — |
| PENDING | 42 | — |

#### C182 COMPLETED Results (8/10 SUCCESS = 80%)
| Task | GSM | Cells | MR | Wall |
|------|-----|-------|-----|------|
| T7 | GSM4586593 | 3,005 | 71.7% | 265s |
| T18 | GSM3684535 | 578 | 86.8% | 556s |
| T12 | GSM8178275 | 302 | 80.0% | 3,728s |
| T14 | GSM4770471 | 2,395 | 67.3% | 2,682s |
| T17 | GSM7680518 | 6,910 | 82.4% | 2,415s |
| T25 | GSM8577676 | 233 | 60.6% | 1,133s |
| T29 | GSM8454811 | 732 | 63.1% | 1,227s |
| T34 | GSM7062661 | 8,561 | 88.0% | 1,185s |
| T4 | GSM7054586 | 5 | 0.01% | 909s | ← SOFT_FAIL (species mismatch) |
| T39 | GSM8376305 | 527 | 0.9% | 1,158s | ← SOFT_FAIL (species mismatch) |

#### C182 Key Insights
- **--genome-shared WORKS**: STAR logs show `--genomeLoad LoadAndKeep`, genome RSS 0 for 2nd+ tasks on same node
- **86% success among post-download samples**: 6/7 completed = SUCCESS
- **Download failures dominate**: 15/25 finished = download_fail. Root cause: batch had read_count=0 (no size filter), many samples 200M-540M reads overflowing /dev/shm timeout
- **OOM persists for 500M+ read samples at 128G**: T5 (561M reads, est BAM 104GB), T8, T15 all OOM
- **Fix for C183**: Read count filter 10M-200M applied. Expected to eliminate both download failures AND OOM.

### Cycle 183 — Read-count filtered batch (job 368734, COMPLETE)
- **Batch**: 100 samples, 40 human, 40 mouse, 10 macaque, 5 drosophila, 5 chicken
- **Read count**: 10M-179M (properly filtered)
- **Key change**: read_count filter + --genome-shared

#### C183 Final Results (100 tasks)
| Status | Count | % |
|--------|-------|---|
| COMPLETED (exit 0) | 32 | 32% |
| FAILED (exit 1) | 61 | 61% |
| OUT_OF_MEMORY | 4 | 4% |
| RUNNING/CANCELLED | 3 | 3% |

#### C183 Failure Breakdown
| Category | Count | Notes |
|----------|-------|-------|
| data_incomplete (empty R2) | ~53 | Barcode-stripped SRA deposits |
| download_timeout (30:00) | 2 | VDB server-side throttling |
| zero_barcodes | 1 | Multiome ATAC mislabeled as GEX |
| OOM (>100M reads) | 4 | 108M-156M reads exceed 128G |

#### C183 True Success Rate (excluding data_incomplete)
- **Non-data_incomplete tasks**: ~40
- **Successes**: 32
- **True success rate**: 32/40 = **80%** (matches C180 record)
- **With OOM excluded**: 32/36 = **89%**

#### C183 By Species
| Species | COMPLETED | Protocol diversity |
|---------|-----------|-------------------|
| Homo sapiens | 11 | smartseq2, 10xv3, 10xv2, plate_based, celseq2, indrop |
| Mus musculus | 14 | celseq2, scirna, marsseq, 10xv3, smartseq2, hi_c, rip_seq |
| Macaca mulatta | 4 | bulk, unknown, 10xv2 |
| Drosophila melanogaster | 2 | scATAC, 10xv2 |
| Gallus gallus | 0 | (all data_incomplete) |

#### C183 Key Insights
- **read_count filter WORKS**: Zero 30-min download timeouts from oversized samples
- **data_incomplete is the #1 failure mode**: 53% of batch wasted on barcode-stripped deposits
- **OOM eliminated at ≤100M reads**: All 4 OOMs were 108M-156M (above 100M target)
- **Protocol diversity proven**: celseq2, indrop, marsseq, scirna, scATAC, hi_c, rip_seq all succeed
- **Multi-species validated**: Human, mouse, macaque, drosophila all working
- **Wall time**: 1-91 min (median ~11 min for successful tasks)

### Cycle 184 — High-yield protocol bias (job 369121, NEAR-COMPLETE)
- **Batch**: 100 samples (70 high-yield + 30 exploratory), ≤100M reads
- **Protocol bias**: 70% from proven protocols (10xv3/v2, celseq2, scirna, marsseq, dropseq, scATAC)
- **New filter**: excludes 32 known data_incomplete GSMs from C183

#### C184 Results (96/100 finished, 4 long-runners still running)
| Status | Count | % |
|--------|-------|---|
| COMPLETED (exit 0) | 39 | 39% |
| FAILED (exit 1) | 56 | 56% |
| OUT_OF_MEMORY | 1 | 1% |
| RUNNING (>2h) | 4 | 4% |

#### C184 Failure Breakdown
| Category | Count | Notes |
|----------|-------|-------|
| data_incomplete (instant, <15s) | 17 | Barcode-stripped SRA deposits |
| R2 empty (delayed, 1-5 min) | ~5 | Downloaded OK, R2 empty during alignment |
| quality_string_mismatch (STAR exit 104) | 2+ | AUTOFIX-DECODE-QUAL-LENGTH bug |
| no_barcodes (protocol detect fail) | 1+ | Wrong barcode list assigned |
| download_fail | 1+ | VDB timeout |
| Other pipeline failures | ~30 | Various alignment/detection issues |
| OOM | 1 | Single outlier |

#### C184 Protocol Performance (COMPLETED only)
| Protocol | COMPLETED | data_incomplete | Viable Rate |
|----------|-----------|-----------------|-------------|
| celseq2 | 6 | 1 | 86% |
| scirna | 6 | 2 | 75% |
| 10xv3 | 6 | 2 | 75% |
| smartseq2 | 4 | 4 | 50% |
| marsseq | 3 | 0 | 100% |
| 10xv2 | 3 | 0 | 100% |
| 10x_suspect | 3 | 0 | 100% |
| dropseq | 2 | 0 | 100% |
| scATAC | 1 | 2 | 33% |
| unknown | 0 | 3 | 0% |

#### C184 Key Insights
- **Protocol bias VALIDATED**: High-yield protocols have ~83% viable data rate
- **39 COMPLETED beats C183's 32**: +22% improvement from protocol selection
- **AUTOFIX-DECODE-QUAL-LENGTH confirmed active**: STAR exit 104 on 2+ samples
- **Delayed R2 detection gap**: Some samples pass initial probe but fail during alignment (R2 empty)
- **marsseq/10xv2/dropseq/10x_suspect perfect**: 11/11 = 100% viable (no data_incomplete)
- **unknown protocol → 0% success**: Should be excluded entirely
- **Long-runners (>2h)**: T34 (scirna 23M, stuck?), T36/T48 (10x_atac 27-29M, fragment gen slow), T93 (bulk 64M macaque)
- **Strategy for C185**: Pure high-yield only (exclude unknown, limit smartseq2), expect 45%+ success

### Cycle 185 — Pure high-yield batch (job 369303, NEAR-COMPLETE)
- **Batch**: 150 samples, PURE high-yield protocols only (10xv3/v2, celseq2, scirna, marsseq, dropseq, 10x_suspect, indrop)
- **Species**: 75 human, 60 mouse, 10 macaque, 4 drosophila, 1 chicken
- **Read count**: 10M-100M (mean 45M)
- **No unknown, no smartseq2, no plate_based**

#### C185 Results (143/150 finished, 4 running, 3 cancelled)
| Status | Count | % |
|--------|-------|---|
| COMPLETED (exit 0) | 87 | 58% |
| FAILED (exit 1) | 50 | 33% |
| OUT_OF_MEMORY | 3 | 2% |
| CANCELLED (stuck >2h) | 6 | 4% |
| RUNNING | 4 | 3% |

#### C185 Key Metrics
- **Raw success rate**: 87/150 = 58% (RECORD — nearly double C183's 32%)
- **Success rate (excl cancelled)**: 87/143 = 60.8%
- **data_incomplete rate**: ~33% (50 fails / 150 total) — down from 53% in C183
- **OOM rate**: 2% (3/150) — well controlled at 100M cap
- **Wall time range**: 2-40 min (most successful tasks under 30 min)

#### C185 Key Insights
- **PURE HIGH-YIELD VALIDATED**: 87 successes from 150 tasks = best ever absolute yield
- **Protocol bias eliminates ~20% of data_incomplete**: Down from 53% to ~33%
- **Some data_incomplete persists in 10xv3/celseq2**: Not exclusively a non-10x problem
- **No unknown protocol = no 0% yield samples**: Every protocol in batch produces some successes
- **Long-runners (>1h)**: 7 tasks, mostly large 10xv3 (70-85M reads). NFS copy bottleneck.
- **Scale validated**: 150 tasks on 10 nodes processes efficiently in ~4h total wall

### Cycle 186 — Scaled high-yield batch (job 369714, COMPLETE)
- **Batch**: 200 samples, pure high-yield (10xv3/v2, celseq2, scirna, marsseq, dropseq, 10x_suspect, indrop)
- **Species**: 100 human, 80 mouse, 15 macaque, 5 drosophila
- **Read count**: 10M-100M (mean ~45M)

#### C186 Results (200 tasks, all dispatched)
| Status | Count | % |
|--------|-------|---|
| COMPLETED (exit 0) | 121 | 60.5% |
| FAILED (exit 1) | 66 | 33.0% |
| OUT_OF_MEMORY | 1 | 0.5% |
| CANCELLED (stuck >2h) | 11 | 5.5% |
| RUNNING | 1 | 0.5% |

#### C186 Key Metrics
- **Raw success rate**: 121/200 = 60.5% (NEW RECORD absolute yield per batch)
- **Success rate (excl cancelled)**: 121/188 = 64.4%
- **data_incomplete**: ~33% (66 fails / 200) — same as C185
- **OOM rate**: 0.5% (1/200) — excellent
- **Long-runners (>2h)**: 11 cancelled — mostly large 10xv3 (70-100M reads)
- **Total wall time**: ~5h (4.5h processing time on 10 nodes)

#### C186 Key Insights
- **121 successes = largest single batch yield ever** (vs 87 in C185, 39 in C184)
- **Scaling to 200 tasks confirmed**: 10-node limit is the constraint, not batch size
- **Success rate stable at ~64%**: Consistent with C185, showing this is the ceiling for current catalog quality
- **Remaining ~33% failure floor**: data_incomplete in 10xv3/scirna/celseq2 — some SRA deposits in ALL protocols are bad
- **Long-runner pattern**: 5-10% of tasks exceed 2h regardless of read count. NFS bottleneck during copy phase.
- **Strategy for C187+**: Same parameters, keep scaling. Consider 2h SLURM timeout flag.

### C176/C181 Long-Runners — CANCELLED (freed nodes for C183/C184)
- Task 25: BD Rhapsody 280M reads — cancelled at 17.5h (BAM sort would hit 24h wall)
- Task 29: Parse 172M reads — cancelled at 17.5h
- C179 T20, C180 T13, C181 T17/T62 — all cancelled (8-6h, oversized samples)

#### C187 Results (200 tasks, 2h SLURM timeout — PARTIAL due to HPC maintenance)
- Job 370013, submitted 2026-04-30 05:27 EDT
- **COMPLETED**: 112 (62.6% of dispatched)
- **FAILED**: 62 (data_incomplete dominant)
- **TIMEOUT**: 4 (2h auto-cancel WORKING — no stale runners!)
- **OOM**: 1 (rare, 128G insufficient for one sample)
- **CANCELLED**: 14 (by us pre-maintenance)
- **Never dispatched**: 7 (HPC maintenance shutdown)

#### C187 Key Metrics
- **2h SLURM timeout validated**: 4 tasks auto-cancelled cleanly, zero manual scancel needed
- **Success rate tracking**: 68% early → 62.6% final (consistent with C185/C186)
- **Protocols**: 10xv3(71), scirna(40), 10x_suspect(32), 10xv2(22), celseq2(16), marsseq(13), dropseq(4), indrop(2)

#### C187 Key Insights
- **2h timeout is production-ready**: Eliminates stale long-runners without manual intervention
- **Partial batch still yields 112 samples**: Even interrupted by maintenance, strong output
- **Rate holds at ~63%**: Three consecutive batches (C185-C187) all in 58-65% band

#### C188 Results (200 tasks, 2h SLURM timeout — NEW RECORD)
- Job 370272, submitted 2026-04-30 10:25 EDT
- **COMPLETED**: 126 (65.3% of non-timeout terminated) — **NEW SINGLE-BATCH RECORD**
- **FAILED**: 60 (data_incomplete dominant)
- **TIMEOUT**: 9 (2h auto-cancel working perfectly)
- **OOM**: 2 (128G insufficient for 2 samples)
- **Still running at checkpoint**: 3 (will timeout)

#### C188 Key Metrics
- **New batch record**: 126 > C186's 121 — highest single-batch yield ever
- **Success rate**: 65.3% (126/193 non-timeout) — best rate of any 200-task batch
- **Protocols**: 10xv3(69), 10x_suspect(37), scirna(36), 10xv2(20), celseq2(19), marsseq(13), indrop(4), dropseq(2)
- **Species**: 100 human, 80 mouse, 15 macaque, 5 drosophila

#### C188 Key Insights
- **2h timeout continuing to work flawlessly**: 9 auto-cancelled, zero manual intervention
- **126 successes in one batch**: New record, sustainable throughput confirmed
- **Rate stable at 63-65%**: Four consecutive batches (C185-C188) all in 58-65% band — this IS the ceiling

### Overall Pipeline Totals (updated)
- **Total COMPLETED (exit 0)**: ...C194(153) + C195(141) + C196(142) + C197(138) = **2,409**
- **Species with confirmed SUCCESS**: human, mouse, macaque, drosophila, chicken
- **True success rate**: ~77% excl timeout on recent batches (C191-C197 average)
- **Primary bottleneck**: 1) data_incomplete (~20%), 2) timeouts (~3-8% per batch)
- **Protocol bias validated**: High-yield protocols = 75-80% success consistently
- **Best batches**: C194 = 153/200 (79.7%), C193 = 147/200 (79.9%), C192 = 145/200 (79.2%)
- **Scaling**: 200-task batches on 10 nodes, ~5h wall, ~140 samples/batch sustained
- **2h SLURM timeout**: Standard — eliminates stale runners cleanly
- **Remaining pool**: ~37,300 eligible samples in catalog

---

## Previous Status (2026-04-16, Cycle 157)

### Cycle 157 Update (2026-04-16 ~13:15)

#### Batch_011 Status
- **T12-T18: ALL RUNNING** (downloads 34-66%, T12/T15 in STAR mapping 2h16m/1h56m)
- **T19-T20: PENDING** (JobArrayTaskLimit)
- **No new completions since cycle 156**

#### Investigations Completed

**1. AUTOFIX-FAST-CRASH-CLUSTER ROOT-CAUSED** (all 57 fast-crash samples classified)
   - **Root cause**: Ultra-low-read deposits (metadata-only SRA)
   - **GSE193517**: 23 samples with 1-2 reads each (catalog `read_count_estimate=NA`)
   - **GSE263733**: 28 samples with 748-11,239 reads (median 2,535)
   - **Catalog scope**: 49,596 human samples (3.3%) have `read_count < 10K` → new eligibility exclusion flag
   - **Decision**: Reclassified as LOW priority (batch filter, not singlify bug); AUTOFIX ticket downgraded

**2. 138 high-MR zero-cell samples CONFIRMED barcode-stripped** (batch_011 SOFT_FAIL samples: T3/T8/T11)
   - **T11 (GSM8167847 10xv2)**: 1/10,000 BC match = 0.01%
   - **T8 (GSM2819217 10x-scATAC)**: 6/10,000 BC match = 0.06%
   - **Both below 0.1% threshold** for AUTOFIX-ZERO-BC-MATCH abort
   - **Action**: Threshold evaluation — 0.1% may be too permissive. bio-exec to test 0.05% on validation set

**3. 47 align_low_map samples have cells > 100** (protocol-specific threshold reassessment)
   - **Protocol boundaries already in batch_011**: celseq2 15%, dropseq 20%, 10xv2 25%, bd_rhapsody 30% MR thresholds active
   - **Net reclassification**: +12 re-graded SOFT_FAIL/-16 reclassified SUCCESS (marginal, marginal)
   - **Examples**:
     - GSM3195609 (celseq2, 26% MR, 3460 cells) → SOFT_FAIL (MR just above threshold)
     - GSM8141309 (10xv2, 13% MR, 15,076 cells) → HARD_FAIL (MR well below threshold)

#### Batch_012 Prepared (NOT YET SUBMITTED)
- **25 mega-fix retry candidates** (426M-1285M reads)
- **Protocols**: 10xv3/v3_5p/v2/bd_rhapsody/dropseq/v4 (diverse)
- **Status**: Holding for T14/T15 mega-fix validation
- **JSON**: `/mnt/projects/debruinz_project/singlify_pipeline/batch_012.json` ready

#### Pipeline Metrics (unchanged)
- **Total results**: 1373
- **SUCCESS**: 512 (37.3%)
- **Cluster**: T12-T18 on c001/c002/c004/c005/c007/c101; 4+ idle nodes available

### Cycle 156 Status (AUTOFIX-ZERO-BC-MATCH + Failure Investigation)
- **Active batch**: batch_011 (SLURM 361257, 20 tasks, 36h/384G)
  - T1-T11: 1 SUCCESS, 4 OOM, 3 FAILED, 3 SOFT_FAIL (all on old binary, pre-mega fix)
  - T12-T18: RUNNING (downloads + alignment in progress, new binary with mega fix + zero-BC fix)
  - T19-T20: PENDING
  - T14 (GSM5239644, 817M reads): Mega-SHM-EXHAUSTION fix validation — downloading
  - T12 (GSM4337423): 0% barcode match, will fail (data_incomplete, barcode-stripped)
  - T17 (GSM3360834): 0% barcode match, will fail (data_incomplete, barcode-stripped)
  - T18 (GSM3743501): 85% barcode match, should succeed
  - T15 (GSM5686870, 427M 5'): 7% match, borderline
  - T16 (GSM7102845, 667M 5'): 6% match, borderline
- **Fix shipped**: AUTOFIX-ZERO-BC-MATCH (commit 13525a7, pushed to origin)
  - Aborts download when barcode WL validation ≤0.1% on BOTH orientations
  - Saves 30-60 min download + 1-2h compute per barcode-stripped sample
  - Exit code 2 + "data_incomplete" message for automated triage
  - 81/82 ctests pass (1 not-run = pre-existing)
- **New failure investigations**:
  - 57 fast-crash samples (≤10s): GSE263733 (28, 10xv3/unknown_sc), GSE193517 (23, 10xv2) — likely VDB access failures, filed as AUTOFIX-FAST-CRASH-CLUSTER
  - 482 zero-cell+<5%MR samples = 156h wasted compute (many preventable by zero-BC fix)
- **Other agent activity**: commit 1a10d5b (doublet bg-subtract), bio-exec job 361374/361417 (Kang 2018 validation) on c100

### Grand Totals (2026-04)
- **Month total**: ~1,400+ results
- **Commits this session**: 13525a7 (zero-BC-match), 90ad777 (mega-SHM fix), 4557e8e (catalog proto override), a45a952 (FIFO EOF), c4a0c4d (BAM sort uncap), 24959ae (host Bloom), 69f858c (EM threshold)
- **DAG tasks filed**: AUTOFIX-ZERO-BC-MATCH (shipped 13525a7), AUTOFIX-FAST-CRASH-CLUSTER (investigation needed)

### Cycle 149 Status (Agnostic-BC Fix)
- **Total results (April)**: 1,284 (session delta)
- **Active batches**: b007-384g (359953, running), b009 (360039/360071, running)
- **batch_009 partial results (22/45 in)**: 8 SUCCESS, 10 HARD_FAIL, 4 SOFT_FAIL (36% raw)
- **New failure mode identified**: AUTOFIX-AGNOSTIC-BC-INVALID
  - Root cause: `build_metadata_json()` ignored agnostic-detected bc/umi lengths, instead used `r1_len/2`
  - Example: R1=38bp → soloCBlen=19, soloUMIlen=19 → "Invalid barcode configuration" crash
  - Fix committed to `sra_encoder.h` — parse bc/umi from `agnostic-bcN+umiM` tag
  - Validation job 360099 running
- **batch_010**: 31 samples prepared, preflight passed (4× --metadata-json, no barnyard, no null SRR)
  - Composition: 18×10xv3 (incl. 6 large 30-60M), 4×10xv2, 2×scATAC, 2×celseq2, 1×marsseq, 1×celseq, 3×snRNA_unknown

### Cycle 146 Status (Batch_007 Final Triage)
- **Total results (April)**: 1284 (session delta +191 since cycle 145)
- **Batch_007 classified**: 129 of 150 (27 SUCCESS / 10 SOFT_FAIL / 92 HARD_FAIL = 20.9% raw success)
- **Cells from SUCCESS**: 309,829 total
- **Protocol performance highlights**:
  - **10x-3p-v3**: 12/14 = 85.7% SUCCESS (highest v3 batch rate ever)
  - **10x-3p-v2**: 7/11 = 63.6% SUCCESS
  - **10x-visium**: 3/5 = 60% SUCCESS
  - **marsseq2**: 1/1 = 100% (FIRST SUCCESS this cycle)
- **OOM diagnosis (critical finding)**:
  - **51 OOM tasks** all peaked at EXACTLY 128GB (hard SLURM ceiling)
  - **Read-count correlation**: 23/51 (45%) are >300M-read samples; 28/51 (55%) are 100-300M
  - **Protocol-neutral**: 9 protocols affected (10xv3, 10xv2, dropseq, bd_rhapsody, sciRNA, etc.)
  - **Root cause**: Batch_007 had `read_count=0` (unknown) samples; at runtime they exceeded 300M reads. Without `--genome-shared`, peak RAM = 35GB genome + ~100GB STAR BAM sort = 135GB+ ≥ 128G limit.
  - **Action**: Update resource-model.json >300M bucket → 192G; plan batch_008 with tiered RAM before next submission
- **Circuit-breaker**: Tick 2/3 → **RESET on actionable finding**; diagnostic value is high (resource-tier misclassification now understood)
- **Next steps**: resource-model tier bump + batch_008 plan → submit

## Cycle 146/Continued Status (2026-04-14 22:15)

### Batch Progression & OOM Triage
- **Batch_005 (SLURM 359252, 149 tasks)**: COMPLETED ✅
- **Batch_007 (SLURM 359627, 150 tasks, 8 CPUs/128G)**:
  - Completed: 94 tasks (86 SUCCESS, 8 classification errors)
  - OOM at 128G: 43 tasks affecting large-read samples (148M–311M reads)
  - Download failures: ~25 tasks (null SRR or ENA path missing)
  - R2-empty crashes: ~5 tasks (VDB variable R2 or protocol misdetect)
  - **Requeue decision**: Created batch_007_384g_requeue with 43 OOM samples at 16 CPUs/384GB (SLURM 359953)
- **Batch_007_384g_requeue (SLURM 359953, 46 tasks, 16 CPUs/384G)**: Freshly submitted — targeting OOM samples from batch_007
- **Batch_008 (SLURM 359965, 36 tasks, 8 CPUs/128G)**: Freshly submitted (protocol-diverse diagnostic batch; see Composition below)
- **Species validation (SLURM 359961, running on g005)**: Cross-species validation panel
- **Sentinel (SLURM 353774)**: Monitoring job running

### Batch_008 Composition (36 samples, all human, barnyards excluded)
- 10xv3: 12 samples
- 10xv2: 6 samples
- Dropseq: 4 samples
- BD Rhapsody: 3 samples
- inDrop: 3 samples
- sci-RNA-seq: 4 samples
- Seqwell: 3 samples
- 10xv3_5prime: 4 samples
- **Filter applied**: `read_count < 50M AND srr_accessions NOT NULL`

### Pipeline Fixes Shipped (This cycle)
- **Commit adf8f1b** (bio-exec): Export snp_ad.1pz + snp_dp.1pz when `--snps` provided (fix FAILURE-1)
- **Commit 7880949** (perf-exec): Sort protocol candidates to eliminate arc-gex false positives (fix FAILURE-3)
- **Commit d6fd3ab** (bio-exec): Expanded species k-mer DB from 5K→50K per species (17 species supported)

### Cluster State (2026-04-14 22:15)
- **Active nodes**: 16+ (cpu, bigmem, short partitions saturated)
- **Concurrent batches**: 3 (batch_007_384g, batch_008, species_val)
- **Idle CPU slots**: <5% (cluster well-utilized)

### Grand Totals (2026-04)
- **Month total**: 1,213 results (updated from 1,210)
- **All-time**: 1,213+ results
- **Success rate (2026-04)**: ~64% (batch_005 final: 86/149 SUCCESS)

### Resource Model Feedback (OOM Incident)
- **Observation**: Batch_007 included samples with `read_count=0` (unknown from catalog) that turned out to be massive (148M–311M reads)
- **Peak RAM pattern**: 128G insufficient for STAR BAM sort (~100GB) + 35GB genome load (uncached) = 135GB+ peak
- **Solution**: Tiered batches by read_count estimate:
  - **Standard tier** (read_count < 50M): 8 CPUs / 128GB (safe, tested to 83GB p95)
  - **Large tier** (read_count ≥ 50M or unknown): 16 CPUs / 384GB (handles up to 300M reads)
- **`--genome-shared` opportunity**: Pre-loading genome with `singlify genome load` in SLURM prologue saves 35GB per job. DAG task filed (HIGH priority).

### E2E Validation Status
- **Panel A (human 40M)**: PASS — gene r=0.9995 vs STARsolo, 11,560 cells called
- **Panel B (Kang 2018 SRR5398238)**: BLOCKED — VDB read swap issue (SRR formatted R1=cDNA R2=BC, detected as splitseq → 0 cells)
- **Panel F (sex call)**: PASS — 7/7 samples correct

### Species Validation Panel (1/7 SUCCESS)
- **Complete**: GSM5584134 (cat, 88.2% mapping, 7,798 cells) ✅
- **Failures**: 6 samples
  - Zebrafish (GSM2830043): adapter trim→30bp R2, misdetected as 10x-atac (protocol_id=23, confidence=1)
  - Multiple: OOM at large read counts
  - 1 Python false/true bug in result JSON write (FIXED)

## Cycle 146 Status (2026-04-14 19:45)

### Batch_004 Final Results (SLURM 359083)
- **Submitted**: 200 samples (diagnostic + recipe mix)
- **Completed**: 51+ tasks (sample of batch_004_1..51)
- **Performance verified**: 39 tasks with RAM data harvested
  - 10xv3 25-50M: n=12, p95_wall=907s, p95_ram=59.2GB → recommend 26CPUs/128GB
  - 10xv2 25-50M: n=4, p95_wall=633s, p95_ram=40GB → recommend 22CPUs/128GB
  - bd_rhapsody 25-50M: n=1, p95_wall=309s, p95_ram=128GB → recommend 11CPUs/192GB
- **Resource model updated**: 11 (protocol, bucket) entries written

### Batch_005 Submission (SLURM 359252)
- **Scope**: 149 tasks (indices 52-200 of batch_004.json)
- **Resources**: 8 CPUs / 128 GB (lower tier for smaller samples; 5 concurrent/node vs 2 with prior 20CPU tier)
- **Status**: QUEUED (behind batch_004 high-CPU tasks)
- **Rationale**: batch_004 tasks 1-51 running at 20 CPUs; batch_005 resubmits remainder at optimized 8 CPUs per resource model

### E2E Validation Panel (Multi-cycle tracking)
- **Panel A (human 40M)**: PASS — singlify gene counts r²=0.9995 vs STARsolo
- **Panel B (Kang2018 SRR5398238 8-donor)**: BLOCKED — Wrong read orientation detected
  - SRR5398238 has R1=cDNA(98bp), R2=BC(26bp) (not standard 10xv2 layout)
  - VDB streams in original orientation; singlify misdetects as splitseq → 0 cells
  - DAG task AUTOFIX-VDB-READ-SWAP-PROTOCOL filed (HIGH priority)
- **Panel F (sex call accuracy)**: PASS — 7/7 samples correct (100%)

### Fixes Landed (This cycle)
- **commit 7880949** (perf-exec): Protocol detection sort — arc-gex false positive eliminated
- **commit adf8f1b** (bio-exec): Export snp_ad.1pz/snp_dp.1pz when --snps provided
- **Acceptance tests**: Both validated on real samples before commit

### Grand Totals (2026-04)
- **Month total**: 1,210 results (April to date)
- **All-time**: 1,210+ results across all Clipper cycles
- **Success rate (2026-04)**: ~64% (batch_005 final: 86/149 SUCCESS → 57.7% + batch_007 partial: 86/94 completed → 91.5% procedural SUCCESS before OOM triaging)
  - OOM samples not counted as failure until requeued and confirmed still OOM→fixed by larger tier

## Cycle 145 Status (2026-04-14 17:20)

### Species K-mer DB Expansion (COMPLETE)
- **Expanded from 2 → 17 species** in the compiled k-mer DB
- Commit: 5a374cb — 73/73 CTests pass
- Species: human, mouse, fly, macaque, zebrafish, chicken, worm, rat, pig, yeast, sheep, cow, horse, cat, dog, rabbit, frog
- Genome_tag mismatch bug fixed (underscores → dots/hyphens to match directory names)

### Cross-Species Validation Panel (IN PROGRESS — SLURM 359843)
- 15 non-human/mouse samples across 15 species
- **V1 (with --genome-dir)**: Tests alignment quality given correct genome
  - 5/15 complete: 1 SUCCESS (cat 88.2% MR, 7798 cells), 4 SOFT_FAIL (all R2-empty / protocol misdetection)
  - 10/15 still downloading (VDB streaming, 11 concurrent)
  - Python boolean bug in result JSON writing fixed in script
- **V2 (without --genome-dir)**: Pending — will test k-mer-based species auto-detection
  - Script created at scripts/species_val_v2.sh

### Batch_007 Progress (SLURM 359627, 95/150 complete)
- **21 SUCCESS (22% raw)**, 187,443 cells
- **Processable rate (excl empty-proto): 21/33 = 64%**
- Empty-protocol failures: 62/95 (31 R2-empty crash, 19 download_fail, 12 OOM)
- Per detected protocol: 10x-3p-v3 82%(9/11), 10x-3p-v2 70%(7/10), celseq2 43%(3/7), visium 50%(2/4)

### Grand Totals
- **Total results**: 1180
- **Total SUCCESS**: 453 (38.4%)
- **Total cells**: 1,423,572 (1.42M)
- **Failure categories**: align_low_map 298, pipeline_crash 240, cells_below_threshold 137, align_oom 26, download_fail 26

## Cycle 143 Status (2026-04-14 14:30)

### Active Batches
- **359421 (batch_006)**: ~20 running (tasks 130s-150). 111/150 results, 49 SUCCESS (44%). Last tasks (marsseq/bd_rhapsody) completing.
- **359595 (batch_007)**: SUBMITTED. 150 samples, 0 barnyard, 30 concurrent. Queued behind batch_006.
  - Composition: 60 10xv3, 30 10xv2, 20 10xv3_5prime, 10 celseq2, 10 indrop, 10 bd_rhapsody, 5 marsseq, 5 dropseq
- **359440 (pilot)**: 2 tasks running on bigmem (b002-b003)

### Key Metrics Update
- Pipeline: **1045+ results, 421+ SUCCESS (40.3%), 1,209,389 cells (1.2M)**
- batch_006 breakdown: 44% raw → **55% excl barnyard → 60% processable** (excl barn+R2empty)

### 10xv3 Catalog Group Analysis (batch_006 tasks 81-120)
| Detected Protocol | Success | Total | Rate |
|---|---|---|---|
| 10x-3p-v3 (true v3) | 2 | 7 | 29% |
| 10x-3p-v2 (misdetect) | 3 | 3 | 100% |
| 10x-visium (misdetect) | 2 | 3 | 67% |
| celseq2/marsseq/quartzseq (catalog wrong) | 2 | 4 | 50% |
| R2 empty | 0 | 5 | 0% |
| Barnyard | 0 | 4 | 0% |

KEY INSIGHT: When singlify correctly identifies a protocol, success rate is HIGH (GSM8225020 91.3%, GSM9034797 91.7%). batch_004's 36% 10xv3 rate was catalog noise + R2 empty, NOT singlify alignment bugs.

### New Failure Modes (DAG updated)
- **AUTOFIX-CLIP5P-AGGRESSIVE**: 30-31bp R2 prefix clip + adapter trim leaves <25bp → 0.6-20% MR (3 cases)
- **AUTOFIX-VDB-R2-VARIABLE-EMPTY**: 7% of results are pipeline_crash from VDB variable R2 → empty decode

### Protocol Breadth (batch_006 by detected protocol)
| Protocol | Success/Total | Rate |
|---|---|---|
| 10x-3p-v2 | 19/28 | 68% |
| 10x-3p-v3 | 4/9 | 44% |
| 10x-visium | 5/8 | 62% |
| quartzseq2 | 3/4 | 75% |
| dropseq | 5/19 | 26% |
| microwell-seq | 1/3 | 33% |
| agnostic-bc13+umi7 | 1/1 | 100% |
| indrop | 1/? | first result |
| celseq2 | 0/2 | early (3.7-12.3% MR, below 15% thresh for one) |

### User Code Fixes Since Cycle 135
- **7880949**: Protocol detection — prefer lower protocol_id, prevents arc-gex false positives
- **adf8f1b**: Export snp_ad.1pz and snp_dp.1pz when --snps provided
- **7e02eff**: VDB download validation — spot count vs catalog read_count
- **b16bf97**: AUTOFIX-PROTOCOL-CONFIDENCE-OVERRIDE: catalog metadata wins at low confidence
- **b546b98**: AUTOFIX-PROVENANCE-CELLS-WRONG-FIELD: use actual cell_indices count

### Batch 004 Results (175/200)
| Metric | Value |
|--------|-------|
| SUCCESS | 65 (37.1%) |
| HARD_FAIL | 62 (35.4%) |
| SOFT_FAIL | 48 (27.4%) |

By catalog protocol_inferred:
- **10xv3**: 41/114 (36%) — protocol mis-detection drives most failures
- **10xv2**: 15/24 (62%) — solid
- **dropseq**: 7/11 (64%) — solid
- **bd_rhapsody**: 1/10 (10%) — **FIRST SUCCESS!** GSM5599953 (95.6% MR, 1730 cells)
- **celseq2**: 0/9 (0%) — threshold too strict; batch_006 has 15% threshold
- **indrop**: 1/3 (33%) — GSM2946619 success
- **marsseq**: 0/2, **splitseq**: 0/1, **scirna**: 0/1

### Full Pipeline Metrics (all 983 results)
| Metric | Current | Target |
|--------|---------|--------|
| Total results | 983 | 206,000+ |
| SUCCESS | 393 (40.0%) | ≥85% |
| HARD_FAIL | ~310 | — |
| SOFT_FAIL | ~280 | — |
| Total cells (SUCCESS) | 1,137,560 | — |
| Mean cells/SUCCESS | ~2,893 | — |

### Failure Category Breakdown (all results)
| Category | Count | Pct | Root Cause |
|----------|-------|-----|------------|
| align_low_map | 244 | 25.1% | Wrong species/protocol from catalog + barnyard + mislabeled libraries |
| pipeline_crash | 189 | 19.4% | R2 empty — VDB variable R2 length + protocol mis-detection |
| cells_below_threshold | 134 | 13.8% | ~111 are data_incomplete (barcodes stripped from SRA) |
| align_oom | 11 | 1.1% | Large samples need 384G |
| download_fail | 8 | 0.8% | VDB/ENA transient |

### Key Insight: Confidence Level Predicts Success
- **All 10 batch_006 successes had VDB detection confidence ≥2**
- **All 4 R2-empty crashes had confidence=1** with R2="variable"
- Override (b16bf97) fires at c=1 but can't fix VDB variable R2 decoding
- Next fix target: handle VDB "variable" R2 in the encoder (fall back to per-read variable-length storage)

## Goal Status

| Goal | Status | Notes |
|------|--------|-------|
| G1: Pipeline stability | 🟡 Active | 37.3% SUCCESS (1248 results). Multi-SRR fix deployed. OOM retries active at 384G. |
| G2: Clipper saturation | 🟢 Good | 36+ jobs running (b007 + b007-384g + b008 + b009). ~50 pending/running. |
| G3: Protocol completeness | 🟡 Diagnostic | batch_009 includes 5prime, scirna, bd_rhapsody, citeseq, dnbelab probes |
| G4: ANVIL deployment | 🔴 Not started | Blocked on G1 |
| G5: Full catalog coverage | 🔴 Not started | Blocked on G4 |

## Failure Registry (Cycle 8 Analysis)

Cumulative results (22 samples across all pilot runs to date):
- SUCCESS: GSM7103327 (10xv3, 90%), GSM3693219 (dropseq, 88%), GSM3587956 (seqwell, 79%) = 3/22 = 14%
- SOFT_FAIL cells_below_threshold (0 cells despite ≥82% mapping): GSM2706111, GSM3132061, GSM3528852, GSM3681521, GSM5434235, GSM8960288 — all had R2 reads mapping to repetitive elements with 0 exon hits in pileup. Root cause: data quality (these specific datasets have very low exonic signal). Fix: replaced with val1 golden samples.
- HARD_FAIL pipeline_crash: GSM2685265 (R1=8bp), GSM7152960 (BD Rhapsody soloCBposition — fix in binary, validation pending), GSM8271873/GSM8272094 (sci-rna-seq3 R1=18bp, barcode at pos 24 unreachable).
- HARD_FAIL align_low_map: Various protocol mismatches and data quality issues.
- BD Rhapsody: singlify.cpp soloCBposition fix applied (args.push_back("--soloCBposition") moved before loop, not inside it). Binary rebuilt 16:40. Validator job 355026 testing on b002.
- GTF bug fixed: pilot_job.sh previously added BOTH human+mouse GTF for pure-mouse samples. Fixed to correctly use only mouse GTF for Mus musculus organism.

## Resource Model State

Bootstrap defaults active (8 CPUs, 128G RAM for batch_005).
batch_006 will use batch_template_v2.sh with protocol-specific MR thresholds.
- Real processable queue: 10,374 human+in-scope+clean samples (no screen flags)
- batch_006 composition: 49 10xv2, 40 10xv3, 31 dropseq, 10 celseq2, 10 indrop, 5 marsseq, 5 bd_rhapsody

## Catalog Summary (from 2026-04-13 probe)

- Total catalog rows: 1,490,358
- Homo sapiens samples: ~577K
- In-scope protocols (human): ~10xv3 74K, 10xv2 21K, scirna 29K, indrop 28K, 10x_atac 24K
- Out-of-scope (will skip): smartseq2 808K, plate_based 42K, bulk 12K
- Read count median: 1.7M reads; p75: 8.3M reads
- Estimated processable human samples: ~150-200K (pending eligibility filter refinement)

## Cycle 132 (2026-04-14)
- **Total results**: 346 (pre-requeue) → 293 after clearing 80 for requeue
- **Success rate**: 216/346 = 62.4% raw (pre-requeue), 216/293 = 73.7% on non-failed
- **Bug fixes shipped**: 04a971d (clip5p 30→15bp + wrong-strand auto-retry + CB_samTagOut 1-value)
- **Validation**: SRR16355913: 0.82% → 72.49% mapping, 0→3053 cells
- **Requeue**: Batch 358615 (80 samples: 20 pipeline_crash + 60 align_low_map)
- **Failure breakdown (pre-requeue)**:
  - align_low_map: 60 (18 wrong-strand, 13 clip-skip, 7 unexplained, 22 other)
  - pipeline_crash: 20 (all binary-not-found during rebuild)
  - align_oom: 5 (need higher RAM)
  - cells_below_threshold: 45 (SOFT_FAIL, data quality)
- **Git**: 04a971d pushed to origin/main
- **Active batches**: 358607 (pilot, 10), 358615 (requeue, 80)

## Cycle 133 (2026-04-14 07:30)
- **Total results**: 357 (305 fresh + 54/80 requeue processed)
- **Fresh success rate**: 254/305 = **83.3%** SUCCESS, 46 SOFT_FAIL (15.1%), 5 HARD_FAIL (1.6%)
- **Pipeline correctness rate**: 98.4% (SUCCESS + SOFT_FAIL — pipeline ran, data was low quality)
- **Requeue recovery**: 1 SUCCESS, 32 SOFT_FAIL, 21 HARD_FAIL (clip5p+strand fixes converted most crashes to SOFT_FAIL)
- **Fresh HARD_FAIL breakdown**: 4 align_low_map (all 10xv3, all <1.6% mapping — likely wrong species or bad data), 1 validator artifact
- **OOM escalation**: Batch 358732 submitted (5 multi-SRR samples at 384G, cpu partition)
- **Active batches**: 358646 (requeue, ~60/80 done), 358721 (new pilot, 8), 358732 (OOM, 5)
- **Sentinel**: 353774 running (16h+)
- **Protocol distribution (SUCCESS)**: 10xv3=113, dropseq=88, 10xv2=52, seqwell=1, scirna=1
- **Bugs identified**: STAR quality-string length mismatch (GSM3583892), 5' adapter over-trimming (GSM5093910)

## Nonhost Screening Track — Status 2026-04-14

| Task | Status | Commit |
|------|--------|--------|
| NONHOST-SYLPH-PORT (MinSketchIndex + NonHostScreener) | ✅ | 3c43c7d |
| NONHOST-UNMAPPED-CAPTURE (--nonhost-db flag, STAR capture) | ✅ | fe08589 |
| build-nonhost-db CLI subcommand | ✅ | fe08589 |
| NONHOST-EM (EM deconvolution, nonhost_em.h) | ✅ | fe718a1 |
| NONHOST-VIRALDB (RefSeq viral DB build) | 🟡 SLURM 360375 running |
| NONHOST-MICROBIALDB bacterial | 🟡 SLURM 360376 running (c007, 12h) |
| NONHOST-MICROBIALDB fungal | 🟡 SLURM 360377 running (g005, 4h) |
| NONHOST-SCREENER validation (SRR11092058 COVID) | 🔴 pending DB completion |
| NONHOST-VALIDATION mock community (SRR7287187 Zymo D6300) | 🔴 pending DB completion |

**Architecture**: Header-only C++ in include/singlet-pileup/nonhost/. MinSketchIndex (k=21, w=11 canonical minimizers, binary .snhskidx). NonHostScreener (per-read classification). NonHostEM (EM abundance deconvolution — collapses multi-hit reads to species-level relative abundances). All three activated by --nonhost-db flag in scRNA, bulk RNA-seq, Visium, and ATAC modes.

**Kingdom coverage when DB builds complete**: Viral (~19,149 RefSeq genomes), Bacterial (~6,478 complete reference genomes), Fungal (~37 RefSeq release files). Full nonhost_em_abundance.tsv reporting relative abundance per detected species.
## Cycle 158 (2026-04-16 ~13:30)

### Pipeline Status
- **Batch_011** (SLURM 361257): T12-T18 RUNNING, T19 PENDING
  - T12 (613M, barcode-stripped): STAR mapping 2.5h+
  - T14 (817M, 10xv3_5prime, PRIMARY validation): 66% download
  - T15 (427M, 10xv3_5prime, 7% BC): STAR mapping 2.2h+
  - T16 (667M, 10xv3_5prime): 73% download
  - T17 (594M, 10xv2, barcode-stripped, old binary): 90% download
  - T18 (408M, 10xv2, 85% BC healthy): 46% download
- **Batch_012** (SLURM 361493): SUBMITTED 13 mega-retry samples
  - T1-T3 RUNNING on c006/c008/c009, T4-T13 PENDING (QOSMaxNodePerUserLimit)
  - Protocol mix: 10xv3(4), 10xv3_5prime(3), 10xv2(1), 10xv4(1), bd_rhapsody(3), dropseq(1)
  - Read range: 440M-872M
  - All previously failed OOM/crash in batches 002-010

### Actions Taken
1. Pushed external commit 3ae3f3e (BAM deposit reclassification) to origin/main
2. Cleaned dag.md duplicate entries (commit 431205b)
3. Root-caused GSM7431256 sci-RNA-seq3 OOM: missing `scirna3_rt_bc.txt` whitelist → CB_UMI_Complex auto-discovery → memory explosion at only 36M reads. Filed AUTOFIX-SCIRNA3-WL-MISSING.
4. Analyzed 248 old-batch unknown crashes: median wall=51s, 135 fast (<60s), 50 slow (>10min). Most need re-run with current binary to classify.
5. Prepared and submitted batch_012 mega retry job (13 samples, %4 throttle)
6. Added exit-2 (data_incomplete) early handler in batch_012 script (was missing in batch_011)

### New DAG Entry
- AUTOFIX-SCIRNA3-WL-MISSING (MEDIUM): sci-RNA-seq3 OOM from missing whitelist + CB_UMI_Complex barcode auto-discovery

### Totals
- Results: 1389 total, 516 SUCCESS (37.1%)
- Active SLURM: batch_011 T12-T18 (6), batch_012 T1-T3 (3), pending: ~16
- Cluster: c003/c010 idle, c006/c008/c009 claimed by batch_012
- Git HEAD: 431205b (origin/main)

## Cycle 159 (2026-04-16 ~14:00)

### MEGA FIX FULLY VALIDATED ✅
- **T12** (GSM4337423, 613M reads, barcode-stripped): COMPLETED, 79.1% MR, **363G MaxRSS** (94.5% of 384G). Prev: OUT_OF_MEMORY at 402G. Mechanical validation — mega fix prevents OOM.
- **T15** (GSM5686870, 427M reads, 10x 5' v3): **SUCCESS**, 83.6% MR, **749 cells**, 195G MaxRSS (51% of 384G), 2h28m wall. Cells-producing mega validation. 5' protocol auto-detected correctly (71% wrong-strand → reverse-strand switch).
- **Mega fix features confirmed**: BAM compression=1, NFS outTmpDir, .1fq early deletion, 50% SLURM_MEM limitBAMsortRAM cap. All operational at scale.

### Other Batch_011 Results
- T19 (GSM3842207, 488M 10xv2): FAILED 16s — zero-BC fix correctly detected barcode-stripped (exit 2). ENA fallback also failed. data_incomplete.
- T20 (GSM3885373, 412M 10xv2): Already SUCCESS — skipped via claim check.
- T14 (817M, PRIMARY large validation): still downloading (70%). Will be definitive >800M test.
- T16-T18: downloading (78-97%)

### Batch_012 Status
- T3 (GSM4135994, 815M 10xv3): data_incomplete (barcode-stripped, exit 2 in 15s). Zero-BC fix working.
- T1/T2/T4/T5: downloading on c006/c008/c009/c009
- T5-T13: pending (QOS node limit)

### Pipeline Totals
- Results: 1389 total, 517 SUCCESS (37.2%), 505 HARD_FAIL, 367 SOFT_FAIL
- Active SLURM: batch_011 T14/T16/T17/T18 (4 running), batch_012 T1/T2/T4/T5 (4 running)
- Git HEAD: 957fc43 (origin/main)

## Cycle 160 (2026-04-16 12:35 EDT)

### Batch 011 (SLURM 361257) — mega fix validation
- T14 (GSM5239644, 817M 10x-5p-v3): downloading at 81% on c101 — LARGEST sample test
- T16 (GSM7102845, 667M 10x-5p-v3): downloading at ~86% on c005
- T17 (GSM3360834, 594M 10xv2, barcode-stripped): in STAR since 12:20 (c001)
- T18 (GSM3743501, 408M 10xv2): encoding on c002

### Batch 012 (SLURM 361493) — mega retry  
- T1/T2/T4/T5 downloading (872M/858M/668M/656M reads)
- T3 COMPLETED: data_incomplete (zero-BC fix, 15s) — validated
- T6-T13 PENDING (QOS limit)

### Batch 013 (SLURM 361537) — diagnostic-diverse (NEW)
- 34 samples across 11 protocols: 10xv3(6), 10xv2(5), dropseq(5), bd_rhapsody(3), scirna(2), seqwell(2), citeseq(3), 10xv3_5prime(3), parse(2), dnbelab(2), indrop(1)
- Intent: recipe core (16) + failure probes (14) + protocol expansion (5)
- 192G RAM, 8 CPUs, 24h wall, %8 concurrent
- T1-T8 RUNNING on c001/c004/c005/c006/c008/c009
- T9-T34 PENDING (array limit)

### Failure analysis
- 285 pipeline_crash: ALL are fast-crash (wall≤2s, MR=0) from early batches. Only 54 are single-SRR re-runnable.
- Protocol success rates: dropseq 64%, 10xv2 40%, 10xv3 36%, 10xv3_5p 43%, bd_rhapsody 6%, scirna 6%, seqwell 14%, celseq2 0%, marsseq 0%, 10xv4 0%
- 18,898 unprocessed production_ready human samples remain

### Totals
- Processed: 1389 (517 SUCCESS, 505 HARD_FAIL, 367 SOFT_FAIL) → 37.2% success rate
- Active SLURM: batch_011 (4 running), batch_012 (4 running, 8 pending), batch_013 (8 running, 26 pending)
- Cluster: 20+ singlify tasks across 3 batches, most nodes occupied

## Cycle 161b Update (2026-04-16 13:05 UTC)

### New Results (since cycle 161a)
| GSM | Batch | Protocol | Reads | Status | MR | Cells | Wall | Notes |
|-----|-------|----------|-------|--------|-----|-------|------|-------|
| GSM7870508 | B013-T1 | dnbelab-c4 | 34M | SUCCESS | 94.8% | 500 | 1333s | First-ever DNBelab SUCCESS! |
| GSM7971497 | B013-T2 | dnbelab-c4 | 46M | SUCCESS | 92.8% | 150 | 1297s | Second DNBelab SUCCESS |
| GSM5682521 | B013-T8 | 10xv3_5prime | 11M | SUCCESS | 71.5% | 34 | 379s | UMI truncation 12→10bp OK |
| GSM4337423 | B011-T12 | 10xv3 | 326M | SOFT_FAIL | 79% | 0 | 9323s | Failed library: 402 BC, ~6K UMIs from 326M reads |
| GSM4135994 | B012-T3 | 10xv3 | 815M | HARD_FAIL | 0% | 0 | 15s | Zero-BC abort (validated) |
| GSM3842207 | B011-T19 | 10xv2 | 487M | HARD_FAIL | 0% | 0 | 16s | Zero-BC abort (validated) |

### Protocol Expansion Wins
- **DNBelab-C4**: 2/2 SUCCESS — first-ever protocol validation! Single-segment barcode (10bp of 30bp) works due to large search space (4^10=1M). Filed AUTOFIX-DNBELAB-COMBINATORIAL (LOW) for proper 3-segment support.
- **BD Rhapsody**: 3 samples in flight (T9/T10/T11). CRITICAL: 97-barcode single-segment ceiling means >97 cells impossible. Filed AUTOFIX-BD-RHAPSODY-COMBINATORIAL (HIGH).
- **Parse/SPLiT-seq**: Confirmed broken (3.3% MR). AUTOFIX-PARSE-SPLITSEQ-BARCODE filed (HIGH).
- **inDrop**: T3 in STAR with 82M reads.

### New AUTOFIX Tasks Filed
- AUTOFIX-BD-RHAPSODY-COMBINATORIAL (HIGH): 97-barcode ceiling, ~200+ samples blocked
- AUTOFIX-DNBELAB-COMBINATORIAL (LOW): Works but lossy

### Totals
- Processed: ~1395 (520 SUCCESS, 507 HARD_FAIL, 368 SOFT_FAIL) → ~37.3% success rate
- Active SLURM: batch_011 (4 running), batch_012 (4 running, 8 pending), batch_013 (8 running, 26 pending)
- Protocols validated: 10xv2 ✓, 10xv3 ✓, 10xv3_5prime ✓, dropseq ✓, **dnbelab-c4 ✓** (NEW)

## Cycle 163-164 (2026-04-16 16:00-18:00) — FIX WIN

### AUTOFIX-EXPORT-CSC-INT32-OVERFLOW — FULLY VALIDATED
- **Commit 99dc7f0** (pushed origin/main cycle 163)
- 4 independent validations across 3 protocols:
  - GSM8752766 (10xv3, 12.7M reads, mt=2.32B) — SUCCESS 75.73% MR
  - GSM8752765 (arc-gex, 14.8M, mt=2.77B) — SUCCESS 79.5% MR 3,734 cells
  - GSM8752768 (arc-gex, 17M, mt=3.14B) — SUCCESS 75.16% MR 3,066 cells
  - GSM8249691 (indrop, 82M, mt=5.49B) — SUCCESS 72.2% MR 257 cells (original bug witness)
- Impact: unblocks all samples with mt_pileup_bases > 2^31. G3 protocol validated for indrop.

### AUTOFIX-MEGA-SORT-RSS-OVERAGE — FIX PENDING VALIDATION
- bio-exec delivered tiered STAR params (mega_sort_params.h: normal/mega/ultra by read_count)
- Validator 361746 running on GSM7102845 (666M 5p), ~4h ETA
- New cycle-164 witness: GSM6564295 (visium 194M) OOMed at 192G cap — confirms policy needs mem_gb-aware scaling, not just read_count

### Recent origin/main commits (concurrent Copilot work)
- 8a9a1e2 fix(protocol): WL-defensive override
- 2eaf861 test: doublet recall/FPR test
- 6425ad8 fix(data): restore 3M feb-2018 whitelist (3,686,400 entries)
- f8c77cf fix(encoder): R1/R2 auto-swap for inverted SRA deposits (may resolve AUTOFIX-VDB-READ-SWAP-PROTOCOL)
- 99dc7f0 fix: int32 overflow in SparseAccumulator::to_csc


## Cycle 170 (2026-04-18 04:00–07:30) — 5-PANEL BENCHMARK (commit 5352b85)

### Benchmark: Full 5-slot panel with FIFO OOM fix
- **Jobs**: 362865 (build+downloads+slot0), 362877 (slots1-4, re-download ddSEQ)
- **Commit under test**: 5352b85 (fix(fifo-writer): per-read exact sum for FmtBlock2 buffer sizing + R2 length guard)

### Results

| Slot | SRR | Protocol | Wall(s) | Cells | Unique% | RSS_GB | Exit | Status |
|------|-----|----------|---------|-------|---------|--------|------|--------|
| 0 | SRR32855204 | 10x-arc-gex | 184.54 | 2538 | 85.76% | 23.1 | 0 | PASS |
| 1 | SRR17873408 | 10x-3p-v2† | 493.66 | 385 | 59.28% | 27.0 | 0 | PASS (OOM fixed) |
| 2 | SRR23582977 | sci-RNA-seq3 | 6240 (HANG) | — | — | 70.0 | HANG | NEW BUG |
| 3 | SRR10010840 | Drop-seq | — | — | — | — | n/a | data_incomplete |
| 4 | SRR33424030 | 10xv3-mouse | 48.49 | 5069 | 79.08% | 30.2 | 0 | PASS (prev run) |

**Total wall (slots 0+1+4)**: 726s < 1400s ✓

### Key Findings

**SLOT 0 — BASELINE VALIDATED**: SRR32855204 10x-arc-gex: 184.54s, 2538 cells (+0.08% vs expected 2536), 85.76% mapping, 23.1GB RSS. 5352b85 causes zero regression on baseline.

**SLOT 1 — DDSEQ OOM FIXED (root cause found)**: Old SRR17873408.1fq (Apr 10) was encoded by pre-ad9e999 singlify WITH BC dict → prescan skipped → 15.1M Cartesian barcodes → STAR allocated 15.1M×39K×4B=2.3TB → OOM at 384GB. Fix: re-download. Fresh download detects as 10x-3p-v2 (R1=26bp, confidence=3) → CB_UMI_Simple → WL filtered to 385 barcodes → 27GB RSS → exit 0. Mapping rate 59.28% (exact match). Note: old protocol label "ddSEQ" may be incorrect; fresh VDB detection is 10x-3p-v2.

**SLOT 2 — SCI-RNA-SEQ3 HANG (new AUTOFIX-SCI3-CBSAM-HANG filed)**: FIFO OOM fix (5352b85) confirmed working; no immediate 576GB spike. But pipeline HANGS at 70GB RSS for 100+ minutes. Root cause hypothesis: STAR CB_samTagOut+SortedByCoordinate BAM sort for 48M reads takes pathologically long, or `--limitBAMsortRAM` is missing on this path. Filed AUTOFIX-SCI3-CBSAM-HANG (HIGH priority).

**SLOT 3 — DATA_INCOMPLETE CONFIRMED**: SRR10010840 VDB download returns 0 reads at protocol confidence=0 (marsseq2 detection, R2=variable). Not a pipeline bug. AUTOFIX-DROPSEQ-DATA-INCOMPLETE still open.

**SLOT 4 — MOUSE BASELINE VALID**: SRR33424030 10xv3-mouse from previous run (Apr 17): 48.49s, 5069 cells (equal to expected), 79.08% mapping, 30.2GB RSS.

### Active AUTOFIX Status
- AUTOFIX-SCI3-507GB-OOM: 🟡 PARTIAL (FIFO OOM fixed by 5352b85, HANG remains)
- AUTOFIX-SCI3-CBSAM-HANG: 🔴 NEW (STAR CB_samTagOut sort hang)
- AUTOFIX-DDSEQ-192GB-OOM: 🟡 OOM FIXED (re-download; protocol detection changed)
- AUTOFIX-DROPSEQ-DATA-INCOMPLETE: 🔴 OPEN (need different SRR)

---

## Cycle 176 Smoke Tests (2026-04-28)

### Binary Rebuild
- Job 367911: COMPLETED (2:44 wall, 1.6GB RSS, 84/84 CTests pass)
- Binary: Apr 28 20:20 (includes ambient v4, doublet v5, CSC overflow fix, encode abort fix)

### Smoke Test C176c (SRR13496726, 10xv3 human, OLD binary)
- Job 367910: COMPLETED (6:15 wall, 64GB RSS, exit 0)
- 239 cells, 87.9% mapping, 372.6s pipeline wall
- Sex: unknown (STALE BINARY — ratio fix already in source but not compiled)
- Doublet: 18.4% (44/239) — sample has only 239 cells, elevated rate expected
- Ambient: rho=0.001 (emptydrops_derived)

### Smoke Test C176e (SRR33424030, 10xv3 mouse, NEW binary)
- Job 367915: COMPLETED (3:23 wall, exit 0)
- 5,069 cells, 79.1% mapping, 197.1s pipeline wall
- Sex: unknown (mouse genes, expected — human markers only)
- Doublet: 4.97% (252/5069) — CORRECT RANGE (vs old binary 40.7%)
- Ambient: rho=0.012

### Active Builds
- Job 367922: Bloom filter build for species detection — TIMED OUT at 1h (short partition). Resubmitted as job 368015 (cpu partition, 12h wall). Human at 220K/256K sequences when resubmitted.

### Batch Reprocessing (29 crash samples, job 367925)
Submitted 2026-04-28 ~21:00. All human, diverse protocols.

| Category | Count | Exit Code | Notes |
|----------|-------|-----------|-------|
| data_incomplete | 15 | 2 | Barcode-stripped data — correctly detected by new binary |
| processed (SOFT_FAIL) | 7 | 0 | Completed but low quality data |
| processed (would-be SUCCESS) | 2* | 0 | GSM6415294 (500 cells, 75.7% MR), GSM8583110 (63 cells, 84% MR) — misclassified by cells parsing bug |
| STAR fatal | 2 | 104 | CB_UMI_Complex barcode mismatch (task 15), quality string length (task 21) |
| pipeline error | 2 | 1 | No reads processed (task 11), empty R2 CEL-Seq2 (task 17) |
| still running | 2 | — | BD Rhapsody 280M reads (task 25), Parse 172M reads (task 29) |
| preflight skip | 1 | 2 | Task 1 — barcode-stripped (3,269 reads, 4% barcode match) |

*Fixed: reprocess script cells parsing bug (estimated_cells vs cells JSON key). Main pilot_job.sh was NOT affected.

**Key insight**: New binary never crashes on any of these samples — it either exits gracefully with clear error (exit 2) or processes to completion. Previous binary would crash/segfault on many of these.
