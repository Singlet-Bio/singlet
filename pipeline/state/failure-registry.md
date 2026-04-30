# Failure Registry — Singlify Catalog Pipeline

> Tracks all distinct failure modes discovered during Phase 1 monitoring. Each entry records root cause, affected samples, and resolution status. Updated per cycle.

## Resolved Failures

### FAILURE-1: snp_ad/dp not exported (RESOLVED 2026-04-14)
- **Root cause**: Export condition in `include/singlet-pileup/export.h` was `&& !export_cfg.pipeline_mode` — inadvertently skipped SNP matrices when `--pipeline` flag was set (despite `--snps` being provided)
- **Observed symptom**: `snp_ad.1pz` and `snp_dp.1pz` files absent from output even when `--snps common_snps.vcf --pipeline` was passed
- **Fix**: Removed erroneous condition; SNP export now triggered by `--snps` flag presence alone
- **Commit**: adf8f1b (bio-exec, validated 2026-04-14)
- **Acceptance test**: Panel A (SRR32855204) now produces snp_ad.1pz (45K SNP sites × 11,560 cells)

### FAILURE-3: arc-gex false positive in protocol detection (RESOLVED 2026-04-14)
- **Root cause**: Protocol tie-break in `src/protocol.h` evaluated candidates by score alone; identical scores between 10x-arc-gex and 10x-3p-v3 led to arc-gex winning due to iteration order. arc-gex has only 737K barcodes vs v3's 3.7M, so false detections resulted in 0 cells.
- **Observed symptom**: Standard 10xv3 samples (e.g., GSM8540924) misclassified as 10x-arc-gex → 0 cells, SOFT_FAIL
- **Fix**: Sort protocol candidates by `protocol_id` ascending before returning; tie-break prefers lower ID (v3 < arc-gex)
- **Commit**: 7880949 (perf-exec, validated 2026-04-14)
- **Following DAG task**: AUTOFIX-ARC-GEX-WHITELIST (still open — true gex_737K-arc-v1.txt must be installed)

---

## NEW Open Failures (Cycle 146/continued)

### AUTOFIX-CATALOG-NULL-SRR (MEDIUM priority)
- **Root cause**: Several catalog rows have `srr_accessions=None` (NULL in parquet). Batch JSON generation did not filter these out. Job script attempted `singlify download None`, which produced error `ERROR: Cannot open SRA as database or table: None`
- **Observed symptom**: 
  - Batch_007 tasks 43, 44, 45, 46, 66: log shows `[singlify-download] None → /dev/shm/GSMXXXX/None.1fq`
  - No download performed
  - Job exits with HARD_FAIL download_fail
- **Affected samples**: ~5–7 in batch_007; catalog check reveals ~2% prevalence (12/506 samples in recent eligible set)
- **Fix**: At batch composition time, apply filter `srr_accessions NOT NULL AND len(srr_accessions) > 0`
- **Workaround applied**: Batch_008 and future batches now filter `read_count > 5M AND srr_accessions IS NOT NULL OR "None"`
- **Status**: CLASSIFICATION (not a singlify code bug; batch composition bug)

### AUTOFIX-OOM-READ-COUNT-TIER-MISS (HIGH priority)
- **Cycle discovered**: 146
- **Root cause**: Batch_007 included 51 samples with `read_count=0` (unknown from catalog) that turned out to be large (100–311M reads). Without pre-loaded genome (`--genome-shared`), each singlify job loads 35GB genome into job RAM. STAR BAM sort on 50M+ reads adds ~100GB temp. Total peak = 135GB+, exceeding 128G SLURM allocation.
- **Diagnosis**: Code-scout triage of all 51 OOM logs shows:
  - All 51 peaked at EXACTLY 128GB (hard SLURM ceiling, not over-subscription)
  - 51% (23/51) are >300M-read samples; 100% are >100M reads
  - Protocol-neutral: 9 different protocols affected (10xv3, 10xv2, dropseq, bd_rhapsody, sciRNA-seq, etc.)
  - Node distribution clean (no hardware hotspot)
- **Observed symptom**: 
  - Batch_007 tasks 9, 15, 17, 21, 28, 33, 35, 43, 44, 45, 46 + 40 others: SLURM exit code 137 (OOM) after 20–53 min
  - Affected samples: GSM7496814 (148M reads), GSM4625992 (167M reads), GSM4090782 (311M reads), etc. (complete list in batch_007_384g_requeue)
- **Resource model feedback**: 
  - Bootstrap tier (8 CPUs / 128GB) designed for <50M reads; inadequate for >100M reads without `--genome-shared`
  - Tiered submission strategy required: catalog `read_count >= 50M OR read_count == 0` → 192G tier
  - With `--genome-shared` pre-loaded: peak RAM drops from 135GB to ~65GB (35GB saved, permits 128G allocation)
- **Fix applied (cycle 146)**: Will update resource-model.json to add `>300M` bucket with `ram_gb: 192` before next batch submission
- **Long-term fix**: (1) Implement `--genome-shared` SLURM prologue (pre-load genome on each node before jobs run), (2) Auto-tier batch composition by catalog read_count, (3) Fallback requeue at higher tier on OOM (implemented in cycle 146)
- **Acceptance test**: Batch_008+ with tiered RAM achieves ≥95% SUCCESS on >100M read samples at 192G, zero OOM recurrence
- **Blocked samples**: 43 OOM samples held in batch_007_384g_requeue (SLURM 359953); requeue at 192G pending resource-model update
- **Status**: CLASSIFICATION + IMMEDIATE FIX (resource-model tier bump in progress)
- **Priority**: HIGH

### AUTOFIX-SPECIES-VAL-R2-SHORT (MEDIUM priority)
- **Root cause**: Species validation task 3 (zebrafish, GSM2830043, SRR13069863): ENA curl fallback encoded the FASTQ with automatic adapter trimming. adapter_type=auto detected phix-adapter at position 30 in R2, resulting in truncated R2 (30bp instead of 58bp expected). singlify decoded the truncated R2 and protocol detector saw short R2 + long R1 → misassigned as `10x-atac` (protocol_id=23, confidence=1). Pipeline proceeded with 0 cells (ATAC has wrong barcode structure for standard chemistry).
- **Observed symptom**: 
  - species_val log: `adapter detected in R2 at position 30 → auto-setting r2_maxlen=30`
  - Result JSON: `protocol_id=23 (10x-atac), cells_called=0, mapping_rate=0%`
- **Affected samples**: 1 confirmed (GSM2830043); likely affects ~3–5% of samples using ENA fallback
- **Fix target**: Protocol detector should reject confidence=1 auto-detections when R2 is truncated/variable from encode phase. Log as encode_fail and recommend re-download with explicit `--protocol` or VDB fix.
- **Blocked samples**: GSM2830043 (zebrafish species validation)

### AUTOFIX-SPECIES-VAL-PYTHON-FALSE (LOW priority)
- **Root cause**: species_val_panel.sh result JSON generation embedded a Python3 command with JavaScript-style boolean literals: `"mapped": false, "success": true` instead of Python `False/True`
- **Observed symptom**: 
  - Result JSON write fails: `NameError: name 'false' is not defined` in Python3
  - Result JSON not created; species_val script logs partial failure
- **Affected samples**: species_val panels; reproducible in any run of species_val_panel.sh
- **Fix**: Replace `false` → `False` and `true` → `True` in embedded Python section of species_val_panel.sh
- **Status**: FIXED (sed command run on login node)

---

## Open Failures (Latest Cycle)

### AUTOFIX-VDB-READ-SWAP-PROTOCOL (HIGH priority)
- **Root cause**: Some SRA depositors submit read files with non-standard orientation: R1 contains cDNA (e.g., 98bp), R2 contains barcodes (26bp). VDB streams them in original orientation. singlify protocol detector sees long R1 reads and may misassign (e.g., splitseq instead of 10xv2).
- **Observed symptom**: 
  - SRR5398238 (Kang 2018 8-donor pool, ~181M reads): detected as splitseq, 0 cells (E2E Panel B blocked)
  - Silent 0-read failure in STAR phase when splitseq decode produces empty R2
- **Fix target**: 
  1. Before writing `.1fq` header, probe both read orientations
  2. Sample 100K reads from VDB; attempt BC match against candidate whitelists for both (R1→BC, R2→BC)
  3. Set `swapped` flag in .1fq header when swap detected
  4. Advance to Stage 1 auto-detection with corrected orientation
- **Acceptance test**: `singlify download SRR5398238` auto-detects as 10xv2 (or inferred protocol) with cell count ≥1000
- **Blocked samples**: Any SRA dataset with non-standard read orientation (likely ~5-10% of catalog based on SRA deposition practices)
- **Priority notes**: Blocks E2E Panel B validation; affects multi-donor samples where demux signal is critical

### AUTOFIX-SPLITSEQ-DECODE-EMPTY (HIGH priority)
- **Root cause**: When singlify incorrectly detects splitseq protocol (e.g., due to VDB read swap), the decode phase produces empty FASTQ files (0 reads in R1/R2). STAR then receives empty input and exits with 0 reads aligned, mapping rate N/A, no cells called. This silent failure is classified as SOFT_FAIL or HARD_FAIL depending on error message content.
- **Observed symptom**: Result JSON shows `mapping_rate: 0.0, cells_called: 0` with no error message; log shows STAR ran successfully but with 0/0 reads
- **Fix target**: 
  1. After decode phase, validate that R1/R2 decoded FASTQ files have ≥1 read
  2. If either file is empty, log HARD_FAIL with reason "decode produced empty FASTQ" and abort before STAR
  3. Recommend VDB re-download with explicit protocol flag as manual recovery
- **Acceptance test**: Running pipeline on SRR5398238 without fix → produces 0 cells; with fix → HARD_FAIL with clear message pointing to decode error
- **Blocked samples**: Samples incorrectly detected as splitseq (depends on AUTOFIX-VDB-READ-SWAP-PROTOCOL fix landing first)
- **Priority**: Dependent on VDB-READ-SWAP; move to MEDIUM after that lands

---

## Monitoring Checklist (Phase 1)
- [ ] Check new result status counts (SUCCESS/SOFT_FAIL/HARD_FAIL)
- [ ] Triage all HARD_FAIL log files (last 5 results)
- [ ] Check for new protocol auto-detection failures
- [ ] Check for new download failures
- [ ] Summarize OOM events (Batch OOM rate vs per-protocol baseline)
- [ ] Note any new error message patterns not previously seen
