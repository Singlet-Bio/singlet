# Anvil (Purdue/ACCESS) Setup Guide

## Cluster Overview

| Resource | Spec |
|----------|------|
| **Login** | `ssh x-USERNAME@anvil.rcac.purdue.edu` (SSH key auth, ACCESS Duo MFA) |
| **CPU nodes** | 750 nodes × 128 cores × 257 GB RAM |
| **Home** | `$HOME` — 25 GB, backed up, not purged |
| **Scratch** | `$SCRATCH` (`/anvil/scratch/`) — 100 TB, **purged after 30 days** |
| **Project** | `$PROJECT` (`/anvil/projects/ALLOCATION/`) — 5 TB, persistent |
| **SU budget** | 100,000 SUs (1 SU = 1 core × 1 hour) |
| **Best partition** | `shared` — up to 128 cores/node, 96 hr walltime, fractional billing |

## Strategy: Run Phase 4a (Mouse) + 4b (Other Organisms)

**Why:** The local Clipper HPC is running Phase 1 (human droplet RNA) — 79 active
jobs processing 28k human samples. Anvil should process the **21,304 eligible
mouse samples** + other-species samples to avoid any contention.

The two clusters use completely separate filesystems and claim ledgers, so there
is zero risk of double-claiming. We just need to copy the catalog over and run
with a separate `SCGEO_BASE`.

### Phase 4a Eligible Set (Post-Curation)

| Protocol | Samples | Historical Success Rate | Risk |
|----------|---------|------------------------|------|
| sci-RNA-seq3 | 12,731 | 91% (human) | Low |
| 10xv3 | 5,879 | ~85% (Phase 1) | Very low |
| 10xv2 | 1,172 | ~85% (Phase 1) | Very low |
| CITE-seq | 473 | ~80% (Phase 1) | Low |
| Drop-seq | 409 | 65% | Medium |
| DNBelab | 169 | 79% | Low |
| 10xv3 5' | 141 | ~85% | Very low |
| BD Rhapsody | 132 | 53% | Medium |
| Parse | 109 | 28% | High |
| InDrop | 41 | 46% (real libs) | Medium |
| 10xv4 | 38 | ~85% | Very low |
| Split-seq | 8 | 46% | Medium |
| ddSEQ | 2 | 24% | High |

**Total:** 21,304 samples — all high confidence (99.9%), all curator-cleared.

**Curation applied 2025-07:** 26,778 problematic samples blocked/triaged:
- 26,347 per-cell demultiplexed InDrop (simpleaf requires multiplexed input)
- 264 10x_multiome → triaged (Phase 2a)
- 78 sci-RNA with 0 reads → blocked_no_fastq
- 51 low-confidence 10x_suspect → blocked
- 21 snRNA_unknown → triaged
- 17 unsupported protocols → blocked/triaged

### Recommended Processing Order

1. **Tier A — Proven 10x protocols** (7,703 samples, ~3,851 SU): Process first
2. **Tier B — sci-RNA-seq3** (12,731 samples, ~6,365 SU): Process second
3. **Tier C — Other validated protocols** (870 samples, ~435 SU): Process last

### SU Budget Analysis

At 4 cores × ~15 min per sample (typical for `shared` partition):
- **1 sample ≈ 1 SU** (4 cores × 0.25 hr)
- **100k SU budget ≈ 100,000 samples** — 5× headroom for 21k samples
- Smoke test (10 samples): ~10 SU
- Full mouse run (21k samples): ~21,000 SU (with retries/failures: ~25,000 SU)
- Budget remaining for Phase 4b (other species): ~75,000 SU

---

## Step-by-Step Setup

All setup is automated via numbered scripts. Run them in order:

```
scripts/anvil/
├── 01_bootstrap.sh       # Install packages + build mouse index
├── 02_globus_setup.sh    # Configure Globus + transfer smoke test
├── 03_smoke_test.sh      # Process 2 small mouse samples (SLURM)
├── 04_launch.sh          # Submit full Phase 4a array job
├── anvil_worker.sh       # SLURM array task (called by 04_launch.sh)
└── ANVIL_SETUP.md        # This document
```

### Quick Start

```bash
ssh x-zdebruine@anvil.rcac.purdue.edu
export ALLOCATION=cis250209  # Your allocation code

# Step 1: Install everything + start index build (~45 min SLURM job)
bash scripts/anvil/01_bootstrap.sh

# Step 2 (while index builds): Set up Globus, run transfer smoke test
bash scripts/anvil/02_globus_setup.sh

# Step 3 (after index + catalog ready): Smoke test pipeline
sbatch -A $ALLOCATION scripts/anvil/03_smoke_test.sh

# Step 4 (after smoke test passes): Full launch
bash scripts/anvil/04_launch.sh
```

## Memory Requirements

**CRITICAL:** Anvil workers need 64GB RAM, not 8GB.

| Stage | Peak RSS | Driver |
|-------|----------|--------|
| simpleaf/piscem mapping | **50–90 GB** | k-mer index + match structures |
| AnnData conversion | 5–15 GB | sparse matrix transpose |
| Kraken2 | ~5 GB | memory-mapped DB |
| Download | 1–5 GB | curl buffering |

On Clipper, proven production config is **64GB + 4 CPUs** on cpu/short partitions
(128GB only on bigmem for outliers). The 50-100GB peak RSS observed in early runs
was at higher thread counts — piscem allocates per-thread working memory for
k-mer matching, so peak RSS scales roughly linearly with thread count.

At 4 CPUs, peak stays under 64GB for nearly all samples. The mouse index (2.0GB
on disk) is smaller than human (3.3GB), providing additional headroom.

**SU impact:** Anvil `shared` partition bills by `max(cores/128, mem/257GB) × 128`.
At 64GB, memory dominates — **CPUs don't affect the SU rate:**

| Memory | CPUs | Effective SU/hr | SU per sample (est) | Notes |
|--------|------|-----------------|---------------------|-------|
| 8 GB | 4 | 4 | 0.5 | OOM guaranteed |
| 32 GB | 4 | 16 | 1.6 | ~10% OOM rate |
| **64 GB** | **4** | **32** | **3.2** | **Matches Clipper production** |
| 64 GB | 8 | 32 | 3.2 | Same cost, higher OOM risk |
| 96 GB | 4 | 48 | 4.8 | Safe but expensive |
| 128 GB | 8 | 64 | 6.4 | Overkill for mouse |

**Total estimate: 21,304 × 3.2 ≈ 68,000 SU** (within 100k budget, ~32k remaining
for Phase 4b).

---

## Key Differences from Clipper

| Aspect | Clipper (local) | Anvil |
|--------|----------------|-------|
| Partition | cpu/bigmem/short | shared (fractional billing!) |
| Cores/task | 4 (cpu/short), 8 (prior) | 4 (memory-dominated billing) |
| Memory/task | 64 GB (cpu/short), 128 GB (bigmem) | 64 GB |
| Downloads | $SCGEO_BASE/pipeline/downloads | $SCRATCH (fast, auto-purged 30d) |
| Indices | $SCGEO_BASE/index/ | $PROJECT/scgeo/index/ |
| Module | miniconda3/25.5.1 | anaconda |
| Max walltime | 2 days | 96 hours (shared) |
| Auth | SSH key | SSH key + ACCESS Duo MFA |
| Phase | 1 (human) | 4a (mouse) + 4b (other) |
| SU cost/sample | N/A (local) | ~3.2 SU |

## Smoke Test Samples (No Contention)

These are mouse/other-species samples NOT being processed on Clipper:

| GSE | Organism | Samples | Protocol | Confidence |
|-----|----------|---------|----------|------------|
| GSE107527 | Mus musculus | 1 | 10xv3 | high |
| GSE312733 | Mus musculus | 1 | 10xv3 | high |
| GSE313063 | Mus musculus | 1 | 10xv3 | high |
| GSE314753 | Mus musculus | 1 | 10xv3 | high |
| GSE104842 | Mus musculus | 1 | 10xv2 | high |
| GSE312374 | Danio rerio | 1 | 10xv3 | high |
| GSE139318 | Rattus norvegicus | 1 | 10xv3 | high |
| GSE133204 | D. melanogaster | 1 | 10xv3 | high |
| GSE139012 | Mus musculus | 5 | 10xv3 | high |
| GSE141111 | Mus musculus | 5 | 10xv3 | high |

All are high-confidence, single-GSE samples with known protocols — ideal for
validating the pipeline on new infrastructure before scaling.

## Portability Notes

`grab_batch.py` reads paths from `SCGEO_BASE` environment variable (with
fallback to Clipper's default). Set `SCGEO_BASE` in your Anvil `~/.bashrc`
and the claim system works without modification.

The `scgeo` package itself also reads `SCGEO_BASE` from the environment
(see `scgeo/config/paths.py`), so `process_samples()` works out of the box.

### Phase 4a Filter Hardening

The Phase 4a eligibility filter (`_is_eligible_phase4a()` in `grab_batch.py`)
has been hardened to match Phase 1 rigor. It now requires:
- Supported protocol regex (10x, Drop-seq, InDrop, sci-RNA, Parse, etc.)
- No ATAC/multiome protocols
- No low-confidence assignments
- Curator clearance (`cleared` or `triaged` status)
- Not multi-species (no semicolons in organism)
- Not screened (screen_any_flag)
- Not permanently failed

---

## Globus Transfer Plan: Anvil → Clipper

### Overview

After Phase 4a processing completes on Anvil, the `.1pz` output files (quantified
single-cell matrices) need to be transferred back to Clipper for integration with
the rest of the Singlet Bio data lake.

| Parameter | Value |
|-----------|-------|
| **Source** | Anvil `$PROJECT/scgeo/pipeline/quant/` |
| **Destination** | Clipper `/mnt/projects/debruinz_project/cellarium/pipeline/quant/` |
| **File format** | `.1pz` (compressed AnnData, ~5-50 MB per sample) |
| **Expected volume** | ~21,000 files, ~100-500 GB total |
| **Transfer method** | Globus Online (web or CLI) |

### Step 1: Identify Globus Endpoints

**Anvil (source):**
- Anvil is an ACCESS resource with a managed Globus endpoint
- Search for "Purdue Anvil" or "ACCESS Anvil" on [app.globus.org](https://app.globus.org)
- Authenticate with your ACCESS credentials (same Duo MFA as SSH)
- Navigate to `$PROJECT/scgeo/pipeline/quant/`

**Clipper (destination):**
- If GVSU has a Globus endpoint, search for "GVSU" or "Grand Valley State" on Globus
- If not, install **Globus Connect Personal** on the Clipper login node:
  ```bash
  # On Clipper login node
  cd /tmp
  wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
  tar xzf globusconnectpersonal-latest.tgz
  cd globusconnectpersonal-*
  ./globusconnectpersonal -setup  # Follow prompts to link to your Globus account
  ./globusconnectpersonal -start &  # Start the endpoint
  ```
- Configure the endpoint to allow access to `/mnt/projects/debruinz_project/cellarium/`
- Edit `~/.globusonline/lta/config-paths` to add the allowed path

### Step 2: Transfer via Globus Web UI

1. Go to [app.globus.org/file-manager](https://app.globus.org/file-manager)
2. Left panel: Select Anvil endpoint → navigate to `$PROJECT/scgeo/pipeline/quant/`
3. Right panel: Select Clipper endpoint → navigate to `/mnt/projects/debruinz_project/cellarium/pipeline/quant/`
4. Select all `.1pz` files → click "Start"
5. Globus handles retries, checksums, and parallel streams automatically

### Step 3: Transfer via Globus CLI (scripted)

```bash
# Install Globus CLI (on any machine with Python)
pip install globus-cli
globus login  # One-time browser-based auth

# Find endpoint IDs
globus endpoint search "Purdue Anvil"     # Note the UUID
globus endpoint search "GVSU"             # Note the UUID (or use Personal endpoint UUID)

# Set variables
ANVIL_EP="<anvil-endpoint-uuid>"
CLIPPER_EP="<clipper-endpoint-uuid>"
SRC="$PROJECT/scgeo/pipeline/quant/"
DST="/mnt/projects/debruinz_project/cellarium/pipeline/quant/"

# Submit recursive transfer
TASK_ID=$(globus transfer "$ANVIL_EP:$SRC" "$CLIPPER_EP:$DST" \
  --recursive --label "Phase 4a mouse .1pz transfer" \
  --sync-level checksum \
  --jmespath 'task_id' --format unix)

echo "Transfer task: $TASK_ID"
globus task show "$TASK_ID"

# Monitor (or just check web UI)
globus task wait "$TASK_ID" --polling-interval 60
```

### Step 4: Post-Transfer Reconciliation

After transfer, reconcile the catalog on Clipper:

```bash
# On Clipper compute node
ssh c005 "module load miniconda3/25.5.1 && source \$(conda info --base)/etc/profile.d/conda.sh && conda activate cellarium && python3 << 'PYEOF'
import pandas as pd
from pathlib import Path

# Check which .1pz files arrived
quant_dir = Path('/mnt/projects/debruinz_project/cellarium/pipeline/quant')
arrived = {p.stem for p in quant_dir.glob('**/*.1pz')}

# Update catalog for arrived mouse samples
cat = pd.read_parquet('/mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet')
mouse_done = cat[(cat['organism'].str.contains('Mus musculus', na=False)) & 
                 (cat['gsm_id'].isin(arrived))]
print(f'Mouse .1pz files arrived: {len(arrived & set(cat[\"gsm_id\"]))}')
print(f'Need catalog update: {(mouse_done[\"processing_status\"] != \"done\").sum()}')
PYEOF"
```

### Alternative: rsync (if Globus not available)

If Globus endpoints aren't available, use rsync through an SSH tunnel:

```bash
# From Clipper login node (pull from Anvil)
rsync -avz --progress \
  x-zdebruine@anvil.rcac.purdue.edu:'$PROJECT/scgeo/pipeline/quant/*.1pz' \
  /mnt/projects/debruinz_project/cellarium/pipeline/quant/

# Bandwidth: Typically 50-200 MB/s between ACCESS sites
# For 500 GB: ~40 min at 200 MB/s, ~2.5 hr at 50 MB/s
```

### Transfer Timing

- **During processing:** No need to wait — transfer completed batches progressively
- **After all done:** Single bulk transfer, verify with checksums
- **$SCRATCH purge risk:** Anvil purges `$SCRATCH` after 30 days; .1pz files should
  be in `$PROJECT` (persistent) not scratch. Only temp FASTQ downloads go to scratch.
