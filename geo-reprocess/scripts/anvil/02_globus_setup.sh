#!/bin/bash
#
# Globus Setup + Smoke Test — verify data transfer between Anvil and Clipper.
#
# This script walks through both sides of the Globus setup and runs a small
# round-trip transfer test before committing to the full pipeline.
#
# Run on the ANVIL LOGIN NODE while the index builds.
#
# Usage:
#   bash 02_globus_setup.sh

set -euo pipefail

echo "════════════════════════════════════════════════════"
echo "  Globus Setup + Transfer Smoke Test"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════
# STEP 1: Install Globus CLI
# ═══════════════════════════════════════════════════════

echo ""
echo "▸ Step 1: Globus CLI installation"
echo ""

module load anaconda
conda activate "$PROJECT/envs/scgeo"

if command -v globus &>/dev/null; then
    echo "  Globus CLI already installed: $(globus --version)"
else
    echo "  Installing globus-cli..."
    pip install globus-cli 2>&1 | tail -3
fi

# ═══════════════════════════════════════════════════════
# STEP 2: Authenticate
# ═══════════════════════════════════════════════════════

echo ""
echo "▸ Step 2: Authentication"
echo ""
echo "  Run this command and follow the browser link:"
echo "    globus login"
echo ""
echo "  (If on a remote terminal with no browser, use:"
echo "    globus login --no-local-server"
echo "  and paste the auth code back.)"
echo ""

# Check if already authenticated
if globus get-identities --linked-identities &>/dev/null 2>&1; then
    echo "  Already authenticated ✓"
    LOGGED_IN=true
else
    echo "  Not authenticated. Run 'globus login' first, then re-run this script."
    LOGGED_IN=false
fi

# ═══════════════════════════════════════════════════════
# STEP 3: Find endpoints
# ═══════════════════════════════════════════════════════

echo ""
echo "▸ Step 3: Discover endpoints"
echo ""

echo "  ── Anvil (source/destination) ──"
echo "  Anvil is an ACCESS resource with managed Globus storage."
echo "  Search for the endpoint:"
echo "    globus endpoint search 'Purdue Anvil'"
echo "    globus endpoint search 'ACCESS Anvil'"
echo "    globus endpoint search 'Purdue RCAC'"
echo ""

echo "  ── Clipper (source/destination) ──"
echo "  GVSU Clipper needs a Globus endpoint. Options:"
echo ""
echo "  Option A — Check if GVSU has a managed endpoint:"
echo "    globus endpoint search 'Grand Valley'"
echo "    globus endpoint search 'GVSU'"
echo ""
echo "  Option B — Install Globus Connect Personal on Clipper:"
echo "  (Run these commands on Clipper login node)"
cat << 'CLIPPER_SETUP'

    # === ON CLIPPER LOGIN NODE ===
    cd /tmp
    wget -q https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
    tar xzf globusconnectpersonal-latest.tgz
    cd globusconnectpersonal-*

    # Setup (one-time — follow browser prompts to link your account)
    ./globusconnectpersonal -setup

    # Configure allowed paths — CRITICAL: must include the project dir
    # Edit ~/.globusonline/lta/config-paths to contain:
    #   /mnt/projects/debruinz_project/cellarium,0,1
    #   (path, readable=0/1, writable=0/1)

    mkdir -p ~/.globusonline/lta
    echo "/mnt/projects/debruinz_project/cellarium,0,1" > ~/.globusonline/lta/config-paths

    # Start the personal endpoint (background)
    ./globusconnectpersonal -start &

    # Note the endpoint UUID printed during setup
    # e.g.: "Endpoint ID: abc12345-6789-..."
    echo "Record this endpoint UUID for transfers!"

CLIPPER_SETUP

echo ""

# ═══════════════════════════════════════════════════════
# STEP 4: Transfer smoke test
# ═══════════════════════════════════════════════════════

echo ""
echo "▸ Step 4: Transfer smoke test"
echo ""
echo "  Once you have both endpoint UUIDs, run the smoke test:"
echo ""

# Create a small test file on Anvil
export SCGEO_BASE="$PROJECT/scgeo"
TESTFILE="$SCGEO_BASE/pipeline/quant/_globus_test_$(date +%s).txt"
echo "Globus smoke test from Anvil — $(date -u)" > "$TESTFILE"
echo "  Created test file: $TESTFILE"

cat << 'SMOKE_TEST'

  # Set your endpoint UUIDs (found via 'globus endpoint search'):
  ANVIL_EP="<anvil-endpoint-uuid>"
  CLIPPER_EP="<clipper-endpoint-uuid>"

  # ── Test 1: Anvil → Clipper (this is the production path) ──
  TASK=$(globus transfer \
    "$ANVIL_EP:$PROJECT/scgeo/pipeline/quant/_globus_test_*.txt" \
    "$CLIPPER_EP:/mnt/projects/debruinz_project/cellarium/pipeline/quant/" \
    --label "Smoke test: Anvil → Clipper" \
    --sync-level checksum \
    --jmespath 'task_id' --format unix)

  echo "Transfer task ID: $TASK"

  # Wait for it (should be <30 seconds for a tiny file)
  globus task wait "$TASK" --polling-interval 5 --timeout 120

  # Check result
  globus task show "$TASK"

SMOKE_TEST

echo ""
echo "  ── Verify on Clipper side ──"
echo "  SSH to Clipper and check:"
echo "    ls -la /mnt/projects/debruinz_project/cellarium/pipeline/quant/_globus_test_*"
echo "    cat /mnt/projects/debruinz_project/cellarium/pipeline/quant/_globus_test_*"
echo "    # Should show: 'Globus smoke test from Anvil — <timestamp>'"
echo ""

# ═══════════════════════════════════════════════════════
# STEP 5: Verify reverse path (Clipper → Anvil for catalog)
# ═══════════════════════════════════════════════════════

echo ""
echo "▸ Step 5: Transfer catalog via Globus (Clipper → Anvil)"
echo ""

cat << 'CATALOG_XFER'

  # Transfer the processing catalog from Clipper to Anvil
  TASK=$(globus transfer \
    "$CLIPPER_EP:/mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet" \
    "$ANVIL_EP:$PROJECT/scgeo/catalog/processing_catalog.parquet" \
    --label "Catalog: Clipper → Anvil" \
    --sync-level checksum \
    --jmespath 'task_id' --format unix)

  echo "Catalog transfer: $TASK"
  globus task wait "$TASK" --polling-interval 10 --timeout 300

  # Verify
  globus task show "$TASK"

CATALOG_XFER

echo ""
echo "  ── Alternative: scp (if Globus isn't working) ──"
echo "  From Clipper login node:"
echo "    scp /mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet \\"
echo "        x-zdebruine@anvil.rcac.purdue.edu:\$PROJECT/scgeo/catalog/"
echo ""

echo "════════════════════════════════════════════════════"
echo "  Globus Smoke Test Checklist"
echo ""
echo "  [ ] globus login completed"
echo "  [ ] Anvil endpoint UUID found"
echo "  [ ] Clipper endpoint UUID found (or GCP installed)"
echo "  [ ] Test file transferred Anvil → Clipper"
echo "  [ ] Test file verified on Clipper"
echo "  [ ] Catalog transferred Clipper → Anvil"
echo "  [ ] Catalog verified on Anvil:"
echo "      python3 -c \"import pandas as pd; c=pd.read_parquet('\$PROJECT/scgeo/catalog/processing_catalog.parquet'); print(len(c), 'rows')\""
echo ""
echo "  Next: bash 03_smoke_test.sh"
echo "════════════════════════════════════════════════════"
