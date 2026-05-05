#!/bin/bash
#
# Anvil Bootstrap — Step 1: Install packages and build mouse index.
#
# Run this INTERACTIVELY on the Anvil login node. It sets up the directory
# structure, installs conda + packages, then submits a SLURM job to build
# the mouse splici index (takes ~45 min on 16 cores).
#
# Prerequisites:
#   - ssh x-zdebruine@anvil.rcac.purdue.edu
#   - $PROJECT is set (Anvil sets this automatically)
#   - Allocation name known (replace YOUR_ALLOCATION below)
#
# Usage:
#   bash 01_bootstrap.sh

set -euo pipefail

ALLOCATION="${ALLOCATION:-YOUR_ALLOCATION}"
if [[ "$ALLOCATION" == "YOUR_ALLOCATION" ]]; then
    echo "ERROR: Set ALLOCATION first:"
    echo "  export ALLOCATION=cis250209   # (or whatever your allocation code is)"
    exit 1
fi

echo "════════════════════════════════════════════════════"
echo "  Anvil Bootstrap — Phase 4a (Mouse)"
echo "  Allocation: $ALLOCATION"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "════════════════════════════════════════════════════"

# ── 1. Directory structure ──
echo ""
echo "▸ Step 1: Creating directory structure..."
export SCGEO_BASE="$PROJECT/scgeo"
mkdir -p "$SCGEO_BASE"/{catalog,pipeline/{claims,downloads,quant,results,logs},af_home,index}

# Symlink scratch for fast downloads (auto-purged after 30 days)
mkdir -p "$SCRATCH/scgeo_downloads"
if [[ ! -L "$SCGEO_BASE/pipeline/downloads" ]]; then
    rmdir "$SCGEO_BASE/pipeline/downloads" 2>/dev/null || true
    ln -sf "$SCRATCH/scgeo_downloads" "$SCGEO_BASE/pipeline/downloads"
fi
echo "  SCGEO_BASE=$SCGEO_BASE"
echo "  Downloads → $SCRATCH/scgeo_downloads (scratch, fast I/O)"

# ── 2. Conda environment ──
echo ""
echo "▸ Step 2: Installing conda environment..."
module load anaconda
if conda env list | grep -q "$PROJECT/envs/scgeo"; then
    echo "  Environment exists, activating..."
    conda activate "$PROJECT/envs/scgeo"
else
    echo "  Creating environment..."
    conda create -p "$PROJECT/envs/scgeo" python=3.11 -y
    conda activate "$PROJECT/envs/scgeo"
fi

# ── 3. Install packages ──
echo ""
echo "▸ Step 3: Installing packages..."

# Clone or update geo-reprocess
if [[ -d "$PROJECT/geo-reprocess" ]]; then
    echo "  geo-reprocess exists, pulling latest..."
    cd "$PROJECT/geo-reprocess"
    git pull
else
    echo "  Cloning geo-reprocess..."
    git clone https://github.com/Singlet-AI/geo-reprocess.git "$PROJECT/geo-reprocess"
    cd "$PROJECT/geo-reprocess"
fi

pip install -e . 2>&1 | tail -5
pip install singlepress 2>&1 | tail -5

# Bioinformatics tools
conda install -c bioconda -c conda-forge simpleaf sra-tools pigz -y 2>&1 | tail -5
echo "  Packages installed"

# Verify tools
echo ""
echo "▸ Verifying installation..."
python3 -c "import scgeo; print(f'  scgeo {scgeo.__version__}')"
python3 -c "import singlepress; print(f'  singlepress OK')"
echo "  simpleaf: $(simpleaf --version 2>&1 | head -1)"
echo "  fasterq-dump: $(which fasterq-dump)"

# ── 4. Configure shell ──
echo ""
echo "▸ Step 4: Configuring shell environment..."
BASHRC_MARKER="# === SCGEO Anvil config ==="
if ! grep -q "$BASHRC_MARKER" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'ENVEOF'

# === SCGEO Anvil config ===
export SCGEO_BASE="$PROJECT/scgeo"
export SCGEO_WORKSPACE="$PROJECT/geo-reprocess"
export SCGEO_INDEX_DIR="$SCGEO_BASE/index"
export ALEVIN_FRY_HOME="$SCGEO_BASE/af_home"
ENVEOF
    echo "  Added SCGEO env vars to ~/.bashrc"
else
    echo "  ~/.bashrc already configured"
fi

# ── 5. Transfer catalog from Clipper ──
echo ""
echo "▸ Step 5: Transfer catalog..."
echo "  Run from Clipper login node:"
echo "    scp /mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet \\"
echo "        x-zdebruine@anvil.rcac.purdue.edu:$SCGEO_BASE/catalog/"
echo ""
echo "  OR via Globus (see 02_globus_setup.sh)"
echo ""

# ── 6. Build mouse index ──
echo "▸ Step 6: Building mouse splici index..."
echo ""
echo "  Option A: Transfer pre-built index from Clipper (~17 GB, faster):"
echo "    scp -r /mnt/projects/debruinz_project/cellarium/index/mouse_splici/ \\"
echo "        x-zdebruine@anvil.rcac.purdue.edu:$SCGEO_BASE/index/"
echo ""
echo "  Option B: Build fresh on Anvil (submitting SLURM job)..."

# Submit index build job
cat > "$SCGEO_BASE/build_mouse_index.sh" << 'IDXEOF'
#!/bin/bash
#SBATCH --job-name=build_idx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH -p shared

set -euo pipefail
module load anaconda
conda activate "$PROJECT/envs/scgeo"

export SCGEO_BASE="$PROJECT/scgeo"
export ALEVIN_FRY_HOME="$SCGEO_BASE/af_home"

# Initialize simpleaf
simpleaf set-paths

# Download Ensembl references
MOUSE_REF="$SCGEO_BASE/index/mouse_ref"
mkdir -p "$MOUSE_REF"
cd "$MOUSE_REF"

echo "Downloading mouse genome + GTF from Ensembl 113..."
wget -q -N "https://ftp.ensembl.org/pub/release-113/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.primary_assembly.fa.gz"
wget -q -N "https://ftp.ensembl.org/pub/release-113/gtf/mus_musculus/Mus_musculus.GRCm39.113.gtf.gz"

# Build splici index
echo "Building piscem splici index (16 threads)..."
OUTPUT_DIR="$SCGEO_BASE/index/mouse_splici"
mkdir -p "$OUTPUT_DIR"

simpleaf index \
  --output "$OUTPUT_DIR" \
  --fasta "Mus_musculus.GRCm39.dna.primary_assembly.fa.gz" \
  --gtf "Mus_musculus.GRCm39.113.gtf.gz" \
  --rlen 91 \
  --threads 16 \
  --use-piscem

echo "Index build complete: $(du -sh $OUTPUT_DIR)"
ls -la "$OUTPUT_DIR/"
IDXEOF

echo "  Submitting index build job..."
INDEX_JOB=$(sbatch -A "$ALLOCATION" "$SCGEO_BASE/build_mouse_index.sh" 2>&1 | grep -oP '\d+')
echo "  Index build job: $INDEX_JOB"
echo "  Monitor: squeue -j $INDEX_JOB"
echo "  Log: cat slurm-${INDEX_JOB}.out"

echo ""
echo "════════════════════════════════════════════════════"
echo "  Bootstrap complete!"
echo ""
echo "  While the index builds (~45 min), proceed to:"
echo "    bash 02_globus_setup.sh"
echo ""
echo "  After index is done + catalog transferred:"
echo "    bash 03_smoke_test.sh"
echo "════════════════════════════════════════════════════"
