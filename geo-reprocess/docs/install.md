# Installation

## Requirements

- Python ≥ 3.10
- Linux (HPC clusters, tested on RHEL 9, Ubuntu 22.04, CentOS 7)
- SLURM job scheduler (for pipeline orchestration)
- External tools: `simpleaf` + `piscem` (quantification), `fasterq-dump` (SRA downloads), `kraken2` (optional, non-host RNA classification)

## From source (recommended)

```bash
git clone https://github.com/Singlet-AI/geo-reprocess.git
cd geo-reprocess
pip install -e ".[dev]"
```

## From PyPI

```bash
pip install sc-geo
```

---

## HPC Cluster Setup

This section covers everything needed to run the full pipeline on a fresh SLURM cluster.

### 1. Create a conda environment

```bash
# Load conda (adjust module name to your cluster)
module load miniconda3

# Create and activate environment
conda create -n scgeo python=3.11 -y
conda activate scgeo

# Install sc-geo from source
git clone https://github.com/Singlet-AI/geo-reprocess.git
cd geo-reprocess
pip install -e .

# Install companion package for .1pz format
pip install singlepress
```

### 2. Install external tools

All tools can be installed into the same conda environment:

```bash
conda activate scgeo

# simpleaf — quantification engine (includes piscem mapper)
conda install -c bioconda -c conda-forge simpleaf -y

# SRA toolkit — FASTQ downloads from NCBI
conda install -c bioconda sra-tools -y

# kraken2 — non-host RNA classification (optional)
conda install -c bioconda kraken2 -y

# pigz — parallel gzip (speeds up FASTQ compression)
conda install -c conda-forge pigz -y
```

Verify installations:

```bash
simpleaf --version   # Should print simpleaf version
fasterq-dump --version
kraken2 --version    # Optional
```

### 3. Build reference genome indices

simpleaf requires pre-built indices for each species you want to quantify.
Download reference genomes from Ensembl and build indices:

```bash
export ALEVIN_FRY_HOME="/path/to/af_home"

# Human (GRCh38)
simpleaf index \
  --output "$ALEVIN_FRY_HOME/human" \
  --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz \
  --gtf Homo_sapiens.GRCh38.113.gtf.gz \
  --threads 16

# Mouse (GRCm39)
simpleaf index \
  --output "$ALEVIN_FRY_HOME/mouse" \
  --fasta Mus_musculus.GRCm39.dna.primary_assembly.fa.gz \
  --gtf Mus_musculus.GRCm39.113.gtf.gz \
  --threads 16
```

The pipeline supports 55+ organisms. See `scgeo/config/species.py` for the
complete list and expected assembly names. Build indices for as many species
as you plan to process.

### 4. Set up Kraken2 database (optional)

```bash
# Download pre-built standard database (~50 GB)
kraken2-build --standard --db /path/to/kraken2_db --threads 16

# Or use a smaller database
kraken2-build --special standard-8 --db /path/to/kraken2_db_8gb --threads 16
```

### 5. Create pipeline directory structure

```bash
export SCGEO_BASE="/path/to/pipeline"

mkdir -p "$SCGEO_BASE/pipeline"/{claims,downloads,quant,results,logs}
mkdir -p "$SCGEO_BASE/catalog"
```

### 6. Configure environment variables

Add to your `.bashrc` or SLURM worker script:

```bash
# Required
export SCGEO_BASE="/path/to/pipeline"              # Pipeline root directory
export SCGEO_WORKSPACE="/path/to/geo-reprocess"     # Source code checkout
export ALEVIN_FRY_HOME="$SCGEO_BASE/af_home"        # simpleaf index directory

# Optional
export NCBI_API_KEY="your_key_here"                  # Faster NCBI API access
export NCBI_EMAIL="you@example.com"                  # NCBI rate limit identity
```

### 7. Verify setup

```bash
conda activate scgeo
python -c "import scgeo; print(scgeo.__version__)"
simpleaf --version
echo "SCGEO_BASE=$SCGEO_BASE"
echo "ALEVIN_FRY_HOME=$ALEVIN_FRY_HOME"
ls "$ALEVIN_FRY_HOME"/human/index  # Should list piscem index files
```

---

## Next Steps

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for how to build the catalog and run quantification at scale.
