# Shared settings for the public benchmark harness on Clipper (sourced by every bench/public/harness/*.sbatch).
WORK=/mnt/projects/debruinz_project/debruinz/singlet
BP=$WORK/bpub                              # data/, runs/, artifacts/, results/, logs of this program
PB=$WORK/pipeline-benchmark                 # pipeline-benchmark session's data and binaries (reused, read-only here)
ENV=$WORK/envs/pbench                       # conda env: STAR 2.7.11b, samtools, bcftools, simpleaf/alevin-fry, python
SRA=$WORK/envs/sratools/bin                 # prefetch / fasterq-dump / fastq-dump 3.x
REF=/mnt/projects/debruinz_project/cellarium/reference   # 10x 2024-A genomes + GENCODE v44 / M33, STAR 2.7.11b index
SRC=${SRC:-$WORK/src/singlet-py/claude__bench-public-cohort-485cf7}   # this branch (harness + compute modules)
# The Singlet 1.0.0 binary under test: the PGO build of prod/campaign (Singlet-Bio/singlet PR #31) by the pipeline-benchmark
# session. Its source tree gives the commit recorded in every result.
SINGLET_BIN=${SINGLET_BIN:-$PB/bin/singlet-campaign}   # PGO build of prod/campaign 02aa9b9, sha256 261d3a50... (pipeline-benchmark, 2026-10-25)
SINGLET_SRC=${SINGLET_SRC:-$WORK/src/singlet-py/prod__campaign}
CELLRANGER=${CELLRANGER:-}                  # empty (default): 10x published outputs are the comparator (Zach, 2026-10-25)
export CONDA_PKGS_DIRS=$WORK/cache/conda_pkgs ALEVIN_FRY_HOME=$PB/af_home
export PATH=$ENV/bin:$SRA:$PATH
COHORT=$SRC/bench/public/cohort.tsv
mkdir -p $BP/{data,runs,artifacts,results,work} $WORK/logs

# cohort row by id -> associative array C[column]
cohort_row() {
  declare -gA C=()
  local hdr row i
  # one field per line: `read -a` with IFS=tab would collapse empty fields and shift every later column
  mapfile -t hdr < <(grep -v '^#' $COHORT | head -1 | awk -F'\t' '{for (i = 1; i <= NF; i++) print $i}')
  mapfile -t row < <(grep -v '^#' $COHORT | awk -F'\t' -v id="$1" '$1==id {for (i = 1; i <= NF; i++) print $i}')
  [ ${#row[@]} -gt 0 ] || { echo "no cohort row $1" >&2; return 1; }
  for i in "${!hdr[@]}"; do C[${hdr[$i]}]="${row[$i]:-}"; done
}

# 10x chemistry -> STARsolo / alevin-fry / whitelist settings
chem_settings() {
  WLDIR=$BP/whitelists   # filled by setup.sbatch (Singlet's bundled lists + 10x's GEM-X v4 list)
  case $1 in
    10x-3p-v1) CBLEN=14; UMISTART=15; UMILEN=10; WL=$WLDIR/737K-april-2014.txt; STRAND=Forward; AFCHEM=10xv1; AFORI=fw ;;
    10x-3p-v2) CBLEN=16; UMISTART=17; UMILEN=10; WL=$WLDIR/737K-august-2016.txt; STRAND=Forward; AFCHEM=10xv2; AFORI=fw ;;
    10x-3p-v3) CBLEN=16; UMISTART=17; UMILEN=12; WL=$WLDIR/3M-february-2018.txt; STRAND=Forward; AFCHEM=10xv3; AFORI=fw ;;
    10x-3p-v4) CBLEN=16; UMISTART=17; UMILEN=12; WL=$WLDIR/3M-3pgex-may-2023.txt; STRAND=Forward; AFCHEM=10xv4-3p; AFORI=fw ;;
    10x-5p-v1|10x-5p-v2) CBLEN=16; UMISTART=17; UMILEN=10; WL=$WLDIR/737K-august-2016.txt; STRAND=Reverse; AFCHEM=10xv2-5p; AFORI=rc ;;
    10x-arc)   CBLEN=16; UMISTART=17; UMILEN=12; WL=$WLDIR/gex_737K-arc-v1.txt; STRAND=Forward; AFCHEM=10xv3; AFORI=fw ;;
    *) CBLEN=; WL= ;;
  esac
}

# Singlet runtime (image + pinned venv), like singlet-agents hpc/clipper/in_runtime.sh but also binding the read-only
# reference tree. Use only inside Slurm jobs.
rt() {
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "rt: refusing to run outside a Slurm job" >&2; return 2; }
  apptainer exec --bind "$WORK,/tmp,$REF" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin"     --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}" --env "HDF5_USE_FILE_LOCKING=FALSE" --env "WORK=$WORK"     --env "PYTHONPATH=${PYTHONPATH:-}" $WORK/images/python-3.12-bookworm.sif "$@"
}

# GNU time wrapper: every run writes time.txt (wall clock, max RSS) next to its outputs
timed() { local out=$1; shift; /usr/bin/time -v -o "$out" "$@"; }
