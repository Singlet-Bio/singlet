#!/bin/bash
#SBATCH --job-name=s14-speciescheck
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# The engine's species check (singlet.cpp, "[species-check]") and the clearer ".1fq processing is missing ..." error, with the real binary built from fix/species-bloom (BIN) and the
# Bloom filters in $FILTERS. singlet is stopped after 90 s: the check line is printed before the alignment starts. Env: BIN (required), FILTERS.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; PB=$W/pipeline-benchmark
BIN=${BIN:?}; FILTERS=${FILTERS:-$W/reference-extra/species_filters}
T=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
export PATH=$W/envs/pbench/bin:$PATH SINGLET_WHITELIST_DIR=$PB/whitelists
run() {   # label, fq, genome (GRCh38-2024-A|GRCm39-2024-A|none), filters (1|0)
  local lab=$1 fq=$2 g=$3 f=$4 gd=()
  [ "$g" != none ] && gd=(--genome-dir $REF/$g/star_2.7.11b --exons $REF/$g/genes/genes.gtf)
  echo "== $lab"
  ( [ "$f" = 1 ] && export SINGLET_SPECIES_FILTERS=$FILTERS; unset SINGLET_REF_BASE
    timeout 90 $BIN $fq "${gd[@]}" --whitelist-dir $PB/whitelists --out-prefix $T/o_$lab/ --threads 4 --pipeline --engine cpu 2>&1 | grep -a "species-check\|ERROR\|missing\|requires" | cut -c1-300 | head -6 )
}
H=$PB/data/pbmc1k_v3/pbmc1k_v3.1fq; M=$PB/data/neuron1k_v3/neuron1k_v3.1fq
run human_on_human $H GRCh38-2024-A 1
run mouse_on_mouse $M GRCm39-2024-A 1
run human_on_mouse_ref $H GRCm39-2024-A 1
run human_no_filters_env $H GRCh38-2024-A 0
run human_no_genome_dir $H none 1
