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
# the engine must be an immutable, identified build: <dir>-<commit>/bin/<binary> (or <dir>/<binary>) beside a `commit` file starting with that commit (never a shared mutable directory)
pin() { local r d c; r=$(readlink -f "$1"); d=$(dirname "$r"); [ "$(basename $d)" = bin ] && d=$(dirname $d); c=$(basename $d); c=${c##*[-_]}; [[ $c =~ ^[0-9a-f]{7,40}$ ]] && [ -x "$r" ] && case "$(cat $d/commit 2>/dev/null)" in $c*) return 0;; esac; echo "ABORT: $1 is not a pinned build (<dir>-<commit>/bin/<binary> with a commit file starting with the commit)"; exit 3; }
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; PB=$W/pipeline-benchmark
BIN=${BIN:?set BIN=$W/builds/<commit8>/bin/<binary> (an immutable build from scripts/engine_gpu/build_clipper.sbatch)}; pin "$BIN"; FILTERS=${FILTERS:-$W/reference-extra/species_filters}
T=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
export PATH=$W/envs/pbench/bin:$PATH SINGLET_WHITELIST_DIR=$PB/whitelists
run() {   # label, fq, genome (GRCh38-2024-A|GRCm39-2024-A|none), filters (1|0)
  local lab=$1 fq=$2 g=$3 f=$4 gd=()
  [ "$g" != none ] && gd=(--genome-dir $REF/$g/star_2.7.11b --exons $REF/$g/genes/genes.gtf)
  echo "== $lab"
  ( [ "$f" = 1 ] && export SINGLET_SPECIES_FILTERS=$FILTERS; unset SINGLET_REF_BASE
    timeout 90 $BIN $fq "${gd[@]}" --out-prefix $T/o_$lab/ --threads 4 --pipeline --output-format 1pz --engine cpu 2>&1 | grep -a "species-check\|ERROR\|missing\|requires" | cut -c1-300 | head -6 )
}
H=$PB/data/pbmc1k_v3/pbmc1k_v3.1fq; M=$PB/data/neuron1k_v3/neuron1k_v3.1fq
run human_on_human $H GRCh38-2024-A 1
run mouse_on_mouse $M GRCm39-2024-A 1
run human_on_mouse_ref $H GRCm39-2024-A 1
run human_no_filters_env $H GRCh38-2024-A 0
run human_no_genome_dir $H none 1
