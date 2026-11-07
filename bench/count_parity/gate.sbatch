#!/bin/bash
#SBATCH --job-name=cp-gate
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Standing STARsolo gate (bench/count_parity/README.md). Runs a Singlet binary on the gate datasets, scores every counting axis
# against frozen STARsolo 2.7.11b outputs with paired bootstrap CIs (parity_score.py) and writes the gate record named by the
# counting-source hash of the commit the binary was built from. Exit 1 if any axis is worse than STARsolo beyond its CI.
#   sbatch --export=ALL,BIN=<.../bin2/<commit8>/singlet-gpu>,BR=<branch>[,DATASETS="hgmm1k_v3 hgmm1k_v2 pbmc1k_cite_v3"][,ENGINE=gpu] gate.sbatch
# The binary's directory holds commit-gpu (gd-build2.sbatch writes it). Record: $W/count-parity/gate/<hash16>.json. Commit it as
# bench/count_parity/gate_results/<hash16>.json so that gate_check.py (CI) passes. ENGINE=gpu adds --engine gpu and the GPU pileup.
set -euo pipefail
: "${BIN:?}" "${BR:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
DATASETS=${DATASETS:-"hgmm1k_v3 hgmm1k_v2 pbmc1k_cite_v3"}
COMMIT=$(cat "$(dirname "$BIN")/commit-gpu"); BARE=$W/repos/singlet-py.git; URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" "+refs/heads/$BR:refs/cp/gate" ) 9>$BARE/.gn-fetch.lock
git -C $BARE cat-file -e "$COMMIT" || { echo "commit $COMMIT of the binary is not in $BR"; exit 2; }
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p "$SCR/src"; trap 'rm -rf "$SCR"' EXIT
git -C $BARE archive "$COMMIT" | tar -x -C "$SCR/src"
PY=$W/envs/pbench/bin/python; CP=$SCR/src/bench/count_parity
HASH=$($PY -I "$CP/gate_check.py" hash "$SCR/src"); H16=${HASH:0:16}
G=$W/count-parity/gate/$H16; mkdir -p "$G"
echo "gate commit $COMMIT source hash $HASH bin $(sha256sum "$BIN" | cut -c1-64) node $(hostname)"
export SINGLET_STALL_TIMEOUT=7200
EXTRA=""
if [ "${ENGINE:-cpu}" = gpu ]; then EXTRA="--engine gpu"; export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=1; fi
rc=0
for DS in $DATASETS; do
  case $DS in hgmm*) SPECIES=GRCh38-GRCm39-barnyard; KIND=barnyard ;; *) SPECIES=GRCh38-2024-A; KIND=cite ;; esac
  O=$G/$DS; rm -rf "$O"; mkdir -p "$O"
  ( cd "$SCR" && "$BIN" "$PB/data/$DS/$DS.1fq" --ref-base $REF --genome-dir $REF/$SPECIES/star_2.7.11b --exons $REF/$SPECIES/genes/genes.gtf \
      --out-prefix "$O/" --threads "$SLURM_CPUS_PER_TASK" --pipeline --output-format mtx --raw-matrix $EXTRA > "$O/run.log" 2>&1 )
  grep -P '\tgene\t' $REF/$SPECIES/genes/genes.gtf > "$SCR/genes.gtf"
  H5=$(ls $PB/data/$DS/*filtered_feature_bc_matrix.h5 | head -1)
  $PY -I "$CP/parity_score.py" --singlet "$O" --starsolo "$W/count-parity/starsolo-ref/$DS" --cr "$H5" --gtf "$SCR/genes.gtf" \
      --kind $KIND --out "$O/parity.json" || rc=1
  # the record keeps the scores and the logs; the matrices are not needed after scoring
  find "$O" -mindepth 1 -maxdepth 1 ! -name parity.json ! -name run.log ! -name pileup_stats.json -exec rm -rf {} +
done
$PY -I "$CP/gate_record.py" "$G" "$COMMIT" "$HASH" "$DATASETS" "${ENGINE:-cpu}"
[ "$rc" = 0 ] && grep -q '"gate": "PASS"' "$G.json"
