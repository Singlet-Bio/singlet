#!/bin/bash
#SBATCH --job-name=rg-cscore
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --array=0-7%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Release gate: extract the counts-family artifacts from the new singlet-public runs (competitors untouched), array task = dataset.
# Code: branch bench/cellcall-independent-truth (the scored 10-26 result), archived from the bare repo, with the binary identity
# in _pubcommon.py patched to the pinned binary so the result does not claim 02aa9b9 / 261d3a50.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/release-gate/jobs/rg_common.sh
SRC=$RG/src/counts
if [ ! -d $SRC/bench ]; then
  mkdir -p $SRC.tmp.$$ && git -C $W/repos/singlet-py.git fetch -q "$(git -C $W/repos/singlet-py.git config remote.origin.url)" +refs/heads/bench/cellcall-independent-truth:refs/rg/counts
  git -C $W/repos/singlet-py.git archive refs/rg/counts bench/public | tar -x -C $SRC.tmp.$$
  SHA=$(sha256sum $BIN | cut -d' ' -f1); COMMIT=$(cat $(dirname $BIN)/commit)
  sed -i -e "s|^SINGLET_COMMIT = .*|SINGLET_COMMIT = \"$COMMIT\"  # pinned release-gate candidate (prod/campaign)|" \
         -e "s|^SINGLET_BINARY_SHA256 = .*|SINGLET_BINARY_SHA256 = \"$SHA\"|" \
         -e "s|\"binary\": \"singlet-campaign\"|\"binary\": \"singlet-cpu\"|" \
         -e "s|\"build_flags\": \"PGO, -O3 -DNDEBUG, -march=x86-64-v3\"|\"build_flags\": \"non-PGO, -O3 -DNDEBUG, -march=x86-64-v3\"|" $SRC.tmp.$$/bench/public/capabilities/_pubcommon.py
  mv $SRC.tmp.$$ $SRC 2>/dev/null || rm -rf $SRC.tmp.$$
fi
ART=$RG/artifacts; mkdir -p $ART
PY=$W/envs/pbench/bin/python
DSS=(pbmc1k_v3 pbmc1k_v2 neuron1k_v3 hgmm1k_v3 pbmc1k_cite_v3 nuclei2k_v2 pbmc5p_v1 kang2018_ctrl)
ID=${DSS[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
CAPS="gex-counts spliced-unspliced junctions-psi cell-calling"; [ $ID = kang2018_ctrl ] && CAPS=cell-calling   # the Kang pool is a cell-calling dataset only (no alevin-fry run)
for CAP in $CAPS; do
  echo "[extract] $CAP $ID"; mkdir -p $ART/$CAP
  $PY -P $SRC/bench/public/capabilities/extract/run_extract.py $CAP $ID --pb $PB --out $ART/$CAP
  flock $ART/.meta.lock $PY -P $SRC/bench/public/capabilities/extract/run_extract.py meta $ID --pb $PB --out $ART/$CAP
done
