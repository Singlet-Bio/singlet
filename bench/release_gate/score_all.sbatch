#!/bin/bash
#SBATCH --job-name=rg-score
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Release gate: score every capability whose artifacts exist into $RG/results/<cap>.json (idempotent; each capability is independent).
# Trees: counts family = bench/cellcall-independent-truth (extract+compute of the 10-26 result), ambient = bench/public-ambient-doublets,
# atac/multiome/spatial = the pinned commit's bench/public. Binary identity is patched to the pin in each tree.
source /mnt/projects/debruinz_project/debruinz/singlet/release-gate/jobs/rg_common.sh
PY=$W/envs/pbench/bin/python; ATACPY=$W/envs/atacbench/bin/python; RES=$RG/results; mkdir -p $RES
COMMIT=$(cat $RG/bin/$PIN8/commit); SHA=$(sha256sum $RG/bin/$PIN8/singlet-cpu | cut -d' ' -f1)
BARE=$W/repos/singlet-py.git
PIN=$RG/src/pin; if [ ! -d $PIN/bench ]; then mkdir -p $PIN; git -C $BARE archive $COMMIT bench/public bench/atac_spatial | tar -x -C $PIN; fi
[ -d $PIN/bench/atac_spatial ] || git -C $BARE archive $COMMIT bench/atac_spatial | tar -x -C $PIN
sc() { # tree cap artifact_dir python
  echo "=== $2"; [ -d "$3" ] || { echo "no artifacts $3"; return; }
  (cd $1 && ${4:-$PY} bench/public/run_capability.py $2 $3 --out $RES/$2.json) || echo "[score] $2 FAILED"
}
# ambient identity -> pin (the Python module stamps what it is told)
AM=$RG/ambient; sed -i -e "s|\"commit\": \"02aa9b9[0-9a-f]*\"|\"commit\": \"$COMMIT\"|" -e "s|\"binary\": \"pipeline-benchmark/bin/singlet-campaign\"|\"binary\": \"release-gate/bin/$PIN8/singlet-cpu\"|" -e "s|\"binary_sha256\": \"261d3a50[0-9a-f]*\"|\"binary_sha256\": \"$SHA\"|" $AM/bench/public/capabilities/_adcommon.py
for c in gex-counts spliced-unspliced junctions-psi cell-calling; do sc $RG/src/counts $c $RG/artifacts/$c; done
sc $AM ambient-rna $AM/artifacts/ambient-rna
# atac family: provenance first (binary hash, commit), then the compute modules
AA=$RG/artifacts-atacsp
if [ -d $AA ]; then
  $ATACPY $PIN/bench/atac_spatial/make_provenance.py $AA $COMMIT $RG/bin/$PIN8/singlet-cpu 2>&1 | tail -3 || true
  for c in atac multiome spatial; do sc $PIN $c $AA/$c $ATACPY; done
fi
ls -la $RES
