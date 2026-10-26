#!/bin/bash
#SBATCH --job-name=adt-fast
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Check a commit of the counter on real data without touching the shared worktree: git archive into scratch, rerun Singlet
# on the public ADT datasets, compare with the outputs of the previous code.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
REV=${REV:-refs/adt/check}
SCR=/tmp/$USER-fast-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $WORK/repos/singlet-py.git fetch -q origin "+refs/heads/claude/adt-modality-0637a6:refs/adt/check" || true
git -C $WORK/repos/singlet-py.git archive $REV | tar -x -C $SCR/src
cp $WORK/src/singlet-py/claude__adt-modality-0637a6/python/singlet/_pz*.so $SCR/src/python/singlet/ 2>/dev/null || true
export PATH=$WORK/envs/adt/bin:$PATH LD_LIBRARY_PATH=$WORK/envs/adt/lib PYTHONPATH=$SCR/src/python HOME=$WORK/cache/condahome
OUT=$WORK/state/adt/fastcheck; mkdir -p $OUT
cp $WORK/bpub/artifacts/adt/adt_*_panel.tsv $OUT/
for D in pbmc1k_cite_v3 pbmc10k_cite_v3; do
  python -P $SCR/src/bench/adt/adt_public_run.py --dataset $D --tool singlet --data $WORK/data/adt_bench --out $OUT --scratch $SCR --threads 8
done
python - <<'PY'
import pandas as pd
W = "/mnt/projects/debruinz_project/debruinz/singlet/bpub/artifacts/"
for d in ("pbmc1k_cite_v3", "pbmc10k_cite_v3"):
    a = pd.read_parquet(W + f"adt/adt_{d}_singlet.parquet").set_index(["barcode", "feature"])["count"]
    b = pd.read_parquet(W + f"../../state/adt/fastcheck/adt_{d}_singlet.parquet").set_index(["barcode", "feature"])["count"]
    j = pd.concat([a, b], axis=1, keys=["old", "new"]).fillna(0)
    print(d, "entries", len(j), "identical", bool((j.old == j.new).all()), "umis", j.old.sum(), j.new.sum())
PY
