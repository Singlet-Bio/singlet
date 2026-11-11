#!/bin/bash
#SBATCH --job-name=cc-cplex
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
O=$W/bench/calls-sota/cell_calling/heldout_cellplex10k; mkdir -p $O
SRC=$W/bench/calls-sota/cell_calling
URL=https://cf.10xgenomics.com/samples/cell-exp/6.0.0/SC3_v3_NextGem_DI_CellPlex_Human_PBMC_10K_Multiplex/SC3_v3_NextGem_DI_CellPlex_Human_PBMC_10K_Multiplex_count_raw_feature_bc_matrix.h5
[ -s $O/raw.h5 ] || curl -sSf -o $O/raw.h5 $URL
sha256sum $O/raw.h5 $SRC/cellcall_v21.py
PY=$W/envs/gpool/venv/bin/python
NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/$USER-nb-$SLURM_JOB_ID $PY -P $SRC/heldout_prep.py $O/raw.h5 $SRC/cellcall_v21.py $O
$W/envs/ambient/bin/Rscript $SRC/emptydrops_ref.R $O
$PY -P - <<PYEOF
import pandas as pd, numpy as np
O="$O"
t=pd.read_csv(f"{O}/table_nocomp.tsv.gz",sep="\t",dtype={"barcode":str})
e=pd.read_csv(f"{O}/emptydrops.tsv",sep="\t")
assert len(e)==len(t) and (e.total.values==t.umi_total.values).all(), "row mismatch"
t["edcr"]=e.edcr.values
t.to_csv(f"{O}/table.tsv.gz",sep="\t",index=False)
print("table written", len(t))
PYEOF
$PY -P $SRC/score_tables.py $O/heldout_scores.json cellplex10k=$O/table.tsv.gz:H
