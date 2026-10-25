#!/bin/bash
# Build the cell-type capability document (bench/public/CONTRACT.md): artifacts -> compute() -> validate.
# CONTRACT_DIR holds contract.py, cohort.tsv and capabilities/_common.py from the bench-public branch.
#SBATCH --job-name=ctnn-pub
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
: "${JOBS:?comma-separated job ids of the method arrays: name=id,...}" "${CONTRACT_DIR:?}"
A=$WORK/bpub/artifacts/cell-type; mkdir -p $A $WORK/bpub/results
RSS=$A/rss.tsv; : > $RSS
for kv in ${JOBS//[,+]/ }; do n=${kv%%=*}; j=${kv#*=}
  sacct -n -P -j $j -o MaxRSS | sed 's/K$//' | awk -v n=$n 'BEGIN{m=0} $1+0>m{m=$1+0} END{printf "%s\t%.2f\n", n, m/1048576}' >> $RSS
done
COMMIT=$(git -C $SRC rev-parse HEAD)
in_runtime python -u $SRC/scripts/celltype_nn/compete/artifacts.py --compete $D/compete --data $D --out $A --commit $COMMIT --rss $RSS
mkdir -p /tmp/$USER-pub-$SLURM_JOB_ID/capabilities
cp $CONTRACT_DIR/contract.py $CONTRACT_DIR/cohort.tsv /tmp/$USER-pub-$SLURM_JOB_ID/
cp $CONTRACT_DIR/_common.py $SRC/bench/public/capabilities/cell_type.py /tmp/$USER-pub-$SLURM_JOB_ID/capabilities/
in_runtime env PYTHONPATH=$PYTHONPATH:/tmp/$USER-pub-$SLURM_JOB_ID/capabilities python - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "/tmp/$USER-pub-$SLURM_JOB_ID/capabilities")
import cell_type
doc = cell_type.compute(Path("$A"))
Path("$WORK/bpub/results/cell-type.json").write_text(json.dumps(doc, indent=1))
for m in doc["metrics"]:
    print(m["id"], m["singlet"]["value"], [(c["tool"], c["value"]) for c in m.get("competitors", [])], m.get("pass"))
PY
in_runtime python /tmp/$USER-pub-$SLURM_JOB_ID/contract.py validate $WORK/bpub/results/cell-type.json --cohort /tmp/$USER-pub-$SLURM_JOB_ID/cohort.tsv
