#!/bin/bash
#SBATCH --job-name=s6-dbg
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
REPO=${REPO:-$W/src/singlet-py/bb__prod__campaign}
D=$(ls -d $W/jobs/s6/e2e_GSM6470208/out/*/GSM6470208)
in_runtime env PYTHONPATH=$REPO/python python - <<PY
import traceback
from singlet.s1.legacy import decode_1pz_bytes
import singlet; print("singlet at", singlet.__file__)
for n in ("exon_counts.1pz", "intron_counts.1pz"):
    b = open("$D/" + n, "rb").read()
    print(n, len(b), b[:16])
    try:
        c = decode_1pz_bytes(b)
        print(" shape", c.m, c.n, "nnz", c.nnz, "check", c.check_canonical(), "sum", int(c.data.sum()))
    except Exception:
        traceback.print_exc()
PY
