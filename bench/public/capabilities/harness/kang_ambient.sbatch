#!/bin/bash
#SBATCH --job-name=bp-kambient
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch kang_ambient.sbatch <cohort-id> <harness dir>
# Ambient competitors on a genotyped lane (real-tissue truth from donor genotypes): droplet inputs, SoupX, DecontX (default and with
# background), and the two CellBender invocations (submitted; they need only $D/raw). Then run harness/artifacts.sbatch kang_ambient.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; BP=$W/bench-public; E=$W/envs/ambient
ID=$1; H=$2; RUN=$BP/runs/$ID/singlet; D=$BP/work/$ID; mkdir -p $D/tools
export PATH=$E/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -I $H/kang_prep.py $RUN $D
NC=$(python -I -c "import json; print(json.load(open('$D/pool.json'))['n_cells'])")
sbatch $H/cellbender.sbatch $ID cellbender || echo "[warn] could not submit CellBender"
sbatch $H/cellbender.sbatch $ID cellbender_rec --expected-cells $NC --total-droplets-included $((NC*6)) || echo "[warn] could not submit CellBender"
T() { n=$1; shift; echo "== $n"; /usr/bin/time -v -o $D/tools/$n.time "$@" 2>&1 | tail -8 || true; grep -E "Elapsed \(wall|Maximum resident" $D/tools/$n.time; }
for t in soupx decontx decontx_bg; do T $t Rscript $H/barnyard_tools.R $t $D; done
ls -la $D/tools
