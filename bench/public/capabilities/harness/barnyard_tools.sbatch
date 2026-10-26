#!/bin/bash
#SBATCH --job-name=bp-btools
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch barnyard_tools.sbatch <cohort-id> <harness dir>
# After run_singlet_10x.sbatch: prepare the common inputs, then run Scrublet, scDblFinder, SoupX, DecontX (default and with the
# empty-droplet background), each in its own timed process (one thread each: these tools are single-threaded at defaults).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; BP=$W/bench-public; E=$W/envs/ambient
ID=$1; H=$2; RUN=$BP/runs/$ID/singlet; D=$BP/work/$ID; mkdir -p $D/tools
export PATH=$E/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl NUMBA_CACHE_DIR=$W/cache/numba-bp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -I $H/barnyard_prep.py $RUN $D
NC=$(python -I -c "import json; print(json.load(open('$D/pool.json'))['n_cells'])")
# CellBender needs only $D/raw: the documented minimal call and the authors' recommended call (expected cells = called cells, 6x droplets)
sbatch $H/cellbender.sbatch $ID cellbender || echo "[warn] could not submit CellBender"
sbatch $H/cellbender.sbatch $ID cellbender_rec --expected-cells $NC --total-droplets-included $((NC*6)) || echo "[warn] could not submit CellBender"
T() { n=$1; shift; echo "== $n"; /usr/bin/time -v -o $D/tools/$n.time "$@" 2>&1 | tail -15 || true; grep -E "Elapsed \(wall|Maximum resident" $D/tools/$n.time; }
T scrublet python -I $H/scrublet_run.py $D
for t in scdblfinder soupx decontx decontx_bg; do T $t Rscript $H/barnyard_tools.R $t $D; done
lscpu | sed -n 's/^Model name: *//p' > $D/tools/node.txt; hostname >> $D/tools/node.txt
ls -la $D/tools
