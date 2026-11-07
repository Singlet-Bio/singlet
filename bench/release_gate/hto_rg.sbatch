#!/bin/bash
#SBATCH --job-name=rg-hto
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Release gate, hto (#87 re-measure): compile hto_demux_cli against the PINNED commit's include/singlet/pileup/hto_demux.h, call the Howitt
# 2023 captures with the held-out (leave-one-dataset-out) settings of bench/adt, rescore against the existing competitors' calls.
# Competitor/truth artifacts are copied from bpub/artifacts/hto (they do not depend on the binary). Output: $RG/results/hto.json.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/release-gate/jobs/rg_common.sh
COMMIT=$(cat $RG/bin/$PIN8/commit); BARE=$W/repos/singlet-py.git
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $COMMIT include bench/adt bench/public | tar -x -C $SCR/src
# bench/adt on the pin is the #87 merge; bench/public/capabilities of the pin has no hto module, take the gate branch's contract + _common
export PATH=$W/envs/pbench/bin:$PATH
g++ -O2 -std=c++17 -I$SCR/src/include -o $SCR/hto_demux_cli $SCR/src/bench/adt/hto_demux_cli.cpp
ART=$RG/artifacts/hto; mkdir -p $ART $RG/results; cp -n $W/bpub/artifacts/hto/* $ART/ || true
rm -f $ART/hto_*_singlet.parquet $ART/resources_hto_*_singlet.json
PY=$W/envs/adt-bench/bin/python
export PYTHONPATH=$RG/jobs/shim:$SCR/src/bench/adt
$PY -P $SCR/src/bench/adt/hto_public_run.py --zenodo $W/data/adt_bench/zenodo_8304003 --exe $SCR/hto_demux_cli --artifacts $ART \
  --selection $SCR/src/bench/adt/results/hto_lodo_selection.json --commit $COMMIT
# score (module: bench/adt/public_hto.py needs the public capability helpers: use the gate generator's tree)
export PYTHONPATH=$PYTHONPATH:$SCR/src/bench/public:$SCR/src/bench/public/capabilities
$PY -P $SCR/src/bench/adt/public_hto.py $ART $RG/results/hto.json
# public_hto takes the commit from the first resources file it reads (a competitor's); the Singlet side here is the pin's header
$PY - <<PYEOF
import json
p = "$RG/results/hto.json"; d = json.load(open(p))
d["singlet"] = {"version": "1.0.0", "commit": "$COMMIT", "branch": "prod/campaign",
                "binary": "hto_demux_cli (bench/adt/hto_demux_cli.cpp compiled against the pin's include/singlet/pileup/hto_demux.h)"}
json.dump(d, open(p, "w"), indent=1)
PYEOF
echo "[hto] artifacts in $ART"; ls $ART | grep -c singlet
