#!/bin/bash
#SBATCH --job-name=bpub-selfrep
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Real-data corroboration for the ancestry benchmark: atlas donors whose GEO characteristics record self-reported race or
# ethnicity (docs/cell_annotation.md 5.2), scored with this commit's singlet.celllabels.ancestry on the restricted donor
# depth tables (read in memory on the compute node; scripts/annotate_ancestry_real.py). Only labels, proportions and site
# counts are kept: $BP/artifacts/genetic-ancestry/selfreport_calls.tsv.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__bench-public-cohort-485cf7/bench/public/harness/common.sh
O=$BP/work/ancestry/selfreport; A=$BP/artifacts/genetic-ancestry; mkdir -p $O $A
[ -f "$HOME/.config/singlet-r2.env" ] && source "$HOME/.config/singlet-r2.env"   # R2 read credentials (600-mode file)
export PYTHONPATH=$SRC/python:$SRC/scripts:$WORK/venvs/annotate-extra R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY
# studies with self-reported labels and donor tables, from the validation run
rt python - <<EOF
import json, glob
g = sorted({r["gse"] for f in glob.glob("$WORK/out/annotate_validation/ancestry_final/anc_*.jsonl")
            for r in map(json.loads, open(f)) if r.get("self_reported")})
open("$O/gses.txt", "w").write("\n".join(g) + "\n"); print(g)
EOF
SCR=/tmp/$USER-selfrep-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
rt python -u $SRC/scripts/annotate_ancestry_real.py --gse-list $O/gses.txt --af $WORK/data/annotate/1kgp_superpop_af.parquet \
  --scratch $SCR --out $O/anc.jsonl --labels $WORK/data/annotate/race_labels.json
rt python - <<EOF
import json, pandas as pd
rows = [r for r in map(json.loads, open("$O/anc.jsonl")) if r.get("self_reported")]
pd.DataFrame([{"dataset": "atlas_selfreport", "gse": r["gse"], "gsm": r["gsm"], "donor": r["donor"],
               "self_reported": r["self_reported"], "tool": "singlet", "call": r.get("group"),
               "q_max": r.get("prob"), "n_sites": r.get("n_sites")} for r in rows]).to_csv("$A/selfreport_calls.tsv", sep="\t", index=False)
print(len(rows), "donors with self-report")
EOF
