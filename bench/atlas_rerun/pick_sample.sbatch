#!/bin/bash
#SBATCH --job-name=s6-pick
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# A small single-run human 10x sample that Alpha's data-Worker policy does not withhold (Beta's Worker still bundles that policy).
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/s6
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
in_runtime python - <<PY
import json, pandas as pd
p = json.load(open("$J/alpha-policy.json")); wg, ex = set(p["withheld_studies"]), set(p["excluded_samples"])
s = pd.read_parquet("$W/data/atlas-rerun/samples.parquet")
s = s[(s.organism == "Homo sapiens") & (s.n_runs == 1) & (s.processing_status == "done") & (s.protocol_inferred == "10xv3")
      & (s.reads.between(15e6, 30e6)) & (s.avg_read_length >= 90) & ~s.screen_vdj.astype(bool) & ~s.screen_adt_read_length.astype(bool) & ~s.screen_non_gex_title.astype(bool) & (s.title_lib_type.fillna("") == "")]
s = s[~s.gse_id.isin(wg) & ~s.gsm_id.isin(ex)].drop_duplicates("gse_id")
print(len(s)); print(s[["gsm_id", "gse_id", "reads", "n_runs", "protocol_inferred"]].head(8).to_string())
PY
