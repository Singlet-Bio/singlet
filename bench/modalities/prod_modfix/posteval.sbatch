#!/bin/bash
#SBATCH --job-name=mf-posteval
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu,short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Scores the finished validate.sbatch runs: modfix_compare (gene level vs singlet-prod-t8), hgmm ambient / doublets /
# barnyard (hgmm_e2e.py), and the sj_features STARsolo-comparable junction set vs STARsolo SJ on PBMC 1k v3.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; RES=$PB/results/modfix
SRC=$W/src/singlet-py/prod__modality-fixes; S=$SRC/bench/modalities/prod_modfix
export PATH=$W/envs/pbench/bin:$PATH
git -C $SRC fetch -q origin 2>/dev/null; git -C $W/repos/singlet-py.git fetch -q --update-head-ok origin +refs/heads/prod/modality-fixes:refs/heads/prod/modality-fixes; git -C $SRC reset -q --hard refs/heads/prod/modality-fixes
for r in pbmc1k_v3/singlet-modfix-t8 pbmc5p_v1/singlet-modfix-t8 pbmc5p_v1/singlet-modfix-t8-noadapter nuclei2k_v2/singlet-modfix-t8 neuron1k_v3/singlet-modfix-t8 hgmm1k_v3/singlet-modfix-t8; do
  ds=${r%%/*}; tag=${r##*/}
  python -P $S/modfix_compare.py $PB/runs/$ds/singlet-prod-t8 $PB/runs/$r --json $RES/$ds.$tag.modfix_compare.json > /dev/null 2>$RES/$ds.$tag.err || cat $RES/$ds.$tag.err
  python - <<EOF
import json; r = json.load(open("$RES/$ds.$tag.modfix_compare.json")); g = r.get("genes", {})
print("$r", "cells", r["called_cells"], "star", r["star"]["ref"].get("Number of splices: Total"), "->", r["star"]["new"].get("Number of splices: Total"),
      "uniq", r["star"]["ref"].get("Uniquely mapped reads %"), "->", r["star"]["new"].get("Uniquely mapped reads %"))
print("   genes:", {k: g.get(k) for k in ("identical", "total_umi_ratio", "median_umi_ref", "median_umi_new", "median_genes_ref", "median_genes_new", "pearson_log_gene_totals", "genes_ge100", "genes_changed_gt10pct", "genes_changed_gt2x")})
print("   up:", g.get("top_up", [])[:5]); print("   down:", g.get("top_down", [])[:5])
o = r.get("outputs", {}); print("   outputs:", {k: o.get(k) for k in ("n_public", "n_private", "bytes_public", "bytes_private")})
print("   prov:", json.dumps(r.get("provenance_extra"))[:400])
EOF
done
$W/envs/ambient/bin/python -P $S/hgmm_e2e.py $PB/runs/hgmm1k_v3/singlet-modfix-t8 $RES/hgmm1k_v3.e2e_ambient_doublets.json
python - <<EOF
import gzip, json
new = "$PB/runs/pbmc1k_v3/singlet-modfix-t8/sj_features.tsv.gz"
solo = "$W/modality-bench/sj_nonhost/pbmc1k_v3/starsolo/Solo.out/SJ/raw/features.tsv"
pas, allj, mol = set(), set(), {}
with gzip.open(new, "rt") as f:
    next(f)
    for l in f:
        x = l.rstrip("\n").split("\t"); k = (x[1], int(x[2]), int(x[3])); allj.add(k); mol[k] = int(x[11])
        if x[10] == "1": pas.add(k)
so = set()
for l in open(solo):
    x = l.split("\t"); so.add((x[0], int(x[1]), int(x[2])))
def jac(a, b): return len(a & b) / max(len(a | b), 1)
R = dict(singlet_all=len(allj), singlet_star_sj_pass=len(pas), starsolo_features=len(so),
         jaccard_all=jac(allj, so), jaccard_pass=jac(pas, so), pass_in_starsolo=len(pas & so), starsolo_in_pass=len(so & pas),
         singletons_all=sum(1 for k in allj if mol[k] == 1), singletons_pass=sum(1 for k in pas if mol[k] == 1))
print("sj_features vs STARsolo SJ features (PBMC 1k v3, unclipped STARsolo):", R)
json.dump(R, open("$RES/pbmc1k_v3.sj_features_vs_starsolo.json", "w"), indent=1)
EOF
