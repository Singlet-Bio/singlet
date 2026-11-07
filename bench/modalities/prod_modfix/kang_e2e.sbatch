#!/bin/bash
#SBATCH --job-name=mf-kang
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# End to end on the Kang 8-donor pool (docs/modality-benchmarks/genotype-pool.md) with bin/singlet-modfix and production
# defaults: --pipeline, no --snps (default panel), no --n-donors (auto K), default BAM sort (disk). Scores donor demux
# + genotype doublets (bench/modalities/demux_fix/eval_demux.py, vireo alongside) and the expression doublet score.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; G=$W/modality-bench/genotype_pool; PB=$W/pipeline-benchmark
SRC=$W/src/singlet-py/prod__modality-fixes; BIN=$PB/bin/singlet-modfix
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
PY=$W/envs/gpool/venv/bin/python
O=$G/singlet_modfix; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cp $G/kang21.1fq $SCR/s.1fq; cd $SCR
export SINGLET_STALL_TIMEOUT=10800
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
  --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline \
  --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "Elapsed|Maximum resident" $O/time.txt
grep -E "Default SNP|\[demux|donor|\[doublet\]|auto-K|K=|\[mt_|\[ambient\]|5' prefix" $O/run.log | head -40
R=$SRC/bench/modalities/prod_modfix/results; mkdir -p $R
cd $SRC/bench/modalities/demux_fix
$PY eval_demux.py $G/truth/GSE96583_batch2.total.tsne.df.tsv.gz $R/kang_e2e_demux.json \
  singlet_modfix=$O/donor/donor_assignments.tsv vireo=$G/ref_k8/vireo/donor_ids.tsv > /dev/null
$PY - <<EOF
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
r = json.load(open("$R/kang_e2e_demux.json"))
for k, v in r.items():
    print(k, {x: (round(v[x], 4) if isinstance(v[x], float) else v[x]) for x in
          ("ari_all_singlets", "ari_assigned_only", "n_donors_found", "dbl_auroc_genotype", "dbl_auprc_genotype",
           "dbl_recall_at_call", "dbl_precision_at_call", "singlets_called_doublet", "unassigned_rate")})
t = pd.read_csv("$G/truth/GSE96583_batch2.total.tsne.df.tsv.gz", sep="\t", index_col=0)
t = t[t["stim"] == "ctrl"].copy(); t.index = [i.split("-")[0] for i in t.index]
d = pd.read_csv("$O/doublet_scores.tsv", sep="\t"); d.index = [b.split("-")[0] for b in d.barcode]
d = d.join(t[["multiplets"]], how="inner"); d = d[d.multiplets.isin(["singlet", "doublet"])]
y = (d.multiplets == "doublet").values; s = d.doublet_score.values.astype(float); c = d.is_doublet.astype(str).isin(["1", "True", "true"]).values
k = int(round(y.mean() * len(y))); top = np.argsort(-s, kind="stable")[:k]
res = dict(cells=len(d), truth_rate=float(y.mean()), auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s),
           recall_at_truth_rate=float(y[top].sum() / y.sum()), native_calls=int(c.sum()), native_rate=float(c.mean()),
           native_recall=float((c & y).sum() / y.sum()), native_precision=float((c & y).sum() / max(c.sum(), 1)))
print("expression doublets:", {a: round(b, 4) if isinstance(b, float) else b for a, b in res.items()})
json.dump(res, open("$R/kang_e2e_expr_doublets.json", "w"), indent=1)
EOF
ls -la $O $O/private $O/private/* $O/donor 2>/dev/null | head -60
